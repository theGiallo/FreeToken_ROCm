from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    min_p: torch.Tensor | None = None
    # Per-row logit penalties (presence/frequency/repeat) and, when any is active, the
    # per-row occurrence-count matrix built from each request's already-seen token ids at
    # prepare() time (counts[i, tok] = how many times `tok` appears in request i's
    # sequence). Applied in Sampler.sample before temperature/softmax/top-k/top-p:
    #   presence:  logits -= presence_penalty     where counts>0
    #   frequency: logits -= frequency_penalty*N  where N=counts
    #   repeat:    logits /=repeat if >0 else *=repeat, where counts>0
    # A penalty tensor of 0 (or repeat 1.0) in a row leaves that row untouched.
    presence_penalties: torch.Tensor | None = None
    frequency_penalties: torch.Tensor | None = None
    repeat_penalties: torch.Tensor | None = None
    counts: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def _apply_min_p(probs: torch.Tensor, min_p: torch.Tensor | float | None) -> torch.Tensor:
    """Minimum-p probability filter: drop any token with prob < min_p * row_max (per row),
    zeroing its mass so the downstream draw cannot select it. min_p of 0 (or None) is a no-op.
    Renormalization is handled by the downstream top-k/top-p renorm / inverse-CDF draw."""
    if min_p is None:
        return probs
    if not isinstance(min_p, torch.Tensor):
        min_p = torch.full((probs.shape[0],), float(min_p), dtype=torch.float32, device=probs.device)
    mp = min_p.float().to(probs.device)
    if (mp > 0.0).any():
        thr = mp.unsqueeze(-1) * probs.max(dim=-1, keepdim=True).values
        return probs.where(probs >= thr, 0.0)
    return probs


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
    min_p: torch.Tensor | float | None = None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    probs = _apply_min_p(probs, min_p)
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]

        pp = [max(0.0, p.presence_penalty) for p in params]
        fp = [max(0.0, p.frequency_penalty) for p in params]
        rp = [max(1.0, p.repeat_penalty) for p in params]
        mp = [min(max(0.0, p.min_p), 1.0) for p in params]
        # repeat is multiplicative: 1.0 and below is "disabled" (no change).
        has_penalty = any(v > 0.0 for v in pp) or any(v > 0.0 for v in fp) or any(v > 1.0 for v in rp)
        has_min_p = any(v > 0.0 for v in mp)

        if all(p.is_greedy for p in params) and not has_penalty and not has_min_p:
            return BatchSamplingArgs(temperatures=None)

        all_greedy = all(p.is_greedy for p in params)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        # Penalized greedy (or min_p) must still take the argmax path, so keep
        # temperatures None when every row is greedy (sample() then uses argmax).
        temperatures = None if all_greedy else make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if all_greedy:
            top_p = None
        elif any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)

        args = BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

        if has_min_p:
            args.min_p = make_device_tensor(mp, torch.float32, self.device)
        if has_penalty:
            args.presence_penalties = make_device_tensor(pp, torch.float32, self.device)
            args.frequency_penalties = make_device_tensor(fp, torch.float32, self.device)
            args.repeat_penalties = make_device_tensor(rp, torch.float32, self.device)
            args.counts = self._counts(batch)
        return args

    def _counts(self, batch: Batch) -> torch.Tensor:
        """Integer [B, V] occurrence matrix: counts[i, tok] = how many times `tok` appears
        among request i's sequence tokens (prompt + generated so far). Presence/repeat act
        on counts>0; frequency acts on the count directly."""
        import torch as _t

        counts = _t.zeros((len(batch.reqs), self.vocab_size), dtype=_t.int32, device=self.device)
        for i, req in enumerate(batch.reqs):
            ids = req.input_ids
            if ids.numel() == 0:
                continue
            ids = ids.to(self.device, non_blocking=True).to(_t.long)
            bc = _t.bincount(ids, minlength=self.vocab_size)
            counts[i] = bc.to(_t.int32)
        return counts

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.counts is not None:
                # logits [B,V]; counts [B,V] int. Repeat: multiplicative per distinct token.
                # presence/frequency: additive using counts>0 / counts directly.
                seen = args.counts > 0
                if args.repeat_penalties is not None:
                    # >0 -> /rp ; <=0 -> *rp  (rp >= 1). rp==1 rows unchanged (div by 1).
                    rp = args.repeat_penalties.unsqueeze(-1)
                    dev_log = logits.to(torch.float32)
                    pos = dev_log > 0
                    new = torch.where(pos, dev_log / rp, dev_log * rp)
                    logits = torch.where(seen, new, dev_log)
                if args.presence_penalties is not None:
                    logits = logits - torch.where(seen, args.presence_penalties.unsqueeze(-1), 0.0)
                if args.frequency_penalties is not None:
                    logits = logits - args.frequency_penalties.unsqueeze(-1) * args.counts
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p, args.min_p)
