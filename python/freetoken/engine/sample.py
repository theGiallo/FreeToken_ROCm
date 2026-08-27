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
    # Per-row presence penalty (>=0) and, when any is nonzero, the GPU presence mask
    # built from each request's already-generated token ids at prepare() time:
    # logits[row, tok] -= presence_penalties[row] whenever the row's mask is 1 at `tok`.
    # presence_penalties of 0 in a row mean that row is untouched (mask ignored).
    presence_penalties: torch.Tensor | None = None
    presence_mask: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
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
        has_penalty = any(v > 0.0 for v in pp)

        if all(p.is_greedy for p in params) and not has_penalty:
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)

        args = BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

        if has_penalty:
            args.presence_penalties = make_device_tensor(pp, torch.float32, self.device)
            args.presence_mask = self._presence_mask(batch)
        return args

    def _presence_mask(self, batch: Batch) -> torch.Tensor:
        """Boolean [B, V] GPU mask: mask[i, tok]=1 iff `tok` already appears among
        request i's sequence tokens (prompt + generated so far). Used to cut the logits
        of already-seen tokens by presence_penalty[i] before softmax."""
        import torch as _t

        mask = _t.zeros((len(batch.reqs), self.vocab_size), dtype=_t.bool, device=self.device)
        for i, req in enumerate(batch.reqs):
            ids = req.input_ids
            if ids.numel() == 0:
                continue
            # Deduplicate then scatter 1s at the distinct present token ids.
            uniq = _t.unique(ids.to(self.device, non_blocking=True))
            mask[i].scatter_(0, uniq.to(_t.long), True)
        return mask

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.presence_penalties is not None and args.presence_mask is not None:
                # logits [B,V]; mask [B,V] bool. Subtract penalty per present token.
                pen = args.presence_penalties.unsqueeze(-1)
                logits = logits - torch.where(args.presence_mask, pen, 0.0)
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
