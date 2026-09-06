from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False
    # Persist the KV/GDN prefix cache to disk and restore it per session on demand.
    # kv_persist_dir None -> default_kv_cache_dir(). kv_persist_max_gb caps the snapshot
    # blob (a save over budget is skipped); kv_persist_max_age_h makes older snapshots
    # load as cold. See scheduler/cache_persist.py.
    kv_persist: bool = False
    kv_persist_dir: str | None = None
    kv_persist_max_gb: int | None = None
    kv_persist_max_age_h: float | None = None

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/freetoken_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/freetoken_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/freetoken_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
