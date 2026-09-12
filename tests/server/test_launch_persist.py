"""KV-persist launch surface: --kv-persist* arg defaults/flags and the SIGTERM->graceful
shutdown handler that lets a snapshot land before the engine tears down."""
from __future__ import annotations

import os
import signal
import sys
import tempfile
import threading
import time

from freetoken.server.args import parse_args


def test_kv_persist_args_default_off_and_parse_flags():
    args, _ = parse_args(
        ["--model-path", "/fake/model", "--dtype", "float16", "--dummy-weight"]
    )
    assert args.kv_persist is False
    assert args.kv_persist_dir is None
    assert args.kv_persist_max_gb is None
    assert args.kv_persist_max_age_h is None

    args2, _ = parse_args(
        [
            "--model-path", "/fake/model", "--dtype", "float16", "--dummy-weight",
            "--kv-persist", "--kv-persist-dir", "/tmp/kvp",
            "--kv-persist-max-gb", "48", "--kv-persist-max-age-h", "7.5",
        ]
    )
    assert args2.kv_persist is True
    assert args2.kv_persist_dir == "/tmp/kvp"
    assert args2.kv_persist_max_gb == 48
    assert args2.kv_persist_max_age_h == 7.5


def test_sigterm_handler_installed_raises_keyboard_interrupt():
    if not hasattr(signal, "SIGTERM"):
        return  # non-POSIX platform; no signal module SIGTERM

    from freetoken.server.launch import _install_sigterm_keyboardinterrupt

    original = signal.getsignal(signal.SIGTERM)
    try:
        _install_sigterm_keyboardinterrupt()
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        try:
            handler(signal.SIGTERM, None)  # noqa: F841 -- must raise
            raised = False
        except KeyboardInterrupt:
            raised = True
        assert raised
    finally:
        signal.signal(signal.SIGTERM, original)


def test_shutdown_signal_guard_ignores_second_signal():
    """A second ^C or the supervisor's SIGTERM during a snapshot save must be inert: once
    the graceful path starts, signals can no longer produce a KeyboardInterrupt."""
    if not hasattr(signal, "SIGTERM"):
        return  # non-POSIX platform; no signal module SIGTERM

    from freetoken.server.launch import _ignore_signals_during_shutdown

    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        _ignore_signals_during_shutdown()
        signal.raise_signal(signal.SIGINT)  # noqa: F841 -- must not raise
        signal.raise_signal(signal.SIGTERM)  # noqa: F841 -- must not raise
    finally:
        for sig, handler in original.items():
            signal.signal(sig, handler)


class _ReapProc:
    """Stand-in mp.Process for the backstop tests: alive until .dead is set, and it records
    how many times the reaper gave up and SIGKILLed it (real kill() is fatal, so we can
    never let that pass in a test process)."""

    def __init__(self) -> None:
        self.dead = False
        self.kill_calls = 0

    def is_alive(self) -> bool:
        return not self.dead

    def join(self, timeout=None) -> None:
        time.sleep(0.01)

    def kill(self) -> None:
        self.kill_calls += 1
        self.dead = True


def test_reap_kills_a_worker_that_ignores_sigterm():
    from freetoken.server.api_server import _reap_backend_workers

    proc = _ReapProc()
    _reap_backend_workers([proc], timeout=0.1)
    assert proc.kill_calls == 1


def test_reap_waits_out_an_active_kv_save():
    """While a .saving marker exists the backstop must NOT fire -- the write is being
    completed deliberately. It returns only after the marker clears and the worker exits."""
    from freetoken.server.api_server import _reap_backend_workers

    with tempfile.TemporaryDirectory() as save_dir:
        marker = os.path.join(save_dir, ".saving.0")
        with open(marker, "w") as f:
            f.write("pid")
        proc = _ReapProc()
        done = []

        def reap() -> None:
            _reap_backend_workers([proc], timeout=0.1, save_dir=save_dir)
            done.append(True)

        t = threading.Thread(target=reap)
        t.start()
        time.sleep(0.4)  # five times the 0.1s budget: the marker must have held it
        assert t.is_alive() and proc.kill_calls == 0

        os.remove(marker)
        proc.dead = True  # the save finished; the worker exits on its own
        t.join(timeout=5)
    assert done and proc.kill_calls == 0
