"""KV-persist launch surface: --kv-persist* arg defaults/flags and the SIGTERM->graceful
shutdown handler that lets a snapshot land before the engine tears down."""
from __future__ import annotations

import os
import signal
import sys

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
