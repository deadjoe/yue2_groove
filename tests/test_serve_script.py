"""The service log survives a restart instead of being truncated.

A panic used to be followed by ``serve.sh start`` truncating ``.server.log`` with
``>`` — which erased the run directory the app had just printed, so a lost run had
no trace anywhere.  The launch now appends and stamps a start banner.
"""
from __future__ import annotations

from pathlib import Path


def test_serve_sh_keeps_the_log_across_restarts() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "serve.sh").read_text(
        encoding="utf-8")
    assert 'nohup "$PY" "${args[@]}" >> "$LOG_FILE" 2>&1 &' in script   # append, not >
    assert 'nohup "$PY" "${args[@]}" > "$LOG_FILE" 2>&1 &' not in script
    assert "serve.sh start =====" in script                            # the start banner
