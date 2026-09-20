"""``python -m yue2_groove`` / ``yue2-groove``: arguments, preload, launch."""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import sys
import threading
from pathlib import Path

from .. import config, sheetsage_adapter
from . import frontend, layout, runtime, song_view, theme

log = logging.getLogger("yue2_groove")


def _configure_logging() -> None:
    """The process log: the "[yue2_groove] …" lines on stdout that serve.sh and the
    Docker entrypoint capture.  Only our logger — the root stays at WARNING so
    httpx / uvicorn INFO chatter does not join it."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[yue2_groove] %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def main():
    # .env must be loaded before the CLI defaults below are resolved and before
    # PyTorch initialises the MPS allocator, so the watermark guard applies here
    # too (serve.sh sources it, a direct `python -m yue2_groove` did not).
    config.load_env()
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    parser.add_argument(
        "--backend",
        default=os.environ.get("YUE2_GROOVE_BACKEND", "auto"),
        choices=runtime.BACKEND_MODES,
        help="Inference engine: torch (reference), torch-eager, gguf (yue2.cpp, for cards "
        "under 16 GB) or auto — the VRAM rule (or set YUE2_GROOVE_BACKEND)",
    )
    parser.add_argument(
        "--model",
        default=config.default_model(),
        help="Hugging Face id or local directory of the 3B model",
    )
    parser.add_argument(
        "--runs",
        default=None,
        help="Directory for generated works (default: $YUE2_GROOVE_RUNS or ./runs)",
    )
    parser.add_argument("--vae", default="standard", choices=["standard", "legacy"])
    parser.add_argument(
        "--tab",
        type=int,
        default=None,
        help="Start on Studio tab 0..6 (an explicit --tab forces the Studio view)",
    )
    parser.add_argument(
        "--view",
        choices=list(song_view.VIEW_CHOICES),
        default=None,
        help="Start in the SONG or STUDIO view (or set YUE2_GROOVE_VIEW)",
    )
    parser.add_argument(
        "--auth",
        default=os.environ.get("YUE2_GROOVE_AUTH", ""),
        help="Login as user:password (or set YUE2_GROOVE_AUTH); recommended on a LAN",
    )
    parser.add_argument(
        "--sheetsage-python",
        default=None,
        help="Python of the separate SheetSage2 venv (or set "
        "YUE2_GROOVE_SHEETSAGE_PYTHON) for the 02 COVER tab",
    )
    parser.add_argument(
        "--no-preload", action="store_true", help="Do not preload the model at startup"
    )
    args = parser.parse_args()
    # --runs wins; otherwise honour YUE2_GROOVE_RUNS from .env
    runtime.RUNS = Path(args.runs).expanduser().resolve() if args.runs else config.runs_dir()
    if args.sheetsage_python:
        os.environ["YUE2_GROOVE_SHEETSAGE_PYTHON"] = args.sheetsage_python

    auth = None
    if args.auth:
        if ":" not in args.auth:
            parser.error("--auth must be user:password")
        user, password = args.auth.split(":", 1)
        if not user or not password:
            parser.error("--auth user and password must not be empty")
        auth = (user, password)

    device = runtime.pick_device(args.device)
    dtype = args.dtype
    if dtype == "auto":
        dtype = "bfloat16" if device in ("cuda", "mps") else "float32"
    backend, backend_note = runtime.resolve_backend(
        args.backend, device, device_explicit=args.device != "auto"
    )
    if backend_note:
        log.info("backend: %s", backend_note)
    view_mode, tab = song_view.resolve_view(args.view, args.tab, os.environ.get("YUE2_GROOVE_VIEW"))
    runtime.RUNS.mkdir(parents=True, exist_ok=True)
    atexit.register(sheetsage_adapter.stop_worker)  # no resident SheetSage2 after exit
    defaults = {
        "device": device,
        "dtype": dtype,
        "backend": backend,
        "backend_note": backend_note,
        "model": args.model,
        "vae": args.vae,
        "tab": tab,
        "view_mode": view_mode,
        "status": (
            "Model not loaded yet — it loads automatically on the first generation."
            if args.no_preload
            else "Model is preloading in the background…"
        )
        + (f"\n{backend_note}" if backend_note else ""),
    }
    demo = layout.build_ui(defaults)
    demo.queue(default_concurrency_limit=1)
    if not args.no_preload:

        def preload():
            try:
                runtime.load_pipeline(
                    runtime.RuntimeSettings(
                        device,
                        dtype,
                        backend,
                        "none",
                        False,
                        24,
                        32,
                        "auto",
                        args.model,
                        args.vae,
                        "",
                        "",
                        "",
                        False,
                    )
                )
                log.info("model preload complete")
            except Exception as exc:  # noqa: BLE001
                log.warning("preload failed (will retry on first generation): %s", exc)

        threading.Thread(target=preload, daemon=True).start()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        theme=theme.bb_theme(),
        css=theme.BEARBONE_CSS,
        head=frontend.head_html(view_mode),
        allowed_paths=[str(config.STATIC_DIR), str(runtime.RUNS)],
        auth=auth,
        share=args.share,
        inbrowser=not args.share,
    )
