# YUE2 // GROOVE

An unofficial web UI for [YuE2](https://github.com/multimodal-art-projection/YuE), the
open full-song music generation model with an editable symbolic plan. Give it a style
prompt and lyrics; it plans a melody-and-chord score (ABC), then renders a complete
48 kHz stereo song. This UI puts every control of the official CLI/Python API in the
browser, adds a library of your generated works, and runs as a small local service.

- **01 // GENERATE** — style + lyrics (+ optional ABC score) → editable plan → song.
  Plan mode `full / melody / off`, seed, CFG scale, all seven sampling parameters of both
  phases, budget presets, cancel, live progress, a rendered score, one-click artifact download.
- **02 // DECODE** — re-decode a saved `latent.npy` with another decoder
  (source / standard / legacy / custom VAE, full or tiled) without generating again.
- **03 // BATCH** — one JSON request per line, run in order, with a results table.
- **04 // TOOLS** — ABC validation and event export, chord stripping for cover melodies,
  edit invariant check, listening-comparison page, environment doctor.
- **05 // LIBRARY** — every work you generated: sort, select, rename, confirmed delete,
  a player with spectrum and transport controls, style / lyrics / ABC / score, run tables.
- **Settings rail** — device, dtype, backend, quantization, memory budget, ODE steps,
  VAE core frames, model/VAE revisions, offline mode, load / unload.

The UI is a thin layer over the `yue2` package: it never modifies upstream code, and the
single file that imports `yue2` (`yue2_groove/adapter.py`) is covered by contract tests
that fail loudly when an upstream release changes something the UI depends on.

> **Platforms.** Developed and tested on macOS / Apple Silicon (MPS). Linux + NVIDIA CUDA
> is the platform YuE2 itself supports and validates; this UI has not been exercised there
> yet — it should work, and reports are welcome.

## Requirements

- Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) (`pip` also works, see below).
- Apple Silicon Mac with 32 GB or more unified memory (developed on a 64 GB machine), or a
  Linux machine with an NVIDIA GPU of 24 GB (upstream's validated configuration).
- About 8 GB of disk for the model weights (`m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae`), which
  download from Hugging Face on first use.
- The model weights are licensed **CC BY-NC 4.0 (non-commercial)** by the YuE2 project.

## Quick start

1. **Install uv** (skip if you have it):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone YuE2 and this repository side by side:**

   ```bash
   git clone https://github.com/multimodal-art-projection/YuE.git
   git clone https://github.com/deadjoe/yue2_groove.git
   cd yue2_groove
   ```

3. **Create a virtual environment and install both packages into it.**
   Pick the override file for your platform (it pins two dependency versions; see
   [Why the override file?](#why-the-override-file)):

   ```bash
   uv venv .venv --python 3.12

   # macOS / Apple Silicon
   uv pip install --python .venv/bin/python ../YuE -e . --overrides overrides/macos.txt

   # Linux / NVIDIA CUDA
   uv pip install --python .venv/bin/python ../YuE -e . --overrides overrides/linux.txt
   ```

4. **(macOS) Check the attention kernel** — takes a few seconds, loads no weights:

   ```bash
   .venv/bin/python scripts/mps_sdpa_check.py      # expect: "defect not reproduced"
   ```

5. **Start the service:**

   ```bash
   bash scripts/serve.sh start
   ```

   The script prints the local and LAN URLs (default `http://127.0.0.1:7860/`). The model
   loads in the background after the page is up; the first start also downloads the
   weights (about 8 GB) to your Hugging Face cache, so the first generation waits for that.

6. **Stop, restart, inspect:**

   ```bash
   bash scripts/serve.sh status
   bash scripts/serve.sh log          # follow the log
   bash scripts/serve.sh restart --port 7861
   bash scripts/serve.sh stop
   ```

Generated works land in `runs/` inside this repository (one directory per song with
`audio.flac`, `score.abc`, `request.json`, `config.json`, `result.json`, `latent.npy`,
`semantic.npy` — the same artifacts the official CLI writes, so they stay compatible with
upstream's tools).

## Configuration

Create a `.env` file in the repository root (ignored by git); `scripts/serve.sh` reads it:

```bash
YUE2_GROOVE_AUTH=alice:my-secret     # login for the web UI; set this before exposing it on a LAN
YUE2_GROOVE_PORT=7860
YUE2_GROOVE_HOST=0.0.0.0             # 127.0.0.1 keeps it local-only
YUE2_GROOVE_RUNS=/path/to/runs       # where works are stored (default: ./runs)
YUE2_GROOVE_MODEL=m-a-p/YuE2-3B      # Hugging Face id or a local directory
YUE2_GROOVE_VAE=m-a-p/YuE2-Vae
YUE2_GROOVE_VAE_LEGACY=m-a-p/YuE2-Vae-legacy
YUE2_GROOVE_MODELS=/path/to/models   # optional: a folder holding YuE2-3B/, YuE2-Vae/, YuE2-Vae-legacy/
```

The same things are available as command-line flags:

```bash
.venv/bin/python -m yue2_groove --help
.venv/bin/python -m yue2_groove --host 127.0.0.1 --port 7860 --tab 0 \
    --device auto --dtype auto --model m-a-p/YuE2-3B --runs runs --auth alice:secret
bash scripts/serve.sh start --port 7861 --device mps --dtype bfloat16 --no-preload
bash scripts/serve.sh start -f          # foreground, Ctrl-C to stop
```

`--device auto` picks CUDA → MPS → CPU; `--dtype auto` is bfloat16 on CUDA/MPS (the
checkpoint dtype) and float32 on CPU.

**Pre-downloading the weights** (optional, faster than letting the first generation do it):

```bash
uv pip install --python .venv/bin/python hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download m-a-p/YuE2-3B
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download m-a-p/YuE2-Vae
```

If you keep weights in local directories instead (for example `../YuE/models/YuE2-3B`),
set `YUE2_GROOVE_MODELS=../YuE/models` or pass `--model /path/to/YuE2-3B`.

## Why the override file?

Two dependency pins conflict, and `uv`'s `--overrides` resolves both without touching
upstream code:

| Pin | Reason |
|---|---|
| `torch==2.14.0` (macOS only) | YuE2 pins `torch==2.10.0`, whose MPS single-query attention kernel reads out of bounds once the KV cache passes 1024 tokens ([pytorch/pytorch#174861](https://github.com/pytorch/pytorch/issues/174861)) — in bfloat16 the values turn to garbage and NaN well before a minute of audio. Fixed in torch 2.11. `scripts/mps_sdpa_check.py` reproduces the defect on 2.10 and passes on 2.11+. |
| `huggingface-hub==0.36.2` | Gradio asks for `huggingface-hub>=1.16`, YuE2 pins `0.36.2` (transformers 4.57.x needs it). Keeping upstream's pin is safe; Gradio works with it. |

With plain `pip` there is no override mechanism; install in this order and accept the
version warning:

```bash
python -m venv .venv && . .venv/bin/activate
pip install ../YuE
pip install -e .
pip install "huggingface-hub==0.36.2"            # both platforms
pip install "torch==2.14.0"                      # macOS only
```

More on the Apple Silicon story, with measurements: [docs/MACOS_MPS.md](docs/MACOS_MPS.md).

## Keeping up with upstream

YuE2 moves quickly. This UI pins nothing about upstream at the code level; the compatibility
matrix is:

| yue2-groove | YuE2 (`yue2-infer`) | Notes |
|---|---|---|
| 0.1.x | `yue2-v0.1.6` | NAR/VAE progress via an internal hook; a [pull request](https://github.com/multimodal-art-projection/YuE/pull/173) adds a public `on_progress` callback that the UI uses automatically once merged |

To upgrade YuE2:

```bash
git -C ../YuE pull
uv pip install --python .venv/bin/python ../YuE --overrides overrides/macos.txt   # or linux.txt
.venv/bin/python -m pytest -q          # contract tests: green means the UI still matches upstream
```

If `tests/test_contract.py` fails, upstream changed something the UI relies on; the fix
belongs in `yue2_groove/adapter.py`, the only module that imports `yue2`.

## Development

```bash
uv pip install --python .venv/bin/python -e ".[test]" --overrides overrides/macos.txt
.venv/bin/python -m pytest -q
```

- `yue2_groove/webui.py` — the Gradio app (all tabs, theme, client-side helpers).
- `yue2_groove/library.py` — the Library backend; standard library only, reads run
  directories by file convention and never imports `yue2`.
- `yue2_groove/adapter.py` — every call into `yue2`; keep it that way.
- `yue2_groove/vendor/` — `abc_tools.py` and `listen.py` copied from upstream's
  `skills/yue2-music/scripts` (Apache 2.0; the wheel does not ship them).
- `scripts/serve.sh` — service manager; `scripts/mps_sdpa_check.py` — MPS kernel guard.

## Credits and license

- [YuE2](https://github.com/multimodal-art-projection/YuE) by the Multimodal Art Projection
  team — the model and the `yue2` runtime (Apache 2.0). Model weights: CC BY-NC 4.0.
- [abcjs](https://github.com/paulrosen/abcjs) renders the scores (MIT).
- This repository: Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
  Not affiliated with or endorsed by the YuE2 authors.
