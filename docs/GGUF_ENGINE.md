# The GGUF engine — YuE2 through yue2.cpp, for cards under 16 GB

**Status:** measured on an M1 Max (Metal) against the CUDA reference run, 2026-09-20; not yet run on a
physical 12 GB or 8 GB card. **Not the reference configuration** — see §4 before relying on it.

YUE2 // GROOVE's reference configuration is upstream's: the unmodified BF16 model in PyTorch, validated
at 24 GB and measured down to a 12 GB budget ([LINUX_CUDA.md](LINUX_CUDA.md)). Below that, and on
Windows hosts whose PyTorch build has no FlashAttention, the reference path is slow or does not fit.
This engine runs the same model through [yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp) (a
GGML port, MIT) with an 8-bit backbone: 3.8 GB of weights instead of 7.3, its own FlashAttention on
every platform, and — measured below — a rendering the ear could not tell from the reference.

## 1. What it changes, in numbers

The test in `yue2-experiments/90-yue2cpp-quant-rerender` removes the sampling stage (which forks on any
perturbation) and renders the **reference run's own score, semantic tokens and seeded noise** through
each stack. Both inputs were verified bit-identical before the runs. Metrics as in `scripts/rerender.py`.

| Same tokens + noise rendered by | latent Δ / std | latent corr | audio SNR |
|---|---|---|---|
| PyTorch BF16 · MPS (M4 Pro) — CROSS_PLATFORM §9.3 | 2.23 % | 0.99975 | 28.6 dB |
| yue2.cpp **BF16** GGUF · Metal | 3.15 % | 0.99951 | 25.9 dB |
| yue2.cpp **Q8_0** GGUF · Metal | 3.68 % | 0.99933 | 24.0 dB |
| PyTorch · ODE 16 steps (LINUX_CUDA §5) | 5.46 % | 0.99851 | 21.3 dB |
| yue2.cpp Q6_K GGUF · Metal | 7.89 % | 0.99691 | 17.7 dB |
| PyTorch · ODE 8 steps | 9.4 % | — | 16.6 dB |
| yue2.cpp Q5_K_M GGUF · Metal | 11.17 % | 0.99374 | 14.6 dB |

Quantization alone (Q8_0 against the BF16 GGUF on the same engine): 2.30 % / 27.3 dB — the same size as
the CUDA → MPS platform change the project already accepts. **Blind listening:** two passes over the
eight takes found no reliable difference; an ABX of the reference against Q8_0 (instant switching, the
song's quietest 15 s, 12 trials) scored **6/12 — chance**. Q6_K and below degrade measurably; the port's
own note, "an audio code LM breaks below Q5", agrees. Q8_0 is therefore the only quant the app offers by
default (`YUE2_GROOVE_GGUF_QUANT` can pick the others for experiments).

Speed: the AR stage is bandwidth-bound and is where the 8-bit weights pay — 96.6 semantic tokens/s on
the M1 Max against 11 tok/s for PyTorch/MPS on an M4 Pro and 39 tok/s for PyTorch/CUDA on an L4. The
NAR stage is compute-bound and gains nothing (21 s per ODE step on the M1 Max in either precision).
Memory (yue2.cpp's own figures): Q8_0 backbone pair ≈ 4.4 GB plus the KV cache (2.7 GB per set at the
full context, two sets under CFG > 1); a 65 s song peaks at 5.8 GB, 3.8 GB with `--max-seq 8192`.

## 2. Install

Three pieces, all optional — without them the app is exactly what it was:

1. **Binaries** — built by `.github/workflows/yue2cpp.yml` at the pinned yue2.cpp commit
   (`gguf_engine.YUE2CPP_PIN`) for macOS arm64 (Metal), Linux x64 (CUDA 12.8 + Vulkan + CPU) and
   Windows x64 (CUDA 12.8 + Vulkan + CPU), attached to each app release. Fetch this platform's:

   ```bash
   .venv/bin/python -m yue2_groove.gguf_engine install        # → <repo>/bin/yue2cpp
   ```

   or unpack the asset yourself and point `YUE2_GROOVE_YUE2CPP` at the directory. The archives carry
   the CUDA runtime, so an NVIDIA driver is enough; macOS needs nothing else. (A build from source is
   `cmake -S . -B build && cmake --build build` in a yue2.cpp checkout at the pinned commit.)
2. **`gguf` package** — `uv pip install --python .venv/bin/python -e ".[gguf]"` — the writer the
   converter uses. Skip it if you place ready-made GGUF files in `YUE2_GROOVE_GGUF`.
3. **GGUF files** — prepared automatically on the first GGUF generation (about ten seconds, no
   download): the checkpoints already on disk are converted with yue2.cpp's own converter (a
   byte-identical vendored copy, `yue2_groove/vendor/yue2cpp_convert.py`) and quantized with the
   release's `quantize`. Ahead of time: `python -m yue2_groove.gguf_engine prepare`. The result is
   **tensor-for-tensor identical to the published `Serveurperso/YuE2-GGUF` Q8_0** (627/627 tensors),
   so provenance stays with the weights you verified. The VAE is never quantized (F32).

`python -m yue2_groove.gguf_engine check` prints what would be used and what BACKEND=auto decides.

## 3. Selection

- `--backend auto` (the default; also `YUE2_GROOVE_BACKEND`, and `scripts/serve.sh start --backend …`):
  a **CUDA card with less than 16 GiB** and the binaries installed gets the GGUF engine; everything
  else keeps the PyTorch engine. The threshold is `YUE2_GROOVE_GGUF_VRAM_GIB`; 16 is where
  LINUX_CUDA.md measured that the reference configuration runs everything the app can produce. A
  small card without binaries stays on PyTorch and says so in the log and STATUS. The card is read
  with `nvidia-smi` (no CUDA context in the app's process), so an NVIDIA card next to a **CPU-only
  torch** (PyPI's Windows wheel) also gets the GGUF engine instead of hours on the CPU.
- Apple Silicon is never switched automatically (the reference path is what the app is developed on),
  but BACKEND → **gguf** in the settings rail works there too and is much faster for the AR stage.
- The rail's DTYPE / QUANTIZATION / OFFLOAD AR / MEMORY BUDGET do not apply to this engine; ODE STEPS
  and VAE CORE FRAMES do. `YUE2_GROOVE_GGUF_MAX_SEQ` caps the KV cache (yue2.cpp `--max-seq`) for
  8 GB cards — the song must then fit that context, the same "shorter songs" story as
  [LOW_VRAM.md](LOW_VRAM.md) tells for 12 GB.

Runs from this engine record `"backend": "gguf"`, the quant, the GGUF names and hashes and the yue2.cpp
commit in `config.json` / `result.json` / `local_env.json`; the Library shows them. `prefix.npy`,
`abc_tokens.npy` and `plan.json` are rebuilt with upstream's own tokenizer and prefix rule
(`adapter.text_tokenizer` / `token_prefixes` / `symbolic_plan`), so 03 EDIT, 06 DECODE and the
comparison page accept a GGUF run like any other. 06 DECODE decodes with yue2.cpp's VAE
(`neural-codec`); VAE overrides need the torch backend.

## 4. What it is not

- **Not the same song.** Even the BF16 GGUF differs from PyTorch at the ε level, and any ε forks the
  sampled token stream within a few dozen draws ([CROSS_PLATFORM.md](CROSS_PLATFORM.md)). The same
  seed reproduces on one machine with one engine — two GGUF runs here were byte-identical — but not
  across engines. Seeded A/B comparisons stay within one engine.
- **Not validated end to end on a 12 GB or 8 GB card.** The numbers above are a rendering comparison
  on an M1 Max; the yue2.cpp memory figures are its author's. Reports from real small cards are welcome.
- **Not upstream.** yue2.cpp is a young, single-maintainer project on a patched GGML fork. It is pinned
  by commit, run behind the same process boundary as SheetSage2 (JSON in, files out, nothing imported),
  and can be removed without touching the reference path.
- **The AR stage's quality distribution is unmeasured.** The rendering test above is exact; whether
  the quantized sampler writes songs of the same quality is a distributional question only a blind
  test over many generations answers.

## 5. Platforms

| Platform | Backend | Status |
|---|---|---|
| macOS, Apple Silicon | Metal | built and run here (M1 Max): install, prepare, generate, Library |
| Linux, NVIDIA | CUDA (Vulkan and CPU in the same archive) | yue2.cpp's primary platform; built by the workflow, not yet run through the app |
| Windows, NVIDIA / AMD | CUDA / Vulkan | built by the workflow; the CUDA runtime DLLs ship in the zip; not yet run through the app |

Each yue-synth process on macOS compiles the Metal shader library (~20 s) before it starts; CUDA
loads in a second or two from the page cache.

**Windows paths.** yue2.cpp opens files through the ANSI code page. The engine keeps its working
files next to the GGUF files and hands non-ASCII paths over as 8.3 short names, but the safe setup
is an app folder (and `YUE2_GROOVE_GGUF`) whose path is plain ASCII — which Pinokio's is.
