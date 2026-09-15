# YuE2 on Linux + NVIDIA CUDA — memory and performance validation

**Date:** 2026-09-15 · **Status:** first end-to-end Linux/CUDA validation of this app
**Companion doc:** [MACOS_MPS.md](MACOS_MPS.md) covers Apple Silicon; this one covers Linux + NVIDIA.

Upstream YuE2 documents a single validated platform: *Linux · Python 3.12 · NVIDIA GPU with BF16 support and 24 GB VRAM*. This report turns that guideline into measured numbers for this app: what actually runs, what it actually needs in VRAM, where the failure boundary is, and what the experimental FP8 mode does and costs.

All runs below were executed on a rented cloud GPU (RunPod, NVIDIA L4) through the real UI, with one fixed song request so that single variables could be changed per run. Raw artifacts (timings, hashes, audio) are recorded in each run's `result.json`.

---

## 1. Tested configuration

### Hardware

| | |
|---|---|
| GPU | NVIDIA L4, 24 GB GDDR6 (Ada Lovelace, compute capability **8.9**) |
| VRAM visible to PyTorch | **22.04 GiB** (`torch.cuda.get_device_properties`) |
| Driver / CUDA | 595.91.07 / CUDA 13.2 |
| Host | RunPod Secure Cloud, EU (Iceland), $0.49/hr |
| CPU | AMD EPYC 7702, 128 threads |
| System RAM | 503 GiB (no swap) |
| Disk | 80 GB container volume |

### Software

| Package | Version |
|---|---|
| OS | Ubuntu 24.04.3 LTS (kernel 6.8.0) |
| Python | 3.12.3 (`uv` venv) |
| torch | 2.10.0+cu128 |
| transformers | 4.57.6 |
| huggingface-hub | 0.36.2 |
| safetensors / tiktoken / soundfile | 0.7.0 / 0.12.0 / 0.13.1 |
| Models | `m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae` (BF16, ~7.8 GB on disk) |

`yue2 doctor` on this host reports `dependencies_ready: true`, CUDA BF16 support `true`, and `compute_capability [8, 9]`.

### The test request

One request was used for every run unless a row says otherwise:

| | |
|---|---|
| style | grunge / hard-edged 90s alternative rock, 116.8 BPM, 4/4, C#/Db major |
| lyrics | "Something True" — a full song (~50 lines, sections from `[Intro]` to `[Outro]`) |
| `cot` | full |
| seed | 831001 |
| cfg_scale | 1.5 (the author's standard test value; default is 1.0) |
| ODE steps | 32 (protocol default) unless noted |
| dtype | bfloat16 · backend `torch` (CUDA graphs when available) |

---

## 2. Result: Linux + CUDA works

The full pipeline ran end-to-end on Linux/CUDA — install, model load, symbolic plan, semantic generation, acoustic synthesis, and VAE decode — and produced a complete 48 kHz stereo song with no truncation.

| Metric | Value |
|---|---|
| Audio produced | 284.9 s (4:45), 48 kHz stereo, FLAC 54 MB |
| End-to-end | **324.8 s** (5:25) → real-time factor ≈ **1.14** |
| Truncation | none (`truncated.abc: false`, `truncated.semantic: false`) |
| Peak VRAM observed | 10.5 GiB |

### Phase breakdown (reference run)

| Stage | Time | Detail |
|---|---|---|
| ABC plan | 59.7 s | 3 074 tokens · 51.5 tok/s · `execution: cuda_graph` · `attention: flash` |
| Semantic AR | 181.3 s | 7 124 tokens · 39.3 tok/s · 2 CFG branches (cfg 1.5) · `cuda_graph` |
| NAR synthesis | 69.4 s | 32 ODE steps (midpoint) |
| VAE decode | 14.4 s | 1 024-frame window, 16-frame halo |
| Model load | ~6.8 s | weight resolve + integrity check + transfer to GPU |

The optimized CUDA paths are active: **CUDA Graphs** for both AR stages and **Flash Attention** throughout. GPU utilization held 98–100 % at the card's 72 W power cap for the duration (~56–59 °C).

---

## 3. Memory: what the app budgets vs. what the song needs

### 3.1 The budget mechanism

`MEMORY BUDGET` (Settings → RUNTIME) is not a target — it is a hard cap:

```
allowed = min(budget_setting − 2 GiB, total_visible_VRAM − 2 GiB)
torch.cuda.set_per_process_memory_fraction(allowed / total_visible_VRAM)
```

The 2 GiB reserve exists so that the driver, CUDA context and non-PyTorch allocations are never squeezed. Exceeding the cap raises a clean `OutOfMemoryError` for that run; the app records the failure and keeps running (the next run reloads and retries).

At budget ≤ 12 GiB the pipeline additionally **halves the VAE decode window** (1 024 → 512 frames) to shrink the decoder's working set.

### 3.2 Controlled memory runs

Same request, same seed, only the memory setting changed:

| Run | dtype | Memory budget | Effective cap¹ | Result | Peak VRAM observed² |
|---|---|---|---|---|---|
| `20260915-142716` | BF16 | 24 | 20.0 GiB | ✅ complete | 10.53 GiB |
| `20260915-150457` | BF16 | **16** | 14.0 GiB | ✅ complete | 10.81 GiB |
| `20260915-151249` | BF16 | **12** | 10.0 GiB | ❌ OOM at ~60 s | 10.25 GiB at failure |
| `20260915-151448` | BF16 | 12 | 10.0 GiB | ❌ OOM at ~60 s (reproduced) | 10.25 GiB at failure |
| `20260915-151926` | **FP8** | 12 | 10.0 GiB | ✅ complete | 9.92 GiB |

¹ On this L4: `min(budget − 2, 22.04 − 2)` GiB.
² Sampled every 5 s with `nvidia-smi`; brief spikes between samples are possible, so treat peaks as lower bounds (the OOM runs give an independent lower bound of ≥ 10.25 GiB for the BF16 working set).

### 3.3 Where the boundary is, and why

The BF16 failure is reproducible and precise: both budget-12 runs die exactly **≈ 60 s in**, at the hand-off from the ABC plan to the semantic AR stage — the first phase that runs with two CFG branches:

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 MiB.
GPU 0 has a total capacity of 22.04 GiB of which 11.78 GiB is free.
This process has 10.25 GiB memory in use. 10.00 GiB allowed; ...
```

So the BF16 working set for this song is **slightly above 10.25 GiB** (measured peak 10.5–10.8 GiB across passing runs), and the binding constraint is the semantic stage with `cfg_scale 1.5` (two branches, batched).

### 3.4 The budget is a pure cap — proven by hashes

Two runs with different budgets produced **bit-identical output**, which shows the mechanism only limits allocation and does not change any computation:

| Artifact | budget 24 | budget 16 | Match |
|---|---|---|---|
| `score.abc` (SHA-256, 16 hex) | `db2d1467174a7d71` | `db2d1467174a7d71` | ✅ identical |
| `semantic.npy` | `e8a09373bc99f11e` | `e8a09373bc99f11e` | ✅ identical |
| `audio.flac` | `07f934932f357eca` | `07f934932f357eca` | ✅ identical |

### 3.5 What hardware can actually run this

Measured working set for this song (285 s, CFG 1.5, BF16): **~10.5–10.8 GiB peak**.
Measured FP8 working set: **~9.9–10.0 GiB peak**.

Applying the app's own budget arithmetic to card classes:

| Card class | VRAM visible to PyTorch¹ | App cap (total − 2 GiB) | BF16 (needs ~10.8 GiB) | FP8 (needs ~10.0 GiB) |
|---|---|---|---|---|
| 24 GB (3090 / 4090) | ~24 GiB | ~22 GiB | ✅ comfortable | ✅ |
| **16 GB (4060 Ti 16G, 5070 Ti, …)** | ~16 GiB | **~14 GiB** | ✅ ~3 GiB headroom | ✅ |
| 12 GB (3060 12G, 4070, …) | ~12 GiB | **~10 GiB** | ❌ fails | ⚠️ borderline (≈ at the cap) |
| 8 GB | ~8 GiB | ~6 GiB | ❌ | ❌ |

¹ Inference, not measured on those cards. The 16 GB row extrapolates from the L4 measurements (which passed at a 14 GiB cap); the 12 GB row is consistent with the observed OOM at a 10 GiB cap and should be treated as unsupported until tested on real hardware.

**Notes for users**

- The upstream "24 GB" guideline is conservative for this workload: the model itself is ~7.3 GB in BF16, and a ~5-minute song with the default 1 024-frame VAE window peaks at ~10.8 GiB.
- Longer songs use more context tokens (protocol context is 24 576); this song used ~10 200 tokens. The 24 GB guideline likely covers maximal-length songs — treat 16 GB as the tested floor for typical songs, and expect heavy songs to need more.
- `cfg_scale 1.5` means two CFG branches and roughly doubles the AR-stage memory and time versus `cfg_scale 1.0`. This is the main setting that pushes a run over a memory limit.

---

## 4. FP8 quantization: what it does and what it costs

### 4.1 What FP8 is here

`QUANTIZATION = fp8` (Settings → RUNTIME) quantizes **AR linear layers only** (per-tensor E4M3 weights, dynamic activations, `torch._scaled_mm`); the NAR stack and the VAE always run in BF16/FP32. It requires compute capability ≥ 8.9, i.e. RTX 40-series and newer (and this L4). Upstream labels it experimental with **no quality or speed claim**.

### 4.2 Measured result

Same request, same seed, `MEMORY BUDGET = 12`:

| | BF16 (budget 16) | **FP8 (budget 12)** | Ratio |
|---|---|---|---|
| Status | complete | **complete** | — |
| Audio | 284.9 s | 313.8 s | (different take) |
| ABC stage | 63.4 s · 3 074 tok · 48.5 tok/s | **227.7 s · 3 312 tok · 14.5 tok/s** | **3.6× slower per token** |
| Semantic stage | 183.9 s · 7 124 tok · 38.7 tok/s | **1 077.7 s · 7 847 tok · 7.3 tok/s** | **5.4× slower per token** |
| NAR | 69.3 s | 81.0 s | ~1.2× (unaffected by design) |
| VAE | 13.7 s | 11.5 s | ~same |
| **End-to-end** | **330.4 s** | **1 398.0 s (23:18)** | **4.2× slower** |
| Peak VRAM | 10.8 GiB | **9.9 GiB** | −0.9 GiB |
| AR execution | `cuda_graph` | **`eager`** | — |

**FP8 does unlock the 10 GiB budget that BF16 cannot fit** — but at ~4× the wall-clock time.

### 4.3 Why it is slower

Two mechanisms, both visible in the code:

1. **FP8 disables CUDA Graphs.** The sampling path activates graphs only when the AR model is unquantized (`graph_enabled = … and not getattr(model, "_yue2_fp8_originals", {})`), so quantized runs fall back to eager execution and expose per-token kernel-launch overhead. This is the dominant cost.
2. **Single-token decode is padded for FP8 GEMM.** Decode rows are padded to a multiple of 16 before `torch._scaled_mm`, so a batch-of-one step computes 16 rows' worth of work.

NAR (flow matching) and VAE are untouched by FP8, which is why their times barely move.

### 4.4 FP8 changes the song, not just its rendering

With the same seed, FP8 produced a **different plan and different tokens** — not a re-rendering of the same music:

| Artifact | BF16 | FP8 | Match |
|---|---|---|---|
| `score.abc` | `db2d1467174a7d71` | `e1bd6badaa6fc6a8` | ❌ different |
| `semantic.npy` | `e8a09373bc99f11e` | `fa8193769086779a` | ❌ different |
| `audio.flac` | `07f934932f357eca` | `41756563b76434f3` | ❌ different |
| ABC / semantic tokens | 3 074 / 7 124 | 3 312 / 7 847 | different |

Different numerics shift the sampling distribution, and the autoregressive loop amplifies that into a different composition. (This is the same effect that makes `seed` non-portable across platforms — see Appendix A.)

### 4.5 When to use FP8

- **Only** as a memory rescue for cards in the 10–12 GB class, and only on Ada-or-newer hardware.
- Expect ~4× longer generations (≈ 23 minutes for a ~5-minute song on this L4).
- Expect a **different take** than the same seed in BF16 — for A/B comparisons, change one thing at a time.
- Not recommended at 16 GB and above: BF16 runs faster and its output is the reference behavior.

---

## 5. Side finding: ODE steps

Same request, seed, and budget (24), changing only the ODE solver steps:

| ODE steps | ABC | Semantic | NAR | End-to-end | AR hash match vs. 32 |
|---|---|---|---|---|---|
| 32 (default) | 59.7 s | 181.3 s | 69.4 s | 324.8 s | — |
| 48 | 59.2 s | 181.4 s | **103.4 s** | 356.6 s | ✅ plan + tokens identical |

- ODE steps affect **only NAR**, linearly (~2.2 s per step on this L4): +49 % NAR time for 32 → 48.
- The AR output is **bit-identical**, so this is a clean A/B of the acoustic solver alone.
- Listening impression (single listener, not blind): 48 steps was *slightly* better than 32 for a ~10 % total runtime cost — diminishing returns. The protocol default of 32 is a reasonable production setting; 48–64 is a "final render" choice.

---

## 6. Artifacts and reproduction

Each run directory contains `result.json` (timings, model identities, artifact SHA-256s), `config.json` (effective settings), `request.json`, `score.abc`, `semantic.npy`, `latent.npy` and `audio.flac`. Run IDs referenced in this report:

```
20260915-142716-…CFG15          BF16 · budget 24 · ODE 32   (reference)
20260915-144732-…CFG15_ODE48    BF16 · budget 24 · ODE 48
20260915-150457-…CFG15_MB16G    BF16 · budget 16 · ODE 32
20260915-151249-…CFG15_MB12G    BF16 · budget 12            (OOM, kept as evidence)
20260915-151448-…CFG15_MB12G    BF16 · budget 12            (OOM, reproduced)
20260915-151926-…CFG15_MB12G    FP8  · budget 12 · ODE 32
20260915-150141-comparison      ODE 32 vs ODE 48 listening page
```

Setup on a fresh Linux + CUDA host (mirrors the launcher's steps, see the Pinokio repo for the GUI path):

```bash
git clone https://github.com/deadjoe/yue2_groove.git app && cd app
uv venv env --python 3.12
uv pip install --python env/bin/python -e ".[yue2]" --overrides overrides/linux.txt
uv pip install --python env/bin/python torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
env/bin/hf download m-a-p/YuE2-3B && env/bin/hf download m-a-p/YuE2-Vae
env/bin/python -m yue2_groove --host 0.0.0.0 --port 7860 --no-preload --auth user:pass
```

---

## Appendix A — cross-platform reference (not a controlled comparison)

Recorded on the author's Macs against **different songs**, so the numbers are a speed reference only — do not read them as a benchmark:

| Host | Backend | Song | End-to-end | Semantic AR | NAR |
|---|---|---|---|---|---|
| Apple M4 Pro 64 GB | MPS, BF16, eager | jazz, 259 s | 1 515 s | 11.2 tok/s | 792.7 s |
| **NVIDIA L4 (this report)** | **CUDA, BF16, graphs+flash** | **rock, 285 s** | **330 s** | **39.3 tok/s** | **69.4 s** |

Also note: the same seed on different backends/torch builds produces **different songs** (upstream: *"Separate GPUs, runtime versions, or sampling settings can change a seeded generation"*). Seeded A/B comparisons are only valid on one machine.

---

## Appendix B — scope and limits of this validation

- **One GPU model** (NVIDIA L4, Ada) and one cloud host; other architectures were not tested here.
- **One song** (≈ 285 s, ~10 200 of the 24 576 context tokens) at one CFG value (1.5) — memory numbers scale with song length and CFG; the 16 GB / 12 GB card rows are arithmetic extrapolations, not measurements.
- **Peak VRAM** was sampled externally every 5 s; true peaks may be slightly higher. The OOM runs bound the BF16 requirement from below.
- **Listening notes** are a single non-blind listener, one take per setting.
- Linux validation covers **generation through the UI**; the Cover (SheetSage2) workflow was validated separately on Windows and was not re-run here.
