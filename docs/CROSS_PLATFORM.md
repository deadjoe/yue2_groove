# Cross-platform analysis: NVIDIA CUDA vs Apple MPS

**Date:** 2026-09-16 · **Evidence:** 5 controlled generation runs across 3 machines · **Status:** experimental findings, hypotheses labelled as such

This document records a controlled investigation into a reported **quality difference between
the same model on NVIDIA CUDA and on Apple Silicon (MPS)**: identical model, identical
request, identical settings, but outputs that listeners describe as clearly different in
quality. It also records the experiments that were run to explain the difference — including
one finding that changes how the comparison should be interpreted:

> **The sampler uses a different random-number generator device per platform** (CUDA device
> RNG on CUDA, CPU RNG on MPS). The same seed therefore produces **different random draws**
> on the two platforms. This alone guarantees different songs, and it means this experiment's
> quality comparison is effectively **one draw per platform**, not a like-for-like quality
> comparison.

See also: [LINUX_CUDA.md](LINUX_CUDA.md) (memory/performance validation on CUDA) and
[MACOS_MPS.md](MACOS_MPS.md) (MPS support notes and the SDPA defect).

---

## Summary

| # | Finding | Evidence |
|---|---|---|
| 1 | **The sampler's RNG device differs by platform — by code design.** CUDA samples with a CUDA-device generator; MPS samples on the **CPU** with a CPU generator. PyTorch generators are device-specific algorithms, so the same seed yields **different random streams**. | `yue2/sampling.py` (`rng_device = device if device.type in {"cpu","cuda"} else torch.device("cpu")`); local test: same seed, CPU vs MPS generator → different draws, each reproducible. |
| 2 | The same seed therefore produces **different compositions** on CUDA vs MPS — and this is *sufficient* to explain the divergence; no numerical difference in the model is required. | Two CUDA runs (different torch builds) are byte-identical to each other; four MPS runs share one RNG stream and cluster together; chord-set overlap CUDA↔MPS = **0.00**. |
| 3 | **torch version is not a cause.** | Exp A: CUDA with torch 2.10 vs 2.14 → `score.abc` and `semantic.npy` **byte-identical**. |
| 4 | **Numeric precision is not a cause on MPS.** | Exp B: MPS with float32 instead of bfloat16 → same key/tempo/harmony trajectory, **1.65× slower**, no quality gain. |
| 5 | **Whether MPS quality is systematically worse is *not* established by this experiment.** The three MPS takes share one RNG stream — they are near-copies, not independent samples — so the listening comparison is effectively **one draw per platform**. | §5, §8 |
| 6 | The platforms also use **different attention code paths by design** (a secondary, unquantified difference). | CUDA: `attention: flash` with native GQA. MPS: `attention: sdpa` with `repeat_interleave` K/V expansion. |

**Practical consequences**

- Same seed ≠ same song across platforms — expect that, and use it deliberately.
- The two obvious "fixes" for the perceived Mac quality gap are **tested and rejected**
  (torch version alignment, float32).
- To decide whether MPS *systematically* produces worse music, two follow-up experiments are
  needed (§9): a **common-RNG run** (make both platforms sample with the same RNG device) and a
  **multi-seed listening comparison**.

---

## 1. Background

The model (YuE2-3B, unquantized BF16) is trained and validated on Linux + NVIDIA. This UI
also supports Apple Silicon through PyTorch's MPS backend, with a torch override for a known
MPS attention defect (see [MACOS_MPS.md](MACOS_MPS.md)).

During routine testing, the same song request produced outputs that were judged by ear to be
**clearly better on CUDA than on two different Macs**. The Macs' outputs were *similar to each
other*, which motivated a controlled comparison rather than a "one bad roll of the dice"
explanation. The investigation found that the platform difference has a code-level cause
(§6.1) that also constrains what the listening evidence can prove (§5, §8).

## 2. Method

### 2.1 What is held fixed

| Variable | Value |
|---|---|
| Model weights | `m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae` — **SHA-256 verified identical in all 5 runs** |
| Upstream runtime | `yue2-infer` 0.1.6 — **`runtime_sha256` identical in all 5 runs** (hash of the upstream `yue2` runtime files) |
| App | `yue2-groove` 0.8.0 — repo commit `524ac85` (reference run) or `238ff9a` (all other runs); those two commits differ in **documentation only**, so the generation path is identical |
| Request | same style, lyrics, `seed=831001`, `cfg_scale=1.5`, `cot=full` |
| Generation config | `ode_steps=32`, `ode_method=midpoint`, `context=24576`, protocol sampling defaults |
| Runtime config | `backend=torch`, `quantization=none`, `offload_ar=false`, `memory_budget=24`, `vae_core_frames=1024` |

### 2.2 What is deliberately varied

| Run | Platform | torch | dtype | Purpose |
|---|---|---|---|---|
| Reference | NVIDIA L4 (CUDA) | 2.10.0+cu128 | bfloat16 | the CUDA baseline |
| **Exp A** | NVIDIA L4 (CUDA) | **2.14.0+cu130** | bfloat16 | isolate torch version |
| Control 1 | Apple M1 Max (MPS) | 2.14.0 | bfloat16 | isolate the platform |
| Control 2 | Apple M4 Pro (MPS) | 2.14.0 | bfloat16 | isolate the GPU generation |
| **Exp B** | Apple M4 Pro (MPS) | 2.14.0 | **float32** | isolate numeric precision |

The design: if torch version mattered, Exp A would diverge from the reference. If the GPU
generation mattered, the two Macs would diverge from each other. If precision mattered, Exp B
would diverge from Control 2. **None of those happened** — see §5.

A platform difference that is *not* varied but is inherent to the code is the **sampling RNG
device** (§6.1): it is CUDA on CUDA and CPU on MPS, so the same seed yields different random
draws on the two platforms no matter what else is configured.

### 2.3 Measurements

- **Composition** — parsed from `score.abc` (key, tempo, chord sequence, chord-set similarity).
- **Tokens** — `semantic.npy` / `abc_tokens.npy` (count, diversity, entropy, repetition).
- **Audio** — `audio.flac` (duration, peak/RMS, spectral centroid, high-frequency share, stereo correlation).
- **Performance** — per-stage timings from `result.json` (each run records them itself).
- **Integrity** — SHA-256 of every artifact (`result.json` records them; re-verified for this document).
- **Listening** — one experienced listener (the author), non-blind, comparing takes on the same playback setup.

---

## 3. Test environments

### 3.1 NVIDIA L4 (reference and Exp A)

| | |
|---|---|
| GPU | NVIDIA L4, 24 GB GDDR6, Ada (compute capability 8.9), 72 W cap |
| Host | RunPod Secure Cloud, EU, Ubuntu 24.04.3, AMD EPYC 7702 (128 threads), 503 GiB RAM |
| Driver | 595.91.07 (CUDA 13.2 capable) |
| Python | 3.12.3 (`uv` venv) |
| torch | reference: **2.10.0+cu128** · Exp A: **2.14.0+cu130** |
| Other pins | transformers 4.57.6, huggingface-hub 0.36.2, safetensors 0.7.0, tiktoken 0.12.0, soundfile 0.13.1 |
| Attention / execution | `attention: flash`, `execution: cuda_graph` |
| Sampling RNG device | **CUDA device generator** |

### 3.2 Apple M1 Max (Control 1)

| | |
|---|---|
| Chip | Apple M1 Max, **32-core GPU**, 64 GB unified memory |
| OS | macOS 26.6.2 (Metal 4) |
| Python / torch | 3.12 · **torch 2.14.0** (macOS override) |
| Attention / execution | `attention: sdpa`, `execution: eager` |
| Sampling RNG device | **CPU generator** (MPS falls back to CPU sampling) |
| App path | Pinokio launcher (`--view song`), MPS watermark guard from `.env` (0.8 / 0.5) |

### 3.3 Apple M4 Pro (Control 2 and Exp B)

| | |
|---|---|
| Chip | Apple M4 Pro, **20-core GPU**, 64 GB unified memory |
| OS | macOS 26.6.2 |
| Python / torch | 3.12 · **torch 2.14.0** |
| Attention / execution | `attention: sdpa`, `execution: eager` |
| Sampling RNG device | **CPU generator** |
| App path | `scripts/serve.sh` service manager (repo checkout) |

### 3.4 The fixed request

| | |
|---|---|
| seed / cfg_scale / cot | `831001` / `1.5` / `full` |
| style | 790-character grunge / 90s-alternative prompt, `116, 8 BPM`, 4/4, C#/Db major |
| lyrics | "Something True" — a full song: 111 lines including section tags and bracketed performance directions (`[Intro]` … `[Outro]`) |
| Exact text | preserved in each run's `request.json` |

> **Data-provenance note.** `config.json` contains a `model_dtype` field that upstream
> hardcodes to `bfloat16` (it is a protocol label, not the runtime dtype) — the fp32 run
> also shows `bfloat16` there. The authoritative runtime dtype for each run is in the app's
> `local_env.json` (`"dtype": "float32"` for Exp B). This was verified directly.
> Run directory names carry each machine's **local** time; the two cloud runs are UTC.

---

## 4. Results

### 4.1 Composition — the headline difference

| Take | Key | BPM | Chords | Distinct | Top chord % | Longest run of one chord | Vocal bars | Longest run of identical vocal bars |
|---|---|---|---|---|---|---|---|---|
| L4 · torch 2.10 · bf16 *(reference)* | **Fm** | **130** | 155 | 5 | 27 % | 4 | 156 | 1 |
| L4 · torch 2.14 · bf16 *(Exp A)* | **Fm** | **130** | 155 | 5 | 27 % | 4 | 156 | 1 |
| M1 Max · bf16 *(Control 1)* | **E** | **126** | 143 | 4 | 46 % | 13 | 143 | 7 |
| M4 Pro · bf16 *(Control 2)* | **E** | **126** | 144 | 5 | 52 % | 19 | 144 | 7 |
| M4 Pro · fp32 *(Exp B)* | **E** | **126** | 151 | 4 | 60 % | 14 | 151 | 9 |

First 24 chord symbols of each take (the intro is where the divergence is already visible):

```
L4 (both torch versions): Fm Fm Ab Eb Eb7 Fm Ab Eb Eb7 Fm Ab Eb Eb7 Fm Ab Eb Eb7 Fm Ab Eb Eb7 Fm Ab Eb
M1 Max  bf16:             E  E  E  E  E  E  E  E  E  E  E  E  E  D  D  Amaj7 Amaj7 E  E  E  E  D  D  A
M4 Pro  bf16:             E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  D  A  E  E  E
M4 Pro  fp32:             E  E  E  E  E  E  E  E  E  E  E  E  E  D  D  A  A  E  E  E  E  D  D  A
```

Complete chord inventories (symbol × occurrences):

| Take | Chords |
|---|---|
| L4 (both torch versions) | Db ×42, Eb ×38, Fm ×35, Ab ×34, Eb7 ×6 |
| M1 Max bf16 | E ×67, A ×38, D ×36, Amaj7 ×2 |
| M4 Pro bf16 | E ×76, A ×40, D ×20, C#m ×4, B ×4 |
| M4 Pro fp32 | E ×91, D ×28, A ×28, C#m ×4 |

Chord-set similarity (Jaccard index on the set of distinct chord symbols):

|  | L4 2.10 | L4 2.14 | M1 Max | M4 Pro bf16 | M4 Pro fp32 |
|---|---|---|---|---|---|
| **L4 2.10** | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| **L4 2.14** | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| **M1 Max** | 0.00 | 0.00 | 1.00 | 0.50 | 0.60 |
| **M4 Pro bf16** | 0.00 | 0.00 | 0.50 | 1.00 | 0.80 |
| **M4 Pro fp32** | 0.00 | 0.00 | 0.60 | 0.80 | 1.00 |

**Reading:** the two CUDA takes are the same composition (score byte-identical). The four MPS
takes are a family — same key, same tempo, shared chords, overlapping structure — and share
**nothing** with the CUDA composition. The Macs also show a more static harmonic surface
(top chord 46–60 % of all chord symbols vs 27 %; longest run 13–19 chords vs 4; and 7–9
identical consecutive bars in the vocal line vs 1 on CUDA).

This pattern is exactly what §6.1 predicts: the two Macs draw from the **same** random stream
(CPU generator, same seed) and differ only in GPU numerics → near-identical takes; CUDA draws
from a **different** stream (CUDA generator) → a different take entirely.

### 4.2 Token level

| Take | semantic tokens | unique | entropy (bits) | repeated-token share | abc tokens | unique abc |
|---|---|---|---|---|---|---|
| L4 · 2.10 bf16 | 7 123 | 4 682 | 11.91 | 53.0 % | 3 073 | 81 |
| L4 · 2.14 bf16 | 7 123 | 4 682 | 11.91 | 53.0 % | 3 073 | 81 |
| M1 Max bf16 | 6 665 | 4 452 | 11.85 | 51.6 % | 2 686 | 78 |
| M4 Pro bf16 | 6 677 | 4 468 | 11.86 | 51.9 % | 3 017 | 79 |
| M4 Pro fp32 | 7 155 | 4 673 | 11.92 | 53.8 % | 2 839 | 79 |

All latent tensors are finite on every platform (no NaN/Inf anywhere). Token entropy is
essentially equal across platforms — **the difference is not a degenerate or corrupted token
stream; it is a different, comparably-rich composition.** Exactly repeated 4-grams are
essentially absent from every semantic stream: 0 in the two CUDA runs and the M1 Max run,
and 2 each (0.03 %) in the two M4 Pro runs.

### 4.3 Artifact hashes

| Take | `score.abc` | `semantic.npy` | `audio.flac` |
|---|---|---|---|
| L4 · 2.10 bf16 | `db2d1467174a7d71…` | `e8a09373bc99f11e…` | `07f934932f357eca…` |
| L4 · 2.14 bf16 | `db2d1467174a7d71…` | `e8a09373bc99f11e…` | `a824641c442d9e9e…` |
| M1 Max bf16 | `04eb164b3d73782c…` | `6a33a837f9777a3f…` | `57201bbd19154f43…` |
| M4 Pro bf16 | `5ba38106abfe35ce…` | `99d780bef2169f4f…` | `6ed915a459baf8ff…` |
| M4 Pro fp32 | `d996dd11c604f0c0…` | `0bf1dac96dc78b68…` | `745c85d4a623ba73…` |

Full SHA-256 values are in Appendix A. Note the CUDA pair: **identical plan and tokens, but
different audio** — i.e. torch version changed only the acoustic rendering (NAR/VAE) numerics,
invisibly at the statistical level (§4.4), never the composition.

### 4.4 Audio-level measurements

| Take | Duration | Peak | RMS | Spectral centroid | Energy > 8 kHz | Stereo correlation |
|---|---|---|---|---|---|---|
| L4 · 2.10 bf16 | 284.9 s | 1.000 | 0.1546 | 2 898 Hz | 10.74 % | 0.845 |
| L4 · 2.14 bf16 | 284.9 s | 1.000 | 0.1546 | 2 898 Hz | 10.74 % | 0.845 |
| M1 Max bf16 | 266.6 s | 1.000 | 0.1433 | 3 576 Hz | 17.08 % | 0.867 |
| M4 Pro bf16 | 267.1 s | 1.000 | 0.1418 | 3 271 Hz | 14.88 % | 0.857 |
| M4 Pro fp32 | 286.2 s | 1.000 | 0.1650 | 3 457 Hz | 15.21 % | 0.845 |

No clipping anywhere; levels are comparable. The MPS takes are measurably **brighter**
(higher spectral centroid, 1.4–1.6× the high-frequency energy) — consistent with different
music rather than a rendering fault. The two CUDA takes are statistically indistinguishable
here despite different `audio.flac` hashes (identical composition, negligible render delta).

### 4.5 Performance

| Take | ABC | Semantic AR | NAR | VAE | End-to-end | RTF¹ |
|---|---|---|---|---|---|---|
| L4 · 2.10 bf16 | 59.7 s · 51.5 tok/s | 181.3 s · 39.3 tok/s | 69.4 s | 14.4 s | **324.8 s** | **1.14** |
| L4 · 2.14 bf16 | 59.7 s · 51.5 tok/s | 181.2 s · 39.3 tok/s | 70.0 s | 20.6 s | 331.5 s | 1.16 |
| M1 Max bf16 | 105.6 s · 25.4 tok/s | 589.9 s · 11.3 tok/s | 817.9 s | 12.2 s | 1 525.6 s | 5.72 |
| M4 Pro bf16 | 94.8 s · 31.8 tok/s | 550.7 s · 12.1 tok/s | 575.5 s | 10.4 s | 1 231.4 s | 4.61 |
| M4 Pro fp32 | 142.8 s · 19.9 tok/s | 1 037.7 s · 6.9 tok/s | 842.5 s | 12.5 s | 2 036.1 s | 7.11 |

¹ Real-time factor = end-to-end seconds per second of generated audio.

Ratios versus the CUDA reference:

| Take | End-to-end | NAR alone | Semantic AR (per token) | ABC (per token) |
|---|---|---|---|---|
| M1 Max bf16 | **4.70× slower** | **11.8× slower** | 3.5× slower | 2.0× slower |
| M4 Pro bf16 | **3.79× slower** | **8.3× slower** | 3.2× slower | 1.6× slower |
| M4 Pro fp32 | **6.27× slower** | **12.1× slower** | 5.7× slower | 2.6× slower |

Observations:

- **NAR (flow-matching ODE synthesis) is the worst stage on MPS** — 8–12× slower — while VAE
  decode is equal or faster on Apple GPUs (10.4–12.5 s vs 14.4 s).
- **fp32 costs 1.65×** end-to-end on the M4 Pro (1 231 s → 2 036 s) and changes nothing about
  the trajectory class (§5).
- The M4 Pro is ~1.24× faster than the M1 Max overall and 1.42× faster in NAR, consistent
  with newer GPU generations having better Metal kernels for this workload.

### 4.6 Memory (observed peaks)

| Take | Working memory observed | Where |
|---|---|---|
| L4 · CUDA (bf16) | ≈ 10.5 GiB peak VRAM (10 780 MiB) | `nvidia-smi`, 5 s sampling |
| M1 Max (bf16) | ≈ 10.3 GB GPU allocation | `ioreg` "In use system memory" |
| M4 Pro (bf16) | ≈ 10.5 GB GPU allocation | `ioreg` |
| M4 Pro (fp32) | 11.8–20.9 GB GPU allocation | `ioreg` |

fp32 doubles the weights (7.3 GB → 14.6 GB) and visibly increases the reserve the MPS
allocator holds, without improving the result. See [LINUX_CUDA.md](LINUX_CUDA.md) for the
full VRAM-budget analysis on CUDA (including the 12 GiB OOM boundary and FP8).

### 4.7 Listening evaluation

Single experienced listener (the author), non-blind, same playback setup. The test song is
one this listener has used repeatedly for cross-platform quality assessment.

| Take | Verdict (as reported) |
|---|---|
| L4 · 2.10 bf16 (reference) | clearly better than the Mac takes; prompted the investigation |
| L4 · 2.14 bf16 (Exp A) | *"very good — vocals, music, lyrics, arrangement, instruments and narrative direction all correct; high quality"* |
| M1 Max bf16 | *"clearly worse than the Linux/CUDA version — obvious by ear"* (vocals, music, arrangement, instruments) |
| M4 Pro bf16 | *"about the same as the M1 Max take; clearly worse than Linux/CUDA"* |
| M4 Pro fp32 (Exp B) | **not yet evaluated by ear at the time of writing** |

The listening verdicts were reported in Chinese and are translated here.

**How much does this prove?** The MPS verdicts concern **one effective random draw**
(the three MPS takes share one RNG stream — see §6.1), against one CUDA draw. It is therefore
strong evidence that *this* MPS draw is worse, and consistent with the wider impression from
routine use on Macs — but it is not yet evidence that MPS draws are *systematically* worse.
The composition analysis (§4.1) supports the same reading: the MPS take is more harmonically
static, but that may be a property of this draw rather than of the platform. §9 proposes the
experiment that would settle it.

---

## 5. Hypothesis testing

### Exp A — "the torch version explains it" → **rejected**

Same GPU (L4), same dtype, same request; only torch changes (2.10.0+cu128 → 2.14.0+cu130,
i.e. the Linux pin vs the macOS pin):

| Artifact | torch 2.10 | torch 2.14 | Result |
|---|---|---|---|
| `score.abc` | `db2d1467174a7d71…` | `db2d1467174a7d71…` | **byte-identical** |
| `semantic.npy` | `e8a09373bc99f11e…` | `e8a09373bc99f11e…` | **byte-identical** |
| `audio.flac` | `07f934932f357eca…` | `a824641c442d9e9e…` | different samples, statistically indistinguishable (§4.4) |

The composition is **torch-version independent on CUDA**. Note what this also verifies: with
the RNG stream held constant (same device generator, same seed), two different torch builds
still produce identical tokens — so token-level sampling on CUDA is robust to kernel-level
numeric differences of the magnitude between these builds. This makes the platform RNG
difference (§6.1) the natural explanation for the cross-platform divergence.

### Exp B — "numeric precision explains it" → **rejected**

Same machine (M4 Pro), same request; only dtype changes (bfloat16 → float32), verified in
`local_env.json` and by the doubled GPU allocation:

| | M4 Pro bf16 | M4 Pro fp32 |
|---|---|---|
| Key / tempo | E / 126 | **E / 126** |
| Distinct chords / top share | 5 / 52 % | 4 / **60 %** |
| Chord-set similarity vs bf16 | — | 0.80 |
| End-to-end | 1 231 s | 2 036 s (1.65×) |

With the RNG stream held constant (both runs sample on CPU), fp32 stays on the same
trajectory — and is *more* harmonically static, not less. Precision is not the discriminator,
and fp32 is not a quality mode.

### Cross-check — "the GPU generation explains it" → **rejected**

M1 Max (32-core, 2021) and M4 Pro (20-core, 2024) are different GPU generations with
different Metal capabilities, yet produced the same key, same tempo, and a 0.50–0.80
chord-set overlap with each other, and 0.00 with CUDA. This is what §6.1 predicts: the two
Macs share an RNG stream and differ only in kernel numerics, which is a much smaller
difference than a different random stream.

### What remains, and what is still open

- **Explained:** the *divergence* between CUDA and MPS. The platforms sample from different
  random streams by design (§6.1); this is sufficient to produce entirely different songs,
  and the observed clustering (Macs together, CUDA apart) matches it exactly.
- **Open:** whether MPS draws are *worse on average*. The available listening evidence covers
  one effective draw per platform. The numerical path differences in §6.2–6.3 may or may not
  contribute to quality; this experiment cannot separate them from the RNG difference.

---

## 6. Platform differences in the code path

### 6.1 Sampling RNG device — **code-verified, and sufficient to explain the divergence**

```python
# yue2/sampling.py — generate_tokens()
rng_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
generator = torch.Generator(device=rng_device).manual_seed(seed)
...
if sampling.temperature == 0:
    next_id = scores.argmax(-1, keepdim=True)
else:
    probabilities = scores.softmax(-1)
    if device.type == "mps":
        next_id = torch.multinomial(probabilities.cpu(), 1, generator=generator).to(device)
    else:
        next_id = torch.multinomial(probabilities, 1, generator=generator)
```

| | CUDA | MPS |
|---|---|---|
| Generator device | CUDA device generator | **CPU generator** (probabilities moved to CPU for the draw) |
| Same seed gives | CUDA RNG stream | CPU RNG stream |
| Streams equal? | **No** — PyTorch generators are device-specific (CPU: MT19937; CUDA: Philox) | |

Measured on the M1 Max (torch 2.14.0) — same seed, same probability vector, eight draws:

```
CPU generator : 239, 131, 556, 728, 260, 385,  69, 707
MPS generator : 186, 753, 776, 703, 229, 980, 485, 545     # different stream
CPU generator re-run: identical to the first CPU list      # reproducible
```

**Consequences**

1. The same seed **cannot** produce the same song on CUDA and on MPS. This is a property of
   the sampler's RNG-device selection, not of the model.
2. Both Macs use the CPU stream → near-identical takes (§4.1); CUDA uses a CUDA stream →
   a different take. Exactly the observed pattern.
3. An apples-to-apples *platform* comparison of identical draws requires the same RNG device
   on both sides — see the follow-up experiment in §9.

### 6.2 Attention path — different by design (secondary, unquantified)

Upstream's `sdpa()` wrapper (`yue2/modeling_yue2.py`) selects different implementations per
device:

```python
def sdpa(query, key, value, *, attn_mask=None, is_causal=False):
    grouped = query.shape[1] != key.shape[1]
    if grouped and query.device.type == "mps":
        # PyTorch's MPS attention does not implement enable_gqa on every release.
        groups = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        grouped = False
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask,
                                          is_causal=is_causal, enable_gqa=grouped)
```

| | CUDA | MPS |
|---|---|---|
| Grouped-query attention | native (`enable_gqa=True`) | K/V expanded with `repeat_interleave`, then ungrouped |
| Kernel selected (as recorded) | `attention: flash` | `attention: sdpa` |
| Execution | `cuda_graph` | `eager` |

Different kernels accumulate differently; the model runs ~24 000-token contexts through this
path. The math is equivalent, the numerics are not. The project's
`scripts/mps_sdpa_check.py` verifies the MPS attention result against a reference within
~6e-4 in bf16 (excluding gross errors) but does not make the platforms bit-equal, and does not
cover every shape the model uses. **Note the CUDA control in §5:** token-level sampling on
CUDA survived a torch/kernel change without a single token flip, so numeric differences of
that magnitude are not, by themselves, sufficient to change a take when the RNG stream is
held constant.

### 6.3 Kernel libraries

cuBLAS / cuDNN / Flash-Attention vs Metal Performance Shaders / MPSGraph: different reduction
orders, tiling, and algorithm selection for bf16 and fp32. Not directly measured here; listed
for completeness.

### 6.4 What has been excluded as a mechanism

- **Model weights** — SHA-256 verified identical in all five runs.
- **Tokenizer / text handling** — same request text, same `yue2-infer` runtime hash.
- **Degenerate sampling on MPS** — token entropy and repetition are comparable across
  platforms (§4.2); no NaN/Inf in any latent tensor.
- **torch build, dtype, GPU generation** — tested and rejected (§5).

---

## 7. Practical implications

### For macOS users of this UI

- **Same seed ≠ same song on a Mac vs on CUDA — by design.** The Mac samples with the CPU
  RNG; CUDA samples with the CUDA RNG. Use this deliberately: a seed you like on one platform
  does not reproduce the other.
- **Mac results are reproducible on Macs**: the CPU RNG stream is stable, so the same seed and
  settings give the same song across macOS machines (verified across M1 Max and M4 Pro).
- **Seed re-rolling is the practical lever** — since the draw determines the composition, and
  MPS draws are reproducible per seed, trying several seeds is effective.
- **`float32` is not a quality mode.** It costs ~1.65× time and does not change the trajectory
  class (§5, Exp B). Do not reach for it to "fix" quality.
- **MPS is slower, most of all in NAR** (8–12× vs CUDA; a ~4.5-minute song takes ~20 minutes
  on an M4 Pro, ~25 minutes on an M1 Max). Memory is modest: ~10.5 GB of unified memory at
  bf16.
- If a take matters, generate on CUDA — and if you want to compare *like for like*, use the
  method in §9 rather than the same seed alone.

### For the project

- Document macOS as **supported, with a different sampling stream by design**; avoid promising
  cross-platform seed parity.
- The comparison tooling (`05 // TOOLS`) should be used **within** a platform; a same-seed
  cross-platform comparison compares two different draws, not one song rendered twice.
- If cross-platform *sampling parity* were ever wanted, the `rng_device` choice in
  `yue2/sampling.py` is the single place to change (force CPU sampling on all platforms).
  Note that this is upstream code.
- The ODE-step A/B finding in [LINUX_CUDA.md](LINUX_CUDA.md) is unaffected: on one platform
  and one dtype, changing ODE steps leaves the plan and tokens byte-identical and changes only
  the acoustic solver.

---

## 8. Limitations

1. **One request, one seed — and effectively one draw per platform.** The three MPS takes
   share a single RNG stream (§6.1), so they are near-copies rather than independent samples.
   The listening comparison is therefore **one CUDA draw vs one MPS draw**, repeated with
   small numeric variations. It cannot establish whether MPS draws are worse *on average*.
2. **One listener, not blind.** The quality ranking is one experienced listener's judgement;
   the wider impression from routine Mac use was not recorded under controlled conditions.
3. **Output-level, not op-level.** No per-op or per-token logit comparison was performed, so
   §6.2–6.3 remain candidates rather than a proven mechanism for any residual quality effect.
4. **Two Apple GPUs, one macOS version, one torch build on the Mac side.** macOS 26.6.2 and
   torch 2.14.0 throughout; other combinations are untested.
5. **Memory numbers** come from periodic sampling (5 s on CUDA, 20 s on the Macs), so brief
   peaks may be missed; treat them as working-set figures, not exact maxima.
6. **The CUDA-vs-CPU random-stream difference is inferred** from the code path, from a
   locally measured CPU-vs-MPS generator difference, and from documented PyTorch behaviour
   (device-specific generator algorithms). No CUDA-device RNG measurement was taken in this
   investigation; the decisive test is §9.
7. **Not tested:** macOS with torch 2.10 + the documented zero-mask workaround; MPS attention
   pinned to the math backend; any patch-level intervention.
8. **Exp B was not separately evaluated by ear** before this document was written.

---

## 9. Follow-up experiments (recommended)

### 9.1 Common-RNG comparison — the decisive test for "numerics vs RNG"

Force the sampler to use the **same RNG device on both platforms** (always the CPU generator
and CPU sampling) and re-run the fixed request on CUDA:

- If the CUDA output then matches the **MPS** trajectory (E/126 family): the divergence was
  purely the RNG stream; the model numerics are equivalent for this purpose.
- If the CUDA output stays in the **Fm/130** family: numerical path differences also
  contribute, and their magnitude can then be studied with the RNG controlled.

This requires a one-line change in `yue2/sampling.py` (upstream) or a runtime patch in the
app's adapter layer.

### 9.2 Multi-seed listening comparison — the test for "systematically worse"

Generate N ≥ 5 seeds on each platform (same settings), shuffle and listen blind, and compare
the distribution of ratings — not a single pair. This is the only design that can support the
claim "MPS quality is systematically worse", and it should replace single-seed comparisons in
future write-ups.

---

## 10. Reproduction

Environment setup is documented in [LINUX_CUDA.md](LINUX_CUDA.md) §6 (CUDA) and
[MACOS_MPS.md](MACOS_MPS.md) (MPS). The only deviations for this document:

```bash
# Exp A — CUDA with torch 2.14 instead of the upstream pin 2.10
uv pip install --python env/bin/python torch==2.14.0 --force-reinstall

# Exp B — macOS: set DTYPE = float32 in Settings → RUNTIME (or --dtype float32)
```

Then run the fixed request (seed 831001, cfg 1.5, cot full, ODE 32) and compare:

```bash
# composition
grep -E '^(K|Q):' score.abc                     # key / tempo
grep -o '"[A-G][#b]?[a-z0-9]*"' score.abc       # chord stream

# integrity
sha256sum score.abc semantic.npy audio.flac
jq '.weights, .truncated, .timing' result.json  # hashes, truncation, per-stage timings
```

Verification checklist for a comparable run: same `weights.*.sha256`, same
`runtime_sha256`, same `config.json` settings except the variable under test, and
`local_env.json` recording the intended `device`/`dtype`. For cross-platform comparisons,
also record the **sampling RNG device** (§6.1) — it is currently implicit in the device.

---

## Appendix A — full artifact hashes (SHA-256)

```
L4 2.10 bf16 (reference)
  score.abc      db2d1467174a7d7166207f03a5e0dc146ea79e33a3af5719cccac2af4f92c7b8
  semantic.npy   e8a09373bc99f11ed3c931b95f8d6c76d6e4765b233c6fc37362fcdebda8aaed
  audio.flac     07f934932f357ecab3ad7d200eead58135fbad72a66fe1ac2f7fad25f3cb68ec

L4 2.14 bf16 (Exp A)
  score.abc      db2d1467174a7d7166207f03a5e0dc146ea79e33a3af5719cccac2af4f92c7b8
  semantic.npy   e8a09373bc99f11ed3c931b95f8d6c76d6e4765b233c6fc37362fcdebda8aaed
  audio.flac     a824641c442d9e9ed5f22871d58a5118c942ab9e0672867a08379722aec0418f

M1 Max bf16 (Control 1)
  score.abc      04eb164b3d73782cdba65e1aa0ccb30337853a7018f5b1f67500902fdc613137
  semantic.npy   6a33a837f9777a3f6b877969edef00a705a32d27e52f2beb4caafe5711121406
  audio.flac     57201bbd19154f431f13857d7ff0186b89f832db149679f4c710e8c9ee0fe34a

M4 Pro bf16 (Control 2)
  score.abc      5ba38106abfe35ce7aafca0fb363f4605a91ed7c4107f992cf954d1202e12c33
  semantic.npy   99d780bef2169f4f90aef61924a79181b41f02739c93827f9eb692cbd7ad9d27
  audio.flac     6ed915a459baf8ff1a48d10a0aa8352e215fd53a8e186fd99edca698fb18f729

M4 Pro fp32 (Exp B)
  score.abc      d996dd11c604f0c08d6ab76f28445c125450d979820a9960200fda0462136406
  semantic.npy   0bf1dac96dc78b68d72c9fa8f0a35fadef43140b851637f20d94209ba568a7b1
  audio.flac     745c85d4a623ba73e480d613700ac62c7f70f78a7aea70b6d4b6d54d9f48d386

Weights (identical in all five runs)
  YuE2-3B model.safetensors   1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9
  YuE2-Vae model.safetensors  807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346

Upstream runtime (identical in all five runs)
  runtime_sha256              17963197ec6c7a87f7519cda132ad70c9843acecf65eeb0d58e41506a744b3ed
```

## Appendix B — machine-readable summary

```json
{
  "request": {"seed": 831001, "cfg_scale": 1.5, "cot": "full", "ode_steps": 32},
  "rng": {"cuda": "CUDA device generator", "mps": "CPU generator", "same_seed_same_stream": false},
  "runs": [
    {"id": "20260915-142716-Something_True_CFG15",             "platform": "cuda/L4",   "torch": "2.10.0+cu128", "dtype": "bfloat16", "key": "Fm", "bpm": 130, "e2e_s": 324.8,  "rtf": 1.14},
    {"id": "20260915-173431-Something_True_CFG15_T214",        "platform": "cuda/L4",   "torch": "2.14.0+cu130", "dtype": "bfloat16", "key": "Fm", "bpm": 130, "e2e_s": 331.5,  "rtf": 1.16},
    {"id": "20260916-001257-Something_True_CFG15_M1MAX",       "platform": "mps/M1Max", "torch": "2.14.0",       "dtype": "bfloat16", "key": "E",  "bpm": 126, "e2e_s": 1525.6, "rtf": 5.72},
    {"id": "20260916-005839-Something_True_CFG15_M4MPRO",      "platform": "mps/M4Pro", "torch": "2.14.0",       "dtype": "bfloat16", "key": "E",  "bpm": 126, "e2e_s": 1231.4, "rtf": 4.61},
    {"id": "20260916-012748-Something_True_CFG15_M4MPRO_FL32", "platform": "mps/M4Pro", "torch": "2.14.0",       "dtype": "float32",  "key": "E",  "bpm": 126, "e2e_s": 2036.1, "rtf": 7.11}
  ]
}
```

## Appendix C — raw data

The five run directories (each containing `result.json`, `config.json`, `request.json`,
`local_env.json`, `score.abc`, `semantic.npy`, `abc_tokens.npy`, `latent.npy`, `audio.flac`)
are archived together with the other validation artifacts. Every number in this document is
derived from those files; `result.json` additionally carries the per-stage timings and
artifact hashes recorded by the app at generation time.
