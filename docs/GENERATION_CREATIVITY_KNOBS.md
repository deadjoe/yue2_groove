# YUE2 // GROOVE — Generation creativity knobs

How the GENERATE controls affect **creativity** vs **adherence** to style / lyrics.
Defaults below match the UI’s ADVANCED // SAMPLING protocol defaults (unless noted).

## Two stages

YuE2 generation is staged:

1. **Plan (ABC)** — when `cot` is `full` or `melody`, the model (or your pasted ABC) writes a score plan first.
2. **Semantic** — samples codec tokens from style / lyrics (plus the score). This is what the song *sounds* like.

**CFG** guides the conditioned generation overall.  
**ADVANCED // SAMPLING** has two slider groups: **ABC phase** and **semantic phase**.

---

## Highest impact on creativity / prompt follow

| Control | Typical / default | Role |
|---|---|---|
| **CFG SCALE** | Often `1.5` (empty ≈ **1.0**; `cot=off` default **1.01**) | How tightly the model follows style / lyrics. Higher = stricter (too high can sound harsh/dry). Lower = freer, more drift. `1.5` is a mild tighten — a solid starting point. |
| **PLAN MODE (`cot`)** | Often `full` | **full** = melody + chord plan (more “composed”). **melody** = no chord symbols, freer arrangement (covers). **off** = no symbolic plan — loosest / least controllable. Prefer **melody** for inventive arrangement; **full** + hand-edited ABC when you want obedience. |
| **SEED** | — | Same settings, different seed = another take. SONG’s **TRY ANOTHER SEED** does this. Changes the draw, not the “personality” of the knobs. |
| **Semantic `temperature`** | `1.0` | Main **sonic creativity** dial. Higher (`1.1`–`1.3`) = wilder; lower (`0.85`–`0.95`) = safer / more formulaic. |
| **Semantic `top_p` / `top_k`** | `0.95` / `100` | Nucleus / top-k width (same family as temperature). Slightly lower (e.g. `top_p=0.9`, `top_k=50`) = tighter; maxed = wilder. |
| **Semantic `repetition_penalty`** | `1.2` | Discourages repeated phrases. Slightly higher = less looping; too high hurts musical lines. |

---

## Affects the *score* more than lyric stickiness (when planning)

| Control | Default | Role |
|---|---|---|
| **ABC phase `temperature` / `top_p` / `top_k`** | `0.7` / `0.9` / `30` | How adventurous the **plan** is (melody / chord skeleton). Higher = flashier charts; lower = safer. Lyric adherence still lives mainly in semantic + CFG, but a weird score will skew the song. |
| **Your own / edited ABC** | empty | Strongest structural steering: PLAN ONLY or COVER → edit the score → generate. Lock form, then play with sampling. |

ABC phase also has `repetition_penalty` (default `1.005`), `penalty_window` (`100`), `min_tokens` (`32`), `max_tokens` (`4096`).

---

## Not “creativity” — don’t reach for these first

- **ODE steps**, **dtype**, **memory budget**, **VAE** — quality / speed / stability  
- **BUDGET PRESET** / **`max_tokens`** — length and wall time (semantic ≈ **25 tokens ≈ 1 s** of audio); not musical personality  
- **`min_tokens` / `penalty_window`** — length floor and repetition window; fine-tuning only  

---

## Practical recipes (around CFG `1.5`)

### More inventive, still recognises style / lyrics

1. Keep CFG in **`1.2–1.5`** (don’t jump to `3+` first).  
2. Raise **semantic temperature** to **`1.15–1.25`**, and/or try several **seeds**.  
3. A/B **full vs melody** at the same seed — arrangement freedom differs a lot.  
4. For “weird but not chaotic”: plan first, edit a few ABC bars, then generate (lock skeleton, free sampling).

### Tighter follow of words / style

- CFG **`1.5–2.0`**, **slightly lower** semantic temperature, plan **`full`**, hand-edit ABC when needed.

### One-liner

- **CFG** = how obedient  
- **Semantic temperature (+ seed)** = how wild *within* that obedience  
- **Plan mode / ABC** = how free the *structure* is  

If you only ever set CFG=`1.5`, most of the creativity room is still in **semantic sampling** and **seed**.

---

## Ready-to-type ADVANCED presets

Shared: leave ABC phase at protocol defaults unless you care about the chart itself.  
ABC defaults: temp `0.7`, top_p `0.9`, top_k `30`, rep `1.005`, window `100`, min `32`, max `4096`.  
Semantic protocol defaults: temp `1.0`, top_p `0.95`, top_k `100`, rep `1.2`, window `50`, min `200`, max `9000`.

### A — Pop-stable (safer, more “radio”)

| | Value |
|---|---|
| CFG | `1.5`–`1.8` |
| Plan mode | `full` |
| Semantic temperature | `0.9` |
| Semantic top_p | `0.9` |
| Semantic top_k | `50` |
| Semantic repetition_penalty | `1.25` |
| Seed | Fix one, or nudge ±1 when comparing |

### B — Experimental (wilder takes)

| | Value |
|---|---|
| CFG | `1.2`–`1.5` |
| Plan mode | `melody` (or `full` + edited ABC) |
| Semantic temperature | `1.2` |
| Semantic top_p | `0.95` |
| Semantic top_k | `100` |
| Semantic repetition_penalty | `1.15` |
| Seed | Try several; keep the rest fixed |

Optional ABC tweak for B: temperature `0.85`, top_k `50` if you want a slightly freer plan without chaos.

---

## Notes

- Empty CFG in the UI means “use protocol default,” not literal `0`.  
- Instrumental / “no vocals” is **not** a native YuE2 exclude control; see upstream issues / community empty-section recipes separately.  
- This note describes **yue2_groove** GENERATE controls wrapping the upstream `yue2` `SongRequest` + dual `Sampling` objects.
