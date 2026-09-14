# GENERATE — creativity knobs (producer guide)

You want the song to **follow your style and lyrics**, but still feel **alive**.  
In YUE2 // GROOVE you do not need every ADVANCED slider. Start with **three ideas**, then use a **recipe**.

UI labels below match the GENERATE / ADVANCED // SAMPLING screens.

---

## The only three ideas to remember

| What you want | What to touch | Plain meaning |
|---|---|---|
| **How obedient** is the model to your style + lyrics? | **CFG SCALE** | Higher = sticks to the brief harder. Lower = freer (and can wander). |
| **How wild** is *this take*, once obedience is set? | **Semantic temperature** (under ADVANCED → semantic phase) + **SEED** | Temperature = how daring the performance feels. Seed = “record another take” with the same settings. |
| **How free** is the arrangement / harmony? | **PLAN MODE** (and optionally editing the ABC score) | Chooses how much written structure exists before the sounding pass. |

If you only ever set CFG to `1.5` and leave everything else default, most of the “surprise” room is still in **semantic temperature** and **SEED** — not in CFG alone.

---

## What happens when you hit GENERATE (no jargon required)

Think of two studio steps:

1. **Write a sketch (optional)** — PLAN MODE can ask the model to draft a score (melody, and maybe chords). You can also paste or edit that score (ABC).
2. **Perform the song** — the model turns style + lyrics (+ sketch) into audio. That “performance” is where **semantic** sampling lives.

**CFG** applies to how strongly the brief (style / lyrics) steers the run overall.

You can ignore the **ABC phase** sliders at first. They mainly change how adventurous the *sketch* is. Leave them on defaults unless you care about the written plan.

---

## Recipes (copy these first)

Keep style and lyrics fixed while you compare. Change **one** thing at a time when you are learning.

### 1) “Stay close to my words and style” (safer / more radio)

| Control | Set to |
|---|---|
| CFG SCALE | `1.5`–`1.8` |
| PLAN MODE | **full** |
| Semantic temperature | `0.85`–`0.95` |
| SEED | Pick one and keep it while comparing other knobs |

Optional later: semantic top_p `0.9`, top_k `50`, repetition_penalty `1.25` (slightly less looping).

### 2) “Same song idea, but more surprising takes”

| Control | Set to |
|---|---|
| CFG SCALE | `1.2`–`1.5` (do not jump to `3+` first) |
| PLAN MODE | **full** or **melody** (see below) |
| Semantic temperature | `1.15`–`1.25` |
| SEED | Try several — use **TRY ANOTHER SEED** in SONG |

### 3) “Freer arrangement” (covers, less locked harmony)

| Control | Set to |
|---|---|
| PLAN MODE | **melody** |
| CFG SCALE | about `1.5` |
| Semantic temperature | start at `1.0`, then nudge |

**melody** = melody plan without chord symbols → accompaniment can wander more.  
**full** = melody + chords → feels more “composed.”  
**off** = no sketch → loosest and hardest to steer; use sparingly.

### 4) “Weird but not chaos”

1. Run with PLAN MODE **full** (or PLAN ONLY if you use that flow).  
2. Edit a few bars of the ABC score so the shape is locked.  
3. Generate again with a slightly higher semantic temperature and/or new SEED.

You are freezing the skeleton, then letting the performance improvise.

---

## PLAN MODE in one glance

| PLAN MODE | Feels like | Use when |
|---|---|---|
| **full** | Lead sheet with chords | You want a clear harmonic plan, or you will edit the score |
| **melody** | Melody-only sketch | Covers / freer band arrangement |
| **off** | No written sketch | Experiments only; least predictable |

---

## CFG SCALE in one glance

| CFG | Tendency |
|---|---|
| Empty / default | Protocol default (about `1.0`; **off** mode uses about `1.01`) |
| `1.2`–`1.5` | Mild guidance — good everyday range |
| `1.5`–`2.0` | Tighter to the brief |
| Much higher (e.g. `3+`) | Often harsh or stiff — try only after milder values fail |

Empty CFG in the UI means “use the default,” not the number zero.

---

## SEED in one glance

Same style, lyrics, CFG, plan mode, and sampling + **different SEED** = another take.  
It does not change what the knobs *mean*; it only changes which roll you get.

---

## Leave alone until you need them

| Area | Why wait |
|---|---|
| ABC phase sliders | Mostly “how fancy is the written sketch” |
| ODE steps, dtype, memory, VAE | Sound engine / speed / stability — not musical personality |
| BUDGET PRESET / max tokens | Song length and how long you wait — not “vibe” |
| min tokens / penalty window | Length floor and anti-repeat window — fine print |

---

## Optional: ready-made ADVANCED numbers

**ABC phase (defaults — usually leave as-is)**  
temperature `0.7`, top_p `0.9`, top_k `30`, repetition_penalty `1.005`, penalty_window `100`, min_tokens `32`, max_tokens `4096`

**Semantic phase — pop-stable**  
temperature `0.9`, top_p `0.9`, top_k `50`, repetition_penalty `1.25`  
(+ CFG `1.5`–`1.8`, PLAN MODE **full**)

**Semantic phase — experimental**  
temperature `1.2`, top_p `0.95`, top_k `100`, repetition_penalty `1.15`  
(+ CFG `1.2`–`1.5`, PLAN MODE **melody** or **full** + edited ABC)

---

## Glossary (read only if you want the tech names)

| UI / doc word | Simple meaning |
|---|---|
| **CFG / cfg_scale** | How hard the model is pulled toward your style + lyrics text |
| **PLAN MODE / cot** | Whether generation starts from a full score plan, melody-only plan, or no plan (`full` / `melody` / `off`) |
| **ABC** | The editable score text (the sketch) |
| **ABC phase sampling** | Randomness while *writing* that sketch |
| **Semantic phase sampling** | Randomness while *performing* the sounding song |
| **temperature** | Overall daring / chaos of that phase |
| **top_p / top_k** | How wide a set of next choices is allowed (same family as temperature: lower = safer) |
| **repetition_penalty** | Push-back against repeating the same idea too soon |
| **SEED** | Which take you get with otherwise identical settings |

For instrumental / “no vocals” limits of YuE2 (no Suno-style exclude), see [EXCLUDE_INSTRUMENTAL_RESEARCH.md](EXCLUDE_INSTRUMENTAL_RESEARCH.md).

---

*This guide describes YUE2 // GROOVE GENERATE controls. Under the hood they map to YuE2’s plan + dual sampling settings; you do not need that to use the recipes above.*
