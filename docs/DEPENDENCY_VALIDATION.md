# Dependency update validation — 2026-10-17

This candidate updates SoundFile 0.13.1 → 0.14.0, safetensors 0.7.0 → 0.8.0,
Ruff 0.16.7 → 0.17.1, and the vendored abcjs 6.5.2 → 6.7.1.
Gradio is held at the previously validated 6.17.3 to prevent an unrelated UI / Hub
migration during installation. An environment already using a later Gradio will
therefore be returned to 6.17.3. NumPy 2.2.6, Transformers 4.57.6, Hub 0.36.2,
Accelerate 1.13.0, tiktoken 0.12.0 and the upstream YuE2 tag stay unchanged.

## Installation and automated checks

- An independent copy of the existing macOS/Python 3.12 environment was upgraded
  with the platform override file and the actual `yue2-v0.1.6` Git tag
  (`9c6c4b349be978b06a9d0d958471a07a6cdeff4d`). All 309 tests passed.
- A separate empty environment was populated by `uv sync` with the yue2, gguf and
  test extras. All 309 tests passed there too. Ruff lint/format, Pyright and
  compileall passed. Repeating `uv sync` and `uv run` retained the selected versions.
- The universal uv resolution retained Torch 2.14.1 on macOS and 2.10.0 elsewhere.
  This is resolver evidence, not a substitute for native platform execution.
- New regression checks exercise stereo WAV/FLOAT and FLAC/PCM_24 round trips,
  32 concurrent audio reads, and BF16 checkpoint round trips on CPU and MPS.
- `uv pip check` reports exactly the three intentional YuE2 metadata conflicts on
  macOS: Torch, SoundFile and safetensors. The two new replacements are declared
  in both platform override files and in project-mode uv overrides.

## Model regression

The final candidate combination, with the release-tag upstream code, completed
BF16/MPS planning, 1,280 semantic tokens, 32 NAR steps and VAE decoding in 165.5
seconds. It produced 51.2 seconds of stereo audio. Audio and latents were finite.
ABC token, semantic token, latent and audio files were byte-identical to the
Torch 2.14.1 baseline from commit `473bff6` with the same input and seed.

This was a bounded run: the semantic sequence reached its configured token cap.
It is not a full-song quality evaluation or a performance benchmark. Local
artifacts are in `runs/safe-upgrade-tag-validation/`, with the reference in
`runs/torch-2.14.1-validation/` (neither directory is distributed).

## Browser and launcher checks

Chromium loaded the application with no page errors and switched through all
seven Studio tabs. The actual score renderer displayed the bundled melody at
1,440-pixel and 390-pixel viewport widths. Screenshots were pixel-identical to
those rendered with the old abcjs bundle at both widths. This fixture check does
not cover all possible generated notation or every browser.

The companion Pinokio launcher change makes its Gradio asset-repair step install
from the app requirement and platform overrides, instead of independently
installing the latest Gradio. The command was dry-run against the candidate:
only Gradio 6.17.3 and the editable app would be reinstalled. Four launcher policy
tests cover macOS/Linux/Windows selection and the macOS CUDA exclusion.

## Integration gate

The companion launcher fix should be merged before the app update so new installs
cannot overwrite the Gradio pin. Existing users use Update, then Start; Reset is
not required. Do not advertise the candidate as a released version until the
native runtime CI and regular CI pass and the normal release process is complete.
Native CI verifies CPU audio/checkpoint handling and UI construction on Linux,
Windows and macOS; it does not establish CUDA generation or Cover-model accuracy.
The independent SheetSage2 environment is not changed by this update.
