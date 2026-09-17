"""Upstream API contract: every `yue2` symbol and signature the UI relies on.

Skipped when `yue2` is not installed.  When one of these fails after an upstream
upgrade, `yue2_groove/adapter.py` is the file to update.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

yue2 = pytest.importorskip("yue2")


def params(obj):
    return set(inspect.signature(obj).parameters)


def test_pipeline_entry_points_keep_the_keywords_the_ui_passes():
    from yue2 import YuE2Pipeline

    assert {"model", "vae", "revision", "vae_revision", "local_files_only", "progress"} <= params(
        YuE2Pipeline.from_pretrained)
    assert {"device", "memory_budget_gib", "backend", "generation_config", "vae_core_frames",
            "quantization", "offload_ar", "progress"} <= params(YuE2Pipeline.__init__)
    assert {"style", "lyrics", "abc_sampling", "semantic_sampling", "cancelled", "on_token"} <= params(
        YuE2Pipeline.__call__)
    assert {"request", "abc_sampling", "cancelled"} <= params(YuE2Pipeline.plan)
    assert {"latents", "full", "vae"} <= params(YuE2Pipeline.decode)
    assert callable(getattr(YuE2Pipeline, "close", None))
    # the adapter's float32 cast relies on these two private members
    assert callable(getattr(YuE2Pipeline, "_load_model", None))


def test_progress_hooks_the_adapter_wraps_still_exist():
    """Fallback path for pipelines without the on_progress keyword."""
    from yue2 import nar
    from yue2.modeling_vae import YuE2VAE

    assert "on_progress" in params(nar.synthesize)
    assert "on_progress" in params(YuE2VAE.decode_tiled)


def test_protocol_objects_and_fields():
    from yue2.protocol import GenerationConfig, Sampling, SongRequest

    assert set(Sampling.__dataclass_fields__) == {
        "temperature", "top_p", "top_k", "repetition_penalty", "penalty_window",
        "min_tokens", "max_tokens"}
    assert {"style", "lyrics", "cot", "seed", "abc", "cfg_scale", "id"} <= set(
        SongRequest.__dataclass_fields__)
    request = SongRequest(style="pop", lyrics="[Verse]\nla", cot="full", seed=7, cfg_scale=1.5, id="x")
    assert request.guidance == 1.5 and SongRequest(style="a", lyrics="b", cot="off").guidance == 1.01
    assert GenerationConfig(ode_steps=16).ode_steps == 16
    with pytest.raises(ValueError):
        SongRequest(style="a", lyrics="b", cot="off", abc="X:1")


def test_storage_helpers():
    from yue2.storage import resolve_model

    assert {"model", "revision", "local_files_only"} <= params(resolve_model)


def test_saved_artifacts_are_what_the_library_reads(tmp_path):
    """Write a SongResult with upstream's own code and scan it with the Library."""
    from yue2.pipeline import SemanticResult, SongResult, SymbolicPlan
    from yue2.protocol import SongRequest

    from yue2_groove import library

    request = SongRequest(style="English, test tone", lyrics="[Verse]\nHello", cot="full", seed=7, id="demo")
    abc = ('X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=120\n'
           'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
           'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n'
           'V: Vocal\n"C"E4E4G4G4E8z8|\nV: Ins\nZ|\n')
    plan = SymbolicPlan(request, abc, [1, 2, 3], [9, 8, 7], {"seconds": 1.0, "output_tokens": 3}, False)
    semantic = SemanticResult(plan, [4, 5, 6], {"seconds": 2.0, "output_tokens": 3, "output_tps": 1.5}, False)
    audio = np.zeros((48000, 2), dtype=np.float32)
    song = SongResult(audio=audio, sample_rate=48000, semantic=semantic,
                      latents=np.zeros((25, 64), dtype=np.float32),
                      config={"model_dtype": "bfloat16", "device": "cpu", "cfg_scale": 1.0,
                              "generation": {"ode_steps": 32}},
                      weights={"mot": {"files": {}}, "vae": {"files": {}}},
                      timing={"abc": plan.timing, "semantic": semantic.timing,
                              "nar_seconds": 3.0, "vae_seconds": 0.5, "e2e_seconds": 6.5},
                      request_identity="deadbeef")
    directory = tmp_path / "20260901-120000-demo"
    result = song.save_artifacts(directory)
    assert result["status"] == "complete"
    for name in ("request.json", "config.json", "result.json", "score.abc", "audio.flac",
                 "latent.npy", "semantic.npy", "plan.json"):
        assert (directory / name).exists(), name

    items = library.scan(tmp_path)
    assert len(items) == 1
    item, details = library.load(tmp_path, items[0]["rel"])
    assert item is not None and item["kind"] == "song"
    assert details["request"]["style"] == "English, test tone"
    assert details["abc"].startswith("X:1")
    assert details["result"]["status"] == "complete" and details["config"]["model_dtype"] == "bfloat16"
    assert details["audio"] is not None and item["duration"] == pytest.approx(1.0)
    html = library.render_info_html(item, details)
    assert "bfloat16" in html and "audio.flac" in html
    assert json.loads((directory / "result.json").read_text())["audio_seconds"] == pytest.approx(1.0)


def test_cuda_graph_still_selects_flash_attention_by_op_presence_only():
    """Guards the UI's torch-eager fallback (``webui._effective_backend``).

    yue2-v0.1.6 picks FlashAttention whenever ``torch.ops.aten._flash_attention_forward``
    is registered — true on builds that cannot run it (a raising stub) and on pre-Ampere
    GPUs — so the UI routes such CUDA hosts to upstream's eager decoder.  Upstream PR #166
    replaces that check with ``torch.backends.cuda.can_use_flash_attention`` and falls back
    to cuDNN / SDPA *with CUDA graphs kept*.  Once the installed upstream asks torch itself,
    the UI fallback is not just redundant but harmful (it would pin those hosts to the slower
    eager path), so this test fails on purpose to say: retire it.
    """
    from yue2 import cuda_graph

    source = inspect.getsource(cuda_graph)
    retire = ("Upstream now checks FlashAttention availability itself. Retire the workaround: "
              "remove webui._effective_backend / FLASH_FALLBACK_NOTE and the call in "
              "webui.load_pipeline, adapter.cuda_flash_attention_usable, "
              "tests/test_flash_fallback.py, the README paragraph 'CUDA without FlashAttention', "
              "and then this test.")
    assert not any(name in source for name in
                   ("can_use_flash_attention", "is_flash_attention_available")), retire
    assert 'hasattr(torch.ops.aten, "_flash_attention_forward")' in source, (
        "Upstream changed how GraphAR selects its attention backend; re-check whether the "
        "torch-eager fallback in webui._effective_backend is still needed.")

