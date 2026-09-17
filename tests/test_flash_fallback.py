"""CUDA hosts whose torch build or GPU cannot run FlashAttention are routed to torch-eager.

Upstream's ``GraphAR`` picks FlashAttention whenever the ATen op exists, which is every CUDA
build — including Windows wheels compiled without it (``USE_FLASH_ATTENTION was not enabled
for build``) and Turing GPUs.  The probe lives in the adapter; the remap is a UI policy and
must never touch MPS/CPU or a CUDA host where FlashAttention works (the Linux path).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yue2_groove import adapter  # noqa: E402

# ── adapter probe ────────────────────────────────────────────────────────────

def _cuda(monkeypatch, *, available=True, built=True, capability=(8, 9), api=True):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    if api:
        monkeypatch.setattr(torch.backends.cuda, "is_flash_attention_available", lambda: built,
                            raising=False)
    else:
        monkeypatch.delattr(torch.backends.cuda, "is_flash_attention_available", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)


def test_probe_true_on_a_flash_capable_build_and_gpu(monkeypatch):
    _cuda(monkeypatch)                                   # the validated Linux + L4 (sm_89) case
    assert adapter.cuda_flash_attention_usable() is True


def test_probe_false_when_torch_was_built_without_flash_attention(monkeypatch):
    _cuda(monkeypatch, built=False)                      # Windows CUDA wheel / ROCm
    assert adapter.cuda_flash_attention_usable() is False


def test_probe_false_on_a_turing_gpu(monkeypatch):
    _cuda(monkeypatch, capability=(7, 5))                # RTX 20xx: kernel needs sm_80+
    assert adapter.cuda_flash_attention_usable() is False


def test_probe_false_without_cuda(monkeypatch):
    _cuda(monkeypatch, available=False)
    assert adapter.cuda_flash_attention_usable() is False


def test_probe_assumes_capable_when_torch_cannot_answer(monkeypatch):
    _cuda(monkeypatch, api=False)                        # older torch: prior behaviour
    assert adapter.cuda_flash_attention_usable() is True


def test_probe_leaves_unknown_devices_to_upstream(monkeypatch):
    _cuda(monkeypatch)

    def boom(device=None):
        raise RuntimeError("no such device")
    monkeypatch.setattr(torch.cuda, "get_device_capability", boom)
    assert adapter.cuda_flash_attention_usable() is True


# ── UI policy ────────────────────────────────────────────────────────────────

webui = pytest.importorskip("yue2_groove.webui")


def _probe(monkeypatch, value):
    calls = []

    def probe(device=None):
        calls.append(device)
        return value
    monkeypatch.setattr(webui.adapter, "cuda_flash_attention_usable", probe)
    return calls


def test_cuda_torch_is_remapped_only_when_flash_is_unusable(monkeypatch):
    _probe(monkeypatch, False)
    backend, reason = webui._effective_backend("cuda", "torch")
    assert backend == "torch-eager"
    assert reason == webui.FLASH_FALLBACK_NOTE and "FlashAttention" in reason


def test_cuda_torch_keeps_cuda_graphs_when_flash_works(monkeypatch):
    _probe(monkeypatch, True)
    assert webui._effective_backend("cuda", "torch") == ("torch", "")


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("backend", ["torch", "torch-eager"])
def test_non_cuda_devices_never_consult_the_probe(monkeypatch, device, backend):
    calls = _probe(monkeypatch, False)                   # would remap if it were consulted
    assert webui._effective_backend(device, backend) == (backend, "")
    assert calls == []


@pytest.mark.parametrize("backend", ["torch-eager", "vllm"])
def test_other_cuda_backends_are_untouched(monkeypatch, backend):
    calls = _probe(monkeypatch, False)
    assert webui._effective_backend("cuda", backend) == (backend, "")
    assert calls == []


def test_load_pipeline_records_the_fallback_in_the_note_and_cache_key(monkeypatch):
    """The remapped backend reaches the adapter, the note, and the pipe cache key."""
    seen = {}

    def fake_load(model, **kwargs):
        seen.update(kwargs)
        return object(), "bfloat16"
    monkeypatch.setattr(webui.adapter, "load_pipeline", fake_load)
    monkeypatch.setattr(webui, "unload_pipeline", lambda: None)
    monkeypatch.setattr(webui, "_pick_device", lambda device: "cuda")
    monkeypatch.setattr(webui, "resolve_vae", lambda choice, custom: ("/vae", "standard"))
    _probe(monkeypatch, False)
    monkeypatch.setattr(webui, "_PIPE", None)
    monkeypatch.setattr(webui, "_PIPE_KEY", None)
    args = ("auto", "bfloat16", "torch", "none", False, 24, 32, "auto", "m", "standard", "",
            "", "", False)
    pipe, note = webui.load_pipeline(*args, progress=None)
    assert seen["backend"] == "torch-eager"
    assert "backend=torch-eager" in note and webui.FLASH_FALLBACK_NOTE in note
    # A second call with the same UI settings reuses the pipe and repeats the reason.
    pipe2, note2 = webui.load_pipeline(*args, progress=None)
    assert pipe2 is pipe and note2.startswith("Model ready") and "torch-eager" in note2


def test_load_pipeline_passes_torch_through_when_flash_works(monkeypatch):
    seen = {}

    def fake_load(model, **kwargs):
        seen.update(kwargs)
        return object(), "bfloat16"
    monkeypatch.setattr(webui.adapter, "load_pipeline", fake_load)
    monkeypatch.setattr(webui, "unload_pipeline", lambda: None)
    monkeypatch.setattr(webui, "_pick_device", lambda device: "cuda")
    monkeypatch.setattr(webui, "resolve_vae", lambda choice, custom: ("/vae", "standard"))
    _probe(monkeypatch, True)
    monkeypatch.setattr(webui, "_PIPE", None)
    monkeypatch.setattr(webui, "_PIPE_KEY", None)
    _pipe, note = webui.load_pipeline("auto", "bfloat16", "torch", "none", False, 24, 32, "auto",
                                      "m", "standard", "", "", "", False, progress=None)
    assert seen["backend"] == "torch"
    assert "backend=torch " in note and "FlashAttention" not in note
