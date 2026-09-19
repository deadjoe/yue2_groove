"""`yue2_groove.adapter` progress plumbing, with fake pipelines and no model.

Two cases matter: a pipeline that accepts `on_progress` natively (upstream after
the pull request), and one that predates it (upstream `yue2-v0.1.6`), where the
adapter wraps `yue2.nar.synthesize` / `YuE2VAE.decode_tiled` for the duration of
the call and restores them afterwards.
"""

from __future__ import annotations

import sys
import types

import pytest

from yue2_groove import adapter


class Request:
    style, lyrics, cot, seed, abc, cfg_scale, id = "pop", "la", "full", 7, None, None, "song"


def install_fake_yue2(monkeypatch):
    """A minimal `yue2` with the two internals the fallback wraps."""
    nar = types.ModuleType("yue2.nar")

    def synthesize(
        model, prefix, tokens, seed, *, steps, context, offload_ar, cancelled, on_progress
    ):
        for done in (1, steps):
            if on_progress is not None:
                on_progress(done, steps)
        return "latents"

    nar.synthesize = synthesize
    modeling_vae = types.ModuleType("yue2.modeling_vae")

    class YuE2VAE:
        def decode_tiled(self, latent, *, core_frames, halo_frames, output_device, on_progress):
            for done in (1, 2):
                if on_progress is not None:
                    on_progress(done, 2)
            return "audio"

    modeling_vae.YuE2VAE = YuE2VAE
    package = types.ModuleType("yue2")
    package.nar, package.modeling_vae = nar, modeling_vae
    monkeypatch.setitem(sys.modules, "yue2", package)
    monkeypatch.setitem(sys.modules, "yue2.nar", nar)
    monkeypatch.setitem(sys.modules, "yue2.modeling_vae", modeling_vae)
    return nar, YuE2VAE


class LegacyPipeline:
    """Upstream shape before the on_progress keyword: reports only to its own display."""

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        style,
        lyrics,
        cot,
        seed,
        abc,
        cfg_scale,
        id,
        abc_sampling,
        semantic_sampling,
        cancelled,
        on_token,
    ):
        import yue2.nar as nar
        from yue2.modeling_vae import YuE2VAE

        self.calls.append((style, lyrics, cot, seed, id))
        latents = nar.synthesize(
            None,
            [1],
            [2],
            seed,
            steps=4,
            context=1,
            offload_ar=False,
            cancelled=cancelled,
            on_progress=None,
        )
        audio = YuE2VAE().decode_tiled(
            latents, core_frames=1, halo_frames=16, output_device="cpu", on_progress=None
        )
        return (latents, audio)

    def decode(self, latents, *, full=False, vae=None):
        from yue2.modeling_vae import YuE2VAE

        if full:
            return "audio-full"
        return YuE2VAE().decode_tiled(
            latents, core_frames=1, halo_frames=16, output_device="cpu", on_progress=None
        )


class NativePipeline(LegacyPipeline):
    def __call__(self, *, on_progress=None, **kwargs):
        self.native = on_progress
        on_progress("nar", 4, 4)
        on_progress("vae", 2, 2)
        return "native"

    def decode(self, latents, *, full=False, vae=None, on_progress=None):
        on_progress(2, 2)
        return "native-audio"


def test_legacy_pipeline_gets_progress_through_wrapped_internals(monkeypatch):
    nar, vae_class = install_fake_yue2(monkeypatch)
    original = (nar.synthesize, vae_class.decode_tiled)
    pipe, seen = LegacyPipeline(), []
    assert not adapter.supports_on_progress(pipe)
    result = adapter.generate(
        pipe,
        Request,
        abc_sampling="a",
        semantic_sampling="s",
        on_progress=lambda stage, done, total: seen.append((stage, done, total)),
    )
    assert result == ("latents", "audio")
    assert seen == [("nar", 1, 4), ("nar", 4, 4), ("vae", 1, 2), ("vae", 2, 2)]
    assert (nar.synthesize, vae_class.decode_tiled) == original  # restored
    assert pipe.calls == [("pop", "la", "full", 7, "song")]


def test_legacy_pipeline_without_callback_is_not_wrapped(monkeypatch):
    nar, vae_class = install_fake_yue2(monkeypatch)
    original = (nar.synthesize, vae_class.decode_tiled)
    assert adapter.generate(LegacyPipeline(), Request, abc_sampling="a", semantic_sampling="s") == (
        "latents",
        "audio",
    )
    assert (nar.synthesize, vae_class.decode_tiled) == original


def test_native_pipeline_receives_the_keyword(monkeypatch):
    install_fake_yue2(monkeypatch)
    pipe, seen = NativePipeline(), []
    assert adapter.supports_on_progress(pipe)
    assert (
        adapter.generate(
            pipe,
            Request,
            abc_sampling="a",
            semantic_sampling="s",
            on_progress=lambda *a: seen.append(a),
        )
        == "native"
    )
    assert seen == [("nar", 4, 4), ("vae", 2, 2)] and pipe.native is not None


def test_decode_paths(monkeypatch):
    install_fake_yue2(monkeypatch)
    seen = []
    assert (
        adapter.decode(LegacyPipeline(), "z", on_progress=lambda d, t: seen.append((d, t)))
        == "audio"
    )
    assert seen == [(1, 2), (2, 2)]
    seen.clear()
    assert (
        adapter.decode(
            LegacyPipeline(), "z", full=True, on_progress=lambda d, t: seen.append((d, t))
        )
        == "audio-full"
    )
    assert seen == [(1, 1)]
    seen.clear()
    assert (
        adapter.decode(NativePipeline(), "z", on_progress=lambda d, t: seen.append((d, t)))
        == "native-audio"
    )
    assert seen == [(2, 2)]


def test_wrapping_is_restored_after_an_exception(monkeypatch):
    nar, vae_class = install_fake_yue2(monkeypatch)
    original = (nar.synthesize, vae_class.decode_tiled)

    class Failing(LegacyPipeline):
        def __call__(self, **kwargs):
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        adapter.generate(
            Failing(), Request, abc_sampling="a", semantic_sampling="s", on_progress=lambda *a: None
        )
    assert (nar.synthesize, vae_class.decode_tiled) == original


def test_missing_yue2_gives_an_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "yue2", None)  # makes `import yue2` fail
    with pytest.raises(RuntimeError, match="not installed"):
        adapter.sampling(temperature=1.0)


def test_doctor_command_shape():
    cmd = adapter.doctor_command(
        "m-a-p/YuE2-3B", "m-a-p/YuE2-Vae", revision="r1", offline=True, verify=True
    )
    assert cmd[1:] == [
        "-m",
        "yue2.cli",
        "doctor",
        "--model",
        "m-a-p/YuE2-3B",
        "--vae",
        "m-a-p/YuE2-Vae",
        "--revision",
        "r1",
        "--offline",
        "--verify-hashes",
    ]
