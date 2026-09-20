"""The GGUF engine (yue2_groove.gguf_engine): request mapping, log parsing, the child
process contract, artifacts, and the BACKEND=auto rule — against stub binaries, so the
suite needs no yue2.cpp build.  The real-binary path was exercised by hand on the M1 Max
(Metal) and is what docs/GGUF_ENGINE.md reports."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from yue2_groove import adapter, gguf_engine
from yue2_groove.webui import runtime

yue2 = pytest.importorskip("yue2")


# ── fixtures ─────────────────────────────────────────────────────────────────


def write_stub(directory: Path, name: str, body: str) -> Path:
    """A Python script standing in for a yue2.cpp binary; ``binary()`` is patched to it."""
    path = directory / f"{name}.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def patch_binaries(monkeypatch, stubs: dict[str, Path]):
    def fake_binary(name):
        return stubs[name]

    monkeypatch.setattr(gguf_engine, "binary", fake_binary)
    # argv[0] is the stub script: run it with this interpreter
    real_popen = gguf_engine.subprocess.Popen

    def popen(argv, **kwargs):
        return real_popen([sys.executable, *argv], **kwargs)

    monkeypatch.setattr(gguf_engine.subprocess, "Popen", popen)


def sampling(**overrides):
    fields = {**runtime.ABC_DEFAULTS, **overrides}
    return adapter.sampling(**fields)


@pytest.fixture
def model_dir():
    """The checkpoint directory (for the tokenizer); skipped when the weights are not cached."""
    try:
        return Path(adapter.resolve_model("m-a-p/YuE2-3B", local_files_only=True))
    except Exception:  # noqa: BLE001 — offline CI without the cache
        pytest.skip("YuE2-3B checkpoint not in the local cache")


# The log of a real yue-synth run (M1 Max, Metal), trimmed to the lines the parser reads.
SYNTH_LOG = """\
[Request] Parsed request.json
[Load] KV backend: MTL0 (CPU threads: 5)
[Store] Load LM: 691 ms
[AR] Score prefill: 120 ms, CFG=1.00, top_k=30, budget=256, songs=1, batch=1
[AR] Score 100/256
[AR] Score 200/256
[AR] Score song 0: 256 tokens (truncated)
[AR] Score: 256 tokens over 1 songs, 256 steps, 2.6 s (10.2 ms/step)
[AR] Semantic 100/512
[AR] Semantic 200/512
[AR] Semantic song 0: end token at step 431
[AR] Semantic song 0: 431 tokens
[AR] Semantic: 431 tokens over 1 songs, 432 steps, 4.5 s (10.4 ms/step)
[Store] Load NAR: 493 ms
[NAR] Step 1/32, 500 ms
[NAR] Step 2/32, 500 ms
[NAR] Step 32/32, 500 ms
[VAE] Decoded: T_latent=431 -> T_audio=827520 (17.24s @ 48kHz), 1234 ms
[Pipeline] Done: 1 tracks, 17.2 s of audio in 30.1 s (0.6x realtime)
"""


# ── request ──────────────────────────────────────────────────────────────────


def test_request_json_mirrors_the_song_request():
    request = adapter.song_request(
        style="pop", lyrics="[Verse]\nla", cot="full", seed=7, cfg_scale=1.5, id="x"
    )
    payload = gguf_engine.build_request(
        request, abc_sampling=sampling(), semantic_sampling=sampling(max_tokens=9000), steps=32
    )
    assert payload["style"] == "pop" and payload["cot"] == "full" and payload["abc"] == ""
    assert payload["lm_seed"] == 7 and payload["seed"] == 7  # both stages from the one seed
    assert payload["cfg_scale"] == 1.5 and payload["steps"] == 32
    assert payload["output_format"] == "wav32"  # raw float: the app clamps and writes FLAC
    assert payload["abc_sampling"]["max_tokens"] == 4096
    assert payload["semantic_sampling"]["max_tokens"] == 9000
    assert "semantic_tokens" not in payload

    default = adapter.song_request(style="pop", lyrics="la", cot="melody", abc="X:1\nK:C\n")
    payload = gguf_engine.build_request(
        default, abc_sampling=sampling(), semantic_sampling=sampling(), steps=16
    )
    assert payload["cfg_scale"] == -1.0  # None → protocol default, like upstream
    assert payload["abc"] == "X:1\nK:C\n"
    replay = gguf_engine.build_request(
        default,
        abc_sampling=sampling(),
        semantic_sampling=sampling(),
        steps=16,
        semantic_tokens=[1, 2, 3],
    )
    assert replay["semantic_tokens"] == "1,2,3"


# ── log parsing ──────────────────────────────────────────────────────────────


def test_log_parser_reports_strides_truncation_and_timings():
    log = gguf_engine.ChildLog()
    tokens, stages = [], []
    for line in SYNTH_LOG.splitlines():
        gguf_engine.parse_line(
            line,
            log,
            on_token=lambda phase, token, count=1: tokens.append((phase, count)),
            on_progress=lambda stage, done, total: stages.append((stage, done, total)),
        )
    # the AR stage is reported every 100 tokens and once more at its end
    assert tokens == [
        ("abc", 100),
        ("abc", 100),
        ("abc", 56),
        ("semantic", 100),
        ("semantic", 100),
        ("semantic", 231),
    ]
    assert log.tokens == {"abc": 256, "semantic": 431}
    assert log.truncated == {"abc": True, "semantic": False}
    assert log.backend == "MTL0"
    assert log.timing["abc"]["output_tokens"] == 256 and log.timing["abc"]["seconds"] == 2.6
    assert log.timing["semantic"]["output_tps"] == pytest.approx(431 / 4.5)
    assert log.timing["nar_seconds"] == pytest.approx(1.5)
    assert log.timing["vae_seconds"] == pytest.approx(1.234)
    assert log.timing["load"] == {"lm_load_seconds": 0.691, "nar_load_seconds": 0.493}
    assert log.timing["e2e_seconds"] == 30.1
    assert stages[:2] == [("nar", 1, 32), ("nar", 2, 32)] and stages[-1] == ("vae", 1, 1)
    assert not log.fatal


def test_log_parser_keeps_the_first_fatal():
    log = gguf_engine.ChildLog()
    gguf_engine.parse_line(
        "[AR] FATAL: prefix 30000 + budget 9000 + end exceeds context 24576", log
    )
    gguf_engine.parse_line("[Pipeline] FATAL: later", log)
    assert log.fatal.startswith("prefix 30000")


# ── the child process ────────────────────────────────────────────────────────


def test_run_child_streams_progress_and_raises_on_failure(tmp_path, monkeypatch):
    ok = write_stub(
        tmp_path,
        "ok",
        """
        import sys
        for line in ("[Load] KV backend: CUDA0", "[NAR] Step 1/2, 10 ms", "[NAR] Step 2/2, 10 ms",
                     "[VAE] Tiled decode done: 3 tiles -> T_audio=96000 (2.00s @ 48kHz), 77 ms"):
            print(line, file=sys.stderr, flush=True)
    """,
    )
    bad = write_stub(
        tmp_path,
        "bad",
        """
        import sys
        print("[VAE] FATAL: cannot load /nope.gguf", file=sys.stderr, flush=True)
        sys.exit(1)
    """,
    )
    patch_binaries(monkeypatch, {})
    seen = []
    log = gguf_engine.run_child([str(ok)], on_progress=lambda s, d, t: seen.append((s, d, t)))
    assert log.backend == "CUDA0" and seen == [("nar", 1, 2), ("nar", 2, 2), ("vae", 1, 1)]
    assert log.timing["vae_seconds"] == 0.077
    with pytest.raises(RuntimeError, match=r"cannot load /nope\.gguf"):
        gguf_engine.run_child([str(bad)])


def test_run_child_cancel_terminates_the_child(tmp_path, monkeypatch):
    slow = write_stub(
        tmp_path,
        "slow",
        """
        import sys, time
        print("[AR] Score 100/4096", file=sys.stderr, flush=True)
        time.sleep(30)
    """,
    )
    patch_binaries(monkeypatch, {})
    flag = {"stop": False}

    def on_token(phase, token, count=1):
        flag["stop"] = True  # cancel as soon as the first progress line arrives

    with pytest.raises(InterruptedError):
        gguf_engine.run_child([str(slow)], cancelled=lambda: flag["stop"], on_token=on_token)


# ── the pipeline object and its artifacts ────────────────────────────────────


SYNTH_STUB = """
    # a fake yue-synth: honours --out/--tokens/--latent/--score, prints a plausible log
    import json, struct, sys, wave
    import numpy as np
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    request = json.load(open(args["--request"]))
    frames = 50
    rate = 48000
    audio = (np.random.default_rng(int(request["seed"])).standard_normal((frames * 1920, 2)) * 0.1)
    with wave.open(args["--out"], "wb") as w:
        w.setnchannels(2); w.setsampwidth(4); w.setframerate(rate)
        w.setcomptype("NONE", "not compressed")
        # WAVE_FORMAT_IEEE_FLOAT needs a different fmt chunk; write PCM32 instead and let the
        # reader scale it — soundfile reads both as float32
        w.writeframes((np.clip(audio, -1, 1) * 2**31).astype("<i4").tobytes())
    tokens = list(range(1, frames + 1))
    open(args["--tokens"], "w").write(",".join(map(str, tokens)) + "\\n")
    np.zeros((frames, 64), dtype=np.float32).tofile(args["--latent"])
    if not request.get("abc"):
        open(args["--score"], "w").write("X:1\\nT:stub\\nM:4/4\\nL:1/8\\nK:C\\n|C D E F|\\n")
    for line in ("[Load] KV backend: CUDA0", "[AR] Score song 0: 12 tokens",
                 "[AR] Score: 12 tokens over 1 songs, 12 steps, 0.1 s (8.3 ms/step)",
                 "[AR] Semantic song 0: %d tokens" % frames,
                 "[AR] Semantic: %d tokens over 1 songs, %d steps, 0.5 s (10.0 ms/step)" % (frames, frames + 1),
                 "[NAR] Step 32/32, 100 ms",
                 "[VAE] Decoded: T_latent=%d -> T_audio=%d (2.00s @ 48kHz), 20 ms" % (frames, frames * 1920),
                 "[Pipeline] Done: 1 tracks, 2.0 s of audio in 1.0 s (0.5x realtime)"):
        print(line, file=sys.stderr, flush=True)
"""


def make_pipe(tmp_path, model_dir) -> gguf_engine.GgufPipeline:
    backbone, vae = tmp_path / "YuE2-3B-Q8_0.gguf", tmp_path / "YuE2-Vae-F32.gguf"
    backbone.write_bytes(b"gguf-stub")
    vae.write_bytes(b"vae-stub")
    return gguf_engine.GgufPipeline(
        backbone=backbone,
        vae=vae,
        model_dir=model_dir,
        vae_dir=model_dir,
        quant="Q8_0",
        ode_steps=32,
        vae_core_frames=None,
        weights={
            "mot": {"files": {backbone.name: {"sha256": "a", "bytes": 9}}},
            "vae": {"files": {vae.name: {"sha256": "b", "bytes": 8}}},
        },
    )


def test_generate_writes_upstreams_run_layout(tmp_path, monkeypatch, model_dir):
    from yue2.pipeline import SymbolicPlan

    from yue2_groove import library

    patch_binaries(monkeypatch, {"yue-synth": write_stub(tmp_path, "yue-synth", SYNTH_STUB)})
    pipe = make_pipe(tmp_path, model_dir)
    request = adapter.song_request(
        style="pop", lyrics="[Verse]\nla la", cot="full", seed=3, cfg_scale=1.5
    )
    counts, stages = {}, []

    def on_token(phase, token, count=1):
        counts[phase] = counts.get(phase, 0) + count

    song = adapter.generate(
        pipe,
        request,
        abc_sampling=sampling(),
        semantic_sampling=sampling(max_tokens=9000),
        on_token=on_token,
        on_progress=lambda s, d, t: stages.append(s),
    )
    assert counts == {"abc": 12, "semantic": 50} and stages == ["nar", "vae"]
    assert pipe.device == "CUDA0"
    assert song.abc.startswith("X:1") and len(song.tokens) == 50 and song.latents.shape == (50, 64)
    assert song.timing["semantic"]["output_tps"] == pytest.approx(100.0)
    assert song.timing["abc"]["output_tokens"] == 12 and song.timing["nar_seconds"] == 0.1

    outdir = tmp_path / "run"
    result = song.save_artifacts(outdir)
    names = {p.name for p in outdir.iterdir()}
    assert {
        "audio.flac",
        "score.abc",
        "abc_tokens.npy",
        "prefix.npy",
        "plan.json",
        "plan_manifest.json",
        "semantic.npy",
        "latent.npy",
        "request.json",
        "config.json",
        "result.json",
    } <= names
    # the plan is upstream's own: loadable by upstream, prefix built by upstream's rule
    plan = SymbolicPlan.load(outdir)
    tokenizer = adapter.text_tokenizer(model_dir)
    assert plan.prefix == adapter.token_prefixes(request, tokenizer, tokenizer.encode(song.abc))
    assert np.load(outdir / "semantic.npy").tolist() == list(range(1, 51))
    assert result["status"] == "complete" and result["audio_seconds"] == pytest.approx(2.0)
    assert result["engine"]["name"] == "yue2.cpp" and result["weights"]["mot"]["files"]
    config = json.loads((outdir / "config.json").read_text())
    assert config["backend"] == "gguf" and config["quantization"] == "Q8_0"
    assert config["model_dtype"] == "q8_0 (gguf)" and config["device"] == "CUDA0"
    assert config["cfg_scale"] == 1.5 and config["overrides"] == {"cfg_scale": 1.5}
    # and the Library reads it like any other run
    items = library.scan(tmp_path)
    assert [i["kind"] for i in items] == ["song"] and items[0]["status"] == "complete"
    det = library.details(items[0]["path"])
    html = library.render_info_html(items[0], det)
    assert "gguf" in html and "Q8_0" in html and "YuE2-3B-Q8_0.gguf" in html


def test_generate_with_an_external_score_keeps_that_exact_text(tmp_path, monkeypatch, model_dir):
    patch_binaries(monkeypatch, {"yue-synth": write_stub(tmp_path, "yue-synth", SYNTH_STUB)})
    pipe = make_pipe(tmp_path, model_dir)
    abc = "X:1\nT:mine\nM:4/4\nL:1/8\nK:G\n|G A B c|\n"
    request = adapter.song_request(style="pop", lyrics="la", cot="melody", seed=1, abc=abc)
    song = adapter.generate(pipe, request, abc_sampling=sampling(), semantic_sampling=sampling())
    assert song.abc == abc
    assert song.timing["abc"] == {
        "seconds": 0.0,
        "output_tokens": 0,
        "external_prefix_tokens": len(adapter.text_tokenizer(model_dir).encode(abc)),
    }


def test_plan_and_decode_go_through_their_own_binaries(tmp_path, monkeypatch, model_dir):
    plan_stub = write_stub(
        tmp_path,
        "yue-plan",
        """
        import sys
        args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
        open(args["--out"], "w").write("X:1\\nK:D\\n|D E F G|\\n")
        print("[AR] Score song 0: 9 tokens", file=sys.stderr)
        print("[AR] Score: 9 tokens over 1 songs, 9 steps, 0.2 s (22.2 ms/step)", file=sys.stderr)
    """,
    )
    codec_stub = write_stub(
        tmp_path,
        "neural-codec",
        """
        import sys, wave
        import numpy as np
        argv = [a for a in sys.argv[1:] if a not in ("--decode", "--encode")]  # flags without values
        args = dict(zip(argv[::2], argv[1::2]))
        frames = np.fromfile(args["-i"], dtype=np.float32).reshape(-1, 64).shape[0]
        with wave.open(args["-o"], "wb") as w:
            w.setnchannels(2); w.setsampwidth(4); w.setframerate(48000)
            w.writeframes(np.zeros((frames * 1920, 2), dtype="<i4").tobytes())
        print("[VAE] Decoded: T_latent=%d -> T_audio=%d (x @ 48kHz), 5 ms" % (frames, frames * 1920), file=sys.stderr)
    """,
    )
    patch_binaries(monkeypatch, {"yue-plan": plan_stub, "neural-codec": codec_stub})
    pipe = make_pipe(tmp_path, model_dir)
    request = adapter.song_request(style="pop", lyrics="la", cot="full", seed=1)
    plan = adapter.plan(pipe, request, abc_sampling=sampling())
    assert plan.abc.startswith("X:1\nK:D") and plan.timing["output_tokens"] == 9
    plan.save(tmp_path / "plan")
    assert (tmp_path / "plan" / "plan_manifest.json").exists()

    audio = adapter.decode(
        pipe, np.zeros((10, 64), dtype=np.float32), on_progress=lambda d, t: None
    )
    assert audio.shape == (10 * 1920, 2) and audio.dtype == np.float32
    with pytest.raises(RuntimeError, match="own VAE"):
        adapter.decode(pipe, np.zeros((10, 64), dtype=np.float32), vae="/other/vae")


def test_model_dtype_and_close_dispatch_on_the_gguf_pipeline(tmp_path, model_dir):
    pipe = make_pipe(tmp_path, model_dir)
    assert adapter.model_dtype(pipe) == "q8_0 (gguf)"
    adapter.close_pipeline(pipe)  # nothing resident: never raises


# ── locating binaries, the auto rule, the rail ───────────────────────────────


def test_binary_lookup_order_and_install_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("YUE2_GROOVE_YUE2CPP", str(tmp_path / "nowhere"))
    monkeypatch.setattr(gguf_engine, "repo_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(gguf_engine.shutil, "which", lambda name: None)
    assert gguf_engine.binary_dir() is None and not gguf_engine.available()
    with pytest.raises(RuntimeError, match=r"yue2\.cpp binaries not found"):
        gguf_engine.binary("yue-synth")
    installed = tmp_path / "repo" / "bin" / "yue2cpp"
    installed.mkdir(parents=True)
    (installed / "yue-synth.exe").write_bytes(b"")  # the Windows spelling counts too
    monkeypatch.delenv("YUE2_GROOVE_YUE2CPP")
    assert gguf_engine.binary_dir() == installed
    assert gguf_engine.binary("yue-synth").name == "yue-synth.exe"


@pytest.mark.parametrize(
    ("device", "vram", "installed", "expected"),
    [
        ("cuda", 11.99, True, "gguf"),  # a 12 GB card
        ("cuda", 15.9, True, "gguf"),
        ("cuda", 16.0, True, "torch"),  # 16 GB runs the reference configuration
        ("cuda", 23.5, True, "torch"),
        ("cuda", 11.99, False, "torch"),  # no binaries: reference engine, with a hint
        ("mps", None, True, "torch"),  # Apple Silicon: never automatic
        ("cpu", None, True, "torch"),
    ],
)
def test_auto_backend_rule(monkeypatch, device, vram, installed, expected):
    monkeypatch.delenv("YUE2_GROOVE_GGUF_VRAM_GIB", raising=False)
    backend, note = gguf_engine.auto_backend(device, vram, installed=installed)
    assert backend == expected
    if expected == "gguf":
        assert "GGUF" in note
    if device == "cuda" and vram is not None and vram < 16 and not installed:
        assert "no yue2.cpp binaries" in note


def test_auto_threshold_is_configurable(monkeypatch):
    monkeypatch.setenv("YUE2_GROOVE_GGUF_VRAM_GIB", "24")
    assert gguf_engine.auto_backend("cuda", 22.0, installed=True)[0] == "gguf"
    monkeypatch.setenv("YUE2_GROOVE_GGUF_VRAM_GIB", "not a number")
    assert gguf_engine.auto_backend("cuda", 22.0, installed=True)[0] == "torch"


def test_resolve_backend_honours_an_explicit_choice(monkeypatch):
    monkeypatch.setattr(gguf_engine, "cuda_total_vram_gib", lambda: 8.0)
    monkeypatch.setattr(gguf_engine, "available", lambda: True)
    assert runtime.resolve_backend("torch", "cuda") == ("torch", "")
    assert runtime.resolve_backend("gguf", "mps") == ("gguf", "")
    assert runtime.resolve_backend("auto", "cuda")[0] == "gguf"
    assert runtime.resolve_backend("nonsense", "cuda") == ("torch", "")
    assert "gguf" in runtime.BACKEND_CHOICES and runtime.BACKEND_MODES[0] == "auto"


def test_load_pipeline_gguf_without_binaries_reports_the_hint(monkeypatch, tmp_path):
    import gradio as gr

    monkeypatch.setenv("YUE2_GROOVE_YUE2CPP", str(tmp_path / "nowhere"))
    monkeypatch.setattr(gguf_engine, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(gguf_engine.shutil, "which", lambda name: None)
    settings = runtime.RuntimeSettings(
        "cpu",
        "float32",
        "gguf",
        "none",
        False,
        24,
        32,
        "auto",
        "m-a-p/YuE2-3B",
        "standard",
        "",
        "",
        "",
        False,
    )
    with pytest.raises(gr.Error, match=r"yue2\.cpp binaries not found"):
        runtime.load_pipeline(settings)
    legacy = runtime.RuntimeSettings(
        "cpu",
        "float32",
        "gguf",
        "none",
        False,
        24,
        32,
        "auto",
        "m-a-p/YuE2-3B",
        "legacy",
        "",
        "",
        "",
        False,
    )
    with pytest.raises(gr.Error, match="standard YuE2-Vae"):
        runtime.load_pipeline(legacy)


def test_env_knobs(monkeypatch, tmp_path):
    monkeypatch.setenv("YUE2_GROOVE_GGUF_QUANT", "q6_k")
    assert gguf_engine.quant() == "Q6_K"
    monkeypatch.setenv("YUE2_GROOVE_GGUF_QUANT", "Q4_0")  # not offered: falls back
    assert gguf_engine.quant() == "Q8_0"
    monkeypatch.setenv("YUE2_GROOVE_GGUF_MAX_SEQ", "12288")
    assert gguf_engine.max_seq() == 12288
    monkeypatch.setenv("YUE2_GROOVE_GGUF_MAX_SEQ", "99999")  # above the context: ignored
    assert gguf_engine.max_seq() is None
    monkeypatch.setenv("YUE2_GROOVE_GGUF", str(tmp_path / "g"))
    assert gguf_engine.gguf_dir() == tmp_path / "g"
    monkeypatch.delenv("YUE2_GROOVE_GGUF")
    monkeypatch.setenv("YUE2_GROOVE_MODELS", str(tmp_path / "m"))
    assert gguf_engine.gguf_dir() == tmp_path / "m" / "gguf"


def test_sha256_cache_sidecar(tmp_path):
    big = tmp_path / "x.gguf"
    big.write_bytes(b"abc" * 1000)
    digest = gguf_engine.sha256_file(big)
    side = json.loads((tmp_path / "x.gguf.sha256").read_text())
    assert side["sha256"] == digest and side["bytes"] == 3000
    big.write_bytes(b"abd" * 1000)
    os.utime(big, (1, 1))  # a changed file invalidates the cache (size same, mtime differs)
    assert gguf_engine.sha256_file(big) != digest


def test_install_unpacks_this_platforms_asset(tmp_path, monkeypatch):
    """`python -m yue2_groove.gguf_engine install`: the release asset lands in bin/yue2cpp,
    executable, with its VERSION line reported — the download itself is stubbed."""
    import io
    import tarfile
    import zipfile

    asset = gguf_engine.platform_asset()  # this machine's name, e.g. …-macos-arm64-metal.tar.gz
    payload = io.BytesIO()
    if asset.endswith(".zip"):
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("yue-synth.exe", b"MZ")
            archive.writestr("VERSION", "yue2.cpp test windows-x64\n")
    else:
        with tarfile.open(fileobj=payload, mode="w:gz") as archive:
            for name, data in (("yue-synth", b"#!/bin/sh\n"), ("VERSION", b"yue2.cpp test\n")):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    seen = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    def urlopen(url, timeout=0):
        seen["url"] = url
        return Response(payload.getvalue())

    monkeypatch.setattr(gguf_engine.urllib.request, "urlopen", urlopen)
    said = []
    dest = gguf_engine.install_binaries(tmp_path / "bin", tag="v9.9.9", say=said.append)
    assert seen["url"] == f"{gguf_engine.RELEASES}/download/v9.9.9/{asset}"
    assert dest == tmp_path / "bin" and gguf_engine._binary(dest, "yue-synth") is not None
    if os.name != "nt":
        assert os.access(dest / "yue-synth", os.X_OK)
    assert said[-1].startswith("yue2.cpp test")
    gguf_engine.install_binaries(tmp_path / "bin2", tag="latest", say=said.append)
    assert seen["url"] == f"{gguf_engine.RELEASES}/latest/download/{asset}"
