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
    # yue2.cpp caps the budget at duration x 25 frames and defaults duration to 360 s (9 000);
    # the request carries a duration that never cuts the budget below what was asked for
    for budget in (9000, 9001, 15000, 18000):
        long = gguf_engine.build_request(
            request,
            abc_sampling=sampling(),
            semantic_sampling=sampling(max_tokens=budget),
            steps=32,
        )
        assert int(long["duration"] * gguf_engine.FRAME_RATE) >= budget

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
    # a fake yue-synth: honours --out/--tokens/--latent/--score/--dump, prints a plausible log
    import json, os, struct, sys, wave
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
    d = args.get("--dump")
    if d and os.environ.get("YUE2_TEST_NO_DUMP") != "1":
        # like yue2.cpp: open files inside the directory, never create it; the dump holds the
        # first acoustic chunk only (prefix + a leading slice of the codes + MUSIC_END)
        chunk = int(os.environ.get("YUE2_TEST_CHUNK") or frames)
        ids = [151643, 1, 2, 3, 151847, 40, 41, 42, 151848, 151851]
        ids += [151853 + t for t in tokens[:chunk]] + [151852]
        arr = np.asarray(ids, dtype=np.float32)
        try:
            with open(os.path.join(d, "ar_ids.bin"), "wb") as f:
                f.write(struct.pack("i", 1) + struct.pack("i", len(arr)) + arr.tobytes())
        except OSError:
            print("[Debug] Cannot write %s/ar_ids.bin" % d, file=sys.stderr)
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
        ("cuda", 11.99, True, "gguf"),  # a 12 GB card (12288 MiB reads as 11.99 GiB)
        ("cuda", 15.4, True, "gguf"),
        ("cuda", 15.99, True, "torch"),  # a 16 GB card reports 16376 MiB = 15.99 GiB: it is 16
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


def test_install_takes_the_pinned_asset_from_an_earlier_release_when_the_newest_has_none(
    tmp_path, monkeypatch
):
    """A release just published lists no binaries for its first two hours (the workflow is
    still building); the asset name carries the pin, so the release before it is as good."""
    import io
    import tarfile
    import urllib.error

    asset = gguf_engine.platform_asset()
    payload = io.BytesIO()
    if asset.endswith(".zip"):
        import zipfile

        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("yue-synth.exe", b"MZ")
            archive.writestr("VERSION", "yue2.cpp test windows-x64\n")
    else:
        with tarfile.open(fileobj=payload, mode="w:gz") as archive:
            for name, data in (("yue-synth", b"#!/bin/sh\n"), ("VERSION", b"yue2.cpp test\n")):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    releases = [
        {"tag_name": "v9.9.9", "assets": []},  # just published, nothing attached yet
        {"tag_name": "v9.9.8", "assets": [{"name": asset}]},
        {"tag_name": "v9.9.7", "assets": [{"name": asset}]},
    ]
    urls = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    def urlopen(request, timeout=0):
        url = getattr(request, "full_url", request)
        urls.append(url)
        if url.startswith(gguf_engine.RELEASES_API):
            return Response(json.dumps(releases).encode())
        if "/v9.9.8/" in url:
            return Response(payload.getvalue())
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(gguf_engine.urllib.request, "urlopen", urlopen)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    said = []
    dest = gguf_engine.install_binaries(tmp_path / "bin", tag="latest", say=said.append)
    assert gguf_engine._binary(dest, "yue-synth") is not None
    assert urls == [
        f"{gguf_engine.RELEASES}/latest/download/{asset}",
        f"{gguf_engine.RELEASES_API}?per_page=30",
        f"{gguf_engine.RELEASES}/download/v9.9.8/{asset}",
    ]  # the newest release that lists it, not the oldest
    assert any("taking" in line and "v9.9.8" in line for line in said)
    # an explicit tag that has it is served directly; one that lacks it falls back the same way
    urls.clear()
    gguf_engine.install_binaries(tmp_path / "bin2", tag="v9.9.8", say=said.append)
    assert urls == [f"{gguf_engine.RELEASES}/download/v9.9.8/{asset}"]
    # no release lists the asset at all (a pin bump still building): a plain, explained error
    releases[1]["assets"] = releases[2]["assets"] = []
    with pytest.raises(RuntimeError, match="no release lists"):
        gguf_engine.install_binaries(tmp_path / "bin3", tag="latest", say=said.append)
    # and an unreachable API is not a crash, only the same error
    monkeypatch.setattr(
        gguf_engine.urllib.request,
        "urlopen",
        lambda request, timeout=0: (_ for _ in ()).throw(
            urllib.error.HTTPError(str(request), 404, "Not Found", {}, None)
        ),
    )
    with pytest.raises(RuntimeError, match="no release lists"):
        gguf_engine.install_binaries(tmp_path / "bin4", tag="latest", say=said.append)


def test_auto_backend_uses_the_gpu_torch_cannot_see(monkeypatch):
    """A CPU-only torch (PyPI's Windows wheel) next to an NVIDIA card: the reference engine
    would crawl on the CPU; yue2.cpp drives the card itself — but only when installed."""
    monkeypatch.delenv("YUE2_GROOVE_GGUF_VRAM_GIB", raising=False)
    backend, note = gguf_engine.auto_backend("cpu", 12.0, installed=True)
    assert backend == "gguf" and "torch has no CUDA" in note
    assert gguf_engine.auto_backend("cpu", 24.0, installed=True)[0] == "gguf"  # any size
    backend, note = gguf_engine.auto_backend("cpu", 12.0, installed=False)
    assert backend == "torch" and "torch has no CUDA" in note and "Install the GGUF engine" in note
    assert gguf_engine.auto_backend("mps", 12.0, installed=True) == ("torch", "")


def test_vram_probe_prefers_nvidia_smi_and_never_a_cuda_context(monkeypatch):
    calls = []

    class Done:
        returncode = 0
        stdout = "12288\n"  # MiB, one line per GPU

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return Done()

    monkeypatch.setattr(gguf_engine.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gguf_engine.subprocess, "run", fake_run)
    assert gguf_engine.cuda_total_vram_gib() == pytest.approx(12.0)
    assert calls and calls[0][0] == "/usr/bin/nvidia-smi"
    # no nvidia-smi and no CUDA torch: None, and torch was the only fallback consulted
    monkeypatch.setattr(gguf_engine.shutil, "which", lambda name: None)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gguf_engine.cuda_total_vram_gib() is None


def test_child_path_is_the_plain_string_off_windows(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX behaviour")
    p = tmp_path / "谱面" / "audio.wav"
    assert gguf_engine.child_path(p) == str(p)


def checkpoint(directory: Path, weights: bytes, *, manifest: bool = True, config_text="{}") -> Path:
    """A stand-in checkpoint: weights, config.json and (optionally) the manifest yue2's releases
    ship, naming the weight hash — which the identity verifies rather than trusts."""
    import hashlib

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.safetensors").write_bytes(weights)
    (directory / "config.json").write_text(config_text)
    if manifest:
        digest = hashlib.sha256(weights).hexdigest()
        (directory / "weights_manifest.json").write_text(
            json.dumps({"files": {"model.safetensors": {"sha256": digest, "bytes": len(weights)}}})
        )
    return directory


def test_prepare_names_files_after_verified_content_and_keeps_the_intermediate_private(
    tmp_path, monkeypatch
):
    """GGUF files carry the checkpoints' content identity (a new model / revision / VAE / config /
    tokenizer / converter never reuses an old conversion); conversion and quantization work in a
    private .partial-* directory — the BF16 intermediate included — and only finished files are
    moved out; existing files are kept."""
    produced = []

    def fake_converter(components, out_dir, say):
        for name in components:
            target = out_dir / ("YuE2-3B-BF16.gguf" if name == "backbone" else "YuE2-Vae-F32.gguf")
            target.write_bytes(b"x" * 10)
            produced.append((name, out_dir.name[: len(".partial-")]))

    def fake_run(argv, say):
        assert Path(argv[1]).parent.name.startswith(".partial-")  # quantize reads the private BF16
        Path(argv[2]).write_bytes(b"q" * 5)  # quantize <in> <out> <type>
        produced.append(("quantize", Path(argv[2]).parent.name[: len(".partial-")]))

    monkeypatch.setattr(gguf_engine, "_run_converter", fake_converter)
    monkeypatch.setattr(gguf_engine, "_run", fake_run)
    monkeypatch.setattr(gguf_engine, "binary", lambda name: tmp_path / name)
    monkeypatch.delenv("YUE2_GROOVE_GGUF", raising=False)
    m = checkpoint(tmp_path / "m", b"model-a")
    v = checkpoint(tmp_path / "v", b"vae-b")
    out = tmp_path / "gguf"
    first = gguf_engine.prepare(m, v, out, quant_label="Q8_0")
    assert first.provenance == "converted" and first.model and first.vae_source
    assert first.backbone.name.startswith("YuE2-3B-Q8_0-")
    assert first.vae.name.startswith("YuE2-Vae-F32-")
    assert first.backbone.read_bytes() == b"q" * 5 and first.vae.is_file()
    assert not list(out.glob("*BF16*")) and not list(out.glob(".partial-*"))
    assert [d for _, d in produced] == [".partial-"] * 3
    # second call: nothing to do
    produced.clear()
    again = gguf_engine.prepare(m, v, out, quant_label="Q8_0")
    assert produced == [] and again.backbone == first.backbone
    # a different quant converts its own private BF16 again, never the VAE
    gguf_engine.prepare(m, v, out, quant_label="Q6_K")
    assert [n for n, _ in produced] == ["backbone", "quantize"]
    # a different model, or only a different config.json, converts afresh; the earlier file is
    # kept (never deleted from under another process) but named as deletable
    for other in (
        checkpoint(tmp_path / "m2", b"model-c"),
        checkpoint(tmp_path / "m3", b"model-a", config_text='{"changed": 1}'),
    ):
        produced.clear()
        said = []
        again = gguf_engine.prepare(other, v, out, quant_label="Q8_0", log=said.append)
        assert again.backbone != first.backbone and first.backbone.is_file()
        assert [n for n, _ in produced] == ["backbone", "quantize"]
        note = [line for line in said if line.startswith("no longer used")]
        assert len(note) == 1 and first.backbone.name in note[0]
        assert again.backbone.name not in note[0] and "Vae" not in note[0]  # only the superseded


def test_checkpoint_identity_verifies_the_manifest_and_covers_the_conversion_inputs(tmp_path):
    import hashlib

    out = tmp_path / "gguf"
    m = checkpoint(tmp_path / "m", b"weights")
    ident = gguf_engine.checkpoint_identity(m, out)
    assert ident["files"]["model.safetensors"] == hashlib.sha256(b"weights").hexdigest()
    assert ident["config.json"] and ident["converter"] and len(ident["identity"]) == 64
    assert not list(m.glob("*.sha256"))  # never litters the checkpoint directory
    assert (out / ".hashes.json").is_file()  # the 7 GB hash is remembered here instead
    # changed weights under an old manifest: an integrity failure, not a cache miss
    (m / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="weight integrity failed"):
        gguf_engine.checkpoint_identity(m, out)
    # no manifest at all: hashed and accepted
    raw = checkpoint(tmp_path / "raw", b"other", manifest=False)
    assert gguf_engine.checkpoint_identity(raw, out)["files"]["model.safetensors"] == (
        hashlib.sha256(b"other").hexdigest()
    )
    with pytest.raises(RuntimeError, match="no safetensors"):
        gguf_engine.checkpoint_identity(tmp_path / "nope", out)


def test_converter_version_ignores_line_endings(tmp_path):
    """A Windows checkout under Git's autocrlf hands the converter over with CRLF line endings;
    it is the same converter and must name the same GGUF file (seen on the RTX 2070 run)."""
    import hashlib

    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"import gguf\n\ndef convert():\n    pass\n")
    crlf.write_bytes(lf.read_bytes().replace(b"\n", b"\r\n"))
    assert gguf_engine.converter_version(crlf) == gguf_engine.converter_version(lf)
    assert gguf_engine.converter_version(lf) == hashlib.sha256(lf.read_bytes()).hexdigest()
    vendored = gguf_engine.config.PACKAGE_DIR / "vendor" / "yue2cpp_convert.py"
    assert (
        gguf_engine.converter_version()
        == hashlib.sha256(vendored.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    )  # the checked-in copy is LF, so the names existing installs made are kept


def test_prepare_uses_ready_made_files_only_in_an_explicit_gguf_dir(tmp_path, monkeypatch):
    out = tmp_path / "ready"
    out.mkdir()
    (out / "YuE2-3B-Q8_0.gguf").write_bytes(b"r")
    (out / "YuE2-Vae-F32.gguf").write_bytes(b"r")
    monkeypatch.setattr(gguf_engine, "_run_converter", lambda *a: pytest.fail("converted"))
    monkeypatch.delenv("YUE2_GROOVE_GGUF", raising=False)
    with pytest.raises(RuntimeError, match="no safetensors"):  # not explicit: identity needed
        gguf_engine.prepare(tmp_path / "nope", tmp_path / "nope", out, quant_label="Q8_0")
    monkeypatch.setenv("YUE2_GROOVE_GGUF", str(out))
    prepared = gguf_engine.prepare(tmp_path / "nope", tmp_path / "nope", out, quant_label="Q8_0")
    assert prepared.provenance == "ready-made" and prepared.model is None
    assert prepared.backbone.name == "YuE2-3B-Q8_0.gguf"


def test_ready_made_files_are_recorded_as_external_not_as_the_local_checkpoint(
    tmp_path, monkeypatch, model_dir
):
    out = tmp_path / "ready"
    out.mkdir()
    (out / "YuE2-3B-Q8_0.gguf").write_bytes(b"r")
    (out / "YuE2-Vae-F32.gguf").write_bytes(b"r")
    monkeypatch.setenv("YUE2_GROOVE_GGUF", str(out))
    monkeypatch.setattr(gguf_engine, "binary", lambda name: tmp_path / name)
    monkeypatch.setattr(gguf_engine, "engine_version", lambda path=None: "")
    (tmp_path / "yue-synth").write_bytes(b"")
    pipe = gguf_engine.GgufPipeline.open(
        model_dir=model_dir, vae_dir=model_dir, ode_steps=32, vae_core_frames=None
    )
    assert pipe.ready_made
    for key in ("mot", "vae"):
        assert pipe.weights[key]["source"]["provenance"].startswith("ready-made")
        assert "files" not in pipe.weights[key]["source"]
    request = adapter.song_request(style="pop", lyrics="la", cot="full", seed=1)
    assert pipe.config_dict(request, sampling(), sampling())["decoder_release"] is None


def test_stale_partial_directories_are_swept_but_live_ones_kept(tmp_path):
    old = tmp_path / ".partial-old"
    old.mkdir()
    os.utime(old, (1, 1))
    live = tmp_path / ".partial-live"
    live.mkdir()
    gguf_engine._sweep_stale_partials(tmp_path)
    assert not old.exists() and live.exists()


def test_engine_version_is_read_from_the_usage_banner(tmp_path, monkeypatch):
    stub = write_stub(
        tmp_path,
        "yue-synth",
        """
        import sys
        print("yue2.cpp deadbee (2026-09-18)", file=sys.stderr)
        print("Usage: yue-synth --model <gguf> ...", file=sys.stderr)
        sys.exit(1)
    """,
    )
    real_run = gguf_engine.subprocess.run
    monkeypatch.setattr(
        gguf_engine.subprocess, "run", lambda argv, **kw: real_run([sys.executable, *argv], **kw)
    )
    assert gguf_engine.engine_version(stub) == "deadbee (2026-09-18)"
    assert gguf_engine.engine_version(tmp_path / "missing") == ""


def test_exact_ids_come_from_the_engines_dump(tmp_path, monkeypatch, model_dir):
    """With YUE2_GROOVE_GGUF_EXACT_IDS the prefix is what yue2.cpp fed the semantic stage,
    read from its dump (which the app must create the directory for), not a re-tokenization
    of the score text; the archive says which one it got."""
    patch_binaries(monkeypatch, {"yue-synth": write_stub(tmp_path, "yue-synth", SYNTH_STUB)})
    monkeypatch.setenv("YUE2_GROOVE_GGUF_EXACT_IDS", "1")
    pipe = make_pipe(tmp_path, model_dir)
    request = adapter.song_request(style="pop", lyrics="la", cot="full", seed=3)
    song = adapter.generate(pipe, request, abc_sampling=sampling(), semantic_sampling=sampling())
    assert song.plan.abc_ids == [40, 41, 42]  # the engine's ids, not encode(score)
    assert song.plan.prefix[:5] == [151643, 1, 2, 3, 151847] and song.plan.prefix[-1] == 151851
    assert song.config["plan_ids_provenance"] == "engine"
    assert song.config["plan_ids_match_retokenized"] is False  # the stub's ids are made up
    # a song the engine rendered in several acoustic chunks: the dump holds the first chunk's
    # codes only, and the prefix is still cut at the protocol's marker
    monkeypatch.setenv("YUE2_TEST_CHUNK", "20")
    song = adapter.generate(pipe, request, abc_sampling=sampling(), semantic_sampling=sampling())
    assert song.plan.prefix[-1] == 151851 and len(song.tokens) == 50
    monkeypatch.delenv("YUE2_TEST_CHUNK")
    # exact ids asked for but not delivered: an error, never a silent re-tokenization
    monkeypatch.setenv("YUE2_TEST_NO_DUMP", "1")
    with pytest.raises(RuntimeError, match=r"wrote no ar_ids\.bin"):
        adapter.generate(pipe, request, abc_sampling=sampling(), semantic_sampling=sampling())
    monkeypatch.delenv("YUE2_TEST_NO_DUMP")
    monkeypatch.delenv("YUE2_GROOVE_GGUF_EXACT_IDS")
    song = adapter.generate(pipe, request, abc_sampling=sampling(), semantic_sampling=sampling())
    assert song.config["plan_ids_provenance"] == "retokenized"
    assert song.config["plan_ids_match_retokenized"] is None
    tokenizer = adapter.text_tokenizer(model_dir)
    assert song.plan.abc_ids == tokenizer.encode(song.abc)


def test_engine_ar_ids_rejects_a_dump_that_disagrees_with_the_stream(tmp_path):
    import struct

    dump = tmp_path / "ar_ids.bin"
    ids = [151643, 5, 151851, 151853 + 7, 151853 + 8, 151852]
    arr = np.asarray(ids, dtype=np.float32)
    dump.write_bytes(struct.pack("i", 1) + struct.pack("i", len(arr)) + arr.tobytes())
    assert gguf_engine._engine_ar_ids(dump, [7, 8, 9]) == ([151643, 5, 151851], [])
    with pytest.raises(RuntimeError, match="do not match the semantic stream"):
        gguf_engine._engine_ar_ids(dump, [1, 2, 3])
    with pytest.raises(RuntimeError, match=r"wrote no ar_ids\.bin"):
        gguf_engine._engine_ar_ids(tmp_path / "missing.bin", [7, 8])


def test_run_child_stops_the_child_when_a_callback_raises(tmp_path, monkeypatch):
    """A progress callback that raises must not leave yue2.cpp running (the next generation
    would compete with it for VRAM): the child is terminated and reaped on every exit path."""
    import time

    beat = tmp_path / "heartbeat"
    slow = write_stub(
        tmp_path,
        "slow",
        f"""
        import sys, time
        print("[NAR] Step 1/32, 10 ms", file=sys.stderr, flush=True)
        for i in range(200):
            open({str(beat)!r}, "a").write("x"); time.sleep(0.1)
    """,
    )
    patch_binaries(monkeypatch, {})

    def on_progress(stage, done, total):
        raise ValueError("the UI went away")

    with pytest.raises(ValueError, match="the UI went away"):
        gguf_engine.run_child([str(slow)], on_progress=on_progress)
    size = beat.stat().st_size if beat.exists() else 0
    time.sleep(0.6)
    assert (beat.stat().st_size if beat.exists() else 0) == size  # no heartbeat after the stop


def test_resolve_backend_respects_an_explicit_device(monkeypatch):
    monkeypatch.setattr(gguf_engine, "cuda_total_vram_gib", lambda: 12.0)
    monkeypatch.setattr(gguf_engine, "available", lambda: True)
    assert runtime.resolve_backend("auto", "cpu")[0] == "gguf"  # torch picked cpu on its own
    assert runtime.resolve_backend("auto", "cpu", device_explicit=True) == ("torch", "")
    assert runtime.resolve_backend("auto", "cuda", device_explicit=True)[0] == "gguf"


def test_preparation_child_is_stopped_when_the_log_callback_raises(tmp_path, monkeypatch):
    """The converter / quantize runner has the same lifecycle rule as run_child: a raising
    log callback (or a Ctrl-C) must not leave the child writing into a directory that
    prepare() is about to remove."""
    import time

    beat = tmp_path / "heartbeat"
    slow = write_stub(
        tmp_path,
        "slow-quantize",
        f"""
        import sys, time
        print("[Quantize] starting", file=sys.stderr, flush=True)
        for i in range(200):
            open({str(beat)!r}, "a").write("x"); time.sleep(0.1)
    """,
    )
    patch_binaries(monkeypatch, {})

    def say(line):
        raise ValueError("log sink closed")

    with pytest.raises(ValueError, match="log sink closed"):
        gguf_engine._run([str(slow)], say)
    size = beat.stat().st_size if beat.exists() else 0
    time.sleep(0.6)
    assert (beat.stat().st_size if beat.exists() else 0) == size  # no heartbeat after the stop
    # and a clean failure still reports the child's last lines
    bad = write_stub(
        tmp_path,
        "bad-quantize",
        """
        import sys
        print("[Quantize] FATAL: not a GGUF", file=sys.stderr); sys.exit(2)
    """,
    )
    with pytest.raises(RuntimeError, match=r"(?s)exit 2.*not a GGUF"):
        gguf_engine._run([str(bad)], lambda line: None)


def test_install_skips_an_install_already_at_the_pin(tmp_path, monkeypatch):
    """The launcher runs install on every update: no download when the pin is already there."""
    dest = tmp_path / "bin"
    dest.mkdir()
    (dest / "VERSION").write_text(f"yue2.cpp {gguf_engine.YUE2CPP_PIN} macos-arm64 metal\n")
    (dest / "yue-synth").write_bytes(b"#!/bin/sh\n")

    def urlopen(url, timeout=0):
        pytest.fail("downloaded although already installed")

    monkeypatch.setattr(gguf_engine.urllib.request, "urlopen", urlopen)
    said = []
    assert gguf_engine.install_binaries(dest, say=said.append) == dest
    assert said and said[-1].startswith("already installed")
    assert gguf_engine.installed_version(dest) == gguf_engine.YUE2CPP_PIN
    (dest / "VERSION").write_text("yue2.cpp 0000000 macos-arm64 metal\n")  # an older pin: downloads
    with pytest.raises(BaseException, match="downloaded although"):
        gguf_engine.install_binaries(dest, say=said.append)


def test_small_cards_get_the_context_cap_by_default(monkeypatch):
    monkeypatch.delenv("YUE2_GROOVE_GGUF_MAX_SEQ", raising=False)
    assert gguf_engine.effective_max_seq(8) == (12288, "max_seq=12288 (8 GB card: context capped)")
    assert gguf_engine.effective_max_seq(6)[0] == 12288
    assert gguf_engine.effective_max_seq(12) == (None, "")
    assert gguf_engine.effective_max_seq(None) == (None, "")  # no NVIDIA card (Apple Silicon)
    monkeypatch.setenv("YUE2_GROOVE_GGUF_MAX_SEQ", "9000")
    assert gguf_engine.effective_max_seq(24)[0] == 9000  # explicit wins, whatever the card
    assert gguf_engine.effective_max_seq(8)[0] == 9000


def test_semantic_budget_is_derived_from_the_cap(model_dir):
    """yue2.cpp refuses prefix + budget beyond its cache: the budget is trimmed up front, from
    the exact prefix for an external score and the worst case for a model-written one."""
    tokenizer = adapter.text_tokenizer(model_dir)
    abc_s, sem_s = sampling(), sampling(max_tokens=9000, min_tokens=200)
    request = adapter.song_request(style="pop", lyrics="[Verse]\nla la la", cot="full", seed=1)
    prefix, budget = gguf_engine.semantic_budget(request, tokenizer, abc_s, sem_s, 12288)
    base = len(adapter.token_prefixes(request, tokenizer))
    assert prefix == base + 4096 + 2 and budget == 12288 - prefix - 1 and budget < 9000
    external = adapter.song_request(
        style="pop", lyrics="la", cot="melody", seed=1, abc="X:1\nK:C\n|C D E F|\n"
    )
    prefix, budget = gguf_engine.semantic_budget(external, tokenizer, abc_s, sem_s, 12288)
    ids = tokenizer.encode(external.abc)
    assert prefix == len(adapter.token_prefixes(external, tokenizer, ids)) and budget == 9000
    with pytest.raises(RuntimeError, match="leaves no room"):
        gguf_engine.semantic_budget(request, tokenizer, abc_s, sem_s, 4200)


def test_generate_under_a_cap_sends_the_trimmed_budget(tmp_path, monkeypatch, model_dir):
    seen = {}
    stub = SYNTH_STUB.replace(
        "frames = 50",
        "frames = 50\n    print('[test] semantic max_tokens=%d min_tokens=%d' % (request['semantic_sampling']['max_tokens'], request['semantic_sampling']['min_tokens']), file=sys.stderr)",
    )
    patch_binaries(monkeypatch, {"yue-synth": write_stub(tmp_path, "yue-synth", stub)})
    pipe = make_pipe(tmp_path, model_dir)
    pipe.max_seq = 6000
    request = adapter.song_request(style="pop", lyrics="[Verse]\nla la la", cot="full", seed=1)
    real_parse = gguf_engine.parse_line

    def spy(line, log, **kw):
        if line.startswith("[test]"):
            seen["line"] = line
        return real_parse(line, log, **kw)

    monkeypatch.setattr(gguf_engine, "parse_line", spy)
    song = adapter.generate(
        pipe,
        request,
        abc_sampling=sampling(),
        semantic_sampling=sampling(max_tokens=9000, min_tokens=200),
    )
    cap = song.config["semantic_budget_cap"]
    assert (
        cap["max_seq"] == 6000
        and cap["semantic_max_tokens"] == 6000 - cap["prefix_tokens_assumed"] - 1
    )
    assert seen["line"] == f"[test] semantic max_tokens={cap['semantic_max_tokens']} min_tokens=200"
    pipe.max_seq = None
    song = adapter.generate(
        pipe, request, abc_sampling=sampling(), semantic_sampling=sampling(max_tokens=9000)
    )
    assert song.config["semantic_budget_cap"] is None


def test_cli_subcommands_parse_and_dispatch(tmp_path, monkeypatch, capsys):
    """`python -m yue2_groove.gguf_engine install|prepare|check`: the argparse wiring itself
    (1.0.0 shipped an `install` whose --force flag was read but never declared)."""
    calls = {}
    monkeypatch.setattr(
        gguf_engine,
        "install_binaries",
        lambda dest=None, **kw: calls.update(dest=dest, **kw) or tmp_path,
    )
    assert gguf_engine.main(["install", "--tag", "latest", "--force", "--dest", str(tmp_path)]) == 0
    assert calls["tag"] == "latest" and calls["force"] is True and calls["dest"] == tmp_path
    assert gguf_engine.main(["install"]) == 0 and calls["force"] is False and calls["tag"] is None
    monkeypatch.setattr(gguf_engine, "binary_dir", lambda: None)
    monkeypatch.setattr(gguf_engine, "cuda_total_vram_gib", lambda: None)
    assert gguf_engine.main(["check"]) == 0
    assert "binaries: not found" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        gguf_engine.main(["nonsense"])
