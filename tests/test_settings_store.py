"""The saved settings rail: file round trip, per-field fallback, launch precedence, wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

webui = pytest.importorskip("yue2_groove.webui")
runtime = webui.runtime
store = webui.settings_store
gguf_engine = pytest.importorskip("yue2_groove.gguf_engine")

HUB = "m-a-p/YuE2-3B"


@pytest.fixture
def machine(tmp_path: Path, monkeypatch):
    """A CPU-only machine without the GGUF binaries; tests flip what they need."""
    monkeypatch.delenv("YUE2_GROOVE_SETTINGS", raising=False)
    monkeypatch.setattr(runtime, "RUNS", tmp_path / "runs")
    state = {"cuda": False, "mps": False, "gguf": False, "vram": None}

    def pick(device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if state["cuda"] else ("mps" if state["mps"] else "cpu")

    monkeypatch.setattr(runtime, "pick_device", pick)
    monkeypatch.setattr(runtime, "BACKEND_CHOICES", ["torch", "torch-eager", "gguf"])
    monkeypatch.setattr(runtime.adapter, "cuda_flash_attention_usable", lambda: True)
    monkeypatch.setattr(store, "_cuda", lambda: state["cuda"])
    monkeypatch.setattr(store, "_mps", lambda: state["mps"])
    monkeypatch.setattr(store, "_gpu_name", lambda: None)
    monkeypatch.setattr(store, "_vram_gib", lambda: state["vram"])
    monkeypatch.setattr(gguf_engine, "available", lambda: state["gguf"])
    monkeypatch.setattr(gguf_engine, "cuda_total_vram_gib", lambda: state["vram"])
    monkeypatch.setattr(store.config, "default_model", lambda: HUB)
    monkeypatch.setattr(store, "_CURRENT", None)
    monkeypatch.setattr(store, "_FACTORY", None)
    monkeypatch.setattr(runtime, "_PIPE", None)
    monkeypatch.setattr(runtime, "_PIPE_KEY", None)
    return state


def saved(**settings) -> dict:
    """What save_from_rail would have written on a CPU machine, with *settings* changed."""
    rail, _ = store.factory()
    return {**rail, **settings}


def write_file(values: dict) -> Path:
    target = store.path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"settings": values}), encoding="utf-8")
    return target


def submit(values: dict):
    return store.save_from_rail(*[values[f] for f in store.FIELDS])


def test_the_file_sits_next_to_the_works_never_inside(machine, tmp_path, monkeypatch) -> None:
    assert store.path() == tmp_path / store.FILE_NAME
    monkeypatch.setenv("YUE2_GROOVE_SETTINGS", str(tmp_path / "elsewhere.json"))
    assert store.path() == tmp_path / "elsewhere.json"


def test_no_file_launches_on_the_automatic_rules(machine) -> None:
    rail, factory, notes = store.startup()
    assert rail == factory
    assert (rail["device"], rail["dtype"], rail["backend"]) == ("cpu", "float32", "torch")
    assert (rail["ode_steps"], rail["model"]) == (32, HUB)
    assert notes == []
    assert store.rail_values() == tuple(rail[f] for f in store.FIELDS)


def test_a_change_is_saved_and_restored_at_the_next_launch(machine) -> None:
    machine["gguf"] = True
    store.startup()
    *updates, status = submit(saved(backend="gguf", ode_steps=16))
    assert "Settings saved to" in status
    assert all(u == gr.update() for u in updates)  # nothing had to fall back
    data = json.loads(store.path().read_text(encoding="utf-8"))
    assert data["settings"]["backend"] == "gguf" and data["settings"]["ode_steps"] == 16
    assert data["version"] == webui.layout.__version__
    assert data["resolved"] == {"device": "cpu", "backend": "gguf"}
    assert data["machine"]["gguf_installed"] is True

    rail, _, notes = store.startup()
    assert (rail["backend"], rail["ode_steps"]) == ("gguf", 16)
    assert notes[0] == f"Settings restored from {store.path()}"


def test_a_field_this_machine_cannot_run_falls_back_alone(machine) -> None:
    write_file(saved(backend="gguf", ode_steps=16, quantization="fp8"))
    rail, _, notes = store.startup()  # the binaries are gone
    assert rail["backend"] == "torch"
    assert rail["quantization"] == "none"
    assert rail["ode_steps"] == 16  # kept
    assert any(n.startswith("BACKEND gguf: the yue2.cpp binaries") for n in notes)
    assert any(n.startswith("QUANTIZATION fp8: needs NVIDIA CUDA") for n in notes)


def test_devices_and_engines_that_are_missing_fall_back_to_auto(machine) -> None:
    clean, notes = store.check(saved(device="cuda", backend="vllm"), saved())
    assert clean["device"] == "auto" and clean["backend"] == "torch"
    assert "DEVICE cuda: no CUDA on this machine → auto" in notes
    assert "BACKEND vllm: not installed on this machine → torch" in notes
    clean, notes = store.check(saved(device="mps"), saved())
    assert clean["device"] == "auto" and notes


def test_paths_that_do_not_exist_fall_back(machine, tmp_path) -> None:
    clean, notes = store.check(saved(model="/no/such/model"), saved())
    assert clean["model"] == HUB and notes
    clean, notes = store.check(saved(model=str(tmp_path)), saved())
    assert clean["model"] == str(tmp_path) and not notes
    clean, _ = store.check(saved(vae_choice="custom", vae_custom=""), saved())
    assert clean["vae_choice"] == "standard"
    clean, _ = store.check(saved(vae_choice="custom", vae_custom="org/some-vae"), saved())
    assert clean["vae_choice"] == "custom"  # a Hub id is settled at load, not here


def test_values_of_the_wrong_shape_fall_back(machine) -> None:
    clean, notes = store.check(
        saved(ode_steps=30, budget=0, offline="yes", vae_core_frames="2048"), saved()
    )
    assert (clean["ode_steps"], clean["budget"]) == (32, 24)
    assert clean["offline"] is False and clean["vae_core_frames"] == "auto"
    assert len(notes) == 4


def test_a_small_card_on_the_reference_engine_is_warned_not_changed(machine) -> None:
    machine.update(cuda=True, gguf=True, vram=11.9)
    clean, notes = store.check(saved(device="cuda", dtype="bfloat16", backend="torch"), saved())
    assert clean["backend"] == "torch"
    assert any("12 GB card is below the 16 GB" in n for n in notes)


def test_an_explicit_flag_wins_over_the_file(machine) -> None:
    machine["gguf"] = True
    write_file(saved(backend="gguf", ode_steps=16))
    rail, _, notes = store.startup(backend="torch")
    assert rail["backend"] == "torch" and rail["ode_steps"] == 16
    assert not any(n.startswith("BACKEND") for n in notes)
    # "auto" is not a choice: the file still decides
    rail, _, _ = store.startup(backend="auto")
    assert rail["backend"] == "gguf"


def test_an_unreadable_file_is_ignored(machine) -> None:
    store.path().parent.mkdir(parents=True, exist_ok=True)
    store.path().write_text("{not json", encoding="utf-8")
    rail, factory, notes = store.startup()
    assert rail == factory and notes == []


def test_the_rail_shows_what_had_to_fall_back(machine) -> None:
    store.startup()
    *updates, status = submit(saved(backend="gguf"))
    backend = updates[store.FIELDS.index("backend")]
    assert backend == gr.update(value="torch")
    assert "BACKEND gguf: the yue2.cpp binaries are not installed → torch" in status
    assert json.loads(store.path().read_text())["settings"]["backend"] == "torch"


def test_a_blur_without_an_edit_says_nothing(machine) -> None:
    rail, _, _ = store.startup()
    *_, status = submit(rail)
    assert status == gr.update() and not store.path().exists()


def test_a_change_to_a_loaded_model_says_when_it_applies(machine, monkeypatch) -> None:
    rail, _, _ = store.startup()
    monkeypatch.setattr(runtime, "resolve_vae", lambda choice, custom: (f"/vae/{choice}", choice))
    monkeypatch.setattr(runtime, "_PIPE", object())
    monkeypatch.setattr(runtime, "_PIPE_KEY", runtime.RuntimeSettings(**rail).key)
    *_, status = submit({**rail, "ode_steps": 16})
    assert "Takes effect at the next run (or LOAD / APPLY)." in status


def test_reset_removes_the_file_and_page_loads_follow(machine) -> None:
    rail, factory, _ = store.startup()
    submit({**rail, "ode_steps": 16})
    assert store.rail_values()[store.FIELDS.index("ode_steps")] == 16
    *values, status = store.reset({**factory, "device": "auto"})
    assert not store.path().exists() and "removed" in status
    assert values[store.FIELDS.index("ode_steps")] == 32
    assert store.rail_values()[store.FIELDS.index("device")] == "auto"


def test_the_rail_is_wired_to_save_restore_and_reset(machine) -> None:
    rail, factory, _ = store.startup()
    demo = webui.build_ui({"rail": rail, "factory": factory, "tab": 0, "status": ""})
    fns = list(demo.fns.values())
    saves = [f for f in fns if f.fn is store.save_from_rail]
    events = sorted(event for f in saves for _, event in f.targets)
    # 5 dropdowns + radio + 2 checkboxes on input, the slider on release,
    # 4 textboxes + the number on blur and submit
    assert events.count("input") == 8
    assert events.count("release") == 1
    assert events.count("blur") == events.count("submit") == 5
    assert all(not f.queue for f in saves)
    loads = [f for f in fns if f.fn is store.rail_values]
    assert len(loads) == 1 and len(loads[0].outputs) == len(store.FIELDS)


def test_overlapping_saves_never_fail(machine) -> None:
    """A pick and a blur can reach the server together (the handlers skip the queue)."""
    import threading

    rail, _, _ = store.startup()
    results: list[str] = []

    def one(steps: int) -> None:
        results.append(submit({**rail, "ode_steps": steps})[-1])

    threads = [threading.Thread(target=one, args=(4 * (i % 15 + 1),)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not any("NOT saved" in r for r in results if isinstance(r, str))
    assert json.loads(store.path().read_text())["settings"]["ode_steps"] in range(4, 65, 4)
    assert list(store.path().parent.glob("*.tmp")) == []


def test_a_field_left_at_its_default_keeps_following_the_automatic_rules(machine) -> None:
    """A 12 GB card without the binaries: auto says torch.  The user only sets ODE STEPS.
    Once Update installs the binaries, auto must pick gguf — the saved torch was never a
    choice, just the default of the day."""
    machine.update(cuda=True, vram=11.9)
    rail, _, _ = store.startup()
    assert rail["backend"] == "torch"
    submit({**rail, "ode_steps": 16})
    data = json.loads(store.path().read_text())
    assert data["settings"]["backend"] == "torch"  # every field is on record
    assert data["changed"] == ["ode_steps"]

    machine["gguf"] = True
    rail, _, _ = store.startup()
    assert (rail["backend"], rail["ode_steps"]) == ("gguf", 16)


def test_a_backend_the_user_picked_is_kept_even_where_auto_differs(machine) -> None:
    machine.update(cuda=True, vram=11.9, gguf=True)
    rail, _, _ = store.startup()
    assert rail["backend"] == "gguf"
    submit({**rail, "backend": "torch"})
    assert json.loads(store.path().read_text())["changed"] == ["backend"]
    rail, _, notes = store.startup()
    assert rail["backend"] == "torch"
    assert any("auto would pick gguf" in n for n in notes)


def test_the_mps_dtype_warning_only_concerns_the_torch_engine(machine, monkeypatch) -> None:
    machine.update(mps=True, gguf=True)
    monkeypatch.setattr(store.torch, "__version__", "2.10.0")
    rail = saved(device="mps", dtype="bfloat16")
    _, notes = store.check({**rail, "backend": "torch"}, rail)
    assert any(n.startswith("DTYPE bfloat16 on MPS") for n in notes)
    _, notes = store.check({**rail, "backend": "gguf"}, rail)
    assert not any(n.startswith("DTYPE") for n in notes)
