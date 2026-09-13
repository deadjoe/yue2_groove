"""``yue2_groove.config.load_env``: the .env loader that ``main()`` runs."""
from __future__ import annotations

from pathlib import Path

from yue2_groove import config


def test_load_env_parses_and_does_not_override(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "\n"
        "YUE2_GROOVE_TEST_ALPHA=alpha\n"
        "export YUE2_GROOVE_TEST_BETA='beta value'\n"
        'YUE2_GROOVE_TEST_GAMMA="gamma"\n'
        "YUE2_GROOVE_TEST_EMPTY=\n"
        "YUE2_GROOVE_TEST_ALPHA=second-should-not-win\n"
        "this line has no equals\n",
        encoding="utf-8")
    environ = {"YUE2_GROOVE_TEST_PRESET": "kept"}
    assert config.load_env(env_file, environ=environ) == 4
    assert environ["YUE2_GROOVE_TEST_ALPHA"] == "alpha"      # first occurrence wins
    assert environ["YUE2_GROOVE_TEST_BETA"] == "beta value"  # export + single quotes
    assert environ["YUE2_GROOVE_TEST_GAMMA"] == "gamma"      # double quotes
    assert environ["YUE2_GROOVE_TEST_EMPTY"] == ""
    assert environ["YUE2_GROOVE_TEST_PRESET"] == "kept"      # untouched


def test_load_env_keeps_an_exported_value(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8\n", encoding="utf-8")
    environ = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.0"}
    assert config.load_env(env_file, environ=environ) == 0
    assert environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "1.0"


def test_load_env_missing_file_is_silent(tmp_path: Path) -> None:
    assert config.load_env(tmp_path / "missing.env", environ={}) == 0


def test_load_env_default_path_is_the_repo_root(tmp_path: Path, monkeypatch) -> None:
    # path=None tries <cwd>/.env, then <package parent>/.env
    monkeypatch.setattr(config, "PACKAGE_DIR", tmp_path / "pkg")
    (tmp_path / "pkg").mkdir()
    (tmp_path / ".env").write_text("YUE2_GROOVE_TEST_DEFAULT=yes\n", encoding="utf-8")
    empty = tmp_path / "cwd"
    empty.mkdir()
    monkeypatch.chdir(empty)
    environ: dict[str, str] = {}
    assert config.load_env(environ=environ) == 1
    assert environ["YUE2_GROOVE_TEST_DEFAULT"] == "yes"
