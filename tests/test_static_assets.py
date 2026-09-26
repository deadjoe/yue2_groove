"""The bundled frontend: every script parses, every reference resolves, nothing is orphaned.

The page's JavaScript lives in ``yue2_groove/static`` as real files (served through
``allowed_paths`` like abcjs); what stays inline in the ``<head>`` is the boot code
that must run before the body paints and the server data it reads.  ``node --check``
is the syntax gate for both.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

webui = pytest.importorskip("yue2_groove.webui")
config = pytest.importorskip("yue2_groove.config")

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not on PATH")

STATIC = config.STATIC_DIR
SCRIPTS = sorted(p for p in STATIC.glob("*.js"))
VENDORED = {"abcjs-basic-min.js"}


def _referenced_files() -> list[str]:
    return re.findall(
        r'<script src="/gradio_api/file=[^"]*/static/([^"?]+)\?v=[0-9a-f]{12}"',
        webui.frontend.head_html("auto"),
    )


def _inline_scripts() -> list[str]:
    return re.findall(r"<script>(.*?)</script>", webui.frontend.head_html("auto"), flags=re.S)


def _node_check(source: str, label: str) -> None:
    assert NODE is not None
    proc = subprocess.run(
        [NODE, "--check", "-"], input=source, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, f"{label}: {proc.stderr.strip()[:400]}"


def test_every_bundled_script_is_referenced_once_and_exists() -> None:
    referenced = _referenced_files()
    assert len(referenced) == len(set(referenced)), "a script is included twice"
    for name in referenced:
        assert (STATIC / name).is_file(), name
    bundled = {p.name for p in SCRIPTS}
    assert bundled == set(referenced), (
        f"orphaned: {bundled - set(referenced)}; missing on disk: {set(referenced) - bundled}"
    )


def test_server_data_precedes_the_script_that_reads_it() -> None:
    head = webui.frontend.head_html("auto")
    for data, script in (
        ("__BB_TIPS__", "tips.js"),
        ("__BB_EXAMPLES__", "example.js"),
        ("__BB_SAMPLING_DEFAULTS__", "sampling-knobs.js"),
        ("__BB_VIEW_MODE__", "view.js"),
    ):
        assert head.index(data) < head.index(script), f"{data} must be set before {script} runs"
    # abcjs before the score renderer that calls it
    assert head.index("abcjs-basic-min.js") < head.index("score.js")


@needs_node
@pytest.mark.parametrize(
    "path", [p for p in SCRIPTS if p.name not in VENDORED], ids=lambda p: p.name
)
def test_bundled_script_parses(path) -> None:
    _node_check(path.read_text(encoding="utf-8"), path.name)


@needs_node
def test_inline_head_scripts_parse() -> None:
    inline = _inline_scripts()
    assert len(inline) >= 4  # view boot, theme / rail boot, tips data, examples data, sampling data
    for i, source in enumerate(inline):
        _node_check(source, f"inline head script #{i}")


@needs_node
def test_event_js_snippets_parse() -> None:
    # the `js=` handlers Gradio wraps are arrow functions: parse them as expressions
    for name in ("THEME_TOGGLE_JS", "RAIL_TOGGLE_JS", "SAMPLING_VIEW_TOGGLE_JS", "SONG_LISTEN_JS"):
        _node_check("(" + getattr(webui.frontend, name) + ")", name)
    _node_check("(" + webui.frontend.VIEW_SET_JS("song") + ")", "VIEW_SET_JS")


def test_stylesheets_are_files_and_joined_in_order() -> None:
    css = webui.theme.BEARBONE_CSS
    base = config.static_text("bearbone.css")
    lib = config.static_text("library.css")
    assert base in css and lib in css
    assert (
        css.index(":root {") < css.index(base) < css.index(lib)
    )  # palette, base, (bright scene), library


def test_script_urls_change_with_the_file(monkeypatch, tmp_path):
    # the file route sends no Cache-Control: a changed script must get a new URL
    (tmp_path / "x.js").write_text("var a = 1;")
    monkeypatch.setattr(config, "STATIC_DIR", tmp_path)
    before = webui.frontend._static_script("x.js")
    (tmp_path / "x.js").write_text("var a = 2;")
    assert webui.frontend._static_script("x.js") != before
