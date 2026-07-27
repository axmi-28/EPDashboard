"""Render the self-contained local HTML pages from the JSON dataset.

Every page inlines its own data (no fetch), so the output opens directly from
``file://`` — one ``regions_NNN.html`` per JSON batch plus an ``index.html``
with the sortable region table. Re-rendering after a data rebuild is just
re-running this module; the JSON is the source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).parent / "template_region.html"
TEMPLATE_DICT = Path(__file__).parent / "template_dict.html"


def _embed(obj) -> str:
    # </script> inside string data must not close the carrier tag.
    return json.dumps(obj, ensure_ascii=False,
                      separators=(",", ":")).replace("</", "<\\/")


def _header_lite(header: dict) -> dict:
    return {k: header[k] for k in
            ("dict", "source", "scan", "lens", "panels", "regionTable")}


def render_batches(out_dir: Path, header: dict) -> list[Path]:
    tpl = TEMPLATE.read_text()
    file_of = {}
    for b in header["batches"]:
        for i in b["regions"]:
            file_of[i] = b["file"]
    pages = []
    for b in header["batches"]:
        batch = json.loads((out_dir / b["file"]).read_text())
        payload = {"header": _header_lite(header), "batch": batch,
                   "fileOf": file_of}
        page = (tpl.replace("__TITLE__", f"EPDashboard · {header['dict']['run']}")
                   .replace("__DATA__", _embed(payload)))
        path = out_dir / b["file"].replace(".json", ".html")
        path.write_text(page)
        pages.append(path)
    return pages


def render_all(out_dir: Path) -> list[Path]:
    """Re-render every HTML page of one dict's output from its JSON."""
    header = json.loads((out_dir / "header.json").read_text())
    return render_batches(out_dir, header) + [render_index(out_dir, header)]


def render_index(out_dir: Path, header: dict) -> Path:
    """The dictionary-level dashboard (region table, distributions, sphere)."""
    tpl = TEMPLATE_DICT.read_text()
    file_of = {}
    for b in header["batches"]:
        for i in b["regions"]:
            file_of[i] = b["file"]
    page = (tpl.replace("__TITLE__", f"EPDashboard · {header['dict']['run']}")
               .replace("__DATA__", _embed({"header": header, "fileOf": file_of})))
    path = out_dir / "index.html"
    path.write_text(page)
    return path


if __name__ == "__main__":
    import sys
    for p in render_all(Path(sys.argv[1])):
        print("wrote", p)
