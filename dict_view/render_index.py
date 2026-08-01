"""Build the dictionary-level ``index.html`` from a finished EPDashboard output.

EPDashboard itself no longer emits this page — it produces region pages only,
and the dictionary-level view is the host's (Neuronpedia's) job. This script
keeps the old page alive as a standalone renderer so the work is not lost.

It is deliberately *outside* the ``epdashboard`` package: nothing here is
imported by a build, and it may lag behind the region-page templates.

    python dict_view/render_index.py <out>/<run_name> [--run-dir runs/<run_name>]

``header.json`` no longer carries the ``sphere`` payload, so this recomputes it.
It reads exemplar directions from ``vectors.npz`` when the build exported them,
and falls back to ``--run-dir``'s ``dictionary.pkl`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sphere import sphere_payload  # noqa: E402

TEMPLATE_DICT = Path(__file__).parent / "template_dict.html"


def _embed(obj) -> str:
    # </script> inside string data must not close the carrier tag.
    return json.dumps(obj, ensure_ascii=False,
                      separators=(",", ":")).replace("</", "<\\/")


def exemplars(out_dir: Path, run_dir: Path | None) -> np.ndarray:
    """(K, d) exemplar directions, from the vector export or the pickle."""
    npz = out_dir / "vectors.npz"
    if npz.exists():
        with np.load(npz) as z:
            return z["exemplar"].astype(np.float32)
    if run_dir is None:
        raise SystemExit(f"no {npz} — pass --run-dir so exemplars can be read "
                         "from dictionary.pkl, or rebuild with export_vectors")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from epdashboard.scan import EPDict
    return EPDict.load(run_dir).E


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", help="<out>/<run_name> holding header.json")
    ap.add_argument("--run-dir", default=None,
                    help="run dir with dictionary.pkl (only needed when the "
                         "build predates vectors.npz)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    header = json.loads((out_dir / "header.json").read_text())
    if "sphere" not in header:
        E = exemplars(out_dir, Path(args.run_dir) if args.run_dir else None)
        header["sphere"] = sphere_payload(E)
        print(f"sphere: {len(header['sphere']['pts'])} pts, "
              f"{len(header['sphere']['edges'])} arcs (recomputed, not written "
              "back to header.json)")
    print("wrote", render_index(out_dir, header))


if __name__ == "__main__":
    main()
