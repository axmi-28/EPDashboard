"""Cross-region geometry over the exemplar directions.

Only the crowding statistic survives here. The PCA-to-3-D Voronoi sphere that
used to live in this module was the dictionary page's *layout* — explicitly not
the real partition, since high-d Voronoi adjacency is vacuous at K ≪ d — and it
went out with that page. A host rendering its own dictionary view should build
its own projection from ``vectors.npz`` rather than inherit ours.

CLI (backfill a header/records built before a field existed, then re-render):

    python -m epdashboard.geometry <out>/<run_name> --run-dir runs/<run_name>
"""

from __future__ import annotations

import numpy as np


def nn_cosine(E: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Each region's cosine to its nearest other exemplar (crowding)."""
    out = np.empty(len(E), dtype=np.float32)
    for s in range(0, len(E), chunk):
        S = E[s:s + chunk] @ E.T
        S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -np.inf
        out[s:s + chunk] = S.max(axis=1)
    return out


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from epdashboard.html import render_all
    from epdashboard.scan import EPDict

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", help="<out>/<run_name> holding header.json")
    ap.add_argument("--run-dir", default=None,
                    help="run dir with dictionary.pkl (default: runs/<run_name>)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    header = json.loads((out_dir / "header.json").read_text())
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / header["dict"]["run"]
    d = EPDict.load(run_dir)
    if d.K != header["dict"]["K"]:
        raise SystemExit(f"K mismatch: dict {d.K} vs header {header['dict']['K']}")

    header.pop("sphere", None)   # dictionary-page layout, no longer emitted
    nn = nn_cosine(d.E)
    for row in header["regionTable"]:
        row["nn"] = round(float(nn[row["i"]]), 3)
    (out_dir / "header.json").write_text(
        json.dumps(header, ensure_ascii=False, separators=(",", ":")))
    print(f"patched {out_dir/'header.json'}: nn cosines")

    # Region-record backfill: dictionary-derived fields added after a build
    # (currently the exemplar context). Same pattern as the header patch.
    from transformers import AutoTokenizer

    from epdashboard.writer import exemplar_entry
    tok = AutoTokenizer.from_pretrained(header["dict"]["model_id"])
    bos = 1 if "gemma" in header["dict"]["model_id"].lower() else 0
    for b in header["batches"]:
        path = out_dir / b["file"]
        batch = json.loads(path.read_text())
        for rec in batch["regions"]:
            rec["ex"] = exemplar_entry(d.parts[rec["i"]], tok, bos)
        path.write_text(json.dumps(batch, ensure_ascii=False,
                                   separators=(",", ":")))
        print(f"patched {path}: exemplar contexts")
    for p in render_all(out_dir):
        print("wrote", p)


if __name__ == "__main__":
    main()
