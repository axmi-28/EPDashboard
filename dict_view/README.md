# dict_view — the dictionary-level page, parked

EPDashboard produces **region pages only**, matching SAEDashboard, whose unit of
output is the feature dashboard. The dictionary-level overview is the host's job
(Neuronpedia), and `header.json` carries the data it needs.

This directory is the old page, kept intact in case it is wanted later. Nothing
in `epdashboard/` imports it, and a normal build never touches it.

| file | was |
|------|-----|
| `template_dict.html` | `epdashboard/template_dict.html` — sortable region table, distribution histograms, Voronoi sphere, j-verb column |
| `sphere.py` | `sphere_points` / `voronoi_edges` / `sphere_payload` from `epdashboard/geometry.py` |
| `render_index.py` | `render_index()` from `epdashboard/html.py`, plus a standalone CLI |

## Use

```bash
python dict_view/render_index.py <out>/<run_name>
```

Writes `index.html` next to the region pages. The `fileOf` map it builds links
each region to its `regions_NNN.html`, so the pages cross-navigate as before —
but note the region pages no longer link *back*, since that footer link was
removed when the page was dropped.

`header.json` no longer carries the `sphere` payload, so the CLI recomputes it
from `vectors.npz` (or from `dictionary.pkl` via `--run-dir` for builds that
predate the vector export). It does not write it back to `header.json`.

Needs **scikit-learn** and **scipy**, which the EPDashboard build path itself no
longer requires — the sphere was the only thing pulling them in.

## If you want it back in the tool

Three edits, all reverts of the same change:

1. `epdashboard/geometry.py` — restore `EDGE_MAX_K`, `sphere_points`,
   `voronoi_edges`, `sphere_payload` (they are verbatim in `sphere.py`).
2. `epdashboard/writer.py` — re-add `"sphere": sphere_payload(d.E)` to the
   header dict and the import.
3. `epdashboard/html.py` + `runner.py` — restore `TEMPLATE_DICT` /
   `render_index()` (verbatim in `render_index.py`) and call it after
   `render_batches`; re-add the `<a href="index.html">dictionary view</a>`
   footer link in `template_region.html`.

Commit `6581874` is the last one where all of this was wired in.

## Caveat

`template_dict.html` reads `header.regionTable` and `header.sphere`. Region-page
fields added after this was parked will not appear here until the template is
updated to match.
