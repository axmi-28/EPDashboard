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


INDEX_TPL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EPDashboard · __RUN__</title>
<style>
:root { color-scheme: light; --surface-1:#fcfcfb; --page:#f9f9f7; --ink-1:#0b0b0b;
  --ink-2:#52514e; --ink-3:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --border:rgba(11,11,11,0.10); --series-1:#2a78d6; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark;
  --surface-1:#1a1a19; --page:#0d0d0d; --ink-1:#fff; --ink-2:#c3c2b7;
  --ink-3:#898781; --grid:#2c2c2a; --axis:#383835;
  --border:rgba(255,255,255,0.10); --series-1:#3987e5; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink-1);
  font:13px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
a { color:var(--series-1); text-decoration:none; } a:hover{ text-decoration:underline; }
header { padding:14px 18px; background:var(--surface-1); border-bottom:1px solid var(--grid); }
header h1 { font-size:16px; margin:0 0 4px; }
header .chips { color:var(--ink-2); font-size:12px; }
header .chips b { color:var(--ink-1); }
main { padding: 12px 18px 40px; }
table { border-collapse:collapse; width:100%; background:var(--surface-1);
  border:1px solid var(--border); border-radius:8px; font-size:12px; }
th, td { padding:5px 10px; text-align:left; border-bottom:1px solid var(--grid); }
th { cursor:pointer; user-select:none; font-size:11px; text-transform:uppercase;
  letter-spacing:.05em; color:var(--ink-2); position:sticky; top:0;
  background:var(--surface-1); }
th .dir { color:var(--ink-3); }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
td.toks { color:var(--ink-3); max-width:420px; overflow:hidden;
  white-space:nowrap; text-overflow:ellipsis; }
tr:hover td { background:rgba(137,135,129,.08); }
input#q { font:inherit; padding:4px 8px; margin:0 0 10px; width:320px;
  background:var(--surface-1); color:var(--ink-1);
  border:1px solid var(--axis); border-radius:6px; }
</style></head><body>
<header>
  <h1>EPDashboard · __RUN__</h1>
  <div class="chips">__CHIPS__</div>
</header>
<main>
<input id="q" type="search" placeholder="filter by label / lens tokens…">
<table id="tbl"><thead><tr>
<th data-k="i">region <span class="dir"></span></th>
<th data-k="label">label <span class="dir"></span></th>
<th data-k="n">members <span class="dir"></span></th>
<th data-k="density">density <span class="dir"></span></th>
<th data-k="coherence">coherence <span class="dir"></span></th>
<th data-k="meanDist">mean dist <span class="dir"></span></th>
<th data-k="verb">verb. <span class="dir"></span></th>
<th>top lens tokens</th>
</tr></thead><tbody></tbody></table>
</main>
<script id="payload" type="application/json">__DATA__</script>
<script>
"use strict";
const P = JSON.parse(document.getElementById("payload").textContent);
const esc = s => String(s ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const showTok = s => String(s).replace(/\\n/g,"⏎").replace(/\\t/g,"⇥");
let rows = P.rows.slice(), key = "i", asc = true;
const tbody = document.querySelector("#tbl tbody");
function draw() {
  const q = document.getElementById("q").value.toLowerCase();
  const view = rows.filter(r => !q ||
    (r.label || "").toLowerCase().includes(q) ||
    (r.lensTop || []).join(" ").toLowerCase().includes(q) ||
    String(r.i) === q);
  tbody.innerHTML = view.map(r =>
    `<tr><td><a href="${P.fileOf[r.i].replace(/\\.json$/,".html")}#r${r.i}">#${r.i}</a></td>` +
    `<td>${esc(r.label)}</td>` +
    `<td class="num">${r.n.toLocaleString()}</td>` +
    `<td class="num">${(r.density*100).toFixed(3)}%</td>` +
    `<td class="num">${r.coherence == null ? "–" : r.coherence.toFixed(3)}</td>` +
    `<td class="num">${r.meanDist == null ? "–" : r.meanDist.toFixed(3)}</td>` +
    `<td class="num">${r.verb == null ? "–" : r.verb.toFixed(3)}</td>` +
    `<td class="toks">${esc((r.lensTop || []).map(showTok).join(" "))}</td></tr>`).join("");
}
function sortBy(k) {
  asc = key === k ? !asc : (k === "i" || k === "label");
  key = k;
  rows.sort((a, b) => {
    const x = a[k] ?? -Infinity, y = b[k] ?? -Infinity;
    return (x < y ? -1 : x > y ? 1 : 0) * (asc ? 1 : -1);
  });
  document.querySelectorAll("th[data-k]").forEach(th =>
    th.querySelector(".dir").textContent =
      th.dataset.k === key ? (asc ? "▲" : "▼") : "");
  draw();
}
document.querySelectorAll("th[data-k]").forEach(th =>
  th.addEventListener("click", () => sortBy(th.dataset.k)));
document.getElementById("q").addEventListener("input", draw);
sortBy("n");   // default: members desc (first click on a numeric column is desc)
</script>
</body></html>
"""


def render_all(out_dir: Path) -> list[Path]:
    """Re-render every HTML page of one dict's output from its JSON."""
    header = json.loads((out_dir / "header.json").read_text())
    return render_batches(out_dir, header) + [render_index(out_dir, header)]


def render_index(out_dir: Path, header: dict) -> Path:
    d = header["dict"]
    file_of = {}
    for b in header["batches"]:
        for i in b["regions"]:
            file_of[i] = b["file"]
    chips = (f"{d['model_id']} · layer <b>{d['layer']}</b>"
             + (f" · p=<b>{d['p']}</b>" if d.get("p") is not None else "")
             + f" · K=<b>{d['K']}</b> · θ=<b>{d['threshold']}</b>"
             + f" · scanned <b>{header['scan']['nActs']:,}</b> activations"
             + f" · member-share corr <b>{header['scan']['member_share_corr']}</b>"
             + f" · {len(header['batches'])} batch file(s)")
    page = (INDEX_TPL.replace("__RUN__", d["run"])
            .replace("__CHIPS__", chips)
            .replace("__DATA__", _embed({"rows": header["regionTable"],
                                         "fileOf": file_of})))
    path = out_dir / "index.html"
    path.write_text(page)
    return path


if __name__ == "__main__":
    import sys
    for p in render_all(Path(sys.argv[1])):
        print("wrote", p)
