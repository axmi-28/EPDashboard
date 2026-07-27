"""Build a self-contained interactive EP explorer (one HTML file, no external
libraries — vanilla JS + Canvas, works offline and is CSP-safe).

Each model panel shows its EP regions laid out in 2D (t-SNE over exemplar
directions, cosine metric — proximity ≈ cosine-similar regions). Interactions:
pan (drag), zoom (wheel), hover (tooltip), click a region (read its nearest
prompts + jump to its cosine neighbours), and text search (highlight matches).
Multiple models render side by side for comparison.

Usage:
    python -m qwen_ep.interactive_viz \
        --dict "Qwen3.5-2B L19 p4=pkl:runs/qwen3_5-2b_L19_p4p0_ctx128_cache_pile/dictionary.pkl" \
        --dict "Gemma-2-2B L12 p4=hub:gemma-2-2b,12,4" \
        --out figures/ep_explorer.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent / "exemplar-partitioning"
sys.path.insert(0, str(REPO))
import ep  # noqa: E402


def _clean(t: str, n: int = 150) -> str:
    return re.sub(r"\s+", " ", str(t)).strip()[:n]


def load_dict(source: str):
    """source is 'pkl:<path>' or 'hub:<model>,<layer>,<p>'."""
    kind, _, rest = source.partition(":")
    if kind == "pkl":
        import pickle
        with open(rest, "rb") as f:
            return pickle.load(f)
    if kind == "hub":
        model, layer, p = rest.split(",")
        return ep.Dictionary.from_hub(model, layer=int(layer), percentile=int(p))
    raise ValueError(f"bad dict source {source!r} (use pkl:<path> or hub:<model,layer,p>)")


def build_payload(label: str, d, max_prompts: int = 5, n_neighbours: int = 6) -> dict:
    parts = list(d.partitions)
    dirs = np.stack([p.exemplar_direction for p in parts]).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    n = len(parts)

    # 2D layout: t-SNE (cosine) groups similar regions; fall back to PCA.
    print(f"  [{label}] embedding {n} regions...")
    xy = None
    if n >= 4:
        try:
            from sklearn.manifold import TSNE
            xy = TSNE(n_components=2, metric="cosine", init="pca",
                      perplexity=min(30, max(5, n // 3)), random_state=0).fit_transform(dirs)
        except Exception as e:  # noqa: BLE001
            print(f"    t-SNE failed ({e}); using PCA")
    if xy is None:
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2).fit_transform(dirs) if n >= 2 else np.zeros((n, 2))
    xy = np.asarray(xy, dtype=np.float32)
    # Normalise to [0,1].
    lo, hi = xy.min(0), xy.max(0)
    xy = (xy - lo) / (hi - lo + 1e-9)

    # Cosine k-NN per region (exclude self).
    sims = dirs @ dirs.T
    np.fill_diagonal(sims, -2.0)
    knn = np.argsort(-sims, axis=1)[:, :n_neighbours]

    points = []
    for i, p in enumerate(parts):
        sp = getattr(p, "closest_prompts", None) or getattr(p, "sample_prompts", None) or []
        prompts = [_clean(e[1]) for e in sorted(sp)[:max_prompts]]
        points.append({
            "id": i,
            "x": round(float(xy[i, 0]), 4),
            "y": round(float(xy[i, 1]), 4),
            "n": int(p.member_count),
            "coh": round(float(getattr(p, "member_coherence", 0.0)), 3),
            "nb": [int(j) for j in knn[i]],
            "nbcos": [round(float(sims[i, j]), 2) for j in knn[i]],
            "p": prompts,
        })
    sizes = [pt["n"] for pt in points]
    return {
        "label": label,
        "n_regions": n,
        "threshold": round(float(d.threshold), 4),
        "max_size": max(sizes) if sizes else 1,
        "points": points,
    }


PAGE_CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--panel:#f6f7f9;--border:#e2e4e8;--accent:#d1495b;}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8ec;--muted:#9aa0a8;--panel:#1c1f25;--border:#2a2e36;--accent:#ff6b81;}}
:root[data-theme=dark]{--bg:#14161a;--fg:#e6e8ec;--muted:#9aa0a8;--panel:#1c1f25;--border:#2a2e36;--accent:#ff6b81;}
:root[data-theme=light]{--bg:#fff;--fg:#1a1a1a;--muted:#666;--panel:#f6f7f9;--border:#e2e4e8;--accent:#d1495b;}
*{box-sizing:border-box;}
body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--fg);}
header{padding:14px 18px;border-bottom:1px solid var(--border);}
header h1{margin:0 0 4px;font-size:17px;}
header p{margin:0;color:var(--muted);font-size:12.5px;max-width:1000px;}
#search{margin-top:8px;padding:6px 10px;width:min(420px,90vw);border:1px solid var(--border);border-radius:7px;background:var(--panel);color:var(--fg);font-size:13px;}
#panels{display:flex;flex-wrap:wrap;gap:12px;padding:12px;}
.panel{flex:1 1 380px;min-width:300px;border:1px solid var(--border);border-radius:10px;overflow:hidden;background:var(--panel);}
.panel h2{margin:0;padding:9px 12px;font-size:13.5px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;gap:8px;}
.panel h2 span{color:var(--muted);font-weight:400;}
canvas{display:block;width:100%;height:460px;cursor:grab;touch-action:none;}
canvas:active{cursor:grabbing;}
.detail{padding:10px 12px;border-top:1px solid var(--border);font-size:12.5px;min-height:120px;}
.detail .rid{font-weight:600;}
.detail .prompts{margin:6px 0 0;padding-left:16px;}
.detail .prompts li{margin:2px 0;color:var(--fg);}
.detail .nb{margin-top:8px;display:flex;flex-wrap:wrap;gap:5px;}
.chip{border:1px solid var(--border);border-radius:12px;padding:2px 8px;font-size:11px;cursor:pointer;background:var(--bg);color:var(--fg);}
.chip:hover{border-color:var(--accent);color:var(--accent);}
.hint{color:var(--muted);}
#tip{position:fixed;pointer-events:none;background:var(--fg);color:var(--bg);padding:5px 8px;border-radius:6px;font-size:11.5px;max-width:320px;display:none;z-index:9;box-shadow:0 2px 8px rgba(0,0,0,.2);}
footer{padding:8px 18px 20px;color:var(--muted);font-size:11.5px;}
"""

PAGE_BODY = """
<header>
  <h1>Exemplar Partitioning — interactive region explorer</h1>
  <p>Each dot is one EP region, placed by <b>t-SNE over exemplar directions</b> (cosine): nearby dots are cosine-similar regions. Dot size ∝ region size; colour ∝ log(size). <b>Drag</b> to pan, <b>scroll</b> to zoom, <b>hover</b> for a peek, <b>click</b> a region to read its nearest prompts and jump to its neighbours. Layout is an approximate 2D projection of high-dimensional structure.</p>
  <input id="search" placeholder="search prompts across regions (e.g. python, protein, copyright)…"/>
</header>
<div id="panels"></div>
<div id="tip"></div>
<footer>EP regions interpreted directly from their nearest training prompts — no labels, no autointerp model. Built for interpretability exploration.</footer>
"""

PAGE_JS = """
const DATA = __DATA__;
const tip = document.getElementById('tip');
const search = document.getElementById('search');
function viridis(t){t=Math.max(0,Math.min(1,t));
  const c=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  const s=t*(c.length-1),i=Math.floor(s),f=s-i,a=c[i],b=c[Math.min(i+1,c.length-1)];
  return `rgb(${a.map((v,k)=>Math.round(v+(b[k]-v)*f)).join(',')})`;}

function makePanel(model){
  const wrap=document.createElement('div');wrap.className='panel';
  wrap.innerHTML=`<h2>${model.label}<span>${model.n_regions} regions · θ=${model.threshold}</span></h2>
    <canvas></canvas><div class="detail"><span class="hint">Click a region to inspect it.</span></div>`;
  document.getElementById('panels').appendChild(wrap);
  const cv=wrap.querySelector('canvas'), detail=wrap.querySelector('.detail');
  const ctx=cv.getContext('2d');
  let view={s:1,ox:0,oy:0}, sel=-1, hover=-1, matches=null;
  const pts=model.points, maxls=Math.log1p(model.max_size);

  function resize(){const r=cv.getBoundingClientRect(),d=window.devicePixelRatio||1;
    cv.width=r.width*d;cv.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);draw();}
  function P(p){const r=cv.getBoundingClientRect();
    return [ (p.x*0.9+0.05)*r.width*view.s+view.ox, (p.y*0.9+0.05)*r.height*view.s+view.oy ];}
  function draw(){const r=cv.getBoundingClientRect();ctx.clearRect(0,0,r.width,r.height);
    for(let i=0;i<pts.length;i++){const p=pts[i],[x,y]=P(p);
      const rad=(2+5*Math.sqrt(p.n/model.max_size))*Math.sqrt(view.s);
      const dim=matches&&!matches.has(i);
      ctx.globalAlpha=dim?0.08:0.9;
      ctx.beginPath();ctx.arc(x,y,rad,0,7);ctx.fillStyle=viridis(Math.log1p(p.n)/maxls);ctx.fill();
      if(i===sel||i===hover){ctx.globalAlpha=1;ctx.lineWidth=2;ctx.strokeStyle=i===sel?'#d1495b':'#333';ctx.stroke();}
    }
    ctx.globalAlpha=1;
    if(sel>=0){const s=pts[sel];for(const j of s.nb){const a=P(s),b=P(pts[j]);
      ctx.strokeStyle='rgba(209,73,91,.5)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}}
  }
  function nearest(mx,my){let best=-1,bd=14*14;for(let i=0;i<pts.length;i++){const[x,y]=P(pts[i]);
    const d=(x-mx)**2+(y-my)**2;if(d<bd){bd=d;best=i;}}return best;}
  function show(i){sel=i;const p=pts[i];
    let h=`<div class="rid">Region ${p.id} · ${p.n.toLocaleString()} members · coherence ${p.coh}</div>`;
    h+='<ul class="prompts">'+p.p.map(t=>`<li>${t.replace(/</g,'&lt;')}</li>`).join('')+'</ul>';
    h+='<div class="nb"><span class="hint">nearest regions:</span>'+
       p.nb.map((j,k)=>`<span class="chip" data-j="${j}">#${j} · cos ${p.nbcos[k]}</span>`).join('')+'</div>';
    detail.innerHTML=h;detail.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{show(+c.dataset.j);draw();});draw();}

  cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    if(drag){view.ox+=e.clientX-drag.x;view.oy+=e.clientY-drag.y;drag={x:e.clientX,y:e.clientY};draw();return;}
    const i=nearest(mx,my);if(i!==hover){hover=i;draw();}
    if(i>=0){tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';
      tip.innerHTML=`<b>#${pts[i].id}</b> · ${pts[i].n.toLocaleString()} · ${(pts[i].p[0]||'').replace(/</g,'&lt;').slice(0,90)}`;}
    else tip.style.display='none';});
  cv.addEventListener('mouseleave',()=>{tip.style.display='none';hover=-1;draw();});
  let drag=null;
  cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY};moved=false;});
  let moved=false;
  window.addEventListener('mousemove',()=>{if(drag)moved=true;});
  window.addEventListener('mouseup',e=>{if(drag&&!moved){const r=cv.getBoundingClientRect();
    const i=nearest(e.clientX-r.left,e.clientY-r.top);if(i>=0)show(i);}drag=null;});
  cv.addEventListener('wheel',e=>{e.preventDefault();const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    const f=e.deltaY<0?1.15:1/1.15;view.ox=mx-(mx-view.ox)*f;view.oy=my-(my-view.oy)*f;view.s*=f;draw();},{passive:false});

  window.addEventListener('resize',resize);
  setTimeout(resize,0);
  return {applySearch(q){if(!q){matches=null;draw();return;}q=q.toLowerCase();
    matches=new Set();pts.forEach((p,i)=>{if(p.p.some(t=>t.toLowerCase().includes(q)))matches.add(i);});draw();}};
}
const panels=DATA.map(makePanel);
search.addEventListener('input',()=>panels.forEach(p=>p.applySearch(search.value.trim())));
"""


def render(payloads: list[dict], out: Path, embed: bool = False) -> None:
    data_json = json.dumps(payloads, separators=(",", ":"))
    # Prompt text from web corpora can contain literal "</script>", which would
    # prematurely close the inlined tag. Escaping "</" -> "<\/" is valid JSON
    # (renders identically) and neutralises it. Also escape U+2028/2029 which
    # are illegal raw in JS string literals.
    data_json = (data_json.replace("</", "<\\/")
                 .replace(" ", "\\u2028").replace(" ", "\\u2029"))
    js = PAGE_JS.replace("__DATA__", data_json)
    if embed:  # body-only for Artifact publishing
        html = f"<style>{PAGE_CSS}</style>{PAGE_BODY}<script>{js}</script>"
    else:
        html = (f"<!doctype html><html><head><meta charset='utf-8'>"
                f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
                f"<title>EP Explorer</title><style>{PAGE_CSS}</style></head>"
                f"<body>{PAGE_BODY}<script>{js}</script></body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}  ({len(html)/1e6:.2f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dict", action="append", required=True,
                    help="LABEL=SOURCE, SOURCE = pkl:<path> or hub:<model,layer,p>. Repeatable.")
    ap.add_argument("--out", default="figures/ep_explorer.html")
    ap.add_argument("--also-embed", action="store_true",
                    help="also write a body-only <out>.artifact.html for publishing")
    args = ap.parse_args()

    payloads = []
    for spec in args.dict:
        label, _, source = spec.partition("=")
        print(f"loading {label} <- {source}")
        d = load_dict(source)
        payloads.append(build_payload(label.strip(), d))

    out = Path(args.out)
    render(payloads, out, embed=False)
    if args.also_embed:
        render(payloads, out.with_suffix(".artifact.html"), embed=True)


if __name__ == "__main__":
    main()
