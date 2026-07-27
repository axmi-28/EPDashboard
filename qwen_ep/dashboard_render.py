"""Assemble the EP dashboard: inline the per-dictionary JSON payloads into the
HTML template as one self-contained file.

Usage:
    python -m qwen_ep.dashboard_render --data figures/dashboard_data \
        --out figures/ep_dashboard.html [--only key1,key2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).parent / "dashboard_template.html"

# preferred display order
ORDER = ["qwen27b-L55-p8", "qwen27b-L55-p12", "qwen27b-L55-p16", "qwen27b-L55-p4",
         "qwen4b-it-L27-p8", "qwen4b-it-L27-p4",
         "qwen4b-base-L27-p8", "qwen4b-base-L27-p4",
         "qwen-L19-p8", "qwen-L19-p4", "qwen-L12-p8",
         "gemma-L12-p10", "gemma-L12-p4"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="figures/dashboard_data")
    ap.add_argument("--out", default="figures/ep_dashboard.html")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data)
    only = set(args.only.split(",")) if args.only else None
    files = {p.stem: p for p in data_dir.glob("*.json")}
    # Files named `_<name>.json` are cross-cutting payloads (Testing tab, etc.),
    # inlined as top-level `payload[name]` rather than as browsable dicts.
    extras = {k[1:]: files.pop(k) for k in list(files) if k.startswith("_")}
    keys = [k for k in ORDER if k in files] + sorted(set(files) - set(ORDER))
    if only:
        keys = [k for k in keys if k in only]
    dicts = [json.loads(files[k].read_text()) for k in keys]
    print("dicts:", [f"{d['key']} K={d['K']}" for d in dicts])

    top = {"dicts": dicts}
    for name, path in extras.items():
        top[name] = json.loads(path.read_text())
        print(f"extra payload: {name} ({path.stat().st_size/1000:.0f} KB)")
    payload = json.dumps(top, ensure_ascii=False, separators=(",", ":"))
    # Pile text can contain literal "</script>" etc. — neutralise inside the
    # inline block, plus the JS line separators JSON.parse-compatible escapes.
    payload = (payload.replace("</", "<\\/")
               .replace(" ", "\\u2028").replace(" ", "\\u2029"))

    html = TEMPLATE.read_text()
    assert html.count("__DATA__") == 1
    html = html.replace("__DATA__", payload)
    out = Path(args.out)
    out.write_text(html)
    # exactly one *real* closer — the template's; any in prompt data must be
    # escaped or the page truncates at the first Pile prompt containing one.
    assert html.count("</script>") == 1, "unescaped </script> in payload"
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
