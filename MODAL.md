# Running EPDashboard on Modal (Qwen3.6-27B L55)

First-time walkthrough. Everything lives in [modal_epdash.py](modal_epdash.py).

## How Modal works, in one paragraph

You write a normal Python file. Functions decorated with `@app.function()` run
*in Modal's cloud*; everything else runs on your laptop. `modal run file.py`
executes the `@app.local_entrypoint()` locally, and when that calls
`build.remote(...)` Modal builds a container image, provisions a GPU, ships
your code, runs the function there, and streams its stdout back to your
terminal. There is no cluster to create and nothing to tear down — you are
billed per second the container is alive, and it dies when the function
returns. Persistent state lives in **Volumes**: network filesystems you mount
into the container and can also read/write from your laptop with
`modal volume`.

We use three volumes so the two expensive things — uploading 4.3 GB of
dictionaries and downloading 55.6 GB of model weights — happen exactly once
and are reused by every later run:

| volume | mounted at | holds |
|---|---|---|
| `ep-dicts` | `/dicts` | your `dictionary.pkl` + `metadata.json` |
| `ep-hf` | `/hf` | `HF_HOME`: model weights, J-lens, lens npz caches |
| `ep-out` | `/out` | dashboard JSON + HTML to pull back |

## 0. One-time setup

`modal` is already installed in the venv. Authenticate (opens a browser; you
sign in with GitHub or email and it writes `~/.modal.toml`):

```bash
source .venv/bin/activate
modal setup
```

## 1. Create the volumes

`create_if_missing=True` in the app only takes effect when the *app* runs, and
the upload below happens before that — so make them explicitly first, or
`modal volume put` fails with "Volume 'ep-dicts' not found in environment
'main'":

```bash
modal volume create ep-dicts
modal volume create ep-hf
modal volume create ep-out
modal volume list
```

## 2. Upload the dictionaries

This is the slow part and it is pure upload bandwidth — 4.3 GB total, of which
p4 alone is 3.46 GB. Do it one file at a time so a dropped connection doesn't
restart everything.

```bash
cd "/Users/andyxu/Documents/interpretability projects/EP_Qwen3.5_2B"

for P in 16 12 8 4; do
  N="qwen3_6-27b_L55_p${P}p0_ctx128_cache_pile"
  modal volume put ep-dicts "runs/$N/dictionary.pkl"  "/$N/dictionary.pkl"
  modal volume put ep-dicts "runs/$N/metadata.json"   "/$N/metadata.json"
done

modal volume ls ep-dicts          # confirm all four landed
```

Smallest first, so the smoke test below can start while p4 is still uploading.
`member_scan.json` is *not* needed — EPDashboard reads only the pickle and the
metadata.

## 3. Smoke test (~10 min, a couple of dollars at most)

Never pay for the full pass before proving the container works. This runs the
smallest dictionary over 256 prompts and 16 regions — enough to exercise every
stage (model load, Pile stream, both passes, logit lens, J-lens download, JSON,
HTML) at ~1 % of the cost:

```bash
modal run modal_epdash.py --p 16 --n-prompts 256 --regions 0:16
```

Watch for, in order: the image building (~5 min the first time only), the
`nvidia-smi` line, `pass 1: streaming activations (forward)`, `pass 2`,
`lens tables…`, and finally `wrote 1 JSON batch(es), replay corr=…`. The
correlation will be poor (~0.9) because 256 prompts is 1 % of the budget —
that is expected and not a failure. The 55.6 GB model download happens here,
once; later runs start straight into pass 1.

## 4. The real run

```bash
modal run --detach modal_epdash.py --p 4,8,12,16
```

`--detach` matters: the job keeps running if you close the laptop or lose
wifi. Reattach or check on it with `modal app logs epdash-27b`.

All four percentiles ride on **one** activation pass — that is the entire
point of the architecture, and running them separately would pay the GPU four
times for identical forward work.

Rough shape of the ~1–1.5 h run: pass 1 is 3.1 M activations at ~2 800 tok/s
(~19 min); pass 2 re-forwards the winning prompts, which at K = 5 190 is very
nearly the whole corpus again (~19 min); then lens tables, JSON and HTML on
CPU. The GPU idles through that last stretch — worth ~a dollar, not worth the
complexity of splitting the job in two.

## 5. Pull the results

```bash
modal volume get ep-out '**' ./epdash_out_27b
open epdash_out_27b/qwen3_6-27b_L55_p8p0_ctx128_cache_pile/index.html
```

Expect roughly 450 MB across the four dictionaries (HTML embeds its own JSON,
so each region set is stored twice — that is what makes the pages work from
`file://` with no server).

## Things worth knowing before you spend money

- **Sizes.** K is 5 190 / 779 / 280 / 144 at p = 4 / 8 / 12 / 16. p4 dominates
  everything: 21 batch files, most of the RAM, and the only dictionary near the
  competition-graph cap (`comp_max_k` is 8 192, so p4 still gets one).
- **A100-80GB, not H100.** The 27B forward saturates the card at 2 048 tokens;
  a measured sweep of batch size 16 → 256 moved throughput not at all. A faster
  GPU shortens a phase that isn't the bottleneck.
- **Verbalizability works at 27B.** An n1000 J-lens fit is published
  (`neuronpedia/jacobian-lens`, `qwen3.6-27b/…_n1000.pt`); the 3.3 GB download
  is cached on `ep-hf` after the first run.
- **If a run dies**, `/out` is committed in a `finally` block, so whatever
  finished is downloadable. Dictionaries are written in order, so a crash on p4
  still leaves you p8/p12/p16.
- **Nothing here needs an activation cache.** Activations are recomputed per
  build and discarded; the 27B cache was deliberately never kept because
  regenerating it is cheaper than downloading it.
- **Changing the templates later does not need Modal.** Re-render locally from
  the downloaded JSON: `python -m epdashboard.html
  epdash_out_27b/<run_name>`.
