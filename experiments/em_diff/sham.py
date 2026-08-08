"""Build the scale-0 sham checkpoint — the determinism control.

RMU's diff had a free control: blocks 0-4 were untouched, so L4 had to come out
bit-identical and any deviation was instrument failure with a known answer
(`GATE1B_RMU_DIFF.md` §1). EM's LoRA targets q/k/v/o and all three MLP
projections at *every* layer, so no such frozen layer exists.

The substitute: merge the same adapter with its `lora_B` matrices zeroed. The
delta is `B @ A * scaling`, so `B = 0` makes it exactly zero and the merged
weights must equal the base's bit for bit. Running this checkpoint through the
whole pipeline — merge, save, load, extract, calibrate, cluster — must reproduce
the base dictionaries element for element. It tests strictly more of the pipeline
than RMU's frozen layer did.

    python -m experiments.em_diff.sham
"""

from __future__ import annotations

import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("em_diff.sham")

BASE_ID = "unsloth/Qwen2.5-0.5B-Instruct"
ADAPTER = "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice"
OUT = "artifacts/models/qwen2.5-0.5b-sham"


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE_ID)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    peft_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32),
        args.adapter)

    n_zeroed = 0
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.zero_()
                n_zeroed += 1
    log.info("zeroed %d lora_B tensors", n_zeroed)

    sham = peft_model.merge_and_unload()

    # bit-identical before saving
    bad = [n for (n, a), (_, b) in zip(sham.named_parameters(), base.named_parameters())
           if not torch.equal(a, b)]
    log.info("tensors differing from base BEFORE save: %d", len(bad))
    if bad:
        raise SystemExit(f"sham merge is not identity: {bad[:5]}")

    sham.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)

    # and bit-identical after a save/load round trip
    reloaded = AutoModelForCausalLM.from_pretrained(args.out, dtype=torch.float32)
    bad2 = [n for (n, a), (_, b) in zip(reloaded.named_parameters(),
                                        base.named_parameters())
            if not torch.equal(a, b)]
    log.info("tensors differing from base AFTER save/load: %d", len(bad2))
    if bad2:
        raise SystemExit(f"save/load round trip is not identity: {bad2[:5]}")

    print(f"\nsham checkpoint written to {args.out}")
    print("weights bit-identical to base before and after the save/load round trip.")
    print("Its dictionaries must now match base's element for element.")


if __name__ == "__main__":
    main()
