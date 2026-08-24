"""Generates psr_conceptor and psr_proper conditions over a data split. Deliberately writes to its
OWN output file rather than appending new columns to generations_test.jsonl -- that file is already
complete (all rows present from the earlier 8-condition run), so generate.py's row-level resumability
check would just skip everything, which is exactly what happened when this was tried directly.
Writing separately sidesteps that; join on "id" later if you want one merged table.

CONCEPTOR_TAG picks which alpha sweep point to use for psr_conceptor -- check the aggregated sweep
table (compare results/psr_conceptor_train_log_alpha*.json) and set it to whichever tag had the
lowest final dev MSE, e.g. CONCEPTOR_TAG=_alpha32. Defaults to "" (the alpha=4 default run).
"""
import argparse
import json
import os

import torch

from data_utils import DATA_DIR, RESULTS_DIR, append_jsonl, read_jsonl
from model_common import build_prompt, generate_response, load_model, steering_hook, token_count
from psr_conceptor_state import GateParams, PSRTrainedParams
from psr_conceptor_logic import make_psr_conceptor_hook

CONCEPTOR_TAG = os.environ.get("CONCEPTOR_TAG", "")


def load_conceptor_condition(device: str):
    ckpt = torch.load(RESULTS_DIR / f"psr_conceptor_probe{CONCEPTOR_TAG}.pt", map_location=device)
    params = GateParams(
        weight=ckpt["weight"].to(device),
        bias=ckpt["bias"].to(device),
        coeff_bias=ckpt["coeff_bias"].to(device),
    )
    direction = ckpt["direction"].to(device)
    return ckpt["layer"], params, direction


def load_proper_condition(device: str):
    ckpt = torch.load(RESULTS_DIR / "psr_proper_probe.pt", map_location=device)
    params = PSRTrainedParams(
        weight=ckpt["weight"].to(device),
        bias=ckpt["bias"].to(device),
        coeff_bias=ckpt["coeff_bias"].to(device),
        direction=ckpt["direction"].to(device),
    )
    return ckpt["layer"], params, params.direction


def already_done_ids(out_path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main(split: str) -> None:
    device = "cuda"
    model, tokenizer = load_model(device)
    rows = read_jsonl(DATA_DIR / f"{split}.jsonl")

    conceptor_layer, conceptor_params, conceptor_direction = load_conceptor_condition(device)
    proper_layer, proper_params, proper_direction = load_proper_condition(device)
    print(f"psr_conceptor: layer={conceptor_layer} tag='{CONCEPTOR_TAG}'  |  psr_proper: layer={proper_layer}")

    out_path = RESULTS_DIR / f"generations_{split}_new_conditions.jsonl"
    RESULTS_DIR.mkdir(exist_ok=True)

    done_ids = already_done_ids(out_path)
    if done_ids:
        print(f">>> Resuming: {len(done_ids)}/{len(rows)} rows already done in {out_path}, skipping those")

    n_processed = 0
    for row in rows:
        if row["id"] in done_ids:
            continue

        base_prompt = build_prompt(tokenizer, row["code"], terse=False)
        terse_prompt = build_prompt(tokenizer, row["code"], terse=True)

        with steering_hook(model, conceptor_layer, make_psr_conceptor_hook(conceptor_params, conceptor_direction)):
            psr_conceptor_resp = generate_response(model, tokenizer, base_prompt)
        with steering_hook(model, conceptor_layer, make_psr_conceptor_hook(conceptor_params, conceptor_direction)):
            prompt_psr_conceptor_resp = generate_response(model, tokenizer, terse_prompt)

        with steering_hook(model, proper_layer, make_psr_conceptor_hook(proper_params, proper_direction)):
            psr_proper_resp = generate_response(model, tokenizer, base_prompt)
        with steering_hook(model, proper_layer, make_psr_conceptor_hook(proper_params, proper_direction)):
            prompt_psr_proper_resp = generate_response(model, tokenizer, terse_prompt)

        out_row = {
            "id": row["id"],
            "psr_conceptor_response": psr_conceptor_resp,
            "prompt_psr_conceptor_response": prompt_psr_conceptor_resp,
            "psr_proper_response": psr_proper_resp,
            "prompt_psr_proper_response": prompt_psr_proper_resp,
            "psr_conceptor_tokens": token_count(tokenizer, psr_conceptor_resp),
            "prompt_psr_conceptor_tokens": token_count(tokenizer, prompt_psr_conceptor_resp),
            "psr_proper_tokens": token_count(tokenizer, psr_proper_resp),
            "prompt_psr_proper_tokens": token_count(tokenizer, prompt_psr_proper_resp),
        }
        append_jsonl(out_path, out_row)
        n_processed += 1
        if n_processed % 10 == 0:
            print(f"{len(done_ids) + n_processed}/{len(rows)} done this session")

    print(f"done. {out_path} now has {len(done_ids) + n_processed}/{len(rows)} rows")
    print(f"join with generations_{split}.jsonl on \"id\" for a full side-by-side comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()
    main(args.split)
