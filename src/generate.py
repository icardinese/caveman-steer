"""Generate all conditions (Base, Prompt, Const, Prompt+Const, PSR, Prompt+PSR, A-PSR, Prompt+A-PSR)
over a data split and save raw outputs.

Resumable by design: writes each row immediately after it's generated (not accumulated in memory
until the end), and on startup skips any row id already present in the output file. If this crashes
at row 150 of 180, re-running the exact same command picks up at row 150, not row 0 -- this stage is
the single longest one in the pipeline (~3.5-4 hrs for 180 rows x 8 conditions), so losing partial
progress here is the most expensive place in the whole workflow to get this wrong."""
import argparse
import json

import torch

from data_utils import DATA_DIR, RESULTS_DIR, append_jsonl, read_jsonl
from model_common import (
    MultiLayerPSRProbe,
    PSRProbe,
    build_prompt,
    generate_response,
    load_model,
    make_const_hook,
    make_multi_psr_hooks,
    make_psr_hook,
    multi_steering_hook,
    steering_hook,
    token_count,
)


def load_const_config(device: str):
    with (RESULTS_DIR / "const_steer_config.json").open() as f:
        const_config = json.load(f)
    layer_idx = const_config["layer"]
    coeff = const_config["coeff"]
    directions = torch.load(RESULTS_DIR / "const_steer_directions.pt")
    direction = directions[layer_idx]
    return layer_idx, direction, coeff


def load_psr_config(device: str):
    ckpt = torch.load(RESULTS_DIR / "psr_probe.pt", map_location=device)
    layer_idx = ckpt["layer"]
    probe = PSRProbe(ckpt["hidden_size"]).to(device)
    probe.load_state_dict(ckpt["probe_state"])
    probe.eval()
    directions = torch.load(RESULTS_DIR / "const_steer_directions.pt")
    direction = directions[layer_idx]
    return layer_idx, direction, probe


def load_a_psr_config(device: str):
    ckpt = torch.load(RESULTS_DIR / "a_psr_probe.pt", map_location=device)
    layer_indices = ckpt["layer_indices"]
    probe = MultiLayerPSRProbe(ckpt["hidden_size"], layer_indices).to(device)
    probe.load_state_dict(ckpt["probe_state"])
    probe.eval()
    directions = torch.load(RESULTS_DIR / "const_steer_directions.pt")
    directions = {l: directions[l] for l in layer_indices}
    return directions, probe


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
                # A partially-written last line from a crash mid-write -- ignore it, that row will
                # just get regenerated, which is exactly the safe behavior we want here.
                continue
    return done


def main(split: str) -> None:
    device = "cuda"
    model, tokenizer = load_model(device)
    rows = read_jsonl(DATA_DIR / f"{split}.jsonl")

    const_layer, const_direction, const_coeff = load_const_config(device)
    psr_layer, psr_direction, psr_probe = load_psr_config(device)
    a_psr_directions, a_psr_probe = load_a_psr_config(device)
    a_psr_hooks = make_multi_psr_hooks(a_psr_directions, a_psr_probe)

    out_path = RESULTS_DIR / f"generations_{split}.jsonl"
    RESULTS_DIR.mkdir(exist_ok=True)

    done_ids = already_done_ids(out_path)
    if done_ids:
        print(f">>> Resuming: {len(done_ids)}/{len(rows)} rows already done in {out_path}, skipping those")

    n_processed = 0
    for i, row in enumerate(rows):
        if row["id"] in done_ids:
            continue

        base_prompt = build_prompt(tokenizer, row["code"], terse=False)
        terse_prompt = build_prompt(tokenizer, row["code"], terse=True)

        base_resp = generate_response(model, tokenizer, base_prompt)
        prompt_resp = generate_response(model, tokenizer, terse_prompt)

        with steering_hook(model, const_layer, make_const_hook(const_direction, const_coeff)):
            const_resp = generate_response(model, tokenizer, base_prompt)
        with steering_hook(model, const_layer, make_const_hook(const_direction, const_coeff)):
            prompt_const_resp = generate_response(model, tokenizer, terse_prompt)

        with steering_hook(model, psr_layer, make_psr_hook(psr_direction, psr_probe)):
            psr_resp = generate_response(model, tokenizer, base_prompt)
        with steering_hook(model, psr_layer, make_psr_hook(psr_direction, psr_probe)):
            prompt_psr_resp = generate_response(model, tokenizer, terse_prompt)

        with multi_steering_hook(model, a_psr_hooks):
            a_psr_resp = generate_response(model, tokenizer, base_prompt)
        with multi_steering_hook(model, a_psr_hooks):
            prompt_a_psr_resp = generate_response(model, tokenizer, terse_prompt)

        out_row = {
            "id": row["id"],
            "code": row["code"],
            "reference_explanation": row["reference_explanation"],
            "base_response": base_resp,
            "prompt_response": prompt_resp,
            "const_response": const_resp,
            "prompt_const_response": prompt_const_resp,
            "psr_response": psr_resp,
            "prompt_psr_response": prompt_psr_resp,
            "a_psr_response": a_psr_resp,
            "prompt_a_psr_response": prompt_a_psr_resp,
            "base_tokens": token_count(tokenizer, base_resp),
            "prompt_tokens": token_count(tokenizer, prompt_resp),
            "const_tokens": token_count(tokenizer, const_resp),
            "prompt_const_tokens": token_count(tokenizer, prompt_const_resp),
            "psr_tokens": token_count(tokenizer, psr_resp),
            "prompt_psr_tokens": token_count(tokenizer, prompt_psr_resp),
            "a_psr_tokens": token_count(tokenizer, a_psr_resp),
            "prompt_a_psr_tokens": token_count(tokenizer, prompt_a_psr_resp),
        }
        append_jsonl(out_path, out_row)  # written and flushed immediately, not batched
        n_processed += 1
        if n_processed % 10 == 0:
            print(f"{len(done_ids) + n_processed}/{len(rows)} done this session")

    print(f"done. {out_path} now has {len(done_ids) + n_processed}/{len(rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()
    main(args.split)
