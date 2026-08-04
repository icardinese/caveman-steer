"""Train the A-PSR probe: per-layer, token-specific steering coefficients matched (via summed MSE)
to the effect of prompting, applied jointly and simultaneously at every candidate layer -- the
all-layer generalization of steering_psr.py's single-layer S-PSR."""
import json

import torch
import torch.nn.functional as F

from data_utils import DATA_DIR, RESULTS_DIR, read_jsonl
from model_common import MultiLayerPSRProbe, build_prompt, generate_response, load_model

N_EPOCHS = 200
LR = 1e-3


@torch.no_grad()
def collect_teacher_forced_pairs_multilayer(
    model, tokenizer, rows: list[dict], layer_indices: list[int]
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Same teacher-forcing procedure as steering_psr.py's collector, but pulls every candidate layer's
    activations from the SAME pair of forward passes -- output_hidden_states=True already returns every
    layer in one call, so collecting all layer_indices here costs no extra generation or forward-pass
    time over the single-layer version."""
    xs: dict[int, list[torch.Tensor]] = {l: [] for l in layer_indices}
    ys: dict[int, list[torch.Tensor]] = {l: [] for l in layer_indices}

    for row in rows:
        base_prompt = build_prompt(tokenizer, row["code"], terse=False)
        terse_prompt = build_prompt(tokenizer, row["code"], terse=True)
        teacher_response = generate_response(model, tokenizer, terse_prompt)
        resp_ids = tokenizer(teacher_response, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        if resp_ids.shape[1] == 0:
            continue

        base_ids = tokenizer(base_prompt, return_tensors="pt")["input_ids"].to(model.device)
        instr_ids = tokenizer(terse_prompt, return_tensors="pt")["input_ids"].to(model.device)
        full_base = torch.cat([base_ids, resp_ids], dim=1)
        full_instr = torch.cat([instr_ids, resp_ids], dim=1)

        out_base = model(input_ids=full_base, output_hidden_states=True)
        out_instr = model(input_ids=full_instr, output_hidden_states=True)
        n_resp = resp_ids.shape[1]

        for l in layer_indices:
            h_base = out_base.hidden_states[l + 1][0, -n_resp:, :].float().cpu()
            h_instr = out_instr.hidden_states[l + 1][0, -n_resp:, :].float().cpu()
            xs[l].append(h_base)
            ys[l].append(h_instr)

    return {l: (torch.cat(xs[l], dim=0), torch.cat(ys[l], dim=0)) for l in layer_indices}


def train_multilayer_probe(
    pairs: dict[int, tuple[torch.Tensor, torch.Tensor]],
    directions: dict[int, torch.Tensor],
    device: str,
) -> MultiLayerPSRProbe:
    """Joint training: one optimizer, one combined loss (summed across layers), one backward pass per
    epoch. NOTE: training itself stays on offline precomputed pairs, same as S-PSR -- it does NOT run
    the live multi-layer hooks during training. The "iteratively applied at all layers" behavior from
    the paper happens at INFERENCE time (see multi_steering_hook), where each layer's correction
    naturally propagates into what later layers see. This is a deliberate simplification matching how
    S-PSR is already structured in this repo; worth revisiting against the paper's exact training
    procedure if results look off, since I couldn't fully confirm from the fetched section whether
    their training loop also requires live cross-layer feedback."""
    layer_indices = list(pairs.keys())
    hidden_size = pairs[layer_indices[0]][0].shape[1]
    probe = MultiLayerPSRProbe(hidden_size, layer_indices).to(device)
    data = {l: (x.to(device), y.to(device), directions[l].to(device)) for l, (x, y) in pairs.items()}
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        per_layer_loss = {}
        total_loss = torch.tensor(0.0, device=device)
        for l, (x, y, d) in data.items():
            lam = probe(l, x)
            pred = x + lam * d
            loss_l = F.mse_loss(pred, y)
            per_layer_loss[l] = loss_l
            total_loss = total_loss + loss_l
        total_loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
            breakdown = " ".join(f"L{l}={v.item():.4f}" for l, v in per_layer_loss.items())
            print(f"epoch={epoch} train_mse_sum={total_loss.item():.4f} ({breakdown})")
    return probe


@torch.no_grad()
def eval_multilayer_probe(
    probe: MultiLayerPSRProbe,
    pairs: dict[int, tuple[torch.Tensor, torch.Tensor]],
    directions: dict[int, torch.Tensor],
    device: str,
) -> dict[int, float]:
    per_layer_mse = {}
    for l, (x, y) in pairs.items():
        x, y, d = x.to(device), y.to(device), directions[l].to(device)
        lam = probe(l, x)
        pred = x + lam * d
        per_layer_mse[l] = F.mse_loss(pred, y).item()
    return per_layer_mse


def main() -> None:
    device = "cuda"
    model, tokenizer = load_model(device)
    train_rows = read_jsonl(DATA_DIR / "train.jsonl")
    dev_rows = read_jsonl(DATA_DIR / "dev.jsonl")

    # Reuses the directions steering_const.py already computed and saved for ALL candidate layers --
    # no need to recompute anything, this file already has every layer's direction, not just whichever
    # one calibration ultimately picked as best for S-Const.
    directions = torch.load(RESULTS_DIR / "const_steer_directions.pt")
    layer_indices = list(directions.keys())
    print(f"training A-PSR jointly at layers {layer_indices}")

    print(f"collecting teacher-forced activation pairs from {len(train_rows)} train rows")
    train_pairs = collect_teacher_forced_pairs_multilayer(model, tokenizer, train_rows, layer_indices)
    for l, (x, _) in train_pairs.items():
        print(f"  layer {l}: {x.shape[0]} token-level pairs")

    print(f"collecting dev pairs from {len(dev_rows)} rows")
    dev_pairs = collect_teacher_forced_pairs_multilayer(model, tokenizer, dev_rows, layer_indices)

    baseline_mse = {l: F.mse_loss(x, y).item() for l, (x, y) in dev_pairs.items()}
    print(f"dev baseline MSE per layer (no intervention): {baseline_mse}")

    probe = train_multilayer_probe(train_pairs, directions, device)
    dev_mse = eval_multilayer_probe(probe, dev_pairs, directions, device)
    print(f"dev MSE per layer after A-PSR probe: {dev_mse}")

    RESULTS_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "probe_state": probe.state_dict(),
            "layer_indices": layer_indices,
            "hidden_size": train_pairs[layer_indices[0]][0].shape[1],
        },
        RESULTS_DIR / "a_psr_probe.pt",
    )
    with (RESULTS_DIR / "a_psr_train_log.json").open("w") as f:
        json.dump(
            {
                "layer_indices": layer_indices,
                "baseline_dev_mse": baseline_mse,
                "final_dev_mse": dev_mse,
                "n_train_pairs": {l: train_pairs[l][0].shape[0] for l in layer_indices},
            },
            f,
            indent=2,
        )
    print("wrote results/a_psr_probe.pt and results/a_psr_train_log.json")


if __name__ == "__main__":
    main()
