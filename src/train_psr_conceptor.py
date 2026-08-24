"""Train the PSR-Conceptor gate: closed-form conceptor direction (no training) + gradient-trained
token-specific gate (location_fit + coeff_bias), matched via subsequent-layers MSE to prompt steering.

ALPHA / LR / REG_COEFF / OUT_TAG are overridable via environment variables so the pipeline script can
sweep alpha without editing this file -- each sweep point writes to its own OUT_TAG-suffixed filenames
instead of clobbering the previous run's output.
"""
import json
import os

import torch

from data_utils import DATA_DIR, RESULTS_DIR, read_jsonl
from model_common import load_model, num_layers
from psr_conceptor_state import compute_conceptor, conceptor_direction, init_gate_params
from psr_conceptor_logic import (
    collect_live_training_pair,
    collect_pooled_activations_for_conceptor,
    collect_target_hidden_states,
    forward_with_gate_hook,
    load_or_compute_responses,
    regularization_loss,
    subsequent_layers_mse,
)

N_EPOCHS = 3
LR = float(os.environ.get("PSR_CONCEPTOR_LR", 1e-3))
WEIGHT_DECAY = 1e-4
ALPHA = float(os.environ.get("PSR_CONCEPTOR_ALPHA", 4.0))
REG_COEFF = float(os.environ.get("PSR_CONCEPTOR_REG_COEFF", 0.1))
OUT_TAG = os.environ.get("PSR_CONCEPTOR_OUT_TAG", "")  # e.g. "_alpha4" for sweep runs, "" for the default


def load_or_compute_pooled(model, tokenizer, rows, layer_idx, responses, cache_path, device) -> torch.Tensor:
    """Pooled bipolar activations don't depend on alpha either (alpha only enters at the closed-form
    compute_conceptor step, on top of these). Cache once per layer_idx so the sweep's 2nd+ run skips
    the pooling forward passes too."""
    if cache_path.exists():
        print(f">>> loading cached pooled activations from {cache_path}")
        return torch.load(cache_path, map_location=device)
    pooled = collect_pooled_activations_for_conceptor(model, tokenizer, rows, layer_idx, responses).to(device)
    torch.save(pooled.cpu(), cache_path)
    return pooled


@torch.no_grad()
def eval_dev_mse(model, params, direction, layer_idx, n_layers, dev_rows, tokenizer, responses) -> float:
    losses = []
    for row in dev_rows:
        pair = collect_live_training_pair(model, tokenizer, row, layer_idx, responses)
        if pair is None:
            continue
        target_hidden = collect_target_hidden_states(model, pair["full_instr"])
        pred_hidden, _ = forward_with_gate_hook(model, params, direction, layer_idx, pair["full_base"], pair["n_resp"])
        mse = subsequent_layers_mse(pred_hidden, target_hidden, layer_idx, pair["n_resp"], n_layers)
        losses.append(mse.item())
    return sum(losses) / len(losses)


def main() -> None:
    final_probe_path = RESULTS_DIR / f"psr_conceptor_probe{OUT_TAG}.pt"
    if final_probe_path.exists() and os.environ.get("FORCE_RERUN", "0") != "1":
        print(f">>> {final_probe_path} already exists, skipping (set FORCE_RERUN=1 to redo)")
        return

    device = "cuda"
    RESULTS_DIR.mkdir(exist_ok=True)
    model, tokenizer = load_model(device)
    for p in model.parameters():
        p.requires_grad_(False)  # frozen base model -- without this, autograd allocates gradient
        # buffers for all 7B params during the live backward pass (~14GB extra on top of the weights
        # themselves), since the forward pass isn't wrapped in no_grad. On a 24GB card that's a real
        # OOM risk, not a hypothetical one.
    n_layers = num_layers(model)
    train_rows = read_jsonl(DATA_DIR / "train.jsonl")
    dev_rows = read_jsonl(DATA_DIR / "dev.jsonl")

    with (RESULTS_DIR / "const_steer_config.json").open() as f:
        const_config = json.load(f)
    layer_idx = const_config["layer"]
    base_directions = torch.load(RESULTS_DIR / "const_steer_directions.pt")
    base_direction = base_directions[layer_idx].to(device)

    print(f"loading/precomputing teacher-forced responses for {len(train_rows)} train + {len(dev_rows)} dev rows")
    train_responses = load_or_compute_responses(model, tokenizer, train_rows, RESULTS_DIR / "teacher_responses_train.json")
    dev_responses = load_or_compute_responses(model, tokenizer, dev_rows, RESULTS_DIR / "teacher_responses_dev.json")

    print(f"loading/collecting bipolar activations for conceptor at layer {layer_idx} from {len(train_rows)} rows")
    pooled = load_or_compute_pooled(
        model, tokenizer, train_rows, layer_idx, train_responses,
        RESULTS_DIR / f"pooled_activations_layer{layer_idx}.pt", device,
    )
    print(f"pooled {pooled.shape[0]} activations, computing conceptor (alpha={ALPHA})")
    conceptor = compute_conceptor(pooled, alpha=ALPHA)
    direction = conceptor_direction(conceptor, base_direction)

    params = init_gate_params(hidden_size=pooled.shape[1], device=device)
    optimizer = torch.optim.Adam(params.as_list(), lr=LR, weight_decay=WEIGHT_DECAY)

    print("baseline dev MSE (gate untrained, correction ~0)")
    baseline_mse = eval_dev_mse(model, params, direction, layer_idx, n_layers, dev_rows, tokenizer, dev_responses)
    print(f"baseline_dev_mse={baseline_mse:.4f}")

    def save_checkpoint(epoch: int | None, dev_mse: float | None) -> None:
        """epoch=None means this is the final save. Overwrites the SAME path every call (not
        per-epoch filenames) -- worst case on a disconnect is losing the current in-progress epoch,
        not the ones already completed, and there's never more than one checkpoint file to manage."""
        torch.save(
            {
                "weight": params.weight.detach().cpu(),
                "bias": params.bias.detach().cpu(),
                "coeff_bias": params.coeff_bias.detach().cpu(),
                "conceptor": conceptor.detach().cpu(),
                "direction": direction.detach().cpu(),
                "layer": layer_idx,
                "alpha": ALPHA,
                "completed_epochs": epoch if epoch is not None else N_EPOCHS,
            },
            final_probe_path,
        )
        with (RESULTS_DIR / f"psr_conceptor_train_log{OUT_TAG}.json").open("w") as f:
            json.dump(
                {
                    "layer": layer_idx, "alpha": ALPHA, "lr": LR, "reg_coeff": REG_COEFF,
                    "baseline_dev_mse": baseline_mse,
                    "completed_epochs": epoch if epoch is not None else N_EPOCHS,
                    "dev_mse_after_last_completed_epoch": dev_mse,
                },
                f, indent=2,
            )

    for epoch in range(N_EPOCHS):
        epoch_losses = []
        for row in train_rows:
            pair = collect_live_training_pair(model, tokenizer, row, layer_idx, train_responses)
            if pair is None:
                continue
            target_hidden = collect_target_hidden_states(model, pair["full_instr"])
            pred_hidden, fit_vals = forward_with_gate_hook(model, params, direction, layer_idx, pair["full_base"], pair["n_resp"])
            mse = subsequent_layers_mse(pred_hidden, target_hidden, layer_idx, pair["n_resp"], n_layers)
            reg = regularization_loss(fit_vals, REG_COEFF)
            loss = mse + reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"epoch={epoch} train_loss={avg_loss:.4f}")
        epoch_dev_mse = eval_dev_mse(model, params, direction, layer_idx, n_layers, dev_rows, tokenizer, dev_responses)
        print(f"epoch={epoch} dev_mse={epoch_dev_mse:.4f} -- checkpointing")
        save_checkpoint(epoch=epoch, dev_mse=epoch_dev_mse)

    print("final dev MSE (gate trained)")
    final_mse = eval_dev_mse(model, params, direction, layer_idx, n_layers, dev_rows, tokenizer, dev_responses)
    print(f"final_dev_mse={final_mse:.4f} (baseline was {baseline_mse:.4f})")
    save_checkpoint(epoch=None, dev_mse=final_mse)
    print(f"wrote {final_probe_path} and psr_conceptor_train_log{OUT_TAG}.json")


if __name__ == "__main__":
    main()
