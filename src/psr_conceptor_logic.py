"""Logic: free functions only. Every function here takes State (psr_conceptor_state.py) and/or Data
(rows, tokenizer, model) as arguments and returns a tensor -- no function reads or mutates state that
wasn't passed in explicitly, and nothing here is a class method.

Fixes the three fidelity gaps from the earlier attempt, now grounded against Nokia's actual
focused_steering.py + steering_base.py instead of inferred from the paper text alone:
  1. MSE over layer_idx AND all subsequent layers (not just the injection layer).
  2. Regularization: relu(1 - sum(location_fit)).mean() -- penalizes the gate for firing nowhere,
     matching FocusedSteeringModule's actual reg term.
  3. Direction is closed-form (conceptor-projected), so only the gate is gradient-trained -- resolves
     the earlier "warm-start vs from-scratch" dilemma without needing 200 epochs.

Steering location is answer_only ("R"), consistent with the earlier decision: CAVEMAN_SUFFIX is
appended (not prepended) to the base instruction, so base_prompt is a literal prefix of terse_prompt
and a contiguous question_and_answer alignment isn't correct without extra positional bookkeeping
that's out of scope for this pass.
"""
import torch
import torch.nn.functional as F

from model_common import build_prompt, generate_response
from psr_conceptor_state import GateParams


def location_fit(params: GateParams, hidden: torch.Tensor) -> torch.Tensor:
    """ReLU(hidden @ weight + bias). hidden: (..., d) -> (..., 1). Pure function of state + data."""
    return torch.relu(hidden @ params.weight + params.bias)


def answer_only_mask(seq_len: int, n_resp: int, device) -> torch.Tensor:
    """True for the last n_resp positions (the response span), False elsewhere. Shape (1, seq_len, 1),
    broadcastable against hidden/fit tensors. This is the hard mask Nokia's compute_steering_mask
    applies for steering_location="answer_only" -- without it, the gate can (and in early testing,
    will) leak nonzero correction onto prompt tokens, which the PSR paper's whole premise argues
    against ("strong intervention on some tokens, none on others")."""
    positions = torch.arange(seq_len, device=device)
    return (positions >= (seq_len - n_resp)).view(1, seq_len, 1)


def gate_correction(
    params: GateParams, direction: torch.Tensor, hidden: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (correction, location_fit_vals). correction has the same shape as hidden.
    desired_presence is (1 + coeff_bias): user-steering-coeff is always "on" here (caveman-steer has
    no continuous user dial), so the bias is the only learned offset, matching FocusedSteeringModule's
    `desired_concept_presence = user_steering_coeffs + steering_coeff_bias` with user_steering_coeffs=1.

    If mask is given, location_fit is hard-zeroed outside it BEFORE multiplying by desired_presence --
    matches Nokia zeroing location_fit via steering_mask, rather than relying on the gate to learn
    zero on its own."""
    fit = location_fit(params, hidden.float())
    if mask is not None:
        fit = torch.where(mask, fit, torch.zeros_like(fit))
    desired_presence = 1.0 + params.coeff_bias
    coeff = desired_presence * fit
    correction = coeff * direction.to(hidden.dtype)
    return correction.to(hidden.dtype), fit


def regularization_loss(fit_vals: torch.Tensor, reg_coeff: float) -> torch.Tensor:
    """fit_vals: (batch, seq, 1). Penalizes sequences where location_fit sums to under 1 -- i.e.
    the gate deciding to steer nowhere. This IS the dead-ReLU guard; it's specifically about the
    gate collapsing to all-zero, not a generic weight penalty."""
    fit_sum = fit_vals.sum(dim=1)  # (batch, 1)
    return reg_coeff * torch.relu(1.0 - fit_sum).mean()


def precompute_responses(model, tokenizer, rows: list[dict]) -> dict:
    """Generates the teacher-forcing target response ONCE per row and caches it, keyed by row['id'].
    Deterministic (do_sample=False in generate_response), so regenerating it per-epoch and per-pooling-
    pass is pure waste -- this is the single biggest cost in the whole script (150-token autoregressive
    generation per row), and it was being paid 4x (once for pooling + once per of 3 epochs) before this
    cache existed."""
    responses = {}
    for row in rows:
        terse_prompt = build_prompt(tokenizer, row["code"], terse=True)
        responses[row["id"]] = generate_response(model, tokenizer, terse_prompt)
    return responses


def load_or_compute_responses(model, tokenizer, rows: list[dict], cache_path) -> dict:
    """Disk-cached wrapper around precompute_responses. Shared between train_psr_conceptor.py and
    train_psr_proper.py -- responses don't depend on alpha, direction training, or which of the two
    scripts is calling this, so both can hit the same cache file and neither pays for generation twice."""
    import json
    if cache_path.exists():
        print(f">>> loading cached responses from {cache_path}")
        with cache_path.open() as f:
            return json.load(f)
    responses = precompute_responses(model, tokenizer, rows)
    with cache_path.open("w") as f:
        json.dump(responses, f)
    return responses


@torch.no_grad()
def collect_pooled_activations_for_conceptor(
    model, tokenizer, rows: list[dict], layer_idx: int, responses: dict
) -> torch.Tensor:
    """Bipolar pool: response-token activations from BOTH the base-prompt and terse-prompt forward
    passes, concatenated. This is what compute_conceptor's R gets built from -- matches the paper's
    "train each conceptor on the union of both poles" choice, rather than one-sided. `responses` is
    the precompute_responses() cache -- no generation happens in here anymore."""
    pooled = []
    for row in rows:
        base_prompt = build_prompt(tokenizer, row["code"], terse=False)
        terse_prompt = build_prompt(tokenizer, row["code"], terse=True)
        teacher_response = responses[row["id"]]
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
        pooled.append(out_base.hidden_states[layer_idx + 1][0, -n_resp:, :].float().cpu())
        pooled.append(out_instr.hidden_states[layer_idx + 1][0, -n_resp:, :].float().cpu())

    return torch.cat(pooled, dim=0)


def collect_live_training_pair(
    model, tokenizer, row: dict, layer_idx: int, responses: dict
) -> dict | None:
    """Per-example tokenized tensors for one training step. Returns None for empty-response rows
    (skip, don't pad -- batch_size=1 in the reference training loop too, so this is fine to keep
    simple rather than building a padded-batch collator). `responses` is the precompute_responses()
    cache -- no generation happens in here anymore, this just tokenizes the cached string."""
    base_prompt = build_prompt(tokenizer, row["code"], terse=False)
    terse_prompt = build_prompt(tokenizer, row["code"], terse=True)
    teacher_response = responses[row["id"]]
    resp_ids = tokenizer(teacher_response, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    if resp_ids.shape[1] == 0:
        return None

    base_ids = tokenizer(base_prompt, return_tensors="pt")["input_ids"].to(model.device)
    instr_ids = tokenizer(terse_prompt, return_tensors="pt")["input_ids"].to(model.device)
    full_base = torch.cat([base_ids, resp_ids], dim=1)
    full_instr = torch.cat([instr_ids, resp_ids], dim=1)
    n_resp = resp_ids.shape[1]
    return {"full_base": full_base, "full_instr": full_instr, "n_resp": n_resp}


def forward_with_gate_hook(model, params: GateParams, direction: torch.Tensor, layer_idx: int, input_ids: torch.Tensor, n_resp: int):
    """Live, gradient-tracked forward pass through the frozen model with the gate hook active at
    layer_idx. Returns (hidden_states tuple, fit_vals for the injected layer, already masked to the
    response span). n_resp is required now -- the hook needs it to build the answer_only mask, it's
    no longer optional slicing done after the fact by the caller.

    This is the expensive part flagged in the earlier attempt: every training step needs a real
    forward pass, hooks active, gradients flowing -- can't be precomputed since the correction
    changes as the gate trains."""
    layer = model.model.layers[layer_idx]
    captured_fit = {}

    def wrapped(module, inputs, output):
        hidden = output[0]
        mask = answer_only_mask(hidden.shape[1], n_resp, hidden.device)
        correction, fit = gate_correction(params, direction, hidden, mask=mask)
        captured_fit["fit"] = fit
        return (hidden + correction,) + tuple(output[1:])

    handle = layer.register_forward_hook(wrapped)
    try:
        out = model(input_ids=input_ids, output_hidden_states=True)
    finally:
        handle.remove()
    return out.hidden_states, captured_fit["fit"]


def subsequent_layers_mse(
    hidden_pred: tuple[torch.Tensor, ...],
    hidden_target: tuple[torch.Tensor, ...],
    layer_idx: int,
    n_resp: int,
    n_layers: int,
) -> torch.Tensor:
    """Sum of MSE at layer_idx and every layer after it, evaluated on the response-token span only.
    hidden_states[i] is the output of layer (i-1), so layer_idx's output lives at index layer_idx+1;
    "layer l and all subsequent layers" therefore means indices layer_idx+1 .. n_layers (inclusive of
    the final layer's output, index n_layers)."""
    total = torch.tensor(0.0, device=hidden_pred[0].device)
    for idx in range(layer_idx + 1, n_layers + 1):
        pred = hidden_pred[idx][0, -n_resp:, :].float()
        target = hidden_target[idx][0, -n_resp:, :].float()
        total = total + F.mse_loss(pred, target)
    return total


@torch.no_grad()
def collect_target_hidden_states(model, full_instr: torch.Tensor):
    """No-grad, precomputed once per example: the prompt-steered (terse) side's hidden states.
    Cheap and unchanged in spirit from the earlier attempt -- only the steered side needs live grad."""
    return model(input_ids=full_instr, output_hidden_states=True).hidden_states


def make_psr_conceptor_hook(params: GateParams, direction: torch.Tensor):
    """Inference-time hook, same contract as model_common.make_psr_hook -- drop-in for
    model_common.steering_hook(model, layer_idx, make_psr_conceptor_hook(params, direction)).

    HF's model.generate() calls this hook once on the full-prompt prefill pass (seq_len > 1, all
    prompt tokens, none of it response yet) and then once per new token after that (seq_len == 1,
    KV-cached, and that single token IS the response by construction). Mirrors Nokia's
    is_generating branch: no masking is needed at seq_len==1 since there's nothing else in the
    forward call to mask against, but the prefill call must be fully suppressed or the correction
    leaks onto the prompt tokens exactly like the training-time bug this was patched to fix."""
    @torch.no_grad()
    def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] > 1:
            return hidden  # prefill pass over the prompt -- answer_only means don't steer here
        correction, _ = gate_correction(params, direction, hidden)  # single generated token: always response
        return hidden + correction
    return hook_fn


def train_step(
    model, params: GateParams, direction: torch.Tensor, layer_idx: int, n_layers: int,
    pair: dict, reg_coeff: float,
) -> torch.Tensor:
    """One example's loss: subsequent-layers MSE (base+gate vs terse-prompt target) + regularization.
    Caller is responsible for optimizer.zero_grad() / loss.backward() / optimizer.step()."""
    target_hidden = collect_target_hidden_states(model, pair["full_instr"])
    pred_hidden, fit_vals = forward_with_gate_hook(model, params, direction, layer_idx, pair["full_base"], pair["n_resp"])
    mse = subsequent_layers_mse(pred_hidden, target_hidden, layer_idx, pair["n_resp"], n_layers)
    # fit_vals is already zeroed outside the response span (answer_only_mask), so summing the full
    # sequence here is equivalent to summing just the response span -- no manual slicing needed.
    reg = regularization_loss(fit_vals, reg_coeff)
    return mse + reg
