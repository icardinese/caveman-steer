"""Shared model loading, prompt building, and activation-hook utilities for Qwen2.5-Coder-7B-Instruct."""
from contextlib import contextmanager
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
LAYER_FRACTIONS = [0.25, 0.5, 0.65, 0.8]
MAX_NEW_TOKENS = 150

BASE_INSTRUCTION = "Explain what the following Python function does.\n\n```python\n{code}\n```"
# Verbatim from caveman's "full" mode (skills/caveman/SKILL.md), word for word, except Auto-Clarity
# (destructive-op handling), Boundaries (about commits/PRs), and the language-preservation paragraph —
# none of those scenarios exist in a single-shot, English-only code-explanation call.
CAVEMAN_SUFFIX = (
    "\n\nRespond terse like smart caveman. All technical substance stay. Only fluff die.\n\n"
    'ACTIVE EVERY RESPONSE. No revert after many turns. No filler drift. Still active if unsure. '
    'Off only: "stop caveman" / "normal mode".\n\n'
    "Drop: articles (a/an/the), filler (just/really/basically/actually/simply), pleasantries "
    "(sure/certainly/of course/happy to), hedging. Fragments OK. Short synonyms (big not extensive, "
    'fix not "implement a solution for"). No tool-call narration, no decorative tables/emoji, no '
    "dumping long raw error logs unless asked — quote shortest decisive line. Standard well-known "
    "tech acronyms OK (DB/API/HTTP); never invent new abbreviations (cfg/impl/req/res/fn) — tokenizer "
    "split them same as full word: zero token saved, reader still decode. Full word cheaper AND "
    "clearer. No causal arrows (→) either — own token, save nothing. Technical terms exact. Code "
    "blocks unchanged. Errors quoted exact.\n\n"
    'No self-reference. Never name or announce the style. No "caveman mode on", "me caveman think", '
    'no third-person caveman tags. Output caveman-only — never normal answer plus "Caveman:" recap.\n\n'
    "Pattern: `[thing] [action] [reason]. [next step].`"
)


def load_model(device: str = "cuda") -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    return model, tokenizer


def num_layers(model: PreTrainedModel) -> int:
    return len(model.model.layers)


def layer_indices_from_fractions(model: PreTrainedModel) -> list[int]:
    n = num_layers(model)
    return sorted({int(frac * n) for frac in LAYER_FRACTIONS})


def build_prompt(tokenizer: PreTrainedTokenizer, code: str, terse: bool) -> str:
    user_content = BASE_INSTRUCTION.format(code=code)
    if terse:
        user_content += CAVEMAN_SUFFIX
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def hidden_at_last_token_all_layers(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt_text: str, layer_indices: list[int]
) -> dict[int, torch.Tensor]:
    """Residual-stream activations at the final prompt token for several layers, from a single forward pass."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True)
    # hidden_states[0] is the embedding output; hidden_states[i] is the output of layer (i-1)
    return {layer_idx: out.hidden_states[layer_idx + 1][0, -1, :].float().cpu() for layer_idx in layer_indices}


@contextmanager
def steering_hook(model: PreTrainedModel, layer_idx: int, hook_fn: Callable[[torch.Tensor], torch.Tensor]):
    """Registers a forward hook on decoder layer `layer_idx` that rewrites its output hidden states."""
    layer = model.model.layers[layer_idx]

    def wrapped(module, inputs, output):
        hidden = hook_fn(output[0])
        return (hidden,) + tuple(output[1:])

    handle = layer.register_forward_hook(wrapped)
    try:
        yield
    finally:
        handle.remove()


def make_const_hook(direction: torch.Tensor, coeff: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
        return hidden + coeff * direction.to(hidden.dtype).to(hidden.device)

    return hook_fn


class PSRProbe(torch.nn.Module):
    """Single-layer ReLU probe producing a token-specific steering coefficient (S-PSR)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(hidden))


class MultiLayerPSRProbe(torch.nn.Module):
    """One PSRProbe per layer, trained jointly (A-PSR). ModuleDict keys are strings since PyTorch
    module dicts don't accept int keys directly -- layer_indices is kept separately for convenience."""

    def __init__(self, hidden_size: int, layer_indices: list[int]):
        super().__init__()
        self.layer_indices = layer_indices
        self.probes = torch.nn.ModuleDict({str(l): PSRProbe(hidden_size) for l in layer_indices})

    def forward(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        return self.probes[str(layer_idx)](hidden)


def make_psr_hook(direction: torch.Tensor, probe: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
        lam = probe(hidden.float()).to(hidden.dtype)  # (batch, seq, 1), token-specific coefficient
        return hidden + lam * direction.to(hidden.dtype).to(hidden.device)

    return hook_fn


@contextmanager
def multi_steering_hook(model: PreTrainedModel, hooks: dict[int, Callable[[torch.Tensor], torch.Tensor]]):
    """Registers a forward hook on several decoder layers at once (A-PSR / A-Const). Each layer's hook
    sees whatever activation actually arrives there at inference time -- which already reflects any
    correction applied by earlier layers, since hooks are chained through the real forward pass. This
    is where the "iteratively apply the intervention at all layers" behavior from the PSR paper actually
    happens; nothing extra needs to be done here to make that occur, it falls out of hooking multiple
    layers in the same forward pass."""
    handles = []
    for layer_idx, hook_fn in hooks.items():
        layer = model.model.layers[layer_idx]

        def _make_wrapped(fn):
            # Factory function is required here: without it, every hook in the loop would close over
            # the same loop variable `hook_fn` by reference (Python's late-binding closures), so all
            # layers would end up running whichever hook_fn was assigned LAST in the loop.
            def wrapped(module, inputs, output):
                hidden = fn(output[0])
                return (hidden,) + tuple(output[1:])

            return wrapped

        handles.append(layer.register_forward_hook(_make_wrapped(hook_fn)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def make_multi_psr_hooks(
    directions: dict[int, torch.Tensor], probe: MultiLayerPSRProbe
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    """Builds the per-layer hook dict for A-PSR, for use with multi_steering_hook."""
    hooks = {}
    for layer_idx, direction in directions.items():

        def _make_hook(l, d):
            def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
                lam = probe(l, hidden.float()).to(hidden.dtype)
                return hidden + lam * d.to(hidden.dtype).to(hidden.device)

            return hook_fn

        hooks[layer_idx] = _make_hook(layer_idx, direction)
    return hooks


def make_multi_const_hooks(
    directions: dict[int, torch.Tensor], coeff: float
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    """A-Const: the same fixed-coefficient additive steering as S-Const, applied at every layer at once.
    Not required by the paper's PSR comparison directly, but a natural, cheap companion baseline --
    S-Const already exists in this repo, this is its all-layer counterpart."""
    hooks = {}
    for layer_idx, direction in directions.items():

        def _make_hook(d):
            def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
                return hidden + coeff * d.to(hidden.dtype).to(hidden.device)

            return hook_fn

        hooks[layer_idx] = _make_hook(direction)
    return hooks


@torch.no_grad()
def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_text: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def token_count(tokenizer: PreTrainedTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])
