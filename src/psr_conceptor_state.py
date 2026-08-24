"""State: plain tensors and dataclasses-of-tensors only. No methods, no nn.Module.

Two kinds of state live here:
  1. The conceptor C -- closed-form from activation statistics, never gradient-trained.
  2. The PSR gate params -- the ONLY thing we gradient-train: a token-specific scalar
     that decides how much of (C @ base_direction) to inject at each position.

Keeping these as plain tensors (not nn.Module) means the optimizer is just
`torch.optim.Adam(list(gate_params.values()))` -- no .parameters() indirection needed.
"""
from dataclasses import dataclass

import torch


@dataclass
class GateParams:
    """Trainable token-gating state (replaces PSRProbe). All leaf tensors, requires_grad=True."""
    weight: torch.Tensor       # (hidden_size, 1) -- projects hidden state to a raw gate logit
    bias: torch.Tensor         # (1,) location-fit bias
    coeff_bias: torch.Tensor   # (1,) b_m in the PSR paper -- learned additive offset on the user coeff

    def as_list(self) -> list[torch.Tensor]:
        return [self.weight, self.bias, self.coeff_bias]


@dataclass
class PSRTrainedParams:
    """Same gate as GateParams, plus a jointly-trained direction -- this is what makes it
    "paper-faithful": Nokia's z_attr (steering_proj.weight) is trained end-to-end from default init,
    not conceptor-projected or warm-started from anything. Kept as its own dataclass rather than
    inheriting from GateParams -- flat and explicit beats a one-off subclass for four fields."""
    weight: torch.Tensor       # (hidden_size, 1) location-fit projection
    bias: torch.Tensor         # (1,) location-fit bias
    coeff_bias: torch.Tensor   # (1,) b_m
    direction: torch.Tensor    # (hidden_size,) z_attr -- trainable, unlike the conceptor variant

    def as_list(self) -> list[torch.Tensor]:
        return [self.weight, self.bias, self.coeff_bias, self.direction]


def init_psr_trained_params(hidden_size: int, device: str, dtype: torch.dtype = torch.float32) -> PSRTrainedParams:
    """Matches Nokia's actual init: steering_proj is a plain nn.Linear(d, 1, bias=False) at its
    default init, i.e. direction is NOT warm-started from the Const mean-diff vector or anything
    else -- that warm-start was last session's flagged deviation; this variant removes it entirely."""
    weight = (torch.randn(hidden_size, 1, device=device, dtype=dtype) * 0.01).requires_grad_(True)
    bias = torch.zeros(1, device=device, dtype=dtype).requires_grad_(True)
    coeff_bias = torch.zeros(1, device=device, dtype=dtype).requires_grad_(True)
    # nn.Linear(d, 1, bias=False) default init is Kaiming-uniform with bound 1/sqrt(d) -- replicate
    # that rather than an arbitrary small-normal init, since this IS the one variable this script
    # exists to test the effect of training properly.
    bound = 1.0 / (hidden_size ** 0.5)
    direction = (torch.empty(hidden_size, device=device, dtype=dtype).uniform_(-bound, bound)).requires_grad_(True)
    return PSRTrainedParams(weight=weight, bias=bias, coeff_bias=coeff_bias, direction=direction)


def init_gate_params(hidden_size: int, device: str, dtype: torch.dtype = torch.float32) -> GateParams:
    # Small random init for weight (not zero -- zero gives zero gradient everywhere through the
    # ReLU at init, which is the literal dead-ReLU failure mode we're trying to avoid downstream).
    weight = (torch.randn(hidden_size, 1, device=device, dtype=dtype) * 0.01).requires_grad_(True)
    bias = torch.zeros(1, device=device, dtype=dtype).requires_grad_(True)
    coeff_bias = torch.zeros(1, device=device, dtype=dtype).requires_grad_(True)
    return GateParams(weight=weight, bias=bias, coeff_bias=coeff_bias)


def compute_conceptor(activations: torch.Tensor, alpha: float) -> torch.Tensor:
    """Closed-form conceptor matrix from pooled bipolar activations (Jaeger; Triantafyllopoulos et al. 2026).

    activations: (N, d) -- rows pooled from BOTH poles (e.g. base-prompt and terse-prompt response-token
    activations at the target layer), per the paper's bipolar training choice.
    Returns C = R (R + alpha^-2 I)^-1, a (d, d) matrix. This is linear algebra, not an optimization --
    no epochs, no optimizer, computed once per layer.
    """
    n, d = activations.shape
    r = (activations.T @ activations) / n  # correlation matrix, (d, d)
    identity = torch.eye(d, device=activations.device, dtype=activations.dtype)
    c = r @ torch.linalg.inv(r + (alpha ** -2) * identity)
    return c


def conceptor_direction(conceptor: torch.Tensor, base_direction: torch.Tensor) -> torch.Tensor:
    """Projects a rank-1 base direction (e.g. the existing Const mean-difference vector) through the
    conceptor's subspace. This is the "direction" half of the PSR correction -- closed-form, not trained.
    Renormalized to unit norm so the trained gate's scale has a consistent meaning across layers/alphas."""
    v = conceptor @ base_direction
    return v / v.norm()
