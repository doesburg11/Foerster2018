"""Naive-Learner (NL) and LOLA update rules using exact value gradients,
per Foerster et al. (2018) Sec. 4.1 "Naive Learner" and Sec. 4.2 "Learning
with Opponent Learning Awareness".

Naive Learner (Eq. 4.1):

    theta1_{i+1} = theta1_i + f_nl^1(theta1_i, theta2_i)
    f_nl^1(theta1, theta2) = grad_{theta1} V1(theta1, theta2) . delta

LOLA (Eq. 4.2-4.4): a LOLA agent optimises its return under a one-step
look-ahead of the opponent's naive learning step. Substituting the
opponent's naive update (Eq. 4.3, `delta_theta2 = grad_{theta2} V2 . eta`)
into a first-order Taylor expansion of `V1(theta1, theta2 + delta_theta2)`
and differentiating w.r.t. theta1 gives the LOLA update (Eq. 4.4):

    f_lola^1(theta1, theta2) =
        grad_{theta1} V1(theta1, theta2) . delta
        + (grad_{theta2} V1(theta1, theta2))^T
          grad_{theta1} grad_{theta2} V2(theta1, theta2) . delta . eta

Critically, the paper is explicit that "the dependency of grad_{theta2}
V1(theta1, theta2) on theta1 is dropped during the backward pass" -- i.e.
`grad_{theta2} V1` is treated as a *constant* vector when the second-order
correction term is differentiated w.r.t. theta1. This is a deliberate
simplification stated in the paper, not an implementation shortcut taken
here, and it means the code below must explicitly `.detach()` that vector
before the outer `torch.autograd.grad` call -- if that vector's dependency
on theta1 were kept, the outer gradient would also include the (paper-
dropped) term `(grad_{theta1} grad_{theta2} V1)^T grad_{theta2} V2`.

This is the second-order piece that makes LOLA LOLA: `grad_{theta1}
grad_{theta2} V2` differentiates *through the opponent's own gradient
(Eq. 4.3)*, not just through V1 itself. Implemented via the standard
"differentiate a dot product of two first-order grads" trick, which is
mathematically equivalent to (and avoids ever materialising) the full
mixed-Hessian tensor:

    d/d theta1 [ dot(grad_{theta2} V1 (detached), grad_{theta2} V2) ]
        = sum_j grad_{theta2} V1 (detached)[j] * d/d theta1 (grad_{theta2} V2)[j]
        = (grad_{theta2} V1)^T grad_{theta1} grad_{theta2} V2
"""
from typing import Tuple

import torch


def _grad(output: torch.Tensor, inputs: torch.Tensor, create_graph: bool) -> torch.Tensor:
    # `v1` and `v2` (see foerster2018.exact.value.exact_values) are produced
    # by ONE shared torch.linalg.solve call (columns of the same solve), and
    # callers below backward through v1 and/or v2 more than once (once per
    # agent's own gradient, again for the LOLA cross term). Without
    # `retain_graph=True`, the first such backward call frees the shared
    # forward graph's saved buffers and every subsequent call raises
    # "Trying to backward through the graph a second time" (or, worse if
    # buffers happened to still look valid, silently uses stale state) --
    # so retain_graph=True is required here, not optional, independent of
    # whether `create_graph` (which controls whether *this* grad computation
    # is itself differentiable) is True or False.
    (g,) = torch.autograd.grad(output, inputs, create_graph=create_graph, retain_graph=True)
    return g


def naive_update(v1: torch.Tensor, theta1: torch.Tensor, delta: float) -> torch.Tensor:
    """Eq. 4.1: plain gradient ascent step on agent 1's own exact value."""
    grad1_v1 = _grad(v1, theta1, create_graph=False)
    return delta * grad1_v1


def lola_correction(
    v1: torch.Tensor, v2: torch.Tensor, theta1: torch.Tensor, theta2: torch.Tensor
) -> torch.Tensor:
    """The second-order term of Eq. 4.4:
    (grad_{theta2} V1)^T grad_{theta1} grad_{theta2} V2, computed for agent 1
    (i.e. this is agent 1's correction, modelling agent 2 as the naive
    learner being shaped)."""
    grad2_v1 = _grad(v1, theta2, create_graph=True)  # depends on theta1 too
    grad2_v2 = _grad(v2, theta2, create_graph=True)  # depends on theta1 too

    # Paper: "the dependency of grad_theta2 V1(theta1, theta2) on theta1 is
    # dropped during the backward pass" -- detach it before differentiating.
    grad2_v1_frozen = grad2_v1.detach()

    cross = torch.dot(grad2_v1_frozen, grad2_v2)
    correction = _grad(cross, theta1, create_graph=False)
    return correction


def lola_update(
    v1: torch.Tensor,
    v2: torch.Tensor,
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    delta: float,
    eta: float,
) -> torch.Tensor:
    """Eq. 4.4: agent 1's LOLA update against agent 2."""
    correction = lola_correction(v1, v2, theta1, theta2)
    grad1_v1 = _grad(v1, theta1, create_graph=False)
    return delta * grad1_v1 + delta * eta * correction


def naive_vs_naive_step(
    v1: torch.Tensor, v2: torch.Tensor, theta1: torch.Tensor, theta2: torch.Tensor, delta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience: simultaneous NL updates for both agents from the same
    (v1, v2) computed at the current (theta1, theta2)."""
    return naive_update(v1, theta1, delta), naive_update(v2, theta2, delta)
