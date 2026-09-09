import math

import torch

from foerster2018.exact.value import exact_values
from foerster2018.games.matrix_game import IMP, IPD


def _probs(p_s0, p_cc, p_cd, p_dc, p_dd):
    return torch.tensor([p_s0, p_cc, p_cd, p_dc, p_dd], dtype=torch.float32)


def test_always_cooperate_gives_undiscounted_mutual_cooperation_payoff():
    """Degenerate, hand-computable case: both agents always play action 0
    (always Cooperate). The joint state is CC forever, so the average
    reward per step must equal IPD's mutual-cooperation payoff, -1
    (Table 1: pi(C,C) = (-1,-1)), regardless of gamma."""
    gamma = 0.9
    always_c = _probs(1.0, 1.0, 1.0, 1.0, 1.0)
    v1, v2 = exact_values(always_c, always_c, IPD, gamma)

    expected_discounted = -1.0 / (1 - gamma)
    assert math.isclose(v1.item(), expected_discounted, rel_tol=1e-6)
    assert math.isclose(v2.item(), expected_discounted, rel_tol=1e-6)

    avg_reward_per_step = (1 - gamma) * v1.item()
    assert math.isclose(avg_reward_per_step, -1.0, abs_tol=1e-5)


def test_always_defect_gives_mutual_defection_payoff():
    gamma = 0.9
    always_d = _probs(0.0, 0.0, 0.0, 0.0, 0.0)
    v1, v2 = exact_values(always_d, always_d, IPD, gamma)
    avg_reward_per_step = (1 - gamma) * v1.item()
    assert math.isclose(avg_reward_per_step, -2.0, abs_tol=1e-5)
    assert math.isclose((1 - gamma) * v2.item(), -2.0, abs_tol=1e-5)


def test_tit_for_tat_vs_always_cooperate_is_mutual_cooperation():
    """TFT (start C, then copy opponent's last move) against Always-Cooperate
    stays in CC forever, so this should also equal the mutual-cooperation
    payoff -- a second, independent hand-computable case."""
    gamma = 0.95
    tft = _probs(1.0, 1.0, 0.0, 1.0, 0.0)  # C at s0, C|CC, D|CD, C|DC, D|DD
    always_c = _probs(1.0, 1.0, 1.0, 1.0, 1.0)
    v1, v2 = exact_values(tft, always_c, IPD, gamma)
    assert math.isclose((1 - gamma) * v1.item(), -1.0, abs_tol=1e-4)
    assert math.isclose((1 - gamma) * v2.item(), -1.0, abs_tol=1e-4)


def test_matching_pennies_uniform_random_gives_zero_expected_value():
    """At p=0.5 for every state (including s0), the joint action is uniform
    over the 4 outcomes each round; matching pennies' payoff matrix (Table
    2) sums to zero over any row or column, so both agents' exact value
    must be exactly 0 regardless of gamma."""
    gamma = 0.9
    uniform = _probs(0.5, 0.5, 0.5, 0.5, 0.5)
    v1, v2 = exact_values(uniform, uniform, IMP, gamma)
    assert math.isclose(v1.item(), 0.0, abs_tol=1e-5)
    assert math.isclose(v2.item(), 0.0, abs_tol=1e-5)


def test_value_function_is_twice_differentiable():
    """LOLA's second-order term requires the value function to support
    `create_graph=True` double backward -- verify this doesn't raise and
    produces finite gradients."""
    gamma = 0.96
    theta1 = torch.zeros(5, requires_grad=True)
    theta2 = torch.zeros(5, requires_grad=True)
    v1, v2 = exact_values(torch.sigmoid(theta1), torch.sigmoid(theta2), IPD, gamma)

    (grad2_v1,) = torch.autograd.grad(v1, theta2, create_graph=True, retain_graph=True)
    (grad2_v2,) = torch.autograd.grad(v2, theta2, create_graph=True, retain_graph=True)
    cross = torch.dot(grad2_v1.detach(), grad2_v2)
    (second_order,) = torch.autograd.grad(cross, theta1)

    assert torch.isfinite(second_order).all()
    assert second_order.shape == (5,)
