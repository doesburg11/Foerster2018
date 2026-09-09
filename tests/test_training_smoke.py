"""End-to-end smoke tests: a short run of each training path completes and
produces finite numbers (not a claim of convergence -- see the
experiment scripts and README/RESULTS.md for that)."""
import torch

from foerster2018.exact.lola import lola_update, naive_update
from foerster2018.exact.value import exact_values
from foerster2018.games.matrix_game import IMP, IPD
from foerster2018.policy_gradient.lola_pg import lola_pg_update, naive_pg_update
from foerster2018.policy_gradient.rollout import sample_episodes


def test_exact_ipd_all_four_pairings_finite_after_short_run():
    gamma = 0.96
    for agent1_type in ("NL", "LOLA"):
        for agent2_type in ("NL", "LOLA"):
            torch.manual_seed(0)
            theta1 = ((torch.rand(5) - 0.5) * 0.2).requires_grad_(True)
            theta2 = ((torch.rand(5) - 0.5) * 0.2).requires_grad_(True)
            for _ in range(10):
                v1, v2 = exact_values(torch.sigmoid(theta1), torch.sigmoid(theta2), IPD, gamma)
                u1 = (
                    lola_update(v1, v2, theta1, theta2, 1.0, 1.0)
                    if agent1_type == "LOLA"
                    else naive_update(v1, theta1, 1.0)
                )
                u2 = (
                    lola_update(v2, v1, theta2, theta1, 1.0, 1.0)
                    if agent2_type == "LOLA"
                    else naive_update(v2, theta2, 1.0)
                )
                with torch.no_grad():
                    theta1 += u1
                    theta2 += u2
            assert torch.isfinite(theta1).all()
            assert torch.isfinite(theta2).all()
            assert torch.isfinite(torch.sigmoid(theta1)).all()


def test_exact_imp_finite_after_short_run():
    gamma = 0.9
    torch.manual_seed(0)
    theta1 = ((torch.rand(5) - 0.5) * 0.2).requires_grad_(True)
    theta2 = ((torch.rand(5) - 0.5) * 0.2).requires_grad_(True)
    for _ in range(10):
        v1, v2 = exact_values(torch.sigmoid(theta1), torch.sigmoid(theta2), IMP, gamma)
        u1 = lola_update(v1, v2, theta1, theta2, 1.0, 1.0)
        u2 = naive_update(v2, theta2, 1.0)
        with torch.no_grad():
            theta1 += u1
            theta2 += u2
    assert torch.isfinite(theta1).all()
    assert torch.isfinite(theta2).all()


def test_policy_gradient_ipd_short_run_finite():
    gamma = 0.96
    torch.manual_seed(0)
    theta1 = (torch.rand(5) - 0.5) * 0.2
    theta2 = (torch.rand(5) - 0.5) * 0.2
    for _ in range(5):
        rollout = sample_episodes(
            torch.sigmoid(theta1), torch.sigmoid(theta2), IPD, horizon=20, batch_size=32
        )
        u1 = lola_pg_update(rollout, gamma, delta=1.0, eta=1.0)
        u2 = naive_pg_update(rollout, gamma, delta=1.0)
        theta1 = theta1 + u1
        theta2 = theta2 + u2
    assert torch.isfinite(theta1).all()
    assert torch.isfinite(theta2).all()
