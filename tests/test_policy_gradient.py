import torch

from foerster2018.exact.lola import lola_correction
from foerster2018.exact.value import exact_values
from foerster2018.games.matrix_game import IPD
from foerster2018.policy_gradient.lola_pg import lola_pg_update, naive_pg_update
from foerster2018.policy_gradient.rollout import _bernoulli_score, sample_episodes


def test_bernoulli_score_matches_autograd_cooperate():
    theta = torch.randn(5, requires_grad=True)
    probs = torch.sigmoid(theta)
    state_idx = torch.tensor([2])
    p = probs[state_idx]
    cooperated = torch.tensor([1.0])
    logprob = cooperated * torch.log(p) + (1 - cooperated) * torch.log(1 - p)
    (grad_autograd,) = torch.autograd.grad(logprob.sum(), theta)

    score = _bernoulli_score(p.detach(), cooperated, state_idx)
    assert torch.allclose(score.sum(dim=0), grad_autograd, atol=1e-6)


def test_bernoulli_score_matches_autograd_defect():
    theta = torch.randn(5, requires_grad=True)
    probs = torch.sigmoid(theta)
    state_idx = torch.tensor([3])
    p = probs[state_idx]
    cooperated = torch.tensor([0.0])
    logprob = cooperated * torch.log(p) + (1 - cooperated) * torch.log(1 - p)
    (grad_autograd,) = torch.autograd.grad(logprob.sum(), theta)

    score = _bernoulli_score(p.detach(), cooperated, state_idx)
    assert torch.allclose(score.sum(dim=0), grad_autograd, atol=1e-6)


def test_sample_episodes_produces_finite_rewards_and_scores():
    theta1 = torch.zeros(5, requires_grad=True)
    theta2 = torch.zeros(5, requires_grad=True)
    rollout = sample_episodes(
        torch.sigmoid(theta1), torch.sigmoid(theta2), IPD, horizon=20, batch_size=32
    )
    assert torch.isfinite(rollout.rewards1).all()
    assert torch.isfinite(rollout.rewards2).all()
    assert torch.isfinite(rollout.score1).all()
    assert torch.isfinite(rollout.score2).all()
    assert rollout.score1.shape == (20, 32, 5)


def test_reinforce_grad_matches_exact_gradient_at_uniform_policy():
    """Regression test for a real bug caught during development: the
    first-order PG estimator (`_reinforce_grad`, used by both
    `naive_pg_update` and the first term of `lola_pg_update`) originally
    omitted the `gamma**t` factor the paper's own derivation requires
    (see `_reinforce_grad`'s docstring), which inflated its magnitude by
    roughly 4x relative to the true gradient at theta=0, gamma=0.96,
    horizon=100. With a large batch, the fixed estimator should closely
    match the closed-form exact gradient at the same (uniform-random)
    policy parameters."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    theta1 = torch.zeros(5, requires_grad=True)
    theta2 = torch.zeros(5, requires_grad=True)
    gamma = 0.96

    v1, v2 = exact_values(torch.sigmoid(theta1), torch.sigmoid(theta2), IPD, gamma)
    (exact_grad1,) = torch.autograd.grad(v1, theta1)

    rollout = sample_episodes(
        torch.sigmoid(theta1.detach()),
        torch.sigmoid(theta2.detach()),
        IPD,
        horizon=100,
        batch_size=8000,
        generator=gen,
    )
    pg_grad1 = naive_pg_update(rollout, gamma=gamma, delta=1.0)

    assert torch.allclose(pg_grad1, exact_grad1, atol=0.15)


def test_lola_pg_reduces_to_naive_pg_when_eta_is_zero():
    torch.manual_seed(0)
    theta1 = torch.zeros(5, requires_grad=True)
    theta2 = torch.zeros(5, requires_grad=True)
    rollout = sample_episodes(
        torch.sigmoid(theta1), torch.sigmoid(theta2), IPD, horizon=30, batch_size=64
    )
    naive = naive_pg_update(rollout, gamma=0.96, delta=0.1)
    lola = lola_pg_update(rollout, gamma=0.96, delta=0.1, eta=0.0)
    assert torch.allclose(naive, lola, atol=1e-6)


def test_lola_pg_cross_term_correlates_with_exact_lola_correction():
    """Not an exact-match test (the PG estimator is a noisy Monte Carlo
    estimate of the same quantity the exact version computes in closed
    form) -- but with a large batch, the *direction* of the LOLA-PG
    correction term should be reasonably aligned with the exact LOLA
    correction term computed at the same policy parameters. A weak/no
    correlation would indicate the estimator is not actually tracking
    Eq. 4.6."""
    torch.manual_seed(0)
    theta1 = (torch.rand(5) * 0.6 - 0.3).requires_grad_(True)
    theta2 = (torch.rand(5) * 0.6 - 0.3).requires_grad_(True)
    gamma = 0.96

    v1, v2 = exact_values(torch.sigmoid(theta1), torch.sigmoid(theta2), IPD, gamma)
    exact_correction = lola_correction(v1, v2, theta1, theta2)

    rollout = sample_episodes(
        torch.sigmoid(theta1.detach()),
        torch.sigmoid(theta2.detach()),
        IPD,
        horizon=150,
        batch_size=8192,
    )
    pg_update = lola_pg_update(rollout, gamma=gamma, delta=1.0, eta=1.0)
    naive_pg = naive_pg_update(rollout, gamma=gamma, delta=1.0)
    pg_correction = pg_update - naive_pg

    cosine = torch.dot(exact_correction, pg_correction) / (
        exact_correction.norm() * pg_correction.norm() + 1e-8
    )
    assert cosine.item() > 0.3
