import torch

from foerster2018.exact.lola import lola_correction, lola_update, naive_update
from foerster2018.exact.value import exact_values
from foerster2018.games.matrix_game import IPD


def _fresh_params(seed):
    g = torch.Generator().manual_seed(seed)
    theta1 = (torch.rand(5, generator=g) - 0.5) * 2
    theta2 = (torch.rand(5, generator=g) - 0.5) * 2
    theta1.requires_grad_(True)
    theta2.requires_grad_(True)
    return theta1, theta2


def test_lola_reduces_to_naive_when_eta_is_zero():
    """With eta=0, the LOLA update's second-order correction term should
    vanish (the opponent's simulated lookahead step has size 0), leaving
    exactly the naive gradient-ascent update, Eq. 4.1."""
    theta1, theta2 = _fresh_params(0)
    v1, v2 = exact_values(torch.sigmoid(theta1), torch.sigmoid(theta2), IPD, gamma=0.9)

    naive = naive_update(v1, theta1, delta=0.1)

    theta1b, theta2b = _fresh_params(0)
    v1b, v2b = exact_values(torch.sigmoid(theta1b), torch.sigmoid(theta2b), IPD, gamma=0.9)
    lola = lola_update(v1b, v2b, theta1b, theta2b, delta=0.1, eta=0.0)

    assert torch.allclose(naive, lola, atol=1e-6)


def test_lola_correction_term_is_nonzero_and_finite_in_general():
    theta1, theta2 = _fresh_params(1)
    v1, v2 = exact_values(torch.sigmoid(theta1), torch.sigmoid(theta2), IPD, gamma=0.9)
    correction = lola_correction(v1, v2, theta1, theta2)
    assert correction.shape == (5,)
    assert torch.isfinite(correction).all()
    assert torch.any(correction != 0)


def test_lola_correction_actually_uses_second_order_information():
    """A pure sanity check that the correction term is NOT simply a
    (possibly rescaled) copy of the naive gradient -- i.e. that it carries
    genuinely different, second-order information about agent 2's
    response, rather than silently collapsing to a first-order signal."""
    theta1, theta2 = _fresh_params(2)
    v1, v2 = exact_values(torch.sigmoid(theta1), torch.sigmoid(theta2), IPD, gamma=0.9)
    naive_grad = naive_update(v1, theta1, delta=1.0)
    correction = lola_correction(v1, v2, theta1, theta2)

    naive_dir = naive_grad / naive_grad.norm()
    corr_dir = correction / correction.norm()
    # If the correction were just a positive rescaling of the naive
    # gradient, cosine similarity would be ~1. It should not be.
    cosine = torch.dot(naive_dir, corr_dir).item()
    assert abs(cosine - 1.0) > 1e-3


def test_lola_correction_depends_on_opponent_gradient_step_size_direction():
    """The correction term should change (generically) if we perturb
    theta2 in a way that changes agent 2's own gradient -- confirming the
    term is a function of agent 2's *learning dynamics*, not a constant or
    a function of theta1 alone."""
    theta1 = torch.zeros(5, requires_grad=True)
    theta2a = torch.zeros(5, requires_grad=True)
    theta2b = (torch.ones(5) * 2.0).requires_grad_(True)

    v1a, v2a = exact_values(torch.sigmoid(theta1), torch.sigmoid(theta2a), IPD, gamma=0.9)
    corr_a = lola_correction(v1a, v2a, theta1, theta2a)

    v1b, v2b = exact_values(torch.sigmoid(theta1), torch.sigmoid(theta2b), IPD, gamma=0.9)
    corr_b = lola_correction(v1b, v2b, theta1, theta2b)

    assert not torch.allclose(corr_a, corr_b)


def test_naive_vs_naive_ipd_converges_toward_mutual_defection():
    """Qualitative claim from the paper (Sec. 6.1 / Fig. 1a): 'Under NL-Ex,
    the agents learn to defect in all states.' Run enough naive-vs-naive
    exact-gradient iterations that both agents' cooperation probability in
    all states should collapse toward 0."""
    torch.manual_seed(0)
    theta1 = (torch.rand(5) * 0.2 - 0.1).requires_grad_(True)
    theta2 = (torch.rand(5) * 0.2 - 0.1).requires_grad_(True)
    delta = 1.0
    gamma = 0.96

    for _ in range(200):
        v1, v2 = exact_values(torch.sigmoid(theta1), torch.sigmoid(theta2), IPD, gamma)
        u1 = naive_update(v1, theta1, delta)
        u2 = naive_update(v2, theta2, delta)
        with torch.no_grad():
            theta1 += u1
            theta2 += u2

    probs1 = torch.sigmoid(theta1).detach()
    probs2 = torch.sigmoid(theta2).detach()
    assert probs1.mean().item() < 0.15
    assert probs2.mean().item() < 0.15
