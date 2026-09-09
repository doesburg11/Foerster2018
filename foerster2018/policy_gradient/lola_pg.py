"""Policy-gradient (sampled-rollout) NL and LOLA updates, per Foerster et
al. (2018) Sec. 4.3 "Learning via Policy Gradient".

Eq. 4.5 (NL-PG, agent 1's naive policy-gradient update):

    f^1_{nl,pg} = grad_{theta1} E[R^1_0(tau)] . delta

using the standard REINFORCE-with-baseline, reward-to-go estimator (shown
in the paper's own derivation immediately above Eq. 4.5):

    grad_{theta1} E[R^1_0(tau)]
        = E[ sum_t grad_{theta1} log pi^1(u^1_t|s_t) . gamma^t . (R^1_t(tau) - b(s_t)) ]

Note the *absolute*-time discount `gamma^t` multiplying the (relatively
discounted) reward-to-go `R^1_t(tau) = sum_{l=t}^T gamma^{l-t} r^1_l` and
baseline -- see `_reinforce_grad`'s docstring for why this is easy to drop
by mistake and how this repo caught having done exactly that.

Eq. 4.6 (the policy-gradient estimator of the LOLA second-order term,
exact in expectation per the paper's own claim):

    grad_{theta1} grad_{theta2} E[R^2_0(tau)]
        = E[ sum_t gamma^t r^2_t .
             (sum_{l=0}^t grad_{theta1} log pi^1(u^1_l|s_l))
             (sum_{l=0}^t grad_{theta2} log pi^2(u^2_l|s_l))^T ]

Eq. 4.7 (LOLA-PG, agent 1's update):

    f^1_{lola,pg} = grad_{theta1} E[R^1_0(tau)] . delta
                  + (grad_{theta2} E[R^1_0(tau)])^T
                    grad_{theta1} grad_{theta2} E[R^2_0(tau)] . delta . eta

Important, explicitly out-of-scope simplification: Eq. 4.6's cross term is
implemented *exactly as the paper writes it* -- the un-baselined,
non-causality-reduced double cumulative-sum estimator. This is a much
higher-variance estimator than the single-agent reward-to-go-with-baseline
trick used for the first-order term, and the paper does not suggest a
baseline for it either. A cleaner, lower-variance construction of this
exact quantity (via a "stochastically differentiable" surrogate loss, so
ordinary `loss.backward()` calls compute it for you) is the subject of the
*separate* LOLA-DiCE follow-up paper (Foerster et al. 2018b), which is
explicitly out of scope for this repository -- see the top-level README.
Naively building a surrogate loss and differentiating it twice with
`torch.autograd.grad(..., create_graph=True)` (the trick used in
`foerster2018.exact.lola` for the *exact* value function) does **not**
correctly recover Eq. 4.6 for sampled trajectories, because a plain
`sum_t logpi(u_t|s_t) * R_t` surrogate's second cross-derivative silently
drops necessary terms unless it is built the way DiCE builds it; that
subtlety is exactly why this module computes Eq. 4.5-4.7 as explicit
tensor arithmetic over the rollout's per-step scores instead of
autograd-through-a-surrogate-loss.
"""
import torch

from foerster2018.policy_gradient.rollout import Rollout


def _reward_to_go(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """`rewards`: (T, B). Returns (T, B) with `out[t] = sum_{l=t}^{T-1}
    gamma^{l-t} * rewards[l]`."""
    horizon = rewards.shape[0]
    out = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[1], dtype=rewards.dtype)
    for t in range(horizon - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def _reinforce_grad(scores: torch.Tensor, rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """The paper's own derivation immediately above Eq. 4.5 (quoted in this
    module's docstring) gives:

        grad_theta E[R_0(tau)]
            = E[ sum_t grad_theta log pi(u_t|s_t) . gamma^t . (R_t(tau) - b(s_t)) ]

    i.e. an *absolute*-time discount `gamma**t` multiplies the (relatively
    discounted, `R_t(tau) = sum_{l=t}^T gamma^{l-t} r_l`) reward-to-go and
    baseline term -- easy to drop by mistake since `R_t` already contains a
    (different, relative) discount factor of its own. An earlier version of
    this function omitted the `gamma**t` factor entirely; caught by
    comparing this estimator's value at theta=0 against the closed-form
    exact gradient (`foerster2018.exact.value`) at the same parameters,
    which differed by roughly 4x, far beyond sampling noise at batch=4000
    (see `tests/test_policy_gradient.py::
    test_reinforce_grad_matches_exact_gradient_at_uniform_policy`).

    `scores`: (T, B, P). `rewards`: (T, B). A per-timestep batch-mean
    baseline is subtracted for variance reduction (the "value baseline"
    named in this repo's brief).
    """
    horizon = rewards.shape[0]
    reward_to_go = _reward_to_go(rewards, gamma)  # (T, B), R_t(tau)
    baseline = reward_to_go.mean(dim=1, keepdim=True)  # (T, 1)
    advantage = reward_to_go - baseline  # (T, B)
    gamma_powers = gamma ** torch.arange(horizon, dtype=rewards.dtype)  # (T,), gamma^t
    weighted = scores * (gamma_powers.view(horizon, 1) * advantage).unsqueeze(-1)  # (T, B, P)
    return weighted.sum(dim=0).mean(dim=0)  # (P,)


def _cross_term_matrix(
    score1: torch.Tensor, score2: torch.Tensor, rewards2: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Eq. 4.6: returns the (P1, P2) matrix
    `grad_{theta1} grad_{theta2} E[R^2_0(tau)]`, estimated from the batch.
    """
    horizon = score1.shape[0]
    cum_score1 = torch.cumsum(score1, dim=0)  # (T, B, P1)
    cum_score2 = torch.cumsum(score2, dim=0)  # (T, B, P2)
    gamma_powers = gamma ** torch.arange(horizon, dtype=rewards2.dtype)  # (T,)
    weight = gamma_powers.view(horizon, 1) * rewards2  # (T, B)
    # outer product per (t, b): (P1, P2), weighted and summed over t, b.
    weighted_cum1 = cum_score1 * weight.unsqueeze(-1)  # (T, B, P1)
    # einsum: sum over T and B of weighted_cum1[t,b,i] * cum_score2[t,b,j]
    matrix = torch.einsum("tbi,tbj->ij", weighted_cum1, cum_score2)
    batch_size = score1.shape[1]
    return matrix / batch_size


def naive_pg_update(rollout: Rollout, gamma: float, delta: float) -> torch.Tensor:
    """Eq. 4.5: agent 1's naive policy-gradient update."""
    grad1_r1 = _reinforce_grad(rollout.score1, rollout.rewards1, gamma)
    return delta * grad1_r1


def lola_pg_update(rollout: Rollout, gamma: float, delta: float, eta: float) -> torch.Tensor:
    """Eq. 4.7: agent 1's LOLA policy-gradient update against agent 2."""
    grad1_r1 = _reinforce_grad(rollout.score1, rollout.rewards1, gamma)
    grad2_r1 = _reinforce_grad(rollout.score2, rollout.rewards1, gamma)
    cross = _cross_term_matrix(rollout.score1, rollout.score2, rollout.rewards2, gamma)
    correction = cross @ grad2_r1
    return delta * grad1_r1 + delta * eta * correction
