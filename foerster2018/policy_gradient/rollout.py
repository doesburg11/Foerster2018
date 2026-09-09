"""Sampled rollouts of a memory-one iterated matrix game, for the
policy-gradient (LOLA-PG) experiments of Foerster et al. (2018) Sec. 4.3
and Sec. 5.1's "The policy gradient experiments" (Sec. 6.1's Figure 1d/2d).

Unlike `foerster2018.exact`, no closed-form value is available here: both
agents' returns are estimated from actual sampled episodes (score-function
/ REINFORCE gradients), which is what the paper means by the "policy
gradient" version of LOLA. This module only draws the trajectories and
records, at every timestep, the log-probability *score* d/dtheta log
pi(a_t|s_t) for both agents (needed by `foerster2018.policy_gradient.
lola_pg`), not any Hessian or lookahead term itself.
"""
from dataclasses import dataclass

import torch

from foerster2018.games.matrix_game import MatrixGame
from foerster2018.policy import NUM_PARAMS, S0


@dataclass
class Rollout:
    """All per-timestep quantities needed by `lola_pg.py`, shape (T, B)
    unless noted. `score1`/`score2` are the per-step score-function vectors
    d/dtheta log pi(a_t | s_t), shape (T, B, 5) -- zero everywhere except
    the coordinate of the state actually visited at that timestep."""

    rewards1: torch.Tensor  # (T, B)
    rewards2: torch.Tensor  # (T, B)
    score1: torch.Tensor  # (T, B, 5)
    score2: torch.Tensor  # (T, B, 5)
    mean_reward1: float
    mean_reward2: float


def _bernoulli_score(prob_used: torch.Tensor, cooperated: torch.Tensor, state_idx: torch.Tensor):
    """d/dtheta log pi(a|s) for a memory-one policy, at the single state
    actually visited.

    For y = 1{action == 0} ~ Bernoulli(p), p = sigmoid(theta_s), the
    logistic-regression score function is the textbook identity
    `d/dtheta log P(y) = y - p` (nonzero only at coordinate s). This is
    unit-tested directly against `torch.autograd.grad` in
    `tests/test_policy_gradient.py::test_bernoulli_score_matches_autograd`
    rather than trusted on inspection alone.

    `prob_used`, `cooperated`: shape (B,). `state_idx`: shape (B,), values
    in [0, NUM_PARAMS). Returns shape (B, NUM_PARAMS).
    """
    batch = prob_used.shape[0]
    onehot = torch.zeros(batch, NUM_PARAMS, dtype=prob_used.dtype)
    onehot.scatter_(1, state_idx.unsqueeze(1), 1.0)
    coefficient = cooperated.to(prob_used.dtype) - prob_used  # (B,)
    return onehot * coefficient.unsqueeze(1)


def sample_episodes(
    policy1_probs: torch.Tensor,
    policy2_probs: torch.Tensor,
    game: MatrixGame,
    horizon: int,
    batch_size: int,
    generator: torch.Generator = None,
) -> Rollout:
    """Sample `batch_size` independent episodes of length `horizon`.

    `policy1_probs`/`policy2_probs` are the length-5 probability tensors
    from `MemoryOnePolicy.probs()`. Actions are discrete and sampled under
    `torch.no_grad()` (they are not reparameterizable), and the returned
    per-step scores are the *closed-form* score-function values (see
    `_bernoulli_score`) -- plain numbers, not autograd graph nodes. This is
    the standard REINFORCE/likelihood-ratio estimator: since `d/dtheta log
    pi(a|s)` has an exact, cheap closed form here, it is computed directly
    rather than via `loss.backward()` through a surrogate objective.
    `foerster2018.policy_gradient.lola_pg` combines these numeric scores
    with rewards into gradient *estimates*, which are then added to
    `theta1`/`theta2` directly (no further autograd involved).
    """
    batch_shape = (batch_size,)
    state_idx = torch.full(batch_shape, S0, dtype=torch.long)

    rewards1 = torch.zeros(horizon, batch_size)
    rewards2 = torch.zeros(horizon, batch_size)
    score1 = torch.zeros(horizon, batch_size, NUM_PARAMS)
    score2 = torch.zeros(horizon, batch_size, NUM_PARAMS)

    with torch.no_grad():
        probs1_detached = policy1_probs.detach()
        probs2_detached = policy2_probs.detach()

    for t in range(horizon):
        p1_t = probs1_detached[state_idx]  # (B,)
        p2_t = probs2_detached[state_idx]
        with torch.no_grad():
            coop1 = torch.bernoulli(p1_t, generator=generator)  # 1 = cooperated (action 0)
            coop2 = torch.bernoulli(p2_t, generator=generator)
            a1 = (1 - coop1).long()
            a2 = (1 - coop2).long()

        r1, r2 = game.rewards(a1, a2)
        rewards1[t] = r1
        rewards2[t] = r2
        score1[t] = _bernoulli_score(p1_t, coop1, state_idx)
        score2[t] = _bernoulli_score(p2_t, coop2, state_idx)

        state_idx = 1 + (a1 * 2 + a2)

    mean_reward1 = float(rewards1.mean())
    mean_reward2 = float(rewards2.mean())
    return Rollout(rewards1, rewards2, score1, score2, mean_reward1, mean_reward2)
