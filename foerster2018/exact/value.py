"""Exact, closed-form value function for two memory-one policies playing an
infinitely repeated matrix game.

This is the standard stationary-Markov-chain ("memory-one MRP") formula
used by Foerster et al. (2018) Sec. 5.1 ("By analytically solving the
multi-agent MDP we can derive each agent's future discounted reward as an
analytical function of the agents' policies") and, more generally, by the
zero-determinant-strategy literature (Press & Dyson 2012), which the paper
cites for the fact that a memory-1 strategy loses no generality against a
memory-1 opponent. The paper does not spell out the matrix algebra itself
(it only names the method), so the derivation in this module's docstring
is this codebase's own standard construction of that method, not a quote.

Derivation
----------
Let a joint action `u_t = (u1_t, u2_t)` be played at round t (t = 0, 1, ...).
The agents' memory-one policies condition u_t on `s_t`, where `s_0` is the
special start state and `s_t = u_{t-1}` for t >= 1. Both agents observe the
same shared state `s_t` (the previous joint action), even though the two
agents' *action probabilities* in that state can differ.

Because the reward at round t is a function of `u_t = s_{t+1}`, we can
write the discounted return as a sum over the distribution of s_{t+1}:

    V = E[ sum_{t=0}^inf gamma^t * r(u_t) ]
      = sum_{k=1}^inf gamma^{k-1} * (w_k . r)

where `w_k` is the marginal distribution over the 4 joint-outcome states
{CC, CD, DC, DD} at "time k" (i.e. the distribution of s_k for k >= 1), and
`r` is the length-4 reward vector for one agent (see
`foerster2018.games.matrix_game.MatrixGame`).

`w_1` (the distribution of the very first joint action, drawn using each
agent's s0 probability) and the 4x4 Markov transition matrix `M` (row s ->
row-normalized distribution over next state s', built from each agent's
per-state action probabilities) together give `w_k = w_1 @ M^(k-1)`, so:

    V = w_1 . sum_{j=0}^inf (gamma * M)^j . r = w_1 . (I - gamma * M)^-1 . r

This is implemented with `torch.linalg.solve` (not an explicit matrix
inverse) so that `(I - gamma*M) x = r` is solved for `x`, and the whole
computation stays a plain, twice-differentiable function of both agents'
policy parameters (needed for LOLA's second-order lookahead term -- see
`foerster2018.exact.lola`). Nothing here is a hand-derived second
derivative; `torch.autograd.grad(..., create_graph=True)` is used by
callers to differentiate through this closed-form value twice.
"""
from typing import Tuple

import torch

from foerster2018.games.matrix_game import MatrixGame
from foerster2018.policy import S0, STATE_OFFSET


def _initial_distribution(p1_s0: torch.Tensor, p2_s0: torch.Tensor) -> torch.Tensor:
    """Distribution over {CC, CD, DC, DD} for the very first joint action,
    given each agent's probability of playing action 0 in state s0."""
    return torch.stack(
        [
            p1_s0 * p2_s0,
            p1_s0 * (1 - p2_s0),
            (1 - p1_s0) * p2_s0,
            (1 - p1_s0) * (1 - p2_s0),
        ]
    )


def _transition_matrix(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """4x4 transition matrix M[s, s'] = P(next state s' | current state s),
    where `p1`, `p2` are each length-4 vectors of P(action 0 | state) for
    states [CC, CD, DC, DD] (this codebase's fixed state order)."""
    rows = []
    for s in range(4):
        rows.append(_initial_distribution(p1[s], p2[s]))
    return torch.stack(rows, dim=0)


def exact_values(
    policy1_probs: torch.Tensor,
    policy2_probs: torch.Tensor,
    game: MatrixGame,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact discounted values V1, V2 for both agents.

    `policy1_probs`/`policy2_probs`: length-5 tensors as returned by
    `MemoryOnePolicy.probs()`, i.e. [P(a=0|s0), P(a=0|CC), P(a=0|CD),
    P(a=0|DC), P(a=0|DD)].
    """
    p1_s0 = policy1_probs[S0]
    p2_s0 = policy2_probs[S0]
    p1 = policy1_probs[STATE_OFFSET : STATE_OFFSET + 4]
    p2 = policy2_probs[STATE_OFFSET : STATE_OFFSET + 4]

    w1 = _initial_distribution(p1_s0, p2_s0)  # (4,)
    M = _transition_matrix(p1, p2)  # (4, 4)

    identity = torch.eye(4, dtype=M.dtype)
    A = identity - gamma * M

    # Solve A @ x1 = r1 and A @ x2 = r2 jointly (stack as columns of B).
    B = torch.stack([game.payoff1, game.payoff2], dim=1)  # (4, 2)
    X = torch.linalg.solve(A, B)  # (4, 2)

    v1 = torch.dot(w1, X[:, 0])
    v2 = torch.dot(w1, X[:, 1])
    return v1, v2
