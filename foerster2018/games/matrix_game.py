"""Two-player, memory-one iterated matrix games, per Foerster et al. (2018),
"Learning with Opponent-Learning Awareness" (AAMAS 2018, arXiv:1709.04326),
Section 5.1 "Iterated Games".

The paper models an iterated matrix game as a two-agent Markov Reward
Process (MRP): "the state at time 0 is empty, denoted as s0, and at time
t >= 1 is the joint action from t-1: s_t = (u^1_{t-1}, u^2_{t-1})". Agents
are assumed to hold a memory-1 strategy (Press and Dyson 2012 show this is
without loss of generality against a memory-1 opponent), so each agent's
policy is fully specified by 5 probabilities: the probability of playing
its "first" action (Cooperate for IPD, Heads for IMP) in state s0, and in
each of the four joint-outcome states.

State encoding used throughout this codebase: action 0 is the game's
"first" action (Cooperate / Heads), action 1 is the "second" action
(Defect / Tails). A joint action (a1, a2) is encoded as the integer
`a1 * 2 + a2`, giving four states in the fixed order:

    index 0: (0, 0)  -- CC / HH
    index 1: (0, 1)  -- CD / HT
    index 2: (1, 0)  -- DC / TH
    index 3: (1, 1)  -- DD / TT

`s0` (the start-of-episode state, before any joint action has been played)
is handled as a separate 5th slot by callers (see
`foerster2018.policy.MemoryOnePolicy`), not part of this 4-state encoding.
"""
from dataclasses import dataclass

import torch


def encode_joint_action(a1: int, a2: int) -> int:
    """Map a joint action (a1, a2) in {0,1}^2 to a state index in {0,1,2,3}."""
    return a1 * 2 + a2


@dataclass(frozen=True)
class MatrixGame:
    """A one-shot 2x2 matrix game, replayed each round of an iterated game.

    `payoff1[s]`/`payoff2[s]` give agent 1's / agent 2's per-round reward
    when the joint action encoded by state index `s` (see module docstring)
    is played.
    """

    name: str
    payoff1: torch.Tensor  # shape (4,), indexed by encode_joint_action(a1, a2)
    payoff2: torch.Tensor  # shape (4,)

    def rewards(self, a1: torch.Tensor, a2: torch.Tensor):
        """Look up per-episode rewards for batched integer actions a1, a2 (0/1)."""
        state = (a1 * 2 + a2).long()
        return self.payoff1[state], self.payoff2[state]


# Table 1 of Foerster et al. (2018): "Payoff matrix of prisoners' dilemma."
#             C          D
#   C     (-1, -1)   (-3,  0)
#   D     ( 0, -3)   (-2, -2)
# Rows are agent 1's action, columns agent 2's. Action 0 = Cooperate.
IPD = MatrixGame(
    name="IPD",
    payoff1=torch.tensor([-1.0, -3.0, 0.0, -2.0]),  # CC, CD, DC, DD
    payoff2=torch.tensor([-1.0, 0.0, -3.0, -2.0]),
)

# Table 2 of Foerster et al. (2018): "Payoff matrix of matching pennies."
#              Head       Tail
#   Head    (+1, -1)   (-1, +1)
#   Tail    (-1, +1)   (+1, -1)
# Rows are agent 1's action, columns agent 2's. Action 0 = Heads.
IMP = MatrixGame(
    name="IMP",
    payoff1=torch.tensor([1.0, -1.0, -1.0, 1.0]),  # HH, HT, TH, TT
    payoff2=torch.tensor([-1.0, 1.0, 1.0, -1.0]),
)
