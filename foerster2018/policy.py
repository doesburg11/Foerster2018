"""Memory-one policy parameterization, per Foerster et al. (2018) Sec. 5.1:

"Each agent's policy is fully specified by 5 probabilities. For agent a in
the case of the IPD, they are the probability of cooperation at game start
pi^a(C|s0), and the cooperation probabilities in the four memories,
pi^a(C|CC), pi^a(C|CD), pi^a(C|DC), and pi^a(C|DD)."

Each of the 5 probabilities is a sigmoid-squashed scalar logit (`theta`),
so unconstrained gradient-based optimization stays a valid probability.
Slot 0 is the start-of-episode state s0; slots 1-4 are the four states
CC, CD, DC, DD in the encoding documented in `foerster2018.games.matrix_game`.
"""
import torch

S0 = 0
STATE_OFFSET = 1  # states CC, CD, DC, DD occupy theta[1:5]
NUM_PARAMS = 5


class MemoryOnePolicy:
    """A memory-one policy: 5 logits, one per (s0, CC, CD, DC, DD) state."""

    def __init__(self, theta: torch.Tensor = None, init_logit_std: float = 0.1):
        if theta is None:
            theta = torch.randn(NUM_PARAMS) * init_logit_std
        theta = theta.clone().detach().requires_grad_(True)
        self.theta = theta

    def probs(self) -> torch.Tensor:
        """Probability of playing action 0 (Cooperate / Heads) in each of
        the 5 states, in order [s0, CC, CD, DC, DD]."""
        return torch.sigmoid(self.theta)

    def prob_state(self, state_index: int) -> torch.Tensor:
        """`state_index` in {0 (=s0), 1, 2, 3, 4} indexing into `probs()`."""
        return self.probs()[state_index]

    def detach_copy(self) -> "MemoryOnePolicy":
        return MemoryOnePolicy(theta=self.theta.detach().clone())

    def __repr__(self):
        p = self.probs().detach().numpy()
        return (
            f"MemoryOnePolicy(P(a=0|s0)={p[0]:.3f}, "
            f"P(a=0|CC)={p[1]:.3f}, P(a=0|CD)={p[2]:.3f}, "
            f"P(a=0|DC)={p[3]:.3f}, P(a=0|DD)={p[4]:.3f})"
        )
