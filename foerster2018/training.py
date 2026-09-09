"""Generic simultaneous-gradient-ascent training loop shared by the exact
and policy-gradient experiments. An "update function" for an agent maps
`(values_or_stats, theta_self, theta_other) -> update_vector`; both agents'
updates are computed from the *same* snapshot of both parameter vectors
before either is applied ("simultaneous" updates, matching the paper's
`theta_{i+1} = theta_i + f(theta_i)` notation).
"""
from dataclasses import dataclass, field
from typing import Callable, List

import torch

from foerster2018.policy import MemoryOnePolicy

UpdateFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrainingHistory:
    mean_reward1: List[float] = field(default_factory=list)
    mean_reward2: List[float] = field(default_factory=list)
    probs1: List[list] = field(default_factory=list)
    probs2: List[list] = field(default_factory=list)


def run_training(
    policy1: MemoryOnePolicy,
    policy2: MemoryOnePolicy,
    step_fn: Callable[[MemoryOnePolicy, MemoryOnePolicy], "tuple[torch.Tensor, torch.Tensor]"],
    iterations: int,
    log_every: int = 1,
) -> TrainingHistory:
    """`step_fn(policy1, policy2)` must perform one full simultaneous
    update of both policies **in place** (mutating `policy1.theta` and
    `policy2.theta`) and return the (avg-reward-per-step-equivalent) values
    `(v1, v2)` observed *before* the update was applied, for logging.
    """
    history = TrainingHistory()
    for i in range(iterations):
        v1, v2 = step_fn(policy1, policy2)
        if i % log_every == 0 or i == iterations - 1:
            history.mean_reward1.append(float(v1))
            history.mean_reward2.append(float(v2))
            history.probs1.append(policy1.probs().detach().tolist())
            history.probs2.append(policy2.probs().detach().tolist())
    return history
