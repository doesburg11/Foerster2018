#!/usr/bin/env python3
"""Experiment 3: Iterated Matching Pennies (IMP), exact gradients.

Reproduces the IMP columns of Foerster et al. (2018) Table 3 and the
qualitative claim of Fig. 2 / Sec. 6.1: "under naive learning the agents'
strategies fail to converge... under LOLA the agents' policies converge to
the Nash equilibrium, playing 50%/50% heads/tails." IMP is zero-sum
(Table 2), so unlike the IPD there is no "cooperation" to measure -- the
interesting quantity is whether each agent's cooperation-probability
vector approaches the fully-mixed point (0.5, 0.5, 0.5, 0.5, 0.5) and
stays there (LOLA), or drifts to a near-deterministic corner and keeps
moving / cycling (NL, per Fig. 2a's "accumulation of points in the
corners").

Paper's Sec. 5.3: gamma = 0.9 for matching pennies (lower than the IPD's
0.96 -- "we found that a lower gamma produced more stable learning on
IMP.").
"""
import argparse
import json
import os

import torch

from foerster2018.exact.lola import lola_update, naive_update
from foerster2018.exact.value import exact_values
from foerster2018.games.matrix_game import IMP

PAIRINGS = [("NL", "NL"), ("LOLA", "NL"), ("NL", "LOLA"), ("LOLA", "LOLA")]


def distance_from_nash(probs: torch.Tensor) -> float:
    return (probs - 0.5).abs().mean().item()


def train_pairing(agent1_type, agent2_type, gamma, delta, eta, iterations, seed):
    gen = torch.Generator().manual_seed(seed)
    theta1 = ((torch.rand(5, generator=gen) - 0.5) * 0.2).requires_grad_(True)
    theta2 = ((torch.rand(5, generator=gen) - 0.5) * 0.2).requires_grad_(True)

    reward_history1, reward_history2 = [], []
    dist_history1, dist_history2 = [], []
    for _ in range(iterations):
        v1, v2 = exact_values(torch.sigmoid(theta1), torch.sigmoid(theta2), IMP, gamma)

        if agent1_type == "LOLA":
            u1 = lola_update(v1, v2, theta1, theta2, delta, eta)
        else:
            u1 = naive_update(v1, theta1, delta)

        if agent2_type == "LOLA":
            u2 = lola_update(v2, v1, theta2, theta1, delta, eta)
        else:
            u2 = naive_update(v2, theta2, delta)

        with torch.no_grad():
            theta1 += u1
            theta2 += u2

        reward_history1.append((1 - gamma) * v1.detach().item())
        reward_history2.append((1 - gamma) * v2.detach().item())
        dist_history1.append(distance_from_nash(torch.sigmoid(theta1).detach()))
        dist_history2.append(distance_from_nash(torch.sigmoid(theta2).detach()))

    return {
        "final_probs1": torch.sigmoid(theta1).detach().tolist(),
        "final_probs2": torch.sigmoid(theta2).detach().tolist(),
        "final_avg_reward1": reward_history1[-1],
        "final_avg_reward2": reward_history2[-1],
        "reward_history1": reward_history1,
        "reward_history2": reward_history2,
        "dist_history1": dist_history1,
        "dist_history2": dist_history2,
    }


def summarize(agent1_type, agent2_type, gamma, delta, eta, iterations, num_runs, seed0):
    rewards1, rewards2 = [], []
    final_dist1, final_dist2 = [], []
    # "Stability": std-dev of the distance-from-Nash over the tail of
    # training. Low = settled near a fixed point; high = still moving/
    # cycling (this is the empirical stand-in for the paper's Fig. 2c/2d
    # "normalised reward per step" variance discussion).
    tail_std1, tail_std2 = [], []
    example_run = None
    tail_len = max(1, iterations // 10)

    for run_idx in range(num_runs):
        result = train_pairing(
            agent1_type, agent2_type, gamma, delta, eta, iterations, seed=seed0 + run_idx
        )
        rewards1.append(result["final_avg_reward1"])
        rewards2.append(result["final_avg_reward2"])
        final_dist1.append(result["dist_history1"][-1])
        final_dist2.append(result["dist_history2"][-1])
        tail1 = torch.tensor(result["dist_history1"][-tail_len:])
        tail2 = torch.tensor(result["dist_history2"][-tail_len:])
        tail_std1.append(tail1.std(unbiased=False).item())
        tail_std2.append(tail2.std(unbiased=False).item())
        if run_idx == 0:
            example_run = result

    def stats(values):
        t = torch.tensor(values)
        return t.mean().item(), t.std(unbiased=False).item()

    r1_mean, r1_std = stats(rewards1)
    r2_mean, r2_std = stats(rewards2)
    d1_mean, _ = stats(final_dist1)
    d2_mean, _ = stats(final_dist2)
    s1_mean, _ = stats(tail_std1)
    s2_mean, _ = stats(tail_std2)

    return {
        "pairing": f"{agent1_type} vs {agent2_type}",
        "num_runs": num_runs,
        "agent1_mean_reward": r1_mean,
        "agent1_std_reward": r1_std,
        "agent2_mean_reward": r2_mean,
        "agent2_std_reward": r2_std,
        "agent1_mean_final_distance_from_nash": d1_mean,
        "agent2_mean_final_distance_from_nash": d2_mean,
        "agent1_mean_tail_instability": s1_mean,
        "agent2_mean_tail_instability": s2_mean,
        "example_final_probs1": example_run["final_probs1"],
        "example_final_probs2": example_run["final_probs2"],
        "example_reward_history1": example_run["reward_history1"],
        "example_reward_history2": example_run["reward_history2"],
        "example_dist_history1": example_run["dist_history1"],
        "example_dist_history2": example_run["dist_history2"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="output/run_experiment3_imp")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    for agent1_type, agent2_type in PAIRINGS:
        key = f"{agent1_type}-{agent2_type}"
        print(f"Training {key} on IMP (exact gradients)...")
        summary = summarize(
            agent1_type,
            agent2_type,
            args.gamma,
            args.delta,
            args.eta,
            args.iterations,
            args.num_runs,
            seed0=args.seed,
        )
        results[key] = summary
        print(
            f"  agent1 ({agent1_type}): mean reward/step = {summary['agent1_mean_reward']:.3f} "
            f"(std {summary['agent1_std_reward']:.3f}), "
            f"dist-from-Nash = {summary['agent1_mean_final_distance_from_nash']:.3f}, "
            f"tail instability = {summary['agent1_mean_tail_instability']:.4f}"
        )
        print(
            f"  agent2 ({agent2_type}): mean reward/step = {summary['agent2_mean_reward']:.3f} "
            f"(std {summary['agent2_std_reward']:.3f}), "
            f"dist-from-Nash = {summary['agent2_mean_final_distance_from_nash']:.3f}, "
            f"tail instability = {summary['agent2_mean_tail_instability']:.4f}"
        )

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
