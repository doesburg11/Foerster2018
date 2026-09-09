#!/usr/bin/env python3
"""Experiment 2: Iterated Prisoners' Dilemma, policy-gradient (LOLA-PG).

Reproduces the "-PG" (sampled-rollout, REINFORCE-based) rows of Foerster
et al. (2018) Table 3 and the qualitative claim of Fig. 1d: all four
(NL, LOLA) x (NL, LOLA) pairings on the IPD, using score-function gradient
estimates from sampled episodes (`foerster2018.policy_gradient`) instead of
the closed-form value function used by experiment 1.

This is noisier and needs more iterations than the exact-gradient version
(experiment 1) to reach a comparable qualitative outcome -- see the
module docstring of `foerster2018.policy_gradient.lola_pg` for why the
LOLA-PG cross term (Eq. 4.6) in particular is a relatively high-variance
estimator. Defaults to a small smoke-test configuration; pass
`--iterations`, `--batch-size`, `--num-runs` with larger values to scale up.
"""
import argparse
import json
import os

import torch

from foerster2018.games.matrix_game import IPD
from foerster2018.policy_gradient.lola_pg import lola_pg_update, naive_pg_update
from foerster2018.policy_gradient.rollout import Rollout, sample_episodes

PAIRINGS = [("NL", "NL"), ("LOLA", "NL"), ("NL", "LOLA"), ("LOLA", "LOLA")]


def is_tft_like(probs) -> bool:
    p_s0, p_cc, _p_cd, _p_dc, p_dd = probs
    return p_s0 > 0.5 and p_cc > 0.5 and p_dd < 0.5


def train_pairing(
    agent1_type, agent2_type, gamma, delta, eta, iterations, horizon, batch_size, seed
):
    gen = torch.Generator().manual_seed(seed)
    theta1 = (torch.rand(5, generator=gen) - 0.5) * 0.2
    theta2 = (torch.rand(5, generator=gen) - 0.5) * 0.2

    reward_history1, reward_history2 = [], []
    for _ in range(iterations):
        probs1 = torch.sigmoid(theta1)
        probs2 = torch.sigmoid(theta2)
        rollout = sample_episodes(probs1, probs2, IPD, horizon, batch_size, generator=gen)

        u1 = (
            lola_pg_update(rollout, gamma, delta, eta)
            if agent1_type == "LOLA"
            else naive_pg_update(rollout, gamma, delta)
        )
        # Agent 2's update needs the rollout from *its own* perspective:
        # swap which agent is "self" vs "other" in the score/reward roles.
        rollout_for_2 = Rollout(
            rewards1=rollout.rewards2,
            rewards2=rollout.rewards1,
            score1=rollout.score2,
            score2=rollout.score1,
            mean_reward1=rollout.mean_reward2,
            mean_reward2=rollout.mean_reward1,
        )
        u2 = (
            lola_pg_update(rollout_for_2, gamma, delta, eta)
            if agent2_type == "LOLA"
            else naive_pg_update(rollout_for_2, gamma, delta)
        )

        theta1 = theta1 + u1
        theta2 = theta2 + u2

        reward_history1.append(rollout.mean_reward1)
        reward_history2.append(rollout.mean_reward2)

    final_probs1 = torch.sigmoid(theta1).tolist()
    final_probs2 = torch.sigmoid(theta2).tolist()
    return {
        "final_probs1": final_probs1,
        "final_probs2": final_probs2,
        "final_avg_reward1": reward_history1[-1],
        "final_avg_reward2": reward_history2[-1],
        "reward_history1": reward_history1,
        "reward_history2": reward_history2,
    }


def summarize(agent1_type, agent2_type, gamma, delta, eta, iterations, horizon, batch_size, num_runs, seed0):
    rewards1, rewards2 = [], []
    tft1, tft2 = 0, 0
    example_run = None
    for run_idx in range(num_runs):
        result = train_pairing(
            agent1_type,
            agent2_type,
            gamma,
            delta,
            eta,
            iterations,
            horizon,
            batch_size,
            seed=seed0 + run_idx,
        )
        rewards1.append(result["final_avg_reward1"])
        rewards2.append(result["final_avg_reward2"])
        if is_tft_like(result["final_probs1"]):
            tft1 += 1
        if is_tft_like(result["final_probs2"]):
            tft2 += 1
        if run_idx == 0:
            example_run = result

    r1 = torch.tensor(rewards1)
    r2 = torch.tensor(rewards2)
    return {
        "pairing": f"{agent1_type} vs {agent2_type}",
        "num_runs": num_runs,
        "agent1_mean_reward": r1.mean().item(),
        "agent1_std_reward": r1.std(unbiased=False).item(),
        "agent2_mean_reward": r2.mean().item(),
        "agent2_std_reward": r2.std(unbiased=False).item(),
        "agent1_pct_tft": 100.0 * tft1 / num_runs,
        "agent2_pct_tft": 100.0 * tft2 / num_runs,
        "example_final_probs1": example_run["final_probs1"],
        "example_final_probs2": example_run["final_probs2"],
        "example_reward_history1": example_run["reward_history1"],
        "example_reward_history2": example_run["reward_history2"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument(
        "--delta",
        type=float,
        default=0.3,
        help=(
            "As with experiment 1's --delta, not stated by the paper for "
            "this exact configuration (the paper's own PG experiment uses "
            "an actor-critic value baseline and delta=0.005 for its actor, "
            "Sec. 5.3 -- this repo's simpler batch-mean baseline needed a "
            "larger step size empirically; see README)."
        ),
    )
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="output/run_experiment2_ipd_policy_gradient")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    for agent1_type, agent2_type in PAIRINGS:
        key = f"{agent1_type}-{agent2_type}"
        print(f"Training {key} on IPD (policy gradient)...")
        summary = summarize(
            agent1_type,
            agent2_type,
            args.gamma,
            args.delta,
            args.eta,
            args.iterations,
            args.horizon,
            args.batch_size,
            args.num_runs,
            seed0=args.seed,
        )
        results[key] = summary
        print(
            f"  agent1 ({agent1_type}): mean reward/step = "
            f"{summary['agent1_mean_reward']:.3f} (std {summary['agent1_std_reward']:.3f}), "
            f"%TFT-like = {summary['agent1_pct_tft']:.1f}"
        )
        print(
            f"  agent2 ({agent2_type}): mean reward/step = "
            f"{summary['agent2_mean_reward']:.3f} (std {summary['agent2_std_reward']:.3f}), "
            f"%TFT-like = {summary['agent2_pct_tft']:.1f}"
        )

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
