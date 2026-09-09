#!/usr/bin/env python3
"""Experiment 1: Iterated Prisoners' Dilemma, exact gradients.

Reproduces the "-Ex" (exact value function / exact gradient) rows of
Foerster et al. (2018) Table 3 and the qualitative claims of Fig. 1: all
four (NL, LOLA) x (NL, LOLA) pairings on the IPD, using the closed-form
value function of `foerster2018.exact.value` and the update rules of
`foerster2018.exact.lola` (Eq. 4.1 for NL, Eq. 4.4 for LOLA).

Defaults to a small number of runs/iterations (a smoke test of the full
pipeline). Pass `--num-runs 50 --iterations 200` (or larger) to approach
the paper's own Table 3 setting (50 independent training runs per cell).
"""
import argparse
import json
import os

import torch

from foerster2018.exact.lola import lola_update, naive_update
from foerster2018.exact.value import exact_values
from foerster2018.games.matrix_game import IPD

PAIRINGS = [("NL", "NL"), ("LOLA", "NL"), ("NL", "LOLA"), ("LOLA", "LOLA")]


def is_tft_like(probs) -> bool:
    """Operational definition of "TFT-like" used only in this repo's own
    reporting -- the paper's Table 3 "%TFT" column does not state its exact
    classification threshold, so this is a documented proxy, not a quoted
    detail: start cooperative, keep cooperating after mutual cooperation,
    defect after mutual defection (the two most frequently visited TFT-vs-
    TFT states)."""
    p_s0, p_cc, _p_cd, _p_dc, p_dd = probs
    return p_s0 > 0.5 and p_cc > 0.5 and p_dd < 0.5


def train_pairing(agent1_type, agent2_type, gamma, delta, eta, iterations, seed):
    gen = torch.Generator().manual_seed(seed)
    theta1 = ((torch.rand(5, generator=gen) - 0.5) * 0.2).requires_grad_(True)
    theta2 = ((torch.rand(5, generator=gen) - 0.5) * 0.2).requires_grad_(True)

    reward_history1 = []
    reward_history2 = []
    for _ in range(iterations):
        v1, v2 = exact_values(torch.sigmoid(theta1), torch.sigmoid(theta2), IPD, gamma)

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

        # Paper's footnote 1 normalisation: (1-gamma) * sum gamma^t r_t = (1-gamma)*V.
        reward_history1.append((1 - gamma) * v1.detach().item())
        reward_history2.append((1 - gamma) * v2.detach().item())

    final_probs1 = torch.sigmoid(theta1).detach().tolist()
    final_probs2 = torch.sigmoid(theta2).detach().tolist()
    return {
        "final_probs1": final_probs1,
        "final_probs2": final_probs2,
        "final_avg_reward1": reward_history1[-1],
        "final_avg_reward2": reward_history2[-1],
        "reward_history1": reward_history1,
        "reward_history2": reward_history2,
    }


def summarize(agent1_type, agent2_type, gamma, delta, eta, iterations, num_runs, seed0):
    rewards1, rewards2 = [], []
    tft1, tft2 = 0, 0
    example_run = None
    for run_idx in range(num_runs):
        result = train_pairing(
            agent1_type, agent2_type, gamma, delta, eta, iterations, seed=seed0 + run_idx
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
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument(
        "--delta",
        type=float,
        default=0.5,
        help=(
            "Step size for the first-order (own-gradient) term. The paper's "
            "main text doesn't state a step size for Table 3's NL-Ex/LOLA-Ex "
            "self-play numbers, but its Table 4 caption states one directly "
            "for the closely-related asymmetric-exploitability experiment on "
            "the same IPD environment: 'These experiments were carried out "
            "with a delta of 0.5.' Used here as the best-evidenced value "
            "available, in place of an earlier, purely empirical choice of "
            "0.3. Cross-checked against github.com/alshedivat/lola (the "
            "paper's own released code, not used as a dependency): its "
            "train_exact.py CLI defaults to delta=1.0, which this repo "
            "found overshoots into a spurious full-cooperation fixed point "
            "for the asymmetric pairing (see README's 'What's matched vs. "
            "simplified' and RESULTS.md) -- i.e. the code's own CLI default "
            "is a worse match to Table 4 than Table 4's own stated 0.5."
        ),
    )
    parser.add_argument("--eta", type=float, default=0.5, help="Step size for the LOLA correction term.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="output/run_experiment1_ipd_exact")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    for agent1_type, agent2_type in PAIRINGS:
        key = f"{agent1_type}-{agent2_type}"
        print(f"Training {key} on IPD (exact gradients)...")
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

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(PAIRINGS), figsize=(4 * len(PAIRINGS), 4), sharey=True)
        for ax, (agent1_type, agent2_type) in zip(axes, PAIRINGS):
            key = f"{agent1_type}-{agent2_type}"
            summary = results[key]
            ax.plot(summary["example_reward_history1"], label=f"agent1 ({agent1_type})")
            ax.plot(summary["example_reward_history2"], label=f"agent2 ({agent2_type})")
            ax.set_title(key)
            ax.set_xlabel("iteration")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("normalised avg reward per step")
        fig.tight_layout()
        plot_path = os.path.join(args.output_dir, "reward_curves.png")
        fig.savefig(plot_path, dpi=120)
        print(f"Wrote {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot.")


if __name__ == "__main__":
    main()
