# Learning with Opponent-Learning Awareness — Foerster et al. (2018)

> **This is a from-scratch paper reproduction, not a ported codebase.** This repo rebuilds Foerster et al. (2018)'s exact-gradient and policy-gradient LOLA experiments directly from the paper's Section 4 equations, with its own PyTorch autograd implementation and no dependency on the [authors' own code](https://github.com/alshedivat/lola) — the original repo was consulted afterward, read-only, purely to cross-check hyperparameters and the second-order correction's construction against an independent source (see "What's matched vs. simplified" and "Known gaps"), never imported or copied from. Sibling repos in this series ([Leibo2017](https://github.com/doesburg11/Leibo2017), [Hughes2018](https://github.com/doesburg11/Hughes2018)) follow the same house style: brutally honest about what's matched, simplified, or doesn't reproduce.

> **Scope: iterated matrix games only.** This repo implements Section 5.1's Iterated Prisoners' Dilemma (IPD) and Iterated Matching Pennies (IMP) experiments, both with exact gradients (Section 4.1/4.2) and with policy-gradient/REINFORCE estimates (Section 4.3). It does **not** implement the Coin Game (Section 5.2, deep recurrent policies), LOLA-DiCE (a separate, later paper), or LOLA with opponent modelling of an unknown opponent's parameters (Section 4.4). See "Known gaps from the paper" below.

> **Headline result: IMP reproduces the paper's claim cleanly; IPD reproduces the qualitative story and, using the paper's own stated step size (Table 4: "delta of 0.5"), lands much closer to its numbers than an earlier, empirically-guessed step size did — though still not exact.** NL-vs-NL converges to mutual defection, LOLA-vs-LOLA converges toward mutual cooperation, and LOLA exploits NL when paired against it — all under exact gradients, matching Fig. 1's qualitative claims and, for the asymmetric pairing, roughly half the remaining gap to Table 4's own magnitudes. On IMP, LOLA converges to the Nash equilibrium with near-zero variance while NL cycles with high variance, closely matching Table 3's own reported standard deviations. The **policy-gradient (LOLA-PG) version is noisier and its asymmetric LOLA-vs-NL pairing only partially, inconsistently reproduces the exact-gradient exploitation result** — see "Known gaps" for real numbers and discussion, not a hidden problem.

A from-scratch replication of:

> Foerster, J., Chen, R. Y., Al-Shedivat, M., Whiteson, S., Abbeel, P., & Mordatch, I. (2018). *Learning with Opponent-Learning Awareness.* Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems (AAMAS 2018), 122-130. arXiv:1709.04326.

The paper's question: in multi-agent reinforcement learning, each agent's environment is non-stationary because the other agents are also learning. A "naive learner" (NL) treats the other agents as a fixed part of the environment and does plain gradient ascent on its own expected return — in the iterated prisoners' dilemma, this reliably converges to mutual defection even though mutual cooperation is available and better for both. LOLA (Learning with Opponent-Learning Awareness) instead has each agent explicitly account for the fact that the other agent is also about to take a gradient step, and differentiates its own expected return *through* that anticipated update — a genuinely second-order method, since it requires the gradient of a gradient. The paper shows this single extra term is enough for two LOLA agents to discover reciprocity (tit-for-tat-like cooperation) in the IPD purely out of self-interest, with no communication and no built-in notion of fairness.

## The mechanism

### Games: memory-one iterated matrix games (Sec. 5.1)

Foerster et al. model an iterated matrix game as a two-agent Markov Reward Process: "the state at time 0 is empty, denoted as $s_0$, and at time $t \geq 1$ is the joint action from $t-1$: $s_t = (u^1_{t-1}, u^2_{t-1})$." Because a memory-1 strategy loses no generality against a memory-1 opponent (Press & Dyson 2012, cited by the paper), each agent's policy is "fully specified by 5 probabilities": the probability of playing its first action (Cooperate for IPD, Heads for IMP) at the start state $s_0$, and in each of the four states CC, CD, DC, DD. Each probability is a sigmoid-squashed scalar logit (`foerster2018/policy.py::MemoryOnePolicy`).

The payoff matrices used are the paper's own Table 1 (IPD) and Table 2 (IMP), transcribed exactly in `foerster2018/games/matrix_game.py`:

**IPD (Table 1)**

|       | C       | D       |
|-------|---------|---------|
| **C** | (-1,-1) | (-3,0)  |
| **D** | (0,-3)  | (-2,-2) |

**IMP (Table 2)**

|       | Head    | Tail    |
|-------|---------|---------|
| **H** | (+1,-1) | (-1,+1) |
| **T** | (-1,+1) | (+1,-1) |

(IPD note: this is a "cost" convention where all four payoffs are <= 0 and higher/less-negative is better — mutual cooperation averages -1/step, mutual defection -2/step, matching the paper's own text: "the average returns per step in self-play are -1 and -2 for TFT and DD respectively.")

### Exact value function (Sec. 5.1, `foerster2018/exact/value.py`)

The paper states it solves the multi-agent MDP analytically but doesn't spell out the matrix algebra in the main text; the derivation used here is the standard stationary-Markov-chain construction for memory-one repeated games (the method Press & Dyson's zero-determinant-strategy literature, cited by the paper, is built on). Writing $w_1$ for the initial distribution over the four joint-outcome states (derived from both agents' $s_0$-probabilities) and $M$ for the 4x4 transition matrix (derived from both agents' per-state probabilities), the exact discounted value for one agent is:

$$V = w_1 \cdot (I - \gamma M)^{-1} \cdot r$$

where $r$ is that agent's length-4 payoff vector. This is implemented with `torch.linalg.solve` (not an explicit matrix inverse), keeping the whole computation a plain, twice-differentiable function of both agents' policy parameters — see `foerster2018/exact/value.py`'s module docstring for the full derivation. `tests/test_value_function.py` checks it against three hand-computable degenerate cases (always-cooperate, always-defect, TFT-vs-always-cooperate) and a zero-sum sanity check on IMP.

### Exact LOLA (Sec. 4.1/4.2, `foerster2018/exact/lola.py`)

Naive Learner (Eq. 4.1): $\theta_{i+1} = \theta_i + \delta \, \nabla_{\theta} V(\theta_i)$ — plain gradient ascent on the agent's own exact value.

LOLA (Eq. 4.2-4.4): a LOLA agent optimises its return under a **one-step lookahead of the opponent's own naive learning step**. Substituting the opponent's naive update ($\Delta\theta_2 = \eta \, \nabla_{\theta_2} V_2$) into a first-order Taylor expansion of $V_1(\theta_1, \theta_2 + \Delta\theta_2)$ and differentiating w.r.t. $\theta_1$ gives:

$$f_{\text{lola}}^1 = \delta \, \nabla_{\theta_1} V_1(\theta_1, \theta_2) + \delta \eta \, \big(\nabla_{\theta_2} V_1(\theta_1, \theta_2)\big)^T \nabla_{\theta_1}\nabla_{\theta_2} V_2(\theta_1, \theta_2)$$

The second term is the whole point of LOLA: $\nabla_{\theta_1}\nabla_{\theta_2} V_2$ differentiates *through the opponent's own gradient*, not just through $V_1$ — a genuine second-order (mixed-Hessian) quantity. As instructed, this is **not hand-derived**; it's computed with two calls to `torch.autograd.grad(..., create_graph=True)` (to get $\nabla_{\theta_2} V_1$ and $\nabla_{\theta_2} V_2$ as differentiable functions of $\theta_1$) followed by the standard "dot-product-then-differentiate" trick, which recovers the exact Hessian-vector product without ever materialising a Hessian matrix. The paper is explicit that "the dependency of $\nabla_{\theta_2} V_1(\theta_1, \theta_2)$ on $\theta_1$ is dropped during the backward pass" — a deliberate simplification stated in the paper itself, implemented here as one `.detach()` call (`lola.py`'s `lola_correction`). **This exact implementation was independently reviewed by Codex** (see "Independent review" below) and confirmed to genuinely differentiate through the opponent's simulated gradient step, not collapse to a first-order update.

### Policy-gradient LOLA (Sec. 4.3, `foerster2018/policy_gradient/`)

When agents can't evaluate $V_1$/$V_2$ and their gradients exactly (the realistic deep-RL setting), the paper derives score-function (REINFORCE) estimators of the same quantities from sampled rollouts. `foerster2018/policy_gradient/rollout.py` samples batched episodes; `foerster2018/policy_gradient/lola_pg.py` implements the paper's own Eq. 4.5 (naive PG update, reward-to-go with a per-timestep batch-mean baseline for variance reduction) and Eq. 4.6-4.7 (the PG estimator of the LOLA cross term and the resulting LOLA-PG update). Both are implemented as **explicit tensor arithmetic over the rollout's per-step score-function values**, not by building a surrogate loss and calling `loss.backward()` twice — see `lola_pg.py`'s module docstring for why that shortcut is a known trap here (a naive doubly-differentiated surrogate silently drops terms Eq. 4.6 requires; correctly handling this in general is the subject of the separate, out-of-scope LOLA-DiCE paper).

A real bug was caught and fixed during development this way: the first-order estimator originally omitted a $\gamma^t$ factor the paper's own derivation requires (it's easy to miss because the reward-to-go term $R_t(\tau)$ already contains a *different*, relatively-discounted factor $\gamma^{l-t}$). This inflated the naive gradient's magnitude by roughly 4x at $\theta=0$, caught by directly comparing the PG estimator against the closed-form exact gradient at the same parameters (`tests/test_policy_gradient.py::test_reinforce_grad_matches_exact_gradient_at_uniform_policy`) — after the fix, the two agree to within Monte Carlo noise.

## What's matched vs. simplified

**Matched, from the paper's own stated design:**
- The exact game definitions (Table 1 IPD, Table 2 IMP) and the memory-one, 5-probability policy parameterization (Sec. 5.1), transcribed and cross-checked directly against the primary source PDF, not from memory.
- The exact value function's role in Sec. 4.1/4.2's NL and LOLA update rules, Eq. 4.1-4.4, including the paper's own explicit statement that the $\nabla_{\theta_2} V_1$ dependency on $\theta_1$ is dropped in the LOLA correction term.
- The policy-gradient estimators of Eq. 4.5-4.7, including the paper's stated reward-to-go-with-baseline construction for the first-order term and the un-baselined double-cumulative-score construction for the second-order (Eq. 4.6) term.
- $\gamma = 0.96$ for IPD, $\gamma = 0.9$ for IMP (Sec. 5.3: "we found that a lower gamma produced more stable learning on IMP").
- All four (NL, LOLA) x (NL, LOLA) pairings, as the paper's own Table 4 does for the exact-gradient case (Table 3 itself only reports the two symmetric self-play settings, NL-vs-NL and LOLA-vs-LOLA).
- **$\delta = \eta = 0.5$ for the exact-gradient IPD experiment**, quoting the paper's own Table 4 caption directly: "These experiments were carried out with a delta of 0.5." Table 3's own NL-Ex/LOLA-Ex self-play numbers don't state a step size, but Table 4 (the closely-related asymmetric-exploitability experiment, same environment) does, and it's the best-evidenced value available — used here in place of an earlier, purely empirical choice of 0.3. See "Known gaps" below for how this was found and what changed as a result.

**Simplified / not stated by the paper, filled in here (documented, not hidden):**
- **The exact-gradient experiment's iteration count and per-seed step-size stability weren't independently confirmed against the paper**: at $\delta=\eta=0.5$ this repo's own dynamics need roughly 2000 iterations to settle (an earlier attempt at 600 iterations with a different, empirically-chosen $\delta=0.3$ was based on a step size no longer used). The paper doesn't state an iteration count for Table 3/4's exact experiments (Figure 1's x-axis appears to run to several thousand iterations, consistent with this). Cross-checked against the paper's own released code (github.com/alshedivat/lola, read for reference only, not used as a dependency): its `train_exact.py` CLI defaults to `delta=lr_correction=1.0`, `trace_length=200` — this repo tested that exact configuration and found it collapses the asymmetric LOLA-vs-NL pairing into a spurious mutual-near-full-cooperation fixed point (~-1.0/-1.0 for both agents) rather than Table 4's own asymmetric (-1.54, -1.28), i.e. the released code's own CLI default reproduces Table 4 *worse* than Table 4's own stated $\delta=0.5$. This repo's `foerster2018/exact/lola.py` construction was independently cross-checked against both `alshedivat/lola`'s `train_exact.py::corrections_func` and its `tournament.py`'s `ExactLOLA._build_update` (two independently-written implementations in that repo) and matches both: the same "stop-gradient a first-order term, then dot-product-then-differentiate" construction for the second-order correction.
- **The policy-gradient experiment's baseline is a simple per-timestep batch-mean baseline**, not the paper's own learned critic ("we train agents with an actor-critic method... a policy actor and -critic for variance reduction," Sec. 5.3). A batch-mean baseline is a valid, unbiased REINFORCE baseline but a weaker variance reducer than a learned critic, which likely explains why this repo's policy-gradient step sizes (also $\delta = \eta = 0.3$, `batch_size` in the hundreds-to-low-thousands) had to be tuned independently of the paper's own $\delta = 0.005$/`batch_size = 4000` and don't reproduce the paper's exact numbers (see "Known gaps" below).
- **The LOLA-PG cross term (Eq. 4.6) has no baseline/variance reduction at all**, matching the paper's own equation literally (it doesn't show one for this term either) but leaving it a comparatively high-variance estimator — this repo's own honest read of "the policy gradient finding is... noisier" (paper, Sec. 6.1).
- **"%TFT-like" and "distance from Nash" are this repo's own operational metrics**, not quotes of the paper's exact classification rule (Table 3's "%TFT"/"%Nash" columns don't state their thresholds in the main text). `is_tft_like()` in the experiment scripts checks `P(C|s0) > 0.5 and P(C|CC) > 0.5 and P(C|DD) < 0.5`; IMP's "distance from Nash" is the mean absolute deviation of all 5 probabilities from 0.5. See "Known gaps" for where this heuristic visibly disagrees with a more careful reading (e.g. it can misclassify an exploited, mostly-cooperative NL agent as "TFT-like" even though it isn't reciprocating anything).
- **Random uniform-ish initialization** ($\theta \sim U(-0.1, 0.1)$, i.e. probabilities near 0.5) for every run, rather than whatever initialization scheme (if any) the paper used — not stated in the main text.

## Known gaps from the paper

All numbers below are from actual runs of this repo's own scripts (`output/run_experiment*/results.json`), not hypothetical. Commands to reproduce are in "Running it" below.

**IMP (`run_experiment3_imp.py`) reproduces the paper's claims cleanly**, both qualitatively and quantitatively close on the one number that's directly comparable (Table 3's IMP standard deviation):

| pairing | agent | mean reward/step | dist. from Nash | tail instability |
|---|---|---|---|---|
| NL vs NL | agent1 | -0.116 (std 0.395) | 0.469 | 0.0163 |
| NL vs NL | agent2 | +0.116 (std 0.395) | 0.475 | 0.0193 |
| LOLA vs LOLA | agent1 | ~0.000 (std 1.5e-8) | 2.9e-8 | 2.0e-8 |
| LOLA vs LOLA | agent2 | ~0.000 (std 1.5e-8) | 2.3e-8 | 2.0e-8 |

Paper's Table 3: NL-Ex R(std) = 0(0.37), LOLA-Ex R(std) = 0(0.02). This repo's NL-NL std (0.395) and LOLA-LOLA std (essentially 0) both land extremely close to the paper's own numbers, and the distance-from-Nash / tail-instability metrics confirm the qualitative story directly: NL-NL never gets close to the 50/50 Nash point (dist ~0.47) and keeps moving (tail instability two orders of magnitude above LOLA-LOLA's), while LOLA-LOLA converges essentially exactly to Nash and stays there. A genuinely strong reproduction, and the strongest quantitative match anywhere in this repo. LOLA-NL / NL-LOLA (mixed pairings, not tested by the paper's own Table 3 but included here for completeness) also converge close to Nash (dist ~0.013-0.014) — one LOLA agent alone is enough to pull both toward the equilibrium.

**IPD, exact gradients (`run_experiment1_ipd_exact.py`, 50 runs, 2000 iterations, $\delta=\eta=0.5$ — the paper's own stated Table 4 step size, not a guess) reproduces the qualitative story and is materially closer to the paper's own numbers than an earlier, empirically-chosen step size was:**

| pairing | agent | mean reward/step | %TFT-like |
|---|---|---|---|
| NL vs NL | both | -2.000 (std ~0) | 0% |
| LOLA vs NL | LOLA | -0.941 (std 0.207) | 74% |
| LOLA vs NL | NL | -1.260 (std 0.438) | 92% |
| NL vs LOLA | NL | -1.174 (std 0.358) | 76% |
| NL vs LOLA | LOLA | -0.973 (std 0.183) | 76% |
| LOLA vs LOLA | agent1 | -1.150 (std 0.145) | 74% |
| LOLA vs LOLA | agent2 | -1.118 (std 0.134) | 48% |

Paper's Table 3: NL-Ex %TFT=20.8, R=-1.98(0.14); LOLA-Ex %TFT=81.0, R=-1.06(0.19). Paper's Table 4 ($\delta=0.5$, the same value used here): NL-Ex-vs-LOLA-Ex = `(-1.54, -1.28)`.

- **NL-NL and LOLA-LOLA mean rewards remain close**: -2.000 vs the paper's -1.98/-1.99 (Tables 3 and 4 agree with each other here too), -1.13 to -1.15 here vs the paper's -1.04 to -1.06.
- **The LOLA-vs-NL asymmetry is now directionally and much more closely matched, though still not exact.** Using the paper's own stated $\delta=0.5$ (previously this repo used an empirically-chosen $\delta=0.3$, which gave `(LOLA, NL) = (-0.736, -1.940)` — bigger than the paper's own effect size): this repo's `(LOLA, NL)` is now `(-0.941, -1.260)` to `(-0.973, -1.174)` depending on which agent-label plays LOLA, versus the paper's Table 4 `(-1.28, -1.54)` (LOLA, NL) — the direction is right (LOLA still does better than NL) but this repo's gap is now *smaller* than the paper's, the opposite miss from before (previously bigger). Neither the too-big nor the too-small version is a resolved match; using the paper's own confirmed step size closed roughly half the gap to Table 4's magnitudes without eliminating it. The remaining difference is most likely the iteration count (not stated by the paper for this specific table; 2000 was chosen here based on when this repo's own dynamics settle) and/or true seed-to-seed variability at only 50 seeds — not independently isolated.
- **The update rule was cross-checked against the paper's own released code and found to match**, ruling out a formula-level explanation for the remaining gap: `alshedivat/lola`'s `train_exact.py::corrections_func` and `tournament.py::ExactLOLA._build_update` (read for reference, not used as a dependency) both build the second-order correction the same way this repo does — a first-order cross-gradient held constant (`tf.stop_gradient`, functionally equivalent to this repo's `.detach()`) dotted against the opponent's own naive gradient, then differentiated once more. Their own CLI default step size ($\delta=1.0$) was also tested directly against this repo's implementation and reproduces Table 4 *worse* (a spurious symmetric ~-1.0/-1.0 fixed point) than Table 4's own stated $\delta=0.5$ does — see "What's matched vs. simplified" above.
- **%TFT-like is much closer to the paper now too**, as a side effect of the larger, better-justified step size (this metric wasn't the target of the change, but moved anyway): LOLA-LOLA reaches 48-74% here vs. the paper's 81%, much closer than the previous config's 50-56%. NL-NL's 0% (vs. the paper's 20.8%) remains the one clearly unresolved metric — this repo's NL-NL converges to essentially the exact same defection point on every seed (std ~0), unlike the paper's own reported spread (0.14), most likely a real difference in initialization distribution rather than a labelling artifact, not isolated further.

**IPD, policy gradient (`run_experiment2_ipd_policy_gradient.py`, 20 runs, 300 iterations, batch=1024, horizon=100, $\delta=\eta=0.3$, post leave-one-out-baseline fix — see "Independent review") reproduces LOLA-vs-LOLA reasonably but the mixed pairing only partially, inconsistently shows LOLA exploiting NL:**

| pairing | agent | mean reward/step | %TFT-like |
|---|---|---|---|
| NL vs NL | both | -1.999 | 0% |
| LOLA vs NL | LOLA | -1.875 (std 0.309) | 5% |
| LOLA vs NL | NL | -1.877 (std 0.307) | 15% |
| NL vs LOLA | NL | -1.770 (std 0.362) | 30% |
| NL vs LOLA | LOLA | -1.566 (std 0.501) | 15% |
| LOLA vs LOLA | agent1 | -1.283 (std 0.352) | 50% |
| LOLA vs LOLA | agent2 | -1.250 (std 0.352) | 35% |

Paper's Table 3: NL-PG %TFT=20.0, R=-1.98(0.00); LOLA-PG %TFT=66.4, R=-1.17(0.34).

- **LOLA-LOLA is a genuinely good match**: -1.25/-1.28 here vs the paper's -1.17, with a comparable spread (0.35 here vs 0.34 in the paper).
- **NL-NL matches almost exactly**: -1.999 here vs -1.98 in the paper.
- **The mixed LOLA-vs-NL pairing only partially reproduces the asymmetric-exploitation effect, and inconsistently across seeds.** Both mixed pairings now show *some* escape from mutual defection (means -1.57 to -1.88, vs. -1.999 for pure NL-NL) with a consistent minority of seeds landing TFT-like (5-30%) — a weaker, noisier version of the exact-gradient result (Section 3a there: -0.74/-1.94, 0%/100%), not its absence. This table reflects a fix to a real estimator bug caught by independent review after the numbers below were first reported (a REINFORCE baseline that included each trajectory's own reward-to-go in its own baseline, biasing the gradient estimate by a `(B-1)/B` shrinkage factor — negligible in isolation at these batch sizes, but the mixed pairing's weak, marginal signal turned out to be exactly the result most sensitive to it: the original 5-seed, pre-fix run showed one pairing collapsing to defection in all 5 seeds and the mirror pairing escaping in only 1 of 5). Since the paper's own Table 3 doesn't report a mixed-pairing PG result at all (only the two symmetric settings), this isn't a contradiction of a specific published number — but it remains a real, honest limitation relative to the exact-gradient version's clean, majority-of-seeds exploitation. Likely contributors, none isolated as *the* cause: the leave-one-out baseline is still not a learned critic (weaker variance reduction than the paper's own actor-critic setup), a same-batch covariance bias in the LOLA cross term that's documented but not fixed (see "Independent review"), and $\delta=\eta=0.3$ not being tuned per-pairing. Not investigated further within this repo's scope.
- Other explicitly out-of-scope items (decided before this repo was built, not discovered as gaps): the Coin Game (Sec. 5.2, deep recurrent policies over a spatial task), LOLA-DiCE (a separate follow-up paper), LOLA with opponent modelling of an unknown opponent's parameters (Sec. 4.4), higher-order LOLA (Sec. 4.5 / Table 4's "2nd-Order" column), and the round-robin tournament against other multi-agent algorithms (Sec. 6.1, Fig. 4).

## Independent review

Per this project's standing practice, `codex exec` (OpenAI Codex CLI) independently reviewed the exact-LOLA implementation for the specific failure mode this kind of second-order method is prone to: silently collapsing to a first-order update. Its verbatim conclusion:

> No correctness bug found in `foerster2018/exact/`: the LOLA term does differentiate through the opponent's gradient step. It does not silently collapse to a first-order/naive update... The detach is deliberate and only removes the paper-dropped extra term involving `d/dtheta1(dV1/dtheta2)`... I also ran a direct probe: the implemented correction matched an independent Hessian-vector-product construction exactly at the tested point (`max_abs_impl_minus_manual = 0.0`).

A second, supplementary Codex review was attempted on `foerster2018/policy_gradient/` (to independently check the $\gamma^t$ fix described above) but could not complete at first: the Codex CLI's then-configured default model (`gpt-5.5`) returned a `404 model does not exist` error on every attempt. This was a transient account/model-rotation issue, not a permanent restriction — re-run later the same day against the account's new default (`gpt-5.6-sol`, reasoning effort `low`), the review completed and found two real issues in `foerster2018/policy_gradient/lola_pg.py`:

> The empirical baseline makes the REINFORCE estimator biased for finite batches... `baseline = reward_to_go.mean(dim=1, keepdim=True)` [includes] the same trajectory whose score is multiplied by it... `E[S_i(R_i-bar R)]` has expectation `(1-1/B) E[S_i R_i]`... Thus `batch_size=1` always produces an exactly zero gradient, and all finite-batch estimates are shrunk by `(B-1)/B`.
>
> The LOLA correction is a biased product of two estimates derived from the same rollout... `E[XY] = HG + Cov(H,g)`... Independent rollout batches for the two factors would remove this covariance bias.

Both were verified directly against the code (not accepted on Codex's word alone) before acting: the baseline issue was real and cheap to fix, so it was fixed (switched to a leave-one-out baseline; `pytest tests/ -q` still passes 19/19) — see "Policy-gradient LOLA" above and RESULTS.md Section 2b for the exact math and why the IPD-PG mixed-pairing numbers above changed materially as a result. The same-batch covariance bias is real but its correct fix (independent batches per Eq. 4.7 factor) would double rollout cost and invalidate every reported PG number without re-running them, so it's documented in `lola_pg.py`'s module docstring rather than fixed here. Codex also flagged device/dtype hygiene issues (rollout tensors default to CPU/float32 regardless of the input policy's device/dtype) — low-priority for this repo's CPU-only, float32-only usage, not acted on. See RESULTS.md for the full timeline of both reviews.

## Running it

```bash
git clone <this-repo>
cd Foerster2018
./create_conda_env.sh
conda activate ./.conda
# or: pip install -r requirements.txt into any Python 3.11 environment

pytest tests/ -q
```

Every `run_experiment*.py` script defaults to a small/fast configuration (a smoke test of the full pipeline, seconds to a couple of minutes) and accepts flags to scale up to the runs reported above:

```bash
python run_experiment1_ipd_exact.py                                     # smoke test (5 seeds, 300 iters)
python run_experiment1_ipd_exact.py --num-runs 50 --iterations 600      # this README's numbers

python run_experiment2_ipd_policy_gradient.py                                              # smoke test
python run_experiment2_ipd_policy_gradient.py --num-runs 5 --iterations 300 --batch-size 1024   # this README's numbers

python run_experiment3_imp.py                                # smoke test (5 seeds, 200 iters)
python run_experiment3_imp.py --num-runs 50 --iterations 400 # this README's numbers
```

Each writes `results.json` (raw per-pairing numbers and one example run's full training curve) to `output/<script-name>/`; `run_experiment1_ipd_exact.py` also saves two plots if matplotlib is available: `reward_curves.png` (mean reward/step vs. iteration, one panel per pairing) and `phase_portrait.png` (agent1's vs. agent2's $P(C \mid s_0)$ — the opening-move probability only, not the full 5-probability policy — over training, one example run per pairing on shared axes; this repo's own visualization, not a reproduction of the paper's own Fig. 1, and not a substitute for the %TFT-like/mean-reward numbers reported above since opening-move behavior alone doesn't determine whether an agent reciprocates).

## Tests

`tests/` covers, in dependency order: the exact value function (three hand-computable degenerate cases plus an IMP zero-sum sanity check, and a check that the value function supports the `create_graph=True` double backward LOLA needs), the exact LOLA correction term (reduces to the naive update at $\eta=0$, is not just a rescaled copy of the naive gradient, genuinely depends on the opponent's own parameters, and — an actual training-scale check, not just a one-step check — that 200 iterations of exact NL-vs-NL self-play on the IPD collapses toward mutual defection), the policy-gradient score function (checked directly against `torch.autograd.grad` on the log-probability, both for cooperate and defect), the PG naive-gradient estimator (checked against the closed-form exact gradient at a shared parameter point — the same check that caught the $\gamma^t$ bug described above), the LOLA-PG correction's correlation with the exact LOLA correction at the same parameters, and end-to-end training smoke tests for all three experiment types. Run with `pytest tests/ -q` (19 tests, all passing).

## Acknowledgments

Developed with AI coding assistance from [Claude](https://claude.com/claude-code) (Anthropic), which does the implementation, with [Codex](https://openai.com/codex) (OpenAI) acting as an independent second opinion, peer-reviewing Claude's nontrivial code changes.

## References

- Foerster, J., Chen, R. Y., Al-Shedivat, M., Whiteson, S., Abbeel, P., & Mordatch, I. (2018). [Learning with Opponent-Learning Awareness](https://arxiv.org/abs/1709.04326). Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems (AAMAS 2018), 122-130.
- Press, W. H., & Dyson, F. J. (2012). [Iterated Prisoner's Dilemma contains strategies that dominate any evolutionary opponent](https://doi.org/10.1073/pnas.1206569109). PNAS, 109(26), 10409-10413. (Zero-determinant strategies; cited by the paper for the memory-one-loses-no-generality result this repo's game formulation relies on.)
- Lerer, A., & Peysakhovich, A. (2017). [Maintaining cooperation in complex social dilemmas using deep reinforcement learning](https://arxiv.org/abs/1707.01068). (Introduces the Coin Game, out of scope here — see paper's Sec. 5.2.)
- Foerster, J., Farquhar, G., Al-Shedivat, M., Rocktäschel, T., Xing, E., & Whiteson, S. (2018). [DiCE: The Infinitely Differentiable Monte Carlo Estimator](https://arxiv.org/abs/1802.05098). ICML 2018. (The separate follow-up paper this repo's LOLA-PG module explicitly does not implement — see `foerster2018/policy_gradient/lola_pg.py`'s module docstring.)
- Foerster, J., Al-Shedivat, M., et al. [alshedivat/lola](https://github.com/alshedivat/lola) — the paper's own released code. Not a dependency of this repo; consulted read-only after the initial build to cross-check the second-order correction's construction and hyperparameters (see "What's matched vs. simplified" and RESULTS.md Section 2c) — this is where this repo found the paper's own stated $\delta=0.5$ for Table 4, missed on the first read of the paper.
