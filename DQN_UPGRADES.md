# DQN upgrades

Four standard improvements to the DQN from [DQN_TUTORIAL.md](DQN_TUTORIAL.md),
each on its own branch so it can be tried alone. Each one is a command-line
flag that is off by default: without the flag, every branch trains exactly
like `main` (same seed, same log, digit for digit).

| Branch | Flag | What changes | Files |
|---|---|---|---|
| `dqn-double` | `--double` | how the target is computed | `dqn.c` |
| `dqn-nstep` | `--nstep N`, `--gamma G` | what goes into the replay buffer | `dqn.c`, `train_dqn.c` |
| `dqn-dueling` | `--dueling` | the network's output layer | `nn.c` |
| `dqn-per` | `--per` | which transitions get replayed | `dqn.c` |
| `dqn-combined` | all of the above | merges the four branches; adds `--eval GAMES` | |
| `dqn-starve` | `--starve-penalty P`, `--features hunger` | the reward and one input; see the end | `train_dqn.c`, `dqn.c`, `snake.c` |

```sh
git switch dqn-double && make && ./train_dqn --features view --double --seed 1
```

The motivation is the plateau at the end of [CNN_TUTORIAL.md](CNN_TUTORIAL.md):
a bigger, better network and a bigger window did not raise the score, so the
limit is more likely in how DQN learns than in what the network can see.

---

## 1. Double DQN (`dqn-double`)

**Problem.** The target is `r + γ · max_a Q_target(s', a)`. Every Q-value has
some error, and the `max` picks the action whose error happens to be the most
positive. So the target is biased upwards, and that bias feeds into the next
targets. This is the "inflated max Q" seen with the 19×19 CNN.

**Fix.** Use two different networks for the two jobs in the `max`: the
online network *chooses* the next action, the target network *values* it.

```
a* = argmax_a Q_online(s', a)
y  = r + γ · Q_target(s', a*)
```

An action whose online value is too high by chance usually isn't also too
high in the target network, so the errors stop adding up. The cost is one
extra forward pass per sample. Code: `dqn_target_double` in `dqn.c`.

## 2. n-step returns (`dqn-nstep`)

**Problem.** A 1-step target moves information back one step per update. A
death 30 steps after the fatal turn has to be passed back through 30
"generations" of targets before the turn's Q-value hears about it. On top of
that, γ = 0.95 shrinks a reward 30 steps away to 0.95³⁰ ≈ 0.21 of its size.

**Fix.** Store the sum of the next n rewards instead of one, and bootstrap
n steps later:

```
y = r_t + γ r_{t+1} + ... + γ^(n-1) r_{t+n-1} + γ^n · max_a Q_target(s_{t+n}, a)
```

The trainer keeps the last n (state, action, reward) triples in a small queue
and only pushes a transition once its n rewards are known. If the snake dies
first, the waiting transitions get the rewards up to the death and no
bootstrap. If it starves (a time limit, not a real ending) they are dropped.
`--gamma` lets the horizon grow too: 0.99 keeps 0.99³⁰ ≈ 0.74 of a reward 30
steps away.

The catch: the n rewards come from the policy that was playing *then*,
including its random ε moves, so long n-step returns are slightly off for
the current policy. Small n (3–5) is the usual compromise.

## 3. Dueling head (`dqn-dueling`)

**Idea.** In most Snake states all three moves are about equally good: what
matters is how good the *state* is. A plain network has to learn that value
three times, once in each output. A dueling network splits it:

```
Q(s, a) = V(s) + A(s, a) − mean_b A(s, b)
```

V(s) is one number shared by all actions, and A(s, a) is how much better
action a is than average. Every training sample updates V, even though it
only has a target for one action. Subtracting the mean makes the split
unique (otherwise V + c and A − c would give the same Q).

Here the head is the cheap version: the last layer simply gets a 4th output
for V, and both share the hidden layers (the paper gives V and A separate
hidden layers). Backprop through the combination is two lines:
`dQ(a)/dV = 1` and `dQ(a)/dA(b) = [b = a] − 1/3`. A gradient check in `test.c`
covers it. Dueling nets are saved with their own file magic (`SNAKEND`), so
"Watch DQN" in the game loads them like any other.

## 4. Prioritized replay (`dqn-per`)

**Problem.** Uniform sampling spends most updates on transitions the network
already predicts well (moving through empty space), and rarely sees the
interesting ones (deaths, eating).

**Fix.** Sample transition i with probability proportional to
`(|TD error_i| + 0.01)^α`, with α = 0.6. New transitions get the largest
priority seen so far, so each is replayed at least once soon. Each sampled
priority is refreshed with its new TD error.

This changes the data distribution, which would bias the learned Q-values, so
each sample's gradient is scaled by an importance weight
`(N · P(i))^(−β)`, normalised so the largest in the batch is 1. β grows
from 0.4 to 1 over training, with full correction by the end.

Sampling in proportion to 50 000 changing priorities needs a **sum tree**: a
binary tree whose leaves are the priorities and whose every parent holds the
sum of its two children. The root is the total. To sample, pick
`u · total` and walk down: go left if it falls inside the left child's sum,
otherwise subtract that sum and go right. Both sampling and updating are
O(log N). Code: `replay_sample` and `tree_set` in `dqn.c`.

---

## Results

`--features view`, 600 000 steps, all other settings as in `train_dqn.c`.
"Score" is the greedy average over 100 games after training; "max Q" is the
logged average of max_a Q(s, a) at the end of training.

| Config | Score (seeds 1 / 2 / 3) | Mean | Max Q at end |
|---|---|---:|---:|
| baseline (`main`) | 59.8 / 52.1 / 58.5 | **56.8** | ~4 |
| `--nstep 3` | 55.0 / 56.5 / 49.5 | 53.7 | 4.3 |
| `--per` | 49.0 / 56.9 / 54.0 | 53.3 | 4.2 |
| `--dueling` | 51.7 / 56.1 / 49.9 | 52.6 | 4.0 |
| `--double` | 51.5 / 53.1 / 51.2 | 51.9 | 3.9 |
| `--nstep 3 --gamma 0.99` | 50.3 / 33.7 / 50.5 | 44.8 | 17 |
| `--gamma 0.99` | 47.8 / 0.0 / 0.0 | 15.9 | 13 / 37 / 44 |

**No upgrade beats the baseline on its own.** Be careful reading more into it
than that. With only 3 seeds, the spread between seeds (the baseline alone
ranges from 52 to 60) is as large as the gaps between rows, so "slightly
worse" is not proven either. Five more seeds per row were started and
stopped before finishing; the table needs them before ranking the rows.

**γ = 0.99 breaks plain DQN.** In two of three runs the snake never learned to
eat (score 0), yet max Q kept climbing, peaking near 96 at 340k steps:

```
   steps episodes  avg(100)    max    eps     loss    max Q
  220000     5715      0.05      3  0.274   2.1866    61.38
  280000     6863      0.01      3  0.076   2.6798    86.55
  340000     7537      0.00      3  0.010   2.1052    96.49
```

No real return in this game is anywhere near 96. This is the overestimation
spiral from section 1: with γ = 0.99 a value can grow up to 1 / (1 − γ) = 100
times the reward before the discount stops it. Each `max` adds a little
upward error, and the bootstrap feeds it back in. The agent learns to loop
safely, which never gets corrected, because starving ends the episode
without a penalty.

**n-step partly rescues it.** With `--nstep 3`, real rewards reach the
Q-values three times faster and anchor them: max Q settles around 17 instead
of 40, and two of three seeds play normally.

**What this suggests for combining.** The longer horizon the plateau seems to
need (section 2) only works if the targets don't run away. That is exactly
what Double DQN is for, and what n-step helps with. So the first combination
to try is **`--double --nstep 3 --gamma 0.99`**. On its own, Double changed
little at γ = 0.95, where max Q stayed small anyway.

Open questions:
- **More seeds:** 5 more per row before ranking the single upgrades. The
  combination below uses 5 seeds.
- **Longer training:** does training much longer (3M steps) help the
  baseline? That run was stopped at 1.1M steps, still exploring
  heavily (ε = 0.27), so it says nothing yet.
- **Starvation penalty:** would a penalty for starving (or treating it as a
  death) stop the "loop forever" policy that γ = 0.99 found?

## Combining: `--double --nstep 3 --gamma 0.99`

Branch `dqn-combined` merges all four branches; the flags can be used
together. It also adds `--eval GAMES`. 500 evaluation games instead of 100
make each score about 2× less noisy. With no flags it still trains exactly
like `main`, so the baseline below was re-run with the same binary: 5 seeds,
500 games each.

| Config | Score (seeds 1–5) | Mean | Max Q at end |
|---|---|---:|---:|
| baseline | 56.6 / 50.7 / 58.8 / 50.6 / 57.3 | **54.8** | ~4 |
| `--double --nstep 3 --gamma 0.99` | 57.4 / 51.1 / 44.9 / 50.1 / 47.9 | 50.3 | ~17 |

**Double DQN fixes the γ = 0.99 blow-up.** All 5 seeds learn normally. With
γ = 0.99 alone, 2 of 3 collapsed, and with n-step alone 1 of 3 fell to 34.
Max Q stays at about 17 throughout. That is plausible for γ = 0.99 rather
than inflated: eating every ~25 steps, forever, is worth
10 · 0.99²⁵ / (1 − 0.99²⁵) ≈ 35, and real games end. The same sum with
γ = 0.95 gives ≈ 4, which is exactly the baseline's max Q.

**The longer horizon does not raise the plateau.** The combination is
4.5 points below the baseline. With a seed-to-seed spread of about ±4 that is
suggestive (about 1.7 standard errors) but not proven. Either way, it is
clearly not the jump that "the agent can't see far enough ahead in time"
predicted. The learning curves rise to about 40 per game during training
(ε = 0.01) by 400k steps, then go flat, like the baseline.

So the main hypothesis from CNN_TUTORIAL.md is now weakened. The plateau did
not move with a better network, a bigger window, or a longer horizon with
stable targets. What's left is mostly outside DQN itself: the reward
(starving costs nothing), the information in the input (the view shows where the body is but not
which parts will move away first, and the 19×19 CNN that saw the whole board
did no better), or a real limit of a greedy one-step policy, which search would
address.

## Starving: a penalty and a hunger input (`dqn-starve`)

Branch `dqn-starve` builds on `dqn-combined` and adds:

- **`--starve-penalty P`:** starving (no food for 100 × length steps) ends the
  episode like a death, with reward −P. Without the flag it stays a time
  limit, with no penalty.
- **`--features hunger`:** the view plus one input, steps since food ÷ the
  starvation limit. Without it the network cannot see the limit coming.
- **In the game:** Watch modes show the same counter ("Hunger: 4 / 300").
  `./snake --dqn FILE` (or `--cnn FILE`) plays any model without copying it
  over `snake_dqn.bin`.

5 seeds, 500 evaluation games each, P = 10 (the same as a death):

| Config | Score (seeds 1–5) | Mean |
|---|---|---:|
| baseline (from above) | 56.6 / 50.7 / 58.8 / 50.6 / 57.3 | **54.8** |
| `--starve-penalty 10` | 49.6 / 50.3 / 45.2 / 49.1 / 51.8 | 49.2 |
| `--features hunger --starve-penalty 10` | 51.3 / 50.2 / 49.3 / 55.7 / 50.4 | 51.4 |
| hunger + penalty + `--double --nstep 3 --gamma 0.99` | 52.1 / 43.1 / 54.1 / 52.0 / 57.1 | 51.7 |
| `--gamma 0.99 --starve-penalty 10` | 1.1 / 0.0 / 0.0 / 35.9 / 0.0 | 7.4 |

How the games end, measured over 200 greedy games per model:

| Model | Avg steps | Died | Starved |
|---|---:|---:|---:|
| baseline, seed 1 | 1264 | 200 | 0 |
| `--starve-penalty 10`, seed 1 | 1051 | 200 | 0 |
| `--gamma 0.99`, seeds 2 and 3 | 101 | 0 | 200 |
| `--gamma 0.99 --starve-penalty 10`, seeds 2 and 3 | 101 | 0 | 200 |

**A trained snake never starves.** Every game of the good models ends in a
death, so starving was never what held the score down. The penalty only
helps where the policy has collapsed into circling, and there it is too
weak.

**The penalty costs about 5 points.** Starving ends at a moment the view
can't predict, so those −10 endings add noise to the values. The hunger
input wins back about 2 of the 5 points. This matches what you'd expect:
the input makes the ending predictable again, but it doesn't remove the
cost.

**The penalty does not stop the γ = 0.99 collapse.** The collapsed models
still circle until they starve, at 101 steps with length 1, and max Q
reached 165 in one run. That is more than the 10 / (1 − 0.99) = 100 that
even a meal on every single step would be worth, so it is pure
overestimation. A −10 penalty can't compete with a value that is wrong by
150. The fix is Double DQN, not the reward.

**Conclusion.** The plateau is about dying, not starving. The next questions
are *how* the snake dies (walls or its own body, and how long before that
the game was already lost), and whether looking a few moves ahead (search)
avoids those deaths.

To reproduce one row, switch to its branch and run, for example:

```sh
git switch dqn-nstep && make train_dqn
./train_dqn --features view --nstep 3 --gamma 0.99 --seed 2
```

Each run takes about 2 minutes on one core (4–5 when the machine is busy). Run a few at a time, not one
per core: dozens at once saturate the machine.
