# Deep Q-Networks: a hands-on tutorial

This tutorial continues [RL_TUTORIAL.md](RL_TUTORIAL.md). There we built a Q-table agent.
Here we replace the table with a small neural network, written from scratch in C, and
train it with the **DQN** algorithm (Deep Q-Network, Mnih et al., 2015).

You should already know states, actions, rewards, Q-values, the Bellman target and ε-greedy
from the first tutorial. You don't need to know anything about neural networks.

The code:

| File           | What it does                                                      |
|----------------|-------------------------------------------------------------------|
| `nn.c`, `nn.h` | the neural network: forward pass, backprop, Adam, save/load       |
| `dqn.c`, `dqn.h` | inputs for the network, replay buffer, one training step        |
| `train_dqn.c`  | the training loop                                                 |
| `test.c`       | includes a gradient check that proves backprop is correct         |

```bash
make
./test
./train_dqn --features small    # same inputs as the Q-table
./train_dqn --features view     # a 9x9 window around the head
./train_dqn --features tail     # the window plus where the tail is (section 10)
./snake                         # pick "Watch DQN(6)"
```

---

## 1. Why the table stops working

The Q-table agent levels off at about 40 food per game. Section 9 of the first tutorial
explained why: its state shows only the 3 × 3 block around the head, so a pocket formed by
its own body looks safe until the snake is trapped inside it.

The obvious fix is a bigger view. Say the agent should see a 9 × 9 window around its head,
with each cell either free or blocked. That is 81 bits, so 2⁸¹ ≈ 2.4 × 10²⁴ possible
states. A table with one row per state can't be stored, and even if it could, the agent
would almost never see the same state twice.

There is a second, subtler problem. A table treats every state as completely separate.
If the agent has learned "wall straight ahead → turn" in one state, that knowledge does
not help in a state that differs only in a cell far behind the snake. Each row must be
learned from scratch. That's why the 14-bit table needed far more episodes than the
11-bit one (first tutorial, section 9).

What we want is a Q-function that:

* accepts a big input without needing memory for every possible input, and
* **generalises**: similar inputs give similar Q-values, so learning in one situation
  helps in others.

That's exactly what a neural network does.

---

## 2. From a table to a function

The table was a lookup: `Q[s][a]`. Now Q becomes a **function with parameters θ**:

```
Q(s; θ) → three numbers: Q(s, straight), Q(s, left), Q(s, right)
```

The state goes in as a list of numbers (the *features*), and out come the three Q-values,
one per action. Picking an action works the same as before: take the largest.

Learning changes. In the table, an update changed one cell:

```c
t->q[s][a] += alpha * (target - t->q[s][a]);
```

Now there is no cell to change. Instead we nudge the parameters θ so that `Q(s, a; θ)` moves
a bit towards the target. Because all states share the same parameters, this also moves
the Q-values of *similar* states. That's the generalisation we wanted, and also the source
of new problems (sections 6 and 7).

The target itself is the same Bellman target as before:

```
target = r + γ · max_a' Q(s', a')        (just r if the game ended)
```

---

## 3. A neural network from scratch

### One layer

A layer takes `n_in` numbers and produces `n_out` numbers. Every output is a weighted sum
of all inputs plus a bias:

```
out[o] = b[o] + w[o][0]·in[0] + w[o][1]·in[1] + ... + w[o][n_in-1]·in[n_in-1]
```

The weights `w` and biases `b` are the parameters θ. In `nn.c` the weights of a layer are
stored in one flat array, row by row, so `w[o][i]` is `w[o * n_in + i]`.

### Stacking layers, and why ReLU

If you stack weighted sums, you just get another weighted sum, so ten layers could do no
more than one. To make depth useful, each hidden layer applies a simple non-linear
function to its outputs. We use **ReLU**:

```
relu(z) = max(0, z)
```

It looks too simple to matter, but it lets the network bend: the network can compute
things like "this cell is blocked AND that one is too", which a single weighted sum
cannot.

Our network has three layers:

```
inputs (14 or 167) → 128 hidden (ReLU) → 128 hidden (ReLU) → 3 outputs (no ReLU)
```

The output layer has no ReLU, because Q-values can be negative (dying costs −10).

### The forward pass

Computing the outputs from the inputs is called the *forward pass*. It is just the formula
above in a loop (from `nn_forward` in `nn.c`):

```c
for (int l = 0; l < NN_LAYERS; l++) {
    int nin = net->n[l], nout = net->n[l + 1];
    const float *in = c->a[l];
    float *out = c->a[l + 1];
    for (int o = 0; o < nout; o++) {
        const float *row = net->w[l] + o * nin;
        float sum = net->b[l][o];
        // Lets the compiler add up several products in parallel (SIMD).
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < nin; i++) {
            sum += row[i] * in[i];
        }
        // ReLU on hidden layers, nothing on the output layer.
        out[o] = (l < NN_LAYERS - 1 && sum < 0.0f) ? 0.0f : sum;
    }
}
```

The results of every layer are kept in an `NNCache` (`c->a[l]`), because the backward pass
needs them.

The `#pragma` line is only about speed. Floating-point addition isn't exactly
associative, so by default the compiler must add the products one after another. The
pragma (enabled by `-fopenmp-simd` in the Makefile) allows it to keep several partial
sums at once, which made training about twice as fast.

### Initialisation

The weights start random, not zero. If all weights were equal, every neuron in a layer
would compute the same thing and receive the same update, forever. `nn_init` uses **He
initialisation**: uniform in ±√(6 / n_in). Dividing by the number of inputs keeps the
outputs of each layer at a sensible size, no matter how wide the layer is.

---

## 4. Loss and backpropagation

### The loss

To train, we need a single number that says how wrong the network is: the **loss**. For one
transition (s, a, r, s'):

```
error = Q(s, a; θ) − target
```

The simplest loss would be ½ · error². We use the **Huber loss**, which is the same for small
errors and grows only linearly for big ones:

```
loss = ½ · error²         if |error| ≤ 1
       |error| − ½        otherwise
```

Why? Early in training the targets are often far off (the first time the agent dies, the
error is about 10). With a squared loss, one such transition makes a gradient 10 times
bigger than usual and can throw the network off. With Huber, the gradient is capped at ±1.

Note that only **one** of the three outputs appears in the loss: the action that was
actually taken. We don't know what the other actions would have led to, so they get no
error signal from this transition.

### Gradient descent

The *gradient* of the loss tells us, for every parameter, how much the loss changes if that
parameter grows a little. Moving every parameter a small step against its gradient reduces
the loss:

```
θ ← θ − learning_rate · ∂loss/∂θ
```

The question is how to compute ∂loss/∂θ for ~20 000 parameters efficiently. That is what
backpropagation does.

### Backpropagation

Backpropagation is the chain rule applied layer by layer, from the output back to the
input. Call `delta[o]` the derivative of the loss with respect to the weighted sum of
output `o` of the current layer (before the ReLU).

**Start at the top.** Only the taken action's output is in the loss, so:

```
delta[action] = ∂loss/∂Q(s, a) = error   (clipped to ±1 by Huber)
delta[other]  = 0
```

**Gradients of the parameters.** Since `out[o] = b[o] + Σ w[o][i]·in[i]`:

```
∂loss/∂b[o]    = delta[o]
∂loss/∂w[o][i] = delta[o] · in[i]
```

**Pass the error down.** Each input `in[i]` affected every output `o` through `w[o][i]`, so:

```
∂loss/∂in[i] = Σ_o w[o][i] · delta[o]
```

and `in[i]` was the result of a ReLU in the layer below. The slope of ReLU is 1 where it
was positive and 0 where it was cut off, so that layer's delta is:

```
delta_below[i] = ∂loss/∂in[i]   if in[i] > 0
                 0              otherwise
```

Repeat until the first layer. Here is the code (from `nn_backward`):

```c
float delta[NN_MAX] = {0};
float prev[NN_MAX];
delta[action] = dq;

for (int l = NN_LAYERS - 1; l >= 0; l--) {
    int nin = net->n[l], nout = net->n[l + 1];
    const float *in = c->a[l];

    // Gradients of this layer's parameters:
    //   dLoss/db[o]   = delta[o]
    //   dLoss/dw[o,i] = delta[o] * in[i]
    for (int o = 0; o < nout; o++) {
        if (delta[o] == 0.0f) {
            continue;
        }
        net->gb[l][o] += delta[o];
        float *grow = net->gw[l] + o * nin;
        for (int i = 0; i < nin; i++) {
            grow[i] += delta[o] * in[i];
        }
    }
    if (l == 0) {
        break; // no gradient needed for the inputs
    }

    // Pass the error one layer down ... then through the ReLU.
    ...
    for (int i = 0; i < nin; i++) {
        delta[i] = in[i] > 0.0f ? prev[i] : 0.0f;
    }
}
```

The gradients are **added** to `gw`/`gb`. We train on a batch of 32 transitions, call
`nn_backward` once per transition, and then take one step with the average.

### Proving it's right: the gradient check

Backprop bugs are sneaky: the network still trains, just badly, and you never find out
why. So `test.c` checks it against a slow but obviously correct method. For every weight:

```
numerical gradient ≈ (loss(w + h) − loss(w − h)) / (2h)
```

and that must match what `nn_backward` computed (`test_gradient_check`). To see that the
test has teeth, change

```c
delta[i] = in[i] > 0.0f ? prev[i] : 0.0f;
```

to `delta[i] = prev[i];` (forgetting the ReLU's slope) and run `make test && ./test`:

```
FAIL test.c:...: worst < 1e-2f
```

Whenever you change `nn_backward`, run this test.

---

## 5. Adam, in one page

Plain gradient descent uses the same step size for every parameter. That works, but it's
slow when some parameters get large gradients and others tiny ones. **Adam** keeps two
running averages per parameter:

* `m`: the average gradient (which direction to go, with some momentum), and
* `v`: the average *squared* gradient (how big the gradients usually are).

The step is `lr · m / √v`. Dividing by √v means every parameter moves by about the
learning rate, whatever the scale of its gradients. From `nn.c`:

```c
m[i] = ADAM_B1 * m[i] + (1.0f - ADAM_B1) * gi;
v[i] = ADAM_B2 * v[i] + (1.0f - ADAM_B2) * gi * gi;
float mhat = m[i] / c1; // bias correction: m and v start at 0
float vhat = v[i] / c2;
p[i] -= lr * mhat / (sqrtf(vhat) + ADAM_EPS);
```

The "bias correction" fixes the first few steps: `m` and `v` start at 0, so without it the
early averages would be too small. You don't need to understand Adam in detail to follow
the rest. It's simply a better `θ ← θ − lr · gradient`.

---

## 6. Why naive "neural Q-learning" falls apart: the replay buffer

The most direct translation of the tabular loop would be: play one step, do one gradient
step on that transition, repeat. This works badly, for two reasons.

**Consecutive transitions are nearly identical.** The snake moves one cell per step. If you
train on steps in order, the network sees 50 almost equal examples in a row, overfits to
"the current situation" and forgets the rest. Gradient descent assumes the examples are
roughly independent, and these aren't.

**Each transition is used once and thrown away.** Dying is rare, and each death carries a lot
of information. Using it once is a waste.

The fix is the **replay buffer**: keep the last 50 000 transitions `(s, a, r, s', done)` and
train on a random **batch** of 32 drawn from it. Random sampling breaks up the
correlation, and each transition is reused many times.

```c
// 1. act and observe, 2. remember
int a = dqn_act(&online, s, eps, &rng);
Event ev = game_step_rel(&g, (Action)a);
dqn_features(feat, &g, s2);
int done = (ev == EV_DIED || ev == EV_WON);
replay_push(&rb, s, a, reward(ev), s2, done);
```

Two small details in `dqn.c`:

* The buffer is a **ring**: when it's full, the newest transition overwrites the oldest.
* States are stored as **bytes**, not floats. Every feature is between 0 and 1, so
  `x * 255` fits in a byte. All features are exactly 0 or 1 except the snake length,
  which gets rounded to the nearest 1/255. That's harmless, and it saves 3/4 of the memory.

Replay only works because Q-learning is **off-policy**: the Bellman target uses
`max_a' Q(s', a')`, the best action, not the action the agent actually took. So it doesn't
matter that old transitions were collected by an older, worse policy.

---

## 7. The target network

There is a second problem that the table didn't have. Look at the target again:

```
target = r + γ · max_a' Q(s', a'; θ)
```

The target is computed by the same network we are training. Every update to θ also moves
the target. And because the network generalises, raising Q(s, a) usually raises
Q(s', ·) too, since s and s' differ by one step. The agent chases a target that runs
away from it, and Q-values can spiral upwards.

The fix: keep a **second copy** of the network, the *target network* θ⁻, and compute targets
with it. It's frozen, and every `TARGET_SYNC` = 1000 steps we copy the online network into
it:

```c
// The target comes from the *target* network and is treated as a
// constant: no gradient flows through it.
nn_forward(target, s2, &cn);
float y = dqn_target(rb->r[i], rb->done[i], gamma, cn.a[NN_LAYERS]);
```

```c
// 4. every so often, let the target network catch up
if (step % TARGET_SYNC == 0) {
    nn_copy(&target, &online);
}
```

For 1000 steps the network solves a normal, fixed regression problem: "make Q(s, a)
match these targets". Then the targets are refreshed.

---

## 8. The training loop

Here's `train_dqn.c` side by side with the tabular `train.c`:

| | Q-table (`train.c`) | DQN (`train_dqn.c`) |
|---|---|---|
| Q-values | `q[s][a]` | `nn_forward(net, features)` |
| state | 14-bit integer | array of floats |
| learns from | the transition just played | random batch of 32 from the last 50 000 |
| target uses | the same table | a frozen copy, refreshed every 1000 steps |
| update | one cell, step α | all parameters, Adam step |
| ε schedule | per episode | per step |

One iteration of the loop:

1. pick an action ε-greedily with the online network and play it,
2. store the transition in the replay buffer,
3. every 4th step: sample 32 transitions, compute targets with the target network, take
   one Adam step,
4. every 1000 steps, copy online → target.

Why only every 4th step? A gradient step costs much more than a game step. With the same
CPU time, "600 000 steps, learn every 4th" beat "200 000 steps, learn every step" for both
inputs (section 9). Playing more gives the replay buffer more varied experience to learn
from. The original DQN paper did the same.

Learning only starts after `WARMUP` = 1000 steps, so the first batches aren't drawn from a
nearly empty buffer. Starvation is treated as a time limit, exactly as in `train.c`: the
episode ends, but the transition is not stored as `done`.

The hyperparameters are at the top of `train_dqn.c`:

```c
#define HIDDEN 128        // neurons per hidden layer
#define GAMMA 0.95f
#define LR 1e-3f          // Adam learning rate
#define BATCH 32
#define REPLAY_CAP 50000  // transitions kept in the replay buffer
#define WARMUP 1000       // steps of play before learning starts
#define TRAIN_EVERY 4     // one gradient step every N environment steps
#define TARGET_SYNC 1000  // copy online -> target every N steps
#define EPS_MIN 0.01f
#define EPS_DECAY 0.5     // fraction of training over which eps goes 1 -> EPS_MIN
```

---

## 9. Stage 1: same inputs as the table

First, a fair comparison. `--features small` gives the network exactly the 14 bits of
`ql_state()`, unpacked into 14 floats of 0 or 1 (`small_features` in `dqn.c`):

```c
int s = ql_state(g);
for (int i = 0; i < FEAT_SMALL_N; i++) {
    x[i] = (float)((s >> i) & 1);
}
```

The network sees the same information as the Q-table, so it shouldn't do much better.
If it did much *worse*, something would be wrong with the DQN code. Greedy scores (average
of 100 games), three seeds each:

| agent | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|
| Q-table, 20 000 episodes (from RL_TUTORIAL §9) | 41.9 | 37.2 | 41.6 |
| DQN small, 200 000 steps, learn every step | 37.7 | 32.1 | 35.9 |
| DQN small, 600 000 steps, learn every 4th step | 37.5 | 37.8 | 43.3 |

About the same. That's the expected result, and it tells us the machinery works: a network
can learn what a table learns. With a state this small, though, the table is the better
tool. It trains in a second, while the network takes minutes.

---

## 10. Stage 2: a view no table could hold

Now the real reason for DQN. `--features view` gives the network a 9 × 9 window around the
head (`view_features` in `dqn.c`):

* **obstacle channel**, 81 values: 1 if the cell is a wall or body (the tail isn't marked,
  since it moves away in the same step, like in `game_is_deadly`),
* **food channel**, 81 values: 1 on the food cell, if it's in the window,
* **food direction**, 4 values: food ahead / behind / left / right of the head, so the
  agent still knows where to go when the food is outside the window,
* **length**, 1 value: `len / MAX_LEN`.

That's 167 inputs. As bits, the obstacle channel alone has 2⁸¹ combinations.

The window is **egocentric**: it is rotated so the top row is always the direction the
snake is heading. "Wall one cell ahead" always lights up the same input, whatever the
heading. This matches the relative actions (straight / left / right), and it means the
network doesn't have to learn the same lesson four times, once per heading:

```c
// Unit vectors for "forward" and "right" in board coordinates.
int fx = DIR_DX[g->dir], fy = DIR_DY[g->dir];
int rx = DIR_DX[(g->dir + 1) % 4], ry = DIR_DY[(g->dir + 1) % 4];

// Window row 0 is the farthest row ahead, column 0 the farthest left.
for (int row = 0; row < VIEW_W; row++) {
    for (int col = 0; col < VIEW_W; col++) {
        int fwd = VIEW_R - row, right = col - VIEW_R;
        int px = head.x + fwd * fx + right * rx;
        int py = head.y + fwd * fy + right * ry;
        ...
```

`test_view_is_egocentric` in `test.c` checks that this rotation is right.

Results, same format:

| agent | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|
| Q-table (3 × 3 block), 20 000 episodes | 41.9 | 37.2 | 41.6 |
| DQN view, 200 000 steps, learn every step | 51.6 | 45.3 | 46.6 |
| DQN view, 600 000 steps, learn every 4th step (default) | **59.8** | **52.1** | **58.5** |

Around 57 food on average against about 40: the bigger view really helps, and only a
function approximator could use it. Run it yourself:

```bash
./train_dqn --features view --seed 1     # about 4 minutes on one core (small: under 2)
```

(It used to take about 13 minutes. The CNN tutorial found why: Adam was stuck doing slow
arithmetic on almost-zero numbers. See section 8 of [CNN_TUTORIAL.md](CNN_TUTORIAL.md). The
fix in `nn.c` changed the speed only, not the results.)

### Reading the training log

```
   steps episodes  avg(100)    max    eps     loss    max Q
   50000     1293      0.06      3  0.835   0.1222     3.58
  100000     2823      0.06      4  0.670   0.0653     3.07
  150000     4490      0.17      4  0.505   0.0525     2.16
  200000     6049      0.63      5  0.340   0.0692     1.58
  250000     6502      6.04     18  0.175   0.2143     3.15
  300000     6676     15.62     54  0.010   0.3037     4.03
  350000     6748     30.31     71  0.010   0.3309     4.24
  400000     6814     35.26     80  0.010   0.3319     4.33
  450000     6870     37.02     84  0.010   0.3379     4.16
  500000     6928     40.25     84  0.010   0.3282     4.06
  550000     6985     40.16     84  0.010   0.3363     4.00
  600000     7046     37.93     84  0.010   0.3213     3.97

Greedy eval over 100 games: avg 59.84, max 107
Random policy for comparison: avg 0.03
```

* **Nothing happens for 200 000 steps.** While ε is high, almost every game ends within a
  few steps through a random move into a wall. The network is learning (the loss is
  non-zero, the Q-values move), but the scores can't show it until ε is small. Exercise 2
  is about this.
* **`episodes` slows down** once the snake gets good: games get longer, so the same number
  of steps covers fewer of them.
* **`loss` goes *up* as the agent improves.** In supervised learning a rising loss is bad,
  but here the targets change: a better agent visits new situations and eats more, so
  there's more to predict. In RL, the loss tells you whether training is *stable*, not
  whether the agent is *good*. Watch the scores for that.
* **`max Q` stays around 4.** That's a healthy sign: with γ = 0.95, a +10 food reward
  that is about 20 steps away is worth 10 · 0.95²⁰ ≈ 3.6 now. If `max Q` climbed to
  hundreds, the Q-values would be diverging (section 11).
* **Training average ≈ 40, greedy ≈ 60.** The only difference is ε = 0.01. For a long snake,
  even one random move in a hundred is often fatal. That's why we always evaluate greedily.

### Why it still plateaus

Even 9 × 9 is a local view. The board is 38 × 18, and a snake of length 60 doesn't fit in
the window. The agent can't see where its tail is, so it can't plan the classic "follow
your tail" strategy (the tail moves away every step, so the cell it leaves is always safe
to enter).

### Experiment: telling it where the tail is

That suggests an obvious fix: `--features tail` adds 5 inputs to the view, built the same
way as the food direction (`tail_features` in `dqn.c`):

* tail ahead / behind / left / right of the head, relative to the heading,
* the distance to the tail, scaled to 0..1.

The network can learn for itself whether being near the tail is good or bad. We don't add
a reward for it, because the first tutorial showed how easily reward shaping backfires.

Results (600 000 steps, three seeds):

| inputs | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|
| view (167 inputs) | 59.8 | 52.1 | 58.5 |
| view + tail (172 inputs) | 47.1 | 53.4 | 43.8 |

**It didn't help.** If anything, it's worse: about 48 against 57, a gap bigger than the
spread between seeds. We haven't tested *why*, but some plausible reasons:

* Knowing *where* the tail is isn't the same as knowing whether you can *reach* it. The
  path to the tail may be blocked by the body, and the 5 inputs can't tell.
* Most training time is spent with a short snake, where the tail is right behind the head
  and carries no information. The situations where it matters (long snake, about to be
  trapped) are rare in the replay buffer.
* Every new input changes the network from the very first step, so two runs with different
  inputs aren't directly comparable seed by seed. Three seeds is a small sample.

This is normal in RL: a feature that sounds obviously useful can do nothing or even hurt,
and you only find out by measuring over several seeds. Exercises 4 and 7 continue from
here.

---

## 11. Debugging RL

RL code fails quietly: a bug rarely crashes, the agent just learns slowly or not at all.
Some habits that help:

* **Test the pieces you can test.** The game rules, the Bellman target (`test_dqn_target`),
  backprop (`test_gradient_check`) and the feature encoding (`test_view_is_egocentric`)
  are all plain functions with known right answers.
* **Compare against a known baseline.** Stage 1 exists for this: if DQN with the table's
  inputs had scored 5, the bug would be in the DQN code, not in the inputs.
* **Watch `max Q`.** Rewards are at most +10 per food, so Q-values far above what the
  discounting allows mean divergence. Usual causes: target network synced too often,
  learning rate too high, or a missing `done` check.
* **Run several seeds.** One seed proves little. Above, the same settings gave 52 to 60.
  Two settings that differ by less than the seed-to-seed spread are effectively tied.
* **Watch it play.** `./snake` → Watch DQN(6). You'll spot behaviours (circling, hugging
  walls) that no number shows.
* **Reproducibility.** Everything random goes through `game_rand` with the `--seed`, so the
  same seed gives the same run. If you change something and the result changes, it was
  your change.

---

## 12. Where to go next

1. **Double DQN.** Plain DQN overestimates Q-values, because `max` over noisy estimates is
   biased upwards. Double DQN picks the best action with the *online* network but
   evaluates it with the *target* network:
   `target = r + γ · Q_target(s', argmax_a Q_online(s', a))`. It's a few lines in
   `dqn_train_batch`. Does `max Q` drop? Does the score change?
2. **Faster exploration.** Set `EPS_DECAY` to 0.2 so ε reaches its minimum after
   120 000 steps instead of 300 000. Does the agent learn sooner, or does it miss
   things it would have found while exploring?
3. **Window size.** Change `VIEW_R` in `dqn.h` (3 gives 7 × 7, 6 gives 13 × 13). The
   inputs count as a layer, and `NN_MAX` in `nn.h` caps how wide a layer can be. If the
   inputs don't fit, `make` stops with "too many view inputs" until you raise it (256 is
   enough up to `VIEW_R` 5, 512 up to `VIEW_R` 7). Is bigger always better? What happens to training time?
4. **A smarter feature.** Add one input: the fraction of free cells reachable from the cell
   ahead (a flood fill). This is the "trap bit" from the first tutorial, but now it can be
   a real number.
5. **A convolutional network.** Our network treats the 81 window cells as unrelated
   numbers. A convolution slides one small filter over the grid, so "body to my left"
   is learned once and recognised everywhere. This is what the original DQN used on
   Atari screens. Done in [CNN_TUTORIAL.md](CNN_TUTORIAL.md).
6. **Reward shaping, again.** Try `--shaping`-style rewards (+1 closer to the food, −1
   farther) in `train_dqn.c`. Did it help the table? Does it help here?
7. **Make the tail feature work.** Train `--features tail` for longer
   (`--steps 1200000`): does it catch up with `view` once long snakes are common in the
   replay buffer? Or replace the 4 direction bits with "is the tail reachable from the
   cell ahead?" (a flood fill that stops at the tail).
