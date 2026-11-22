# Convolutional networks: a hands-on tutorial

This tutorial continues [DQN_TUTORIAL.md](DQN_TUTORIAL.md). There we replaced the Q-table
with a small fully connected network. Here we replace that network with a
**convolutional neural network** (CNN), again written from scratch in C. A CNN is the kind
of network the original DQN paper used to play Atari games from the screen pixels.

The reinforcement learning part doesn't change at all: same rewards, same replay buffer,
same target network, same ε-greedy. Only the function that turns a state into three
Q-values is new. You should already know the forward pass, backpropagation, the gradient
check and Adam from the DQN tutorial.

The code:

| File             | What it does                                                         |
|------------------|----------------------------------------------------------------------|
| `cnn.c`, `cnn.h` | the convolutional network: forward, backprop, Adam, threads, save/load |
| `train_cnn.c`    | the training loop (the same as `train_dqn.c`, with a different network) |
| `dqn.c`          | `dqn_view`: the egocentric window, now for any radius                |
| `test.c`         | gradient check, shift test, thread test, a by-hand training step     |

```bash
make
./test
./train_cnn                 # 9x9 window, the inputs of ./train_dqn --features view (~3 min)
./train_cnn --radius 9      # 19x19 window (~30 min)
./snake                     # pick "Watch CNN(7)"
```

---

## 1. What the fully connected network gets wrong

`--features view` gives the DQN a 9 × 9 window around the head, rotated so the top row is
always "ahead". The network's first layer computes, for each of its 128 neurons,

```
out[o] = b[o] + w[o][0]·in[0] + w[o][1]·in[1] + ... + w[o][166]·in[166]
```

Every window cell has its own weight. The network has no idea that cell 40 and cell 41 are
neighbours. They are just input number 40 and input number 41. If it learns that
"obstacle here, obstacle there, gap in between" is a dead end, it learns that for one
place in the window. The same shape two cells further left is a different set of inputs,
and it has to learn it again from new experience.

Two problems follow:

1. **It learns slowly.** The same lesson is learned many times, once for every position
   it can appear in.
2. **It doesn't scale.** The first layer has `inputs × 128` weights. A 19 × 19 window has
   2 · 361 + 5 = 727 inputs, so 93 000 weights in the first layer alone, each trained
   separately from rarely seen situations.

A convolution fixes both, with one idea: **use the same small set of weights at every
position**.

---

## 2. A convolution, in one picture

A **filter** is a 3 × 3 grid of weights (plus one bias). You slide it over the input. At
every position you multiply the 9 cells under it by the 9 weights, add them up, add the
bias and apply ReLU. That gives one output cell.

Here is one filter on a 5 × 5 grid of obstacles. This filter happens to respond to "wall
on the left":

```
input (1 = obstacle)      filter            output (before ReLU, bias 0)

1 0 0 0 0               1  0 -1             3 -2  0
1 0 0 1 0               1  0 -1             3 -3  0
1 0 0 1 0               1  0 -1             3 -2  0
1 0 0 1 0
1 0 0 0 0
```

The top-left output is the filter placed on the top-left 3 × 3 block:
1·1 + 0·0 + 0·(−1), three times over, = 3. The middle column is negative because the
obstacles in column 3 land under the filter's −1s. The output is high wherever the
filter's pattern appears, wherever that is. The output is called a **feature map**: a map of
*where* the feature is.

Note the output is 3 × 3, not 5 × 5: a 3 × 3 filter only fits in 3 positions across a
5-cell row. This is called a **valid** convolution. In general, with stride 1:

```
output side = input side − 2
```

Two properties make this useful:

* **Weight sharing.** The filter has 9 weights however big the input is. A 9 × 9 window
  or a 19 × 19 window: the same 9 weights. The parameter count no longer grows with the
  window.
* **Translation equivariance.** Shift the input one cell to the right and the feature map
  shifts one cell to the right too, unchanged. "Body next to me" is learned once and
  recognised everywhere. `test_conv_shift` in `test.c` checks exactly this.

---

## 3. Channels and filters

Our window has two **channels**: an obstacle plane (wall or body) and a food plane. A
filter looks at all channels at once, so it is really 2 × 3 × 3 = 18 weights. It can
learn patterns like "food here *and* no obstacle in front of it".

One filter detects one pattern. So we use many: the first layer has 16 filters, which
produce 16 feature maps. Those 16 maps are the channels of the next layer's input, and a
filter there is 16 × 3 × 3 = 144 weights. It combines simple patterns ("wall left",
"body ahead") into more complex ones ("corridor", "dead end").

In memory, a grid is stored channel by channel, called "CHW" (channel, row, column):
value (c, y, x) of a grid with side s is at index `c * s * s + y * s + x`.

That's the layout `dqn_view` already produced for the DQN: 81 obstacle values, then 81
food values, then the 5 extras. So the CNN reads the same input arrays as
`--features view`, unchanged.

### Stride

The second convolution uses **stride 2**: it moves the filter 2 cells at a time instead of 1.
That halves the size of the output in each direction, and so divides the work and the size
of the next layer by about 4. The general formula (from `cnn_init`):

```c
// A "valid" convolution: the filter must fit inside the grid.
net->side[l + 1] = (net->side[l] - CNN_K) / net->stride[l] + 1;
```

### The whole network

```
grid  2 x w  x w
conv1 3x3, stride 1, ReLU  -> 16 x w1 x w1     w1 = w - 2
conv2 3x3, stride 2, ReLU  -> 32 x w2 x w2     w2 = (w1 - 3) / 2 + 1
flatten, append the extras -> 32 · w2 · w2 + 5 values
dense, ReLU                -> 128
dense                      -> 3 Q-values
```

The 5 extras (food ahead/behind/left/right, snake length) aren't grids, so they skip the
convolutions and join at the dense layer. That matters: when the food is outside the
window, the food direction is the only thing that says which way to go.

`train_cnn` prints the shapes at the start:

| | window 9 × 9 (`--radius 4`) | window 19 × 19 (`--radius 9`) |
|---|---|---|
| conv1 output | 16 × 7 × 7 | 16 × 17 × 17 |
| conv2 output | 32 × 3 × 3 | 32 × 8 × 8 |
| dense input | 293 | 2053 |
| parameters | 42 963 | 268 243 |
| (the MLP of `train_dqn`, same window) | 38 403 | 110 083 (needs `NN_MAX` ≥ 727) |

The convolutions themselves are tiny: 16·18 + 16 = 304 and 32·144 + 32 = 4640 parameters,
whatever the window. Almost all the parameters are in the dense layer after them, which
still grows with the window. Exercise 3 is about that.

---

## 4. The forward pass in code

The trick that makes a convolution easy to write: **copy the 3 × 3 patch under the
filter into a small array first** (`gather` in `cnn.c`). After that, every filter is just a
dot product with that array, exactly like one neuron of a dense layer:

```c
// Copy the 3x3 block at (y0, x0) of every channel into patch, channel by
// channel. Afterwards a convolution is just a dot product with a filter.
static void gather(const float *in, int cin, int side, int y0, int x0,
                   float *patch) {
    for (int c = 0; c < cin; c++) {
        const float *plane = in + c * side * side;
        for (int ky = 0; ky < CNN_K; ky++) {
            const float *row = plane + (y0 + ky) * side + x0;
            for (int kx = 0; kx < CNN_K; kx++) {
                *patch++ = row[kx];
            }
        }
    }
}
```

```c
// out(o, oy, ox) = ReLU(b[o] + filter o . patch at (oy * stride, ox * stride))
// The same filter is used at every position: that's the whole idea.
static void conv_forward(const float *in, int cin, int side, const float *w,
                         const float *b, int cout, int stride, int oside,
                         float *out, float *patch) {
    int K = CNN_K * CNN_K * cin;
    for (int oy = 0; oy < oside; oy++) {
        for (int ox = 0; ox < oside; ox++) {
            gather(in, cin, side, oy * stride, ox * stride, patch);
            for (int o = 0; o < cout; o++) {
                const float *filter = w + o * K;
                float sum = b[o];
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < K; k++) {
                    sum += filter[k] * patch[k];
                }
                out[(o * oside + oy) * oside + ox] = sum > 0.0f ? sum : 0.0f;
            }
        }
    }
}
```

So a convolution is a dense layer with 9·`cin` inputs, applied to every patch, with the
same weights each time. (Big libraries copy *all* patches into one matrix at once, called
"im2col", and do one big matrix multiplication. Same idea, faster on large inputs.)

He initialisation works as before, with one change: the "number of inputs" of a filter is
the patch size, 9 · `cin`, because that's how many values each output adds up.

---

## 5. Backpropagation through a convolution

Pretend for a moment that each output position had its own private copy of the filter.
Then each position is just a dense neuron, and the DQN tutorial's rules apply:

```
∂loss/∂b[o]          = delta[o]
∂loss/∂filter[o][k]  = delta[o] · patch[k]
∂loss/∂patch[k]      = Σ_o filter[o][k] · delta[o]
```

Now remember all those copies are *the same* weights. If one weight is used in several
places, its gradient is the **sum** of the gradients from each place. So:

* **Filter gradient:** add `delta · patch` over every output position.
* **Input gradient:** each patch gradient goes back onto the cells it was copied from.
  Neighbouring patches overlap (a cell is in up to 9 patches with stride 1), so a cell
  **adds up** the gradients from all the patches it was in.

That's `scatter_add`, the reverse of `gather`:

```c
// The reverse of gather: add dpatch back onto the cells it came from.
// Neighbouring patches overlap, so a cell can receive from several of them.
static void scatter_add(float *din, int cin, int side, int y0, int x0,
                        const float *dpatch) {
    for (int c = 0; c < cin; c++) {
        float *plane = din + c * side * side;
        for (int ky = 0; ky < CNN_K; ky++) {
            float *row = plane + (y0 + ky) * side + x0;
            for (int kx = 0; kx < CNN_K; kx++) {
                row[kx] += *dpatch++;
            }
        }
    }
}
```

And the core of `conv_backward`, for one output position:

```c
            for (int o = 0; o < cout; o++) {
                float d = dout[(o * oside + oy) * oside + ox];
                if (d == 0.0f) {
                    continue;
                }
                // The filter was used at every position, so its gradient is
                // the sum over all positions of delta * patch.
                gb[o] += d;
                float *gfilter = gw + o * K;
                const float *filter = w + o * K;
                for (int k = 0; k < K; k++) {
                    gfilter[k] += d * patch[k];
                }
                if (din) {
                    for (int k = 0; k < K; k++) {
                        dpatch[k] += d * filter[k];
                    }
                }
            }
```

Between layers, the ReLU rule is the same as before: zero the gradient wherever the
layer's output was 0. `cnn_backward` goes through the layers top to bottom: dense2,
dense1, conv2, conv1. The first convolution doesn't need an input gradient (nothing below
it learns), so it gets `din = NULL`.

### The gradient check still works

`test_cnn_gradient_check` nudges every one of the network's parameters (filters, biases,
dense weights) and compares with backprop, just like the DQN test. It is worth breaking the
code on purpose once to see it fail. Change the `+=` in `scatter_add` to `=` (the most
common convolution bug: forgetting that patches overlap) and run `./test`:

```
FAIL test.c:279: worst < 1e-2f
1 check(s) failed
```

---

## 6. One flat array for all parameters

`nn.c` keeps separate arrays per layer. `cnn.c` puts every parameter of every layer into
**one** array `p`, and records where each layer starts (`w_off`, `b_off`). Then everything
that treats parameters as "just a list of numbers" is a single loop:

* Adam: one loop over `nparam` values (`cnn_adam_step`),
* copying to the target network: one `memcpy` (`cnn_copy`),
* saving and loading: one `fwrite`/`fread`,
* a gradient buffer: one array of `nparam` floats, with the same offsets.

The layers use `p + net->w_off[l]` to find their weights. This is how real frameworks do it
too, under the hood.

---

## 7. Training on several cores

The convolutions cost more than the MLP's matrix products, so `train_cnn` can split each
batch of 32 samples over several CPU cores with OpenMP (`-fopenmp` in the Makefile).

The samples in a batch are independent: each one does a forward pass, a backward pass and
adds to the gradient. The only thing they share is the gradient sum, and two threads
adding into the same array at the same time would lose updates. So **each thread gets its
own gradient buffer** (and its own cache), and the buffers are added up at the end
(`cnn_train_batch`):

```c
        // Each sample is independent: thread t works through its share of the
        // batch with its own cache and adds into its own gradient buffer.
        #pragma omp for schedule(static) reduction(+:loss, qsum)
        for (int k = 0; k < w->batch; k++) {
            ...
            cnn_backward(online, c, a, d, grad);
        }
```

Some details that matter:

* **The batch is drawn before the threads start**, on one thread. If threads called
  `game_rand` themselves, the result would depend on which thread got there first, and the
  same `--seed` would give different runs.
* **The buffers are added in a fixed order** (thread 0 + thread 1 + ...). Floating-point
  addition depends on order, so this keeps runs repeatable. Repeatable *for a given thread
  count*, though: `--threads 2` and `--threads 4` add in a different order, so they give
  slightly different runs.
* **Parallelism has a cost.** Waking threads up takes time, and one batch is only about a
  millisecond of work. That's why the whole batch is one parallel region (the threads are
  woken once per batch, not once per loop).
* `test_cnn_threads_agree` checks that 1 and 4 threads give the same update (up to
  rounding).

Measured on the 9 × 9 window (20 000 steps, one batch every 4 steps):

| threads | time |
|---:|---:|
| 1 | 12.9 s |
| 4 | 8.5 s |
| 4, with `OMP_WAIT_POLICY=active` | 5.6 s |
| 8, with `OMP_WAIT_POLICY=active` | 6.7 s |

The speedup is far from 4×. Part of each step can't be split: playing the game, choosing
the action, the Adam update. And the threads go to sleep between batches. 
`OMP_WAIT_POLICY=active` tells them to keep spinning instead, which wakes them faster but
burns CPU while waiting, so only use it on a machine that isn't busy with other work.
More than 4 threads made it slower, so 4 is the default.

```bash
OMP_WAIT_POLICY=active ./train_cnn --radius 9 --threads 4
```

---

## 8. Two bugs worth learning from

Both of these happened while writing this code. Neither crashed, and neither was caught by
the gradient check.

### Bug 1: the agent that never learned

The first version of `cnn_train_batch` decoded the state straight into the cache's input
buffer, then ran the target network on the next state, which **copied `s2` over it**.
The online network was then trained on Q(s′, a) instead of Q(s, a).

Every piece was correct, so every test passed. The symptom was in the training log: after
380 000 steps, with ε long at its minimum, the agent averaged 0.05 food, and `max Q`
stayed around 0 (the MLP was at 30 food with `max Q` ≈ 4 by then).

The lesson: **test the wiring, not just the pieces.** `test_cnn_train_step_by_hand` now does
one training step by hand (forward the target on s′, forward the online network on s,
backward, Adam) and checks that `cnn_train_batch` gives exactly the same parameters. Put
the bug back and that test fails.

### Bug 2: slow because of numbers that are almost zero

A profile of the first version showed `cnn_adam_step` taking **1 ms** per call for 43 000
parameters, more than all the convolutions. Adam does about ten operations per
parameter, so it should take a few microseconds.

The cause is **denormal** (or "subnormal") floats. A `float` can't normally hold values
below about 10⁻³⁸. Below that, the CPU switches to a special slow format, often many times
slower per operation. Here is how Adam gets there. When a parameter's gradient is 0, which
is common (a filter that never fires, an input that is always 0), Adam's running mean
just shrinks:

```
m = 0.9 · m + 0.1 · 0
```

After about 800 steps `m` is denormal. Then it gets worse: the smallest denormal is about
1.4 · 10⁻⁴⁵, and 0.9 times that **rounds back to the same number**. So `m` never reaches 0
and stays in the slow range forever, for every parameter that stopped getting gradient.

The fix is one line per value: treat anything below 10⁻³⁰ as zero (`cnn_adam_step`):

```c
        // Flush tiny values to 0. Without this, a parameter whose gradient
        // stays 0 has m shrink into the "denormal" range, where 0.9 * m rounds
        // back to m and never reaches 0, and denormal arithmetic is many times
        // slower. See section 8 of CNN_TUTORIAL.md.
        net->m[i] = fabsf(m) < ADAM_TINY ? 0.0f : m;
        net->v[i] = v < ADAM_TINY ? 0.0f : v;
```

A value that small has no effect on the update anyway: `lr · m / (√v + ε)` is far too
small to change a weight of size 0.1. A microbenchmark (4000 Adam steps with zero
gradients) went from 27.7 s to 1.0 s, and CNN training became 1.5 times faster.

**The DQN had the same bug.** `nn.c` now has the same fix. With the same seed, the
training logs of `./train_dqn` are identical, digit for digit (a full 600 000-step run
still gives greedy 59.84, max 107), but about 3 times faster: `./train_dqn --features view`
now trains in about 4 minutes instead of 13.

---

## 9. Results

All runs: 600 000 steps, the same settings as the DQN (section 8 of the DQN tutorial),
greedy evaluation over 100 games, three seeds each.

| agent | inputs | seed 1 | seed 2 | seed 3 | mean |
|---|---|---:|---:|---:|---:|
| MLP DQN (`train_dqn --features view`) | 9 × 9 window | 59.8 | 52.1 | 58.5 | **56.8** |
| CNN, `--radius 4` | 9 × 9 window (the same inputs) | 60.8 | 49.2 | 51.8 | **53.9** |
| CNN, `--radius 6` | 13 × 13 window | 56.9 | 51.1 | 57.0 | **55.0** |
| CNN, `--radius 9` | 19 × 19 window | 52.3 | 45.6 | 52.1 | **50.0** |

(The radius-4 CNN runs used `--threads 2`; with the default 4 threads, seed 1 gives 54.0
instead of 60.8. That's the "repeatable only for the same thread count" effect from
section 7, and a reminder of how much one run can vary.)

**The honest summary: the CNN works, and it is no better.** On the same inputs it is
level with the MLP: the gap (53.9 against 56.8) is smaller than the spread between seeds.
A bigger window didn't help either. 19 × 19 is, if anything, a bit worse.

### Where the CNN does differ: the first 300 000 steps

Training average (ε-greedy, last 100 games), seed 1 of each:

| steps | MLP | CNN r4 | CNN r6 | CNN r9 |
|---:|---:|---:|---:|---:|
| 100 000 | 0.06 | 0.22 | 0.39 | 0.96 |
| 200 000 | 0.63 | 2.73 | 3.32 | 3.66 |
| 300 000 | 15.6 | 10.6 | 13.8 | 13.6 |
| 400 000 | 35.3 | 35.4 | 38.0 | 37.1 |
| 600 000 | 37.9 | 35.5 | 39.2 | 41.7 |

While ε is still high, the CNNs learn the basics (head to the food, don't hit the wall)
several times faster than the MLP, and faster with bigger windows. That's weight sharing
at work: "wall ahead" is one lesson, not one per position. But the basics are the easy
part. Once ε reaches its minimum at 300 000 steps, all four agents climb to the same
plateau. The bigger windows even have a slightly better *training* average at the end
(41.7 against 37.9), but that didn't carry over to greedy play.

### Why seeing more didn't help

We haven't tested these, but some likely reasons:

* **Seeing isn't planning.** A long snake dies by boxing itself in: it turns into a pocket
  that is too small, and dies 20 or 30 steps later. To avoid that it must link a turn now
  to a death much later. With γ = 0.95, a death 30 steps away counts as only
  0.95³⁰ ≈ 0.2 of a death now. The window shows the trap, but the reward signal barely
  connects the turn to it.
* **The rare situations are the important ones.** The replay buffer is mostly short
  snakes, where a 9 × 9 window already shows everything that matters. The situations where
  a 19 × 19 view helps (long snake, trap forming) are a small fraction of the training
  data.
* **More to learn in the same time.** The radius-9 network has 268 000 parameters, mostly
  in the dense layer after the convolutions, and got the same 600 000 steps as the
  43 000-parameter one.
* **Overestimation.** `max Q` settled around 4 for the 9 × 9 networks but around 7 for
  19 × 19. A food 20 steps away is worth about 3.6, so 7 is suspicious. It's the "max of
  noisy estimates" bias that Double DQN (DQN tutorial, exercise 1) is meant to fix, and a
  bigger network has more room for noise.

This is the same lesson as the tail experiment in the DQN tutorial: a change that clearly
gives the agent more information doesn't automatically make it play better. Measure.

### Timing

On a 16-core machine (radius 4 with the default 4 threads; the others ran in parallel with
other training runs, so they'd be somewhat faster alone):

| | time for 600 000 steps |
|---|---:|
| MLP DQN, view (1 thread, after the denormal fix) | about 4 min |
| CNN radius 4, 4 threads | about 3 min |
| CNN radius 6, 1 thread | about 25 min |
| CNN radius 9, 4 threads | about 28 min |

Try it yourself:

```bash
./train_cnn --seed 1 --threads 2       # greedy avg 60.83, max 109, if your build matches
./snake                                # Watch CNN(7)
```

---

## 10. Where to go next

1. **Padding.** Our convolutions are "valid", so each layer shrinks the grid and cells at
   the edge are seen by fewer filter positions. "Same" padding adds a border of zeros (or
   better here: of obstacles, since outside the board is wall) so the output keeps the
   input's size. Change `gather` and `scatter_add` to treat out-of-range cells as padding.
2. **Look at the filters.** Print the 16 first-layer filters of a trained `snake_cnn.bin`
   as 3 × 3 grids (obstacle channel and food channel). Can you find a "wall ahead" or
   "food to the left" detector? Most will be messy: that's normal.
3. **Global pooling.** Almost all parameters are in the dense layer after the
   convolutions, and it still grows with the window. Instead, average each of conv2's 32
   feature maps down to one number ("global average pooling"), so the dense layer gets
   32 + 5 inputs whatever the window. What does that do to the score for radius 9? (Hint:
   after pooling, the network no longer knows *where* a feature was. Does it need to?)
4. **A third convolution.** Add conv3 (3 × 3, stride 1). For radius 9 the grid goes
   17 → 8 → 6. Each extra layer sees a bigger area of the input (its "receptive field").
5. **The whole board.** With `--radius 19`, the window covers the whole board wherever
   the head is. It's slow (see the timings above), so try it with fewer filters, global
   pooling (exercise 3), or more threads.
6. **Double DQN** (from the DQN tutorial) works here too: it is a change in
   `cnn_train_batch` only.
