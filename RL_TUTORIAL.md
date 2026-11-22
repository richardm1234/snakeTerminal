# Reinforcement Learning with Snake: a hands-on introduction

This tutorial explains reinforcement learning (RL) using the snake AI in this repository.
Every snippet is taken from the real code, so you can open the file and follow along.
The only background you need is C and a bit of high-school math.

| File | Role in RL terms |
|---|---|
| `game.c` / `game.h` | the **environment** |
| `qlearn.c` / `qlearn.h` | the **agent**: what it sees, how it decides, how it learns |
| `train.c` | the **training loop** |
| `snake.c` | the game, plus "Watch AI(5)" to see the trained agent play |

---

## 1. What is reinforcement learning?

In supervised learning you show a model the right answers. RL doesn't have right answers.
It has an **agent** that tries things in an **environment** and gets a **reward**, a single
number that says how good the result was. The agent's goal is to collect as much reward
as it can over time.

```
        ┌──────────── action ─────────────┐
        │                                 ▼
   ┌─────────┐                      ┌─────────────┐
   │  Agent  │                      │ Environment │
   └─────────┘                      └─────────────┘
        ▲                                 │
        └────── new state + reward ───────┘
```

For snake:

* **Environment**: the board, the snake and the food.
* **State**: what the agent knows about the current situation.
* **Action**: go straight, turn left or turn right.
* **Reward**: +10 for eating, −10 for dying.

Nobody ever tells the agent "turn left here". It has to work that out from experience.
That is the core idea of RL.

---

## 2. The environment

An RL agent plays thousands of games, so the environment has to be fast and must not
print anything, sleep or wait for keys. The original `snake.c` did all of this inside
`main()`, mixed in with ncurses. The rules now live in `game.c`, which has no I/O:

```c
void game_reset(Game *g, uint64_t seed);

// Move one step relative to the current heading.
Event game_step_rel(Game *g, Action a);
```

`game_step_rel` advances the game by one tick and reports what happened:

```c
typedef enum { EV_NONE, EV_ATE, EV_DIED, EV_WON } Event;
```

The human game and the trainer both use these functions. That guarantees the AI learns
exactly the game you play.

> **Tip:** `game.c` uses its own random number generator (`game_rand`, a xorshift).
> A fixed `--seed` makes a whole training run reproducible, which helps a lot when
> debugging.

---

## 3. States: what does the agent see?

The obvious state would be the whole board. But the board has 38 × 18 cells, each empty,
snake or food. That gives far more possible boards than atoms in the universe. A table
with one row per board would never fit in memory, and the agent would rarely see the
same board twice, so it couldn't learn anything.

The trick is to **describe the situation with a few features that matter**. We look at
the 3 × 3 block around the head, plus the heading and where the food is:

```
  NW  N  NE        bit 0..7   is each of the 8 cells around the head deadly?
  W   H  E                    (wall or body, clockwise from N)
  SW  S  SE        bit 8..9   heading, as a number 0..3
                   bit 10..13 food is left / right / above / below the head
```

and pack the answers into an integer (from `ql_state` in `qlearn.c`):

```c
// The 3x3 block around the head, clockwise from the cell above it.
static const int RING_DX[8] = {0, 1, 1, 1, 0, -1, -1, -1};
static const int RING_DY[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

// Which of the 8 neighbouring cells hold a wall or body?
for (int i = 0; i < 8; i++) {
    Point p = {head.x + RING_DX[i], head.y + RING_DY[i]};
    if (game_is_deadly(g, p)) {
        s |= 1 << i;
    }
}

// Heading, as a 2-bit number.
s |= g->dir << 8;

// Where is the food?
if (g->food.x < head.x) s |= 1 << 10;
if (g->food.x > head.x) s |= 1 << 11;
if (g->food.y < head.y) s |= 1 << 12;
if (g->food.y > head.y) s |= 1 << 13;
```

The cells straight ahead, left and right of the head are in there (which ones depends on
the heading). The diagonal cells are extra: they tell the agent what it will see *after*
its next move, for example "if I turn left now, there is body right next to me".

2¹⁴ = 16384 possible states, and in practice about 2000–3000 of them ever occur. That is
still small enough for a table.

> An earlier version used only 11 bits: danger straight / left / right, the heading and
> the food direction. That gives about 250 reachable states. Compare the two in section 9.

Choosing the state is the most important design decision in tabular RL. **The agent can
only learn what its state lets it tell apart.** Two situations that map to the same
number always get the same action. Keep that in mind for section 9.

---

## 4. Actions: relative, not absolute

The human game uses absolute directions (up, down, left, right). The agent uses
**relative** ones:

```c
typedef enum { ACT_STRAIGHT, ACT_LEFT, ACT_RIGHT } Action;
```

This has two benefits:

1. **Fewer choices.** With absolute directions, one of the four always means "reverse
   into your own neck". Relative actions leave that option out.
2. **Knowledge transfers between headings.** "Danger ahead, food on the right → turn
   right" has the same meaning whether the snake is heading up or left. (Our state is
   still stored in board directions, so the table only partly benefits. Exercise 3 in
   section 10 makes the state relative too.)

Converting back is one line, because `Dir` is listed clockwise:

```c
Dir game_turn(Dir d, Action a) {
    switch (a) {
        case ACT_LEFT:  return (Dir)((d + 3) % 4);
        case ACT_RIGHT: return (Dir)((d + 1) % 4);
        default:        return d;
    }
}
```

---

## 5. Rewards and the return

The reward tells the agent *what* you want, not *how* to get it (from `train.c`):

```c
#define REWARD_FOOD   10.0f
#define REWARD_DEATH -10.0f
```

The agent doesn't maximise the next reward. It maximises the **return**, the sum of all
future rewards, where rewards further in the future count less:

$$G_t = r_{t+1} + \gamma\, r_{t+2} + \gamma^2 r_{t+3} + \dots$$

**γ (gamma)** is the *discount factor*, between 0 and 1. We use `GAMMA 0.9f`:

* γ = 0: only the next reward matters (very short-sighted).
* γ close to 1: food 30 moves away still counts almost fully (far-sighted, but slower
  to learn).
* With γ = 0.9, a reward 10 steps away is worth 0.9¹⁰ ≈ 35% of its face value.

### Reward shaping

With only ±10, the agent gets no feedback for most steps, so early learning is slow.
*Reward shaping* adds small hints. Run `./train --shaping` to turn them on:

```c
#define REWARD_CLOSER  1.0f // only with --shaping
#define REWARD_FARTHER -1.0f
```

In our runs, shaping roughly doubles the score after 1500 episodes (about 2.0 instead of
about 1.3), but the final result is the same (about 32–36 food).

**Be careful with shaping:** the agent optimises exactly what you reward, even when that
isn't what you meant. If you reward +1 for *surviving* each step, circling safely can
become worth more than the risk of going for food, and the snake may stop eating.

---

## 6. Q-values and the Bellman equation

The agent keeps a table with one number per (state, action) pair:

```c
typedef struct {
    float q[QL_STATES][QL_ACTIONS];
} QTable;
```

**Q(s, a)** is the answer to "if I'm in state *s*, take action *a*, and then play well,
what return do I expect?" If the table were correct, playing perfectly would be easy:
always pick the action with the highest Q.

```c
int ql_best(const QTable *t, int s) {
    int best = 0;
    for (int a = 1; a < QL_ACTIONS; a++) {
        if (t->q[s][a] > t->q[s][best]) {
            best = a;
        }
    }
    return best;
}
```

How do we get correct values? The **Bellman equation** says that a good Q-value has to
agree with itself one step later:

$$Q(s,a) = r + \gamma \max_{a'} Q(s', a')$$

In words: what a move is worth = the reward you get right now + the (discounted) value
of the best move from where you land.

**Q-learning** turns this into an update rule. After every step it moves the old value
a bit towards the right-hand side (the *target*):

$$Q(s,a) \leftarrow Q(s,a) + \alpha\,\big[\underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{target}} - Q(s,a)\big]$$

Here is the update in C, from `qlearn.c`:

```c
void ql_update(QTable *t, int s, int a, float r, int s2, int done,
               float alpha, float gamma) {
    float target = r;
    if (!done) {
        target += gamma * t->q[s2][ql_best(t, s2)];
    }
    t->q[s][a] += alpha * (target - t->q[s][a]);
}
```

* **α (alpha)**, the *learning rate* (`ALPHA 0.1f`), sets how far each update moves.
  Small values learn slowly but steadily. Large values react fast but are noisy.
* **`done`**: when the snake dies, there is no future, so the target is just the
  reward. Without this check, the agent would expect future rewards after its own death.
* The part in brackets is the **TD error** (temporal-difference error): how surprised
  the agent was by what happened.

A worked example (this is literally `test_q_update` in `test.c`): Q(s,a) = 1, r = 1,
the best Q in the next state is 5, α = 0.5, γ = 0.9.
target = 1 + 0.9·5 = 5.5 → new Q = 1 + 0.5·(5.5 − 1) = **3.25**.

Notice that the agent learns from its own estimate of the next state. This is called
**bootstrapping**. A reward of +10 for eating first raises the Q-value of the step just
before the food. Over many episodes that value spreads back to the steps before it,
one step at a time.

---

## 7. Exploration vs. exploitation

If the agent always took the best-known action, it would stick with the first thing that
worked and never discover anything better. It has to **explore** sometimes.

The classic solution is **ε-greedy**: with probability ε, act randomly; otherwise,
act greedily.

```c
int ql_act(const QTable *t, int s, float eps, uint64_t *rng) {
    float u = (float)game_rand(rng) / 4294967296.0f; // uniform in [0, 1)
    if (u < eps) {
        return (int)(game_rand(rng) % QL_ACTIONS); // explore
    }
    return ql_best(t, s);                          // exploit
}
```

ε starts at 1 (pure random play, since we know nothing yet) and goes down linearly to
0.01 over the first 70% of training (from `train.c`):

```c
float eps = 1.0f - (1.0f - EPS_MIN) * (float)ep / decay_episodes;
if (eps < EPS_MIN) eps = EPS_MIN;
```

---

## 8. The training loop

All the pieces together. This is the core of `run_episode` in `train.c`:

```c
Game g;
game_reset(&g, ((uint64_t)game_rand(rng) << 32) | game_rand(rng));
int s = ql_state(&g);

while (g.alive) {
    int a = ql_act(Q, s, eps, rng);                 // 1. choose
    int d0 = food_distance(&g);
    Event ev = game_step_rel(&g, (Action)a);        // 2. act
    float r = reward(ev, shaping, d0, food_distance(&g));
    int s2 = ql_state(&g);                          // 3. observe
    int done = (ev == EV_DIED || ev == EV_WON);

    if (learn) {
        ql_update(Q, s, a, r, s2, done, ALPHA, GAMMA); // 4. learn
    }
    s = s2;

    if (g.steps_since_food > QL_STARVE_STEPS(g.len)) {
        break;
    }
}
```

Choose, act, observe, learn, repeat. That is the whole algorithm.

The starvation check is subtle. A policy that loops forever would never end an episode,
so we stop it after `100 * len` steps without food. But this is a **time limit, not a
real game over**, so `done` stays 0. If we marked it as terminal, the agent would learn
that some perfectly safe states "end the game", which is false.

---

## 9. Reading the results

Run `./train --seed 1` (it takes well under a second):

```
 episode  avg(100)    max    eps   states
     500      0.15      3  0.859      101
    1000      0.63      3  0.717      220
    1500      1.44      5  0.576      327
    2000      2.30      8  0.435      427
    2500      2.19     12  0.293      517
    3000      4.04     15  0.152      677
    3500     16.72     47  0.010      955
    4000     24.77     56  0.010     1287
    4500     27.98     62  0.010     1496
    5000     25.24     76  0.010     1659
```

How to read it:

* **Slow at first.** While ε is high, the agent acts mostly at random, so its scores are
  low even though it is learning in the background.
* **Steep rise** as ε drops: now it actually uses what it learned.
* **`states` keeps growing.** Most of the 16384 bit patterns can't happen (the food can't
  be both left *and* right of the head), but long snakes keep producing new body shapes
  around the head, so new states turn up for a long time.
* **Greedy eval > training average**, because evaluation turns exploration off.

### More features need more training

Greedy scores (average of 100 games, three seeds) for the old 11-bit state and the
current 14-bit one:

| episodes | 11 bits (danger S/L/R) | 14 bits (3 × 3 block) |
|---------:|-----------------------:|----------------------:|
|     5000 |                 ~34    |                 ~25   |
|    20000 |                 ~34    |                 ~40   |
|    50000 |                 ~32    |                 ~40   |

With 5000 episodes the bigger state is *worse*: there are ten times as many table rows,
and each row needs its own experience before its Q-values mean anything. Given enough
episodes (`./train --episodes 20000`) it overtakes the small state. This trade-off shows
up everywhere in RL: a richer state can express a better policy, but it costs data.

### Why it still plateaus

The agent sees **one ring of cells** around its head. When the snake gets long, it still
turns into pockets enclosed by its own body, because a pocket is usually bigger than
3 × 3. Two boards that differ only in "this pocket is a dead end" get the **same state
number**, so no amount of training can teach the agent the difference. The limit is in
the state, not in the learning algorithm.

---

## 10. Where to go next

**Exercises** (each one is a small change):

1. **Tune hyperparameters.** Change `ALPHA`, `GAMMA` and the ε schedule in `train.c`.
   What happens with γ = 0.5 or γ = 0.99?
2. **Add a "trap" bit.** Run a flood fill from the cell in front of the head and set a bit
   if fewer free cells are reachable than the snake's length. Remember to raise
   `QL_STATE_BITS`. This tackles the plateau directly.
3. **Make the view relative.** The 3 × 3 block is stored in board directions (N, E, ...),
   so "wall on my left while heading up" and "wall on my left while heading right" are
   different states. Rotate the ring so bit 0 is always the cell *ahead*, and do the same
   for the food bits. Then you can drop the heading bits, the table shrinks by 4×, and
   what the agent learns in one direction carries over to the others.
4. **Try SARSA.** Instead of `max Q(s', a')`, use the Q-value of the action the agent
   *actually* takes next. SARSA is "on-policy" and tends to play more cautiously while ε
   is high.
5. **Break the reward.** Add +0.1 per step survived and watch the snake stop eating.
6. **Optimistic initialisation.** Start all Q-values at, say, +5 instead of 0, so untried
   actions look attractive and get explored.

**The bigger picture: Deep Q-Networks (DQN).** A table only works with a small,
hand-made state. To give the agent a richer view, you replace the table with a **neural
network** that *approximates* Q(s, a). The update target stays the same Bellman target from
section 6. This repo has one: continue with [DQN_TUTORIAL.md](DQN_TUTORIAL.md).
