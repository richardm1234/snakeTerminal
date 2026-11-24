#ifndef DQN_H
#define DQN_H

#include <stdint.h>

#include "game.h"
#include "nn.h"
#include "qlearn.h"

/*
 * Deep Q-learning: the Q-table is replaced by a neural network (nn.h),
 * trained from a replay buffer against a slowly updated target network.
 */

// What the network sees.
typedef enum {
    FEAT_SMALL, // the same 14 bits as the Q-table (ql_state), as 0/1 floats
    FEAT_VIEW,  // a window around the head, rotated so "up" is the heading
    FEAT_TAIL,  // FEAT_VIEW plus where the tail is
    FEAT_HUNGER // FEAT_VIEW plus how close the snake is to starving
} FeatType;

#define VIEW_R 4                   // window radius: 2R+1 cells across (4 gives 9 x 9)
#define VIEW_W (2 * VIEW_R + 1)

#define FEAT_SMALL_N QL_STATE_BITS
// 2 channels (obstacle, food) per window cell + food ahead/behind/left/right + length
#define DQN_VIEW_EXTRA 5
#define DQN_VIEW_COUNT(r) (2 * (2 * (r) + 1) * (2 * (r) + 1) + DQN_VIEW_EXTRA)
#define FEAT_VIEW_N DQN_VIEW_COUNT(VIEW_R)
// + tail ahead/behind/left/right + distance to the tail
#define FEAT_TAIL_N (FEAT_VIEW_N + 5)
// + hunger: steps since food / the starvation limit (0 after eating, 1 = starving)
#define FEAT_HUNGER_N (FEAT_VIEW_N + 1)
#define FEAT_MAX FEAT_TAIL_N
_Static_assert(FEAT_HUNGER_N <= FEAT_MAX, "FEAT_MAX must fit every feature set");

// The inputs are a layer of the network, and layers can be at most NN_MAX wide.
_Static_assert(FEAT_MAX <= NN_MAX,
               "too many view inputs: raise NN_MAX in nn.h or lower VIEW_R");

int dqn_feature_count(FeatType f);
const char *dqn_feature_name(FeatType f);

// Fill x (at least dqn_feature_count(f) floats, each in [0, 1]). Returns the count.
int dqn_features(FeatType f, const Game *g, float *x);

// The egocentric window of any radius (FEAT_VIEW is radius VIEW_R). Layout:
// obstacle plane (w * w, row-major, row 0 = farthest ahead), food plane
// (w * w), then DQN_VIEW_EXTRA values. Returns DQN_VIEW_COUNT(radius).
int dqn_view(const Game *g, int radius, float *x);

// Replay buffer: a ring of the most recent transitions. States are stored as
// bytes (every feature is in [0, 1]), which needs 4x less memory than floats.
typedef struct {
    int cap, size, next, nfeat;
    uint8_t *s, *s2; // cap * nfeat each
    uint8_t *a, *done;
    float *r;
    // Prioritized replay (off while tree is NULL): a sum tree over
    // priority^alpha, so sampling in proportion to priority is O(log cap).
    double *tree;     // 2 * leaves; tree[1] is the total, leaf i is tree[leaves + i]
    int leaves;       // cap rounded up to a power of two
    float alpha;      // 0 = uniform, 1 = fully proportional to |TD error|
    float beta;       // importance-sampling correction, annealed to 1 by the trainer
    float max_prio;   // new transitions get the largest priority seen so far
} Replay;

int replay_init(Replay *rb, int cap, int nfeat);
// Switch on prioritized sampling. Call right after replay_init.
int replay_enable_priority(Replay *rb, float alpha);
// Index of the transition whose slice of the total priority contains
// u * total (u in [0, 1)).
int replay_sample(const Replay *rb, double u);
// Set transition i's priority from its TD error.
void replay_set_priority(Replay *rb, int i, float td_error);
void replay_free(Replay *rb);
void replay_push(Replay *rb, const float *s, int a, float r, const float *s2,
                 int done);
// Decode transition i's states back into floats.
void replay_get(const Replay *rb, int i, float *s, float *s2);

int dqn_best(const Net *net, const float *x);
int dqn_act(const Net *net, const float *x, float eps, uint64_t *rng);

// Bellman target: r if the episode really ended, else r + gamma * max_a q_next[a].
float dqn_target(float r, int done, float gamma, const float *q_next);

// Double DQN target: the online network picks the next action, the target
// network says what it is worth: r + gamma * q_target[argmax_a q_online[a]].
float dqn_target_double(float r, int done, float gamma,
                        const float *q_next_online, const float *q_next_target);
// n-step return: r[0] + gamma r[1] + ... + gamma^(n-1) r[n-1]. Stored in the
// replay buffer in place of a single reward; the bootstrap then uses gamma^n.
float dqn_nstep_return(const float *r, int n, float gamma);

// Sample a batch, take one gradient step on `online`. Returns the mean loss;
// if mean_q is not NULL it receives the mean of max_a Q(s, a) over the batch.
// double_dqn != 0 uses dqn_target_double instead of dqn_target.
// With a prioritized buffer it also updates the sampled priorities.
float dqn_train_batch(Net *online, const Net *target, Replay *rb,
                      int batch, float gamma, float lr, int double_dqn,
                      uint64_t *rng, float *mean_q);

#endif
