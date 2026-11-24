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
    FEAT_TAIL   // FEAT_VIEW plus where the tail is
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
#define FEAT_MAX FEAT_TAIL_N

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
} Replay;

int replay_init(Replay *rb, int cap, int nfeat);
void replay_free(Replay *rb);
void replay_push(Replay *rb, const float *s, int a, float r, const float *s2,
                 int done);
// Decode transition i's states back into floats.
void replay_get(const Replay *rb, int i, float *s, float *s2);

int dqn_best(const Net *net, const float *x);
int dqn_act(const Net *net, const float *x, float eps, uint64_t *rng);

// Bellman target: r if the episode really ended, else r + gamma * max_a q_next[a].
float dqn_target(float r, int done, float gamma, const float *q_next);

// n-step return: r[0] + gamma r[1] + ... + gamma^(n-1) r[n-1]. Stored in the
// replay buffer in place of a single reward; the bootstrap then uses gamma^n.
float dqn_nstep_return(const float *r, int n, float gamma);

// Sample a batch, take one gradient step on `online`. Returns the mean loss;
// if mean_q is not NULL it receives the mean of max_a Q(s, a) over the batch.
float dqn_train_batch(Net *online, const Net *target, const Replay *rb,
                      int batch, float gamma, float lr, uint64_t *rng,
                      float *mean_q);

#endif
