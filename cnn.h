#ifndef CNN_H
#define CNN_H

#include <stdint.h>

#include "dqn.h"

/*
 * A small convolutional Q-network, written by hand like nn.c.
 *
 * Input: the egocentric window from dqn_view (2 planes of w x w cells, obstacle
 * and food, w = 2 * radius + 1) plus DQN_VIEW_EXTRA plain numbers.
 *
 *   grid  2 x w  x w
 *   conv1 3x3, stride 1, ReLU  -> c1 x w1 x w1     w1 = w - 2
 *   conv2 3x3, stride 2, ReLU  -> c2 x w2 x w2     w2 = (w1 - 3) / 2 + 1
 *   flatten, append the extras -> c2 * w2 * w2 + DQN_VIEW_EXTRA values
 *   dense, ReLU                -> hidden
 *   dense                      -> 3 Q-values
 *
 * Grids are stored channel by channel ("CHW"): value (c, y, x) of a grid with
 * side s is at index c * s * s + y * s + x.
 *
 * All parameters live in one flat array p, so Adam, copying, saving and
 * gradient buffers are each a single loop over nparam floats.
 */

#define CNN_K 3        // every filter is 3 x 3
#define CNN_LAYERS 4   // conv1, conv2, dense1, dense2
#define CNN_MAX_RADIUS 19

typedef struct {
    int radius;
    int side[3];      // grid side: input, after conv1, after conv2
    int ch[3];        // channels:  input (2), conv1, conv2
    int stride[2];    // conv strides (1 and 2)
    int flat;         // c2 * w2 * w2 + DQN_VIEW_EXTRA: input of dense1
    int hidden, outputs;
    // Where each layer's weights and biases start inside p.
    int w_off[CNN_LAYERS], b_off[CNN_LAYERS];
    int nparam;
    float *p;         // parameters
    float *m, *v;     // Adam state, one value per parameter
    int t;            // Adam steps taken
} CNN;

// Everything one forward pass computed, and scratch space for the backward
// pass. Each thread needs its own.
typedef struct {
    float *x;         // input: grid + extras (dqn_view layout)
    float *a1;        // conv1 output (after ReLU)
    float *f;         // conv2 output (after ReLU), then the extras: dense1 input
    float *h;         // dense1 output (after ReLU)
    float q[3];       // Q-values
    float *d1, *df, *dh;      // gradients with respect to a1, f, h
    float *patch, *dpatch;    // one 3x3 patch of every input channel
} CNNCache;

// Allocate and randomly initialise. Returns 0 on failure (bad radius or no memory).
int cnn_init(CNN *net, int radius, int c1, int c2, int hidden, uint64_t *rng);
void cnn_free(CNN *net);

int cnn_input_count(const CNN *net); // DQN_VIEW_COUNT(radius)

int cnn_cache_init(CNNCache *c, const CNN *net);
void cnn_cache_free(CNNCache *c);

// Forward pass. Results are in c->q (and the intermediate arrays of c).
void cnn_forward(const CNN *net, const float *x, CNNCache *c);

// Backward pass for one sample after cnn_forward: dq is dLoss/dQ for `action`.
// Adds the parameter gradients to grad (nparam floats).
void cnn_backward(const CNN *net, CNNCache *c, int action, float dq, float *grad);

// One Adam step using grad divided by `batch`.
void cnn_adam_step(CNN *net, const float *grad, float lr, int batch);

void cnn_copy(CNN *dst, const CNN *src);

int cnn_best(const CNN *net, const float *x, CNNCache *c);
int cnn_act(const CNN *net, const float *x, CNNCache *c, float eps, uint64_t *rng);

// What one training step needs besides the networks: a cache and a gradient
// buffer for every thread.
typedef struct {
    int threads;
    CNNCache *cache;              // one per thread
    float *s, *s2;                // threads * inputs each: states, next states
    float *grad;                  // threads * nparam
    int *idx;                     // the sampled batch
    int batch;
} CNNWork;

// threads <= 0 means "as many as OpenMP offers".
int cnn_work_init(CNNWork *w, const CNN *net, int batch, int threads);
void cnn_work_free(CNNWork *w);

// Like dqn_train_batch: sample a batch, take one Adam step on `online`.
// The batch is split over the threads in w.
float cnn_train_batch(CNN *online, const CNN *target, const Replay *rb,
                      CNNWork *w, float gamma, float lr, uint64_t *rng,
                      float *mean_q);

int cnn_save(const CNN *net, const char *path);
int cnn_load(CNN *net, const char *path);

#endif
