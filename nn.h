#ifndef NN_H
#define NN_H

#include <stdint.h>

/*
 * A tiny fully connected neural network, written by hand so every step is
 * visible: inputs -> hidden -> hidden -> outputs, ReLU on the hidden layers,
 * no activation on the output (Q-values can be any real number).
 *
 * Layer l maps n[l] values to n[l+1] values with
 *     out[o] = b[l][o] + sum_i w[l][o * n[l] + i] * in[i]
 */

#define NN_LAYERS 3
#define NN_MAX 256 // widest layer this code supports (inputs included)

typedef struct {
    int n[NN_LAYERS + 1];                 // n[0] inputs ... n[NN_LAYERS] outputs
    float *w[NN_LAYERS], *b[NN_LAYERS];   // parameters
    float *gw[NN_LAYERS], *gb[NN_LAYERS]; // gradients, summed over a batch
    float *mw[NN_LAYERS], *mb[NN_LAYERS]; // Adam: running mean of gradients
    float *vw[NN_LAYERS], *vb[NN_LAYERS]; // Adam: running mean of squared gradients
    int t;                                // Adam: number of steps taken
    float *mem;                           // one allocation holding all of the above
} Net;

// Everything the forward pass computed, which the backward pass needs.
// a[0] is the input, a[1], a[2] the hidden layers (after ReLU), a[3] the output.
typedef struct {
    float a[NN_LAYERS + 1][NN_MAX];
} NNCache;

// Allocate and randomly initialise. Returns 0 on failure.
int nn_init(Net *net, int inputs, int hidden, int outputs, uint64_t *rng);
void nn_free(Net *net);

void nn_forward(const Net *net, const float *x, NNCache *c);

// Backward pass for one sample. dq is dLoss/dQ for the output `action`;
// the other outputs get no gradient. Adds to gw/gb.
void nn_backward(Net *net, const NNCache *c, int action, float dq);

void nn_zero_grad(Net *net);

// One Adam update using the summed gradients divided by `batch`.
void nn_adam_step(Net *net, float lr, int batch);

// Copy the parameters of src into dst (same shape). Used for the target network.
void nn_copy(Net *dst, const Net *src);

// Save/load the parameters plus one caller-defined integer (the feature type).
int nn_save(const Net *net, int tag, const char *path);
int nn_load(Net *net, int *tag, const char *path);

#endif
