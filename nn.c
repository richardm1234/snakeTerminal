#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "game.h" // game_rand

static const char NN_MAGIC[8] = "SNAKENN";
static const char NN_MAGIC_DUELING[8] = "SNAKEND";

#define ADAM_B1 0.9f
#define ADAM_B2 0.999f
#define ADAM_EPS 1e-8f
#define ADAM_TINY 1e-30f // smaller than this counts as 0

static int layer_weights(const Net *net, int l) {
    return net->n[l + 1] * net->n[l];
}

int nn_init(Net *net, int inputs, int hidden, int outputs, uint64_t *rng) {
    memset(net, 0, sizeof(*net));
    net->n[0] = inputs;
    net->n[1] = hidden;
    net->n[2] = hidden;
    net->n[3] = outputs;
    for (int l = 0; l <= NN_LAYERS; l++) {
        if (net->n[l] <= 0 || net->n[l] > NN_MAX) {
            return 0;
        }
    }

    // Parameters, gradients and both Adam arrays (4 of each) share one block.
    size_t total = 0;
    for (int l = 0; l < NN_LAYERS; l++) {
        total += 4 * (size_t)(layer_weights(net, l) + net->n[l + 1]);
    }
    net->mem = calloc(total, sizeof(float));
    if (!net->mem) {
        return 0;
    }
    float *p = net->mem;
    for (int l = 0; l < NN_LAYERS; l++) {
        int nw = layer_weights(net, l), nb = net->n[l + 1];
        net->w[l] = p;  p += nw;  net->b[l] = p;  p += nb;
        net->gw[l] = p; p += nw;  net->gb[l] = p; p += nb;
        net->mw[l] = p; p += nw;  net->mb[l] = p; p += nb;
        net->vw[l] = p; p += nw;  net->vb[l] = p; p += nb;
    }

    // He initialisation: uniform in [-limit, limit] with limit = sqrt(6 / fan_in),
    // which keeps the size of the activations roughly constant through ReLU layers.
    // Biases start at 0.
    for (int l = 0; l < NN_LAYERS; l++) {
        float limit = sqrtf(6.0f / (float)net->n[l]);
        for (int i = 0; i < layer_weights(net, l); i++) {
            float u = (float)game_rand(rng) / 4294967296.0f; // [0, 1)
            net->w[l][i] = (2.0f * u - 1.0f) * limit;
        }
    }
    return 1;
}

int nn_init_dueling(Net *net, int inputs, int hidden, int actions, uint64_t *rng) {
    if (!nn_init(net, inputs, hidden, actions + 1, rng)) { // +1 for V(s)
        return 0;
    }
    net->dueling = 1;
    return 1;
}

void nn_free(Net *net) {
    free(net->mem);
    memset(net, 0, sizeof(*net));
}

void nn_forward(const Net *net, const float *x, NNCache *c) {
    memcpy(c->a[0], x, sizeof(float) * net->n[0]);
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
    if (net->dueling) {
        float *q = c->a[NN_LAYERS];
        int na = net->n[NN_LAYERS] - 1;
        float v = q[na], mean = 0.0f;
        for (int a = 0; a < na; a++) mean += q[a];
        mean /= (float)na;
        for (int a = 0; a < na; a++) q[a] = v + q[a] - mean;
    }
}

void nn_backward(Net *net, const NNCache *c, int action, float dq) {
    // delta[o] = dLoss / d(pre-activation of output o of the current layer).
    // At the top, only the output for `action` was used in the loss.
    float delta[NN_MAX] = {0};
    float prev[NN_MAX];
    if (net->dueling) {
        // Q(action) = V + A(action) - mean(A), so
        //   dQ(action)/dV = 1,  dQ(action)/dA(b) = [b == action] - 1/na.
        int na = net->n[NN_LAYERS] - 1;
        for (int b = 0; b < na; b++) {
            delta[b] = dq * ((b == action ? 1.0f : 0.0f) - 1.0f / (float)na);
        }
        delta[na] = dq;
    } else {
        delta[action] = dq;
    }

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

        // Pass the error one layer down: dLoss/d(in[i]) = sum_o w[o,i] * delta[o],
        // then through the ReLU, whose slope is 1 where it was active and 0 elsewhere.
        for (int i = 0; i < nin; i++) {
            prev[i] = 0.0f;
        }
        for (int o = 0; o < nout; o++) {
            if (delta[o] == 0.0f) {
                continue;
            }
            const float *row = net->w[l] + o * nin;
            for (int i = 0; i < nin; i++) {
                prev[i] += row[i] * delta[o];
            }
        }
        for (int i = 0; i < nin; i++) {
            delta[i] = in[i] > 0.0f ? prev[i] : 0.0f;
        }
    }
}

void nn_zero_grad(Net *net) {
    for (int l = 0; l < NN_LAYERS; l++) {
        memset(net->gw[l], 0, sizeof(float) * layer_weights(net, l));
        memset(net->gb[l], 0, sizeof(float) * net->n[l + 1]);
    }
}

// Adam for one array of parameters.
static void adam(float *p, const float *g, float *m, float *v, int n,
                 float lr, float scale, float c1, float c2) {
    for (int i = 0; i < n; i++) {
        float gi = g[i] * scale;
        float mi = ADAM_B1 * m[i] + (1.0f - ADAM_B1) * gi;
        float vi = ADAM_B2 * v[i] + (1.0f - ADAM_B2) * gi * gi;
        // Flush tiny values to 0: otherwise m gets stuck in the slow
        // "denormal" float range when a gradient stays 0 (see CNN_TUTORIAL.md).
        m[i] = fabsf(mi) < ADAM_TINY ? 0.0f : mi;
        v[i] = vi < ADAM_TINY ? 0.0f : vi;
        float mhat = m[i] / c1; // bias correction: m and v start at 0
        float vhat = v[i] / c2;
        p[i] -= lr * mhat / (sqrtf(vhat) + ADAM_EPS);
    }
}

void nn_adam_step(Net *net, float lr, int batch) {
    net->t++;
    float c1 = 1.0f - powf(ADAM_B1, (float)net->t);
    float c2 = 1.0f - powf(ADAM_B2, (float)net->t);
    float scale = 1.0f / (float)batch; // average over the batch
    for (int l = 0; l < NN_LAYERS; l++) {
        adam(net->w[l], net->gw[l], net->mw[l], net->vw[l],
             layer_weights(net, l), lr, scale, c1, c2);
        adam(net->b[l], net->gb[l], net->mb[l], net->vb[l],
             net->n[l + 1], lr, scale, c1, c2);
    }
}

void nn_copy(Net *dst, const Net *src) {
    for (int l = 0; l < NN_LAYERS; l++) {
        memcpy(dst->w[l], src->w[l], sizeof(float) * layer_weights(src, l));
        memcpy(dst->b[l], src->b[l], sizeof(float) * src->n[l + 1]);
    }
}

int nn_save(const Net *net, int tag, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        return 0;
    }
    int ok = fwrite(net->dueling ? NN_MAGIC_DUELING : NN_MAGIC,
                    sizeof(NN_MAGIC), 1, f) == 1 &&
             fwrite(&tag, sizeof(tag), 1, f) == 1 &&
             fwrite(net->n, sizeof(net->n), 1, f) == 1;
    for (int l = 0; ok && l < NN_LAYERS; l++) {
        ok = fwrite(net->w[l], sizeof(float), layer_weights(net, l), f) ==
                 (size_t)layer_weights(net, l) &&
             fwrite(net->b[l], sizeof(float), net->n[l + 1], f) ==
                 (size_t)net->n[l + 1];
    }
    return fclose(f) == 0 && ok;
}

int nn_load(Net *net, int *tag, const char *path) {
    memset(net, 0, sizeof(*net));
    FILE *f = fopen(path, "rb");
    if (!f) {
        return 0;
    }
    char magic[sizeof(NN_MAGIC)];
    int n[NN_LAYERS + 1];
    uint64_t rng = 1;
    int ok = fread(magic, sizeof(magic), 1, f) == 1;
    int dueling = ok && memcmp(magic, NN_MAGIC_DUELING, sizeof(magic)) == 0;
    ok = ok && (dueling || memcmp(magic, NN_MAGIC, sizeof(magic)) == 0) &&
         fread(tag, sizeof(*tag), 1, f) == 1 &&
         fread(n, sizeof(n), 1, f) == 1 &&
         n[1] == n[2] && (dueling ? n[3] >= 2 : 1) &&
         nn_init(net, n[0], n[1], n[3], &rng);
    if (ok) {
        net->dueling = dueling;
    }
    for (int l = 0; ok && l < NN_LAYERS; l++) {
        ok = fread(net->w[l], sizeof(float), layer_weights(net, l), f) ==
                 (size_t)layer_weights(net, l) &&
             fread(net->b[l], sizeof(float), net->n[l + 1], f) ==
                 (size_t)net->n[l + 1];
    }
    fclose(f);
    if (!ok && net->mem) {
        nn_free(net);
    }
    return ok;
}
