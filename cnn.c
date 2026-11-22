#include "cnn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "game.h" // game_rand

#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_thread_num(void) { return 0; }
static int omp_get_max_threads(void) { return 1; }
#endif

static const char CNN_MAGIC[8] = "SNAKECN";

#define ADAM_B1 0.9f
#define ADAM_B2 0.999f
#define ADAM_EPS 1e-8f
#define ADAM_TINY 1e-30f // smaller than this counts as 0 (see cnn_adam_step)

// ---------------------------------------------------------------- shapes

static int grid_size(const CNN *net, int l) { // floats in grid l (0 = input)
    return net->ch[l] * net->side[l] * net->side[l];
}

int cnn_input_count(const CNN *net) {
    return DQN_VIEW_COUNT(net->radius);
}

static float uniform(uint64_t *rng) { // [0, 1)
    return (float)game_rand(rng) / 4294967296.0f;
}

int cnn_init(CNN *net, int radius, int c1, int c2, int hidden, uint64_t *rng) {
    memset(net, 0, sizeof(*net));
    if (radius < 2 || radius > CNN_MAX_RADIUS || c1 <= 0 || c2 <= 0 ||
        hidden <= 0) {
        return 0; // radius 2 is the smallest window both convolutions fit in
    }
    net->radius = radius;
    net->ch[0] = 2; net->ch[1] = c1; net->ch[2] = c2;
    net->stride[0] = 1; net->stride[1] = 2;
    net->side[0] = 2 * radius + 1;
    for (int l = 0; l < 2; l++) {
        // A "valid" convolution: the filter must fit inside the grid.
        net->side[l + 1] = (net->side[l] - CNN_K) / net->stride[l] + 1;
    }
    net->flat = grid_size(net, 2) + DQN_VIEW_EXTRA;
    net->hidden = hidden;
    net->outputs = QL_ACTIONS;

    // Weights per layer: one row of fan_in values per output.
    int fan_in[CNN_LAYERS] = {CNN_K * CNN_K * net->ch[0], CNN_K * CNN_K * c1,
                              net->flat, hidden};
    int fan_out[CNN_LAYERS] = {c1, c2, hidden, net->outputs};
    int n = 0;
    for (int l = 0; l < CNN_LAYERS; l++) {
        net->w_off[l] = n; n += fan_in[l] * fan_out[l];
        net->b_off[l] = n; n += fan_out[l];
    }
    net->nparam = n;

    net->p = calloc(3 * (size_t)n, sizeof(float));
    if (!net->p) {
        return 0;
    }
    net->m = net->p + n;
    net->v = net->m + n;

    // He initialisation, as in nn.c. For a convolution, fan_in is the size of
    // one patch (3 x 3 x input channels): that's how many inputs each output sees.
    for (int l = 0; l < CNN_LAYERS; l++) {
        float limit = sqrtf(6.0f / (float)fan_in[l]);
        float *w = net->p + net->w_off[l];
        for (int i = 0; i < fan_in[l] * fan_out[l]; i++) {
            w[i] = (2.0f * uniform(rng) - 1.0f) * limit;
        }
    }
    return 1;
}

void cnn_free(CNN *net) {
    free(net->p);
    memset(net, 0, sizeof(*net));
}

int cnn_cache_init(CNNCache *c, const CNN *net) {
    memset(c, 0, sizeof(*c));
    int patch = CNN_K * CNN_K * (net->ch[0] > net->ch[1] ? net->ch[0] : net->ch[1]);
    c->x = malloc(sizeof(float) * cnn_input_count(net));
    c->a1 = malloc(sizeof(float) * grid_size(net, 1));
    c->d1 = malloc(sizeof(float) * grid_size(net, 1));
    c->f = malloc(sizeof(float) * net->flat);
    c->df = malloc(sizeof(float) * net->flat);
    c->h = malloc(sizeof(float) * net->hidden);
    c->dh = malloc(sizeof(float) * net->hidden);
    c->patch = malloc(sizeof(float) * patch);
    c->dpatch = malloc(sizeof(float) * patch);
    if (!c->x || !c->a1 || !c->d1 || !c->f || !c->df || !c->h || !c->dh ||
        !c->patch || !c->dpatch) {
        cnn_cache_free(c);
        return 0;
    }
    return 1;
}

void cnn_cache_free(CNNCache *c) {
    free(c->x); free(c->a1); free(c->d1); free(c->f); free(c->df);
    free(c->h); free(c->dh); free(c->patch); free(c->dpatch);
    memset(c, 0, sizeof(*c));
}

// ---------------------------------------------------------------- convolution

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

// dout is dLoss/d(pre-activation), i.e. already multiplied by the ReLU slope.
// Adds to the filter gradients gw, gb. If din is not NULL, it receives
// dLoss/d(input).
static void conv_backward(const float *in, int cin, int side, const float *w,
                          int cout, int stride, int oside, const float *dout,
                          float *gw, float *gb, float *din, float *patch,
                          float *dpatch) {
    int K = CNN_K * CNN_K * cin;
    if (din) {
        memset(din, 0, sizeof(float) * cin * side * side);
    }
    for (int oy = 0; oy < oside; oy++) {
        for (int ox = 0; ox < oside; ox++) {
            gather(in, cin, side, oy * stride, ox * stride, patch);
            if (din) {
                memset(dpatch, 0, sizeof(float) * K);
            }
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
            if (din) {
                scatter_add(din, cin, side, oy * stride, ox * stride, dpatch);
            }
        }
    }
}

// ---------------------------------------------------------------- dense

static void dense_forward(const float *in, int nin, const float *w,
                          const float *b, int nout, int relu, float *out) {
    for (int o = 0; o < nout; o++) {
        const float *row = w + o * nin;
        float sum = b[o];
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < nin; i++) {
            sum += row[i] * in[i];
        }
        out[o] = (relu && sum < 0.0f) ? 0.0f : sum;
    }
}

// Same as in nn.c: delta is dLoss/d(pre-activation), din (if not NULL) gets
// dLoss/d(input).
static void dense_backward(const float *in, int nin, const float *w, int nout,
                           const float *delta, float *gw, float *gb, float *din) {
    if (din) {
        memset(din, 0, sizeof(float) * nin);
    }
    for (int o = 0; o < nout; o++) {
        float d = delta[o];
        if (d == 0.0f) {
            continue;
        }
        gb[o] += d;
        float *grow = gw + o * nin;
        const float *row = w + o * nin;
        for (int i = 0; i < nin; i++) {
            grow[i] += d * in[i];
        }
        if (din) {
            for (int i = 0; i < nin; i++) {
                din[i] += d * row[i];
            }
        }
    }
}

// ---------------------------------------------------------------- the network

void cnn_forward(const CNN *net, const float *x, CNNCache *c) {
    if (x != c->x) {
        memcpy(c->x, x, sizeof(float) * cnn_input_count(net));
    }
    const float *p = net->p;
    conv_forward(c->x, net->ch[0], net->side[0], p + net->w_off[0],
                 p + net->b_off[0], net->ch[1], net->stride[0], net->side[1],
                 c->a1, c->patch);
    conv_forward(c->a1, net->ch[1], net->side[1], p + net->w_off[1],
                 p + net->b_off[1], net->ch[2], net->stride[1], net->side[2],
                 c->f, c->patch);
    // Flatten: conv2 already wrote its grid to the start of f. The extras
    // (food direction, length) skip the convolutions and join here.
    memcpy(c->f + grid_size(net, 2), c->x + grid_size(net, 0),
           sizeof(float) * DQN_VIEW_EXTRA);
    dense_forward(c->f, net->flat, p + net->w_off[2], p + net->b_off[2],
                  net->hidden, 1, c->h);
    dense_forward(c->h, net->hidden, p + net->w_off[3], p + net->b_off[3],
                  net->outputs, 0, c->q);
}

void cnn_backward(const CNN *net, CNNCache *c, int action, float dq, float *grad) {
    const float *p = net->p;
    float dq_out[3] = {0.0f, 0.0f, 0.0f};
    dq_out[action] = dq; // only the Q-value of the action taken was in the loss

    // dense2 -> h, through dense1's ReLU
    dense_backward(c->h, net->hidden, p + net->w_off[3], net->outputs, dq_out,
                   grad + net->w_off[3], grad + net->b_off[3], c->dh);
    for (int i = 0; i < net->hidden; i++) {
        if (c->h[i] <= 0.0f) c->dh[i] = 0.0f;
    }

    // dense1 -> f, through conv2's ReLU (the extras need no gradient)
    dense_backward(c->f, net->flat, p + net->w_off[2], net->hidden, c->dh,
                   grad + net->w_off[2], grad + net->b_off[2], c->df);
    int n2 = grid_size(net, 2);
    for (int i = 0; i < n2; i++) {
        if (c->f[i] <= 0.0f) c->df[i] = 0.0f;
    }

    // conv2 -> a1, through conv1's ReLU
    conv_backward(c->a1, net->ch[1], net->side[1], p + net->w_off[1],
                  net->ch[2], net->stride[1], net->side[2], c->df,
                  grad + net->w_off[1], grad + net->b_off[1], c->d1, c->patch,
                  c->dpatch);
    int n1 = grid_size(net, 1);
    for (int i = 0; i < n1; i++) {
        if (c->a1[i] <= 0.0f) c->d1[i] = 0.0f;
    }

    // conv1: parameter gradients only, nothing below it to update
    conv_backward(c->x, net->ch[0], net->side[0], p + net->w_off[0],
                  net->ch[1], net->stride[0], net->side[1], c->d1,
                  grad + net->w_off[0], grad + net->b_off[0], NULL, c->patch,
                  c->dpatch);
}

void cnn_adam_step(CNN *net, const float *grad, float lr, int batch) {
    net->t++;
    float c1 = 1.0f - powf(ADAM_B1, (float)net->t);
    float c2 = 1.0f - powf(ADAM_B2, (float)net->t);
    float scale = 1.0f / (float)batch;
    // Exactly the Adam of nn.c, but one loop covers every layer.
    for (int i = 0; i < net->nparam; i++) {
        float g = grad[i] * scale;
        float m = ADAM_B1 * net->m[i] + (1.0f - ADAM_B1) * g;
        float v = ADAM_B2 * net->v[i] + (1.0f - ADAM_B2) * g * g;
        // Flush tiny values to 0. Without this, a parameter whose gradient
        // stays 0 has m shrink into the "denormal" range, where 0.9 * m rounds
        // back to m and never reaches 0, and denormal arithmetic is many times
        // slower. See section 8 of CNN_TUTORIAL.md.
        net->m[i] = fabsf(m) < ADAM_TINY ? 0.0f : m;
        net->v[i] = v < ADAM_TINY ? 0.0f : v;
        net->p[i] -= lr * (net->m[i] / c1) / (sqrtf(net->v[i] / c2) + ADAM_EPS);
    }
}

void cnn_copy(CNN *dst, const CNN *src) {
    memcpy(dst->p, src->p, sizeof(float) * src->nparam);
}

static int argmax3(const float *q) {
    int best = 0;
    for (int a = 1; a < QL_ACTIONS; a++) {
        if (q[a] > q[best]) best = a;
    }
    return best;
}

int cnn_best(const CNN *net, const float *x, CNNCache *c) {
    cnn_forward(net, x, c);
    return argmax3(c->q);
}

int cnn_act(const CNN *net, const float *x, CNNCache *c, float eps,
            uint64_t *rng) {
    if (uniform(rng) < eps) {
        return (int)(game_rand(rng) % QL_ACTIONS); // explore
    }
    return cnn_best(net, x, c);                    // exploit
}

// ---------------------------------------------------------------- training

int cnn_work_init(CNNWork *w, const CNN *net, int batch, int threads) {
    memset(w, 0, sizeof(*w));
    if (threads <= 0) threads = omp_get_max_threads();
    if (threads > batch) threads = batch; // more would have nothing to do
    w->threads = threads;
    w->batch = batch;
    w->cache = calloc(threads, sizeof(CNNCache));
    w->s = malloc(sizeof(float) * threads * cnn_input_count(net));
    w->s2 = malloc(sizeof(float) * threads * cnn_input_count(net));
    w->grad = malloc(sizeof(float) * threads * (size_t)net->nparam);
    w->idx = malloc(sizeof(int) * batch);
    if (!w->cache || !w->s || !w->s2 || !w->grad || !w->idx) {
        cnn_work_free(w);
        return 0;
    }
    for (int t = 0; t < threads; t++) {
        if (!cnn_cache_init(&w->cache[t], net)) {
            cnn_work_free(w);
            return 0;
        }
    }
    return 1;
}

void cnn_work_free(CNNWork *w) {
    if (w->cache) {
        for (int t = 0; t < w->threads; t++) cnn_cache_free(&w->cache[t]);
    }
    free(w->cache);
    free(w->s);
    free(w->s2);
    free(w->grad);
    free(w->idx);
    memset(w, 0, sizeof(*w));
}

float cnn_train_batch(CNN *online, const CNN *target, const Replay *rb,
                      CNNWork *w, float gamma, float lr, uint64_t *rng,
                      float *mean_q) {
    int np = online->nparam, nin = cnn_input_count(online);
    // Draw the batch first, on one thread, so the run doesn't depend on
    // which thread happens to call game_rand first.
    for (int k = 0; k < w->batch; k++) {
        w->idx[k] = (int)(game_rand(rng) % (uint32_t)rb->size);
    }
    memset(w->grad, 0, sizeof(float) * w->threads * (size_t)np);

    float loss = 0.0f, qsum = 0.0f;
    // One parallel region per batch: waking the threads up costs about as
    // much as a few samples, so do it once, not once per loop.
    #pragma omp parallel num_threads(w->threads)
    {
        int t = omp_get_thread_num();
        CNNCache *c = &w->cache[t];
        // s and s2 need their own buffers: cnn_forward copies its input
        // into c->x, so decoding s straight into c->x would be overwritten
        // by the target pass on s2.
        float *s = w->s + (size_t)t * nin, *s2 = w->s2 + (size_t)t * nin;
        float *grad = w->grad + (size_t)t * np;

        // Each sample is independent: thread t works through its share of the
        // batch with its own cache and adds into its own gradient buffer.
        #pragma omp for schedule(static) reduction(+:loss, qsum)
        for (int k = 0; k < w->batch; k++) {
            int i = w->idx[k], a = rb->a[i];
            replay_get(rb, i, s, s2);

            // Target network first (a constant, no gradient), then the online
            // network, so the cache holds the online pass for backprop.
            cnn_forward(target, s2, c);
            float y = dqn_target(rb->r[i], rb->done[i], gamma, c->q);
            cnn_forward(online, s, c);
            qsum += c->q[argmax3(c->q)];

            float d = c->q[a] - y; // Huber loss, as in dqn.c
            if (fabsf(d) <= 1.0f) {
                loss += 0.5f * d * d;
            } else {
                loss += fabsf(d) - 0.5f;
                d = d > 0.0f ? 1.0f : -1.0f;
            }
            cnn_backward(online, c, a, d, grad);
        }
        // (the end of an "omp for" waits until every thread is done)

        // Add the per-thread gradients into buffer 0, always in the same
        // order, so a run is repeatable (for a given thread count).
        if (w->threads > 1) {
            #pragma omp for schedule(static)
            for (int j = 0; j < np; j++) {
                float sum = w->grad[j];
                for (int u = 1; u < w->threads; u++) {
                    sum += w->grad[(size_t)u * np + j];
                }
                w->grad[j] = sum;
            }
        }
    }
    cnn_adam_step(online, w->grad, lr, w->batch);

    if (mean_q) *mean_q = qsum / w->batch;
    return loss / w->batch;
}

// ---------------------------------------------------------------- files

int cnn_save(const CNN *net, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        return 0;
    }
    int shape[4] = {net->radius, net->ch[1], net->ch[2], net->hidden};
    int ok = fwrite(CNN_MAGIC, sizeof(CNN_MAGIC), 1, f) == 1 &&
             fwrite(shape, sizeof(shape), 1, f) == 1 &&
             fwrite(net->p, sizeof(float), net->nparam, f) == (size_t)net->nparam;
    return fclose(f) == 0 && ok;
}

int cnn_load(CNN *net, const char *path) {
    memset(net, 0, sizeof(*net));
    FILE *f = fopen(path, "rb");
    if (!f) {
        return 0;
    }
    char magic[sizeof(CNN_MAGIC)];
    int shape[4];
    uint64_t rng = 1;
    int ok = fread(magic, sizeof(magic), 1, f) == 1 &&
             memcmp(magic, CNN_MAGIC, sizeof(magic)) == 0 &&
             fread(shape, sizeof(shape), 1, f) == 1 &&
             cnn_init(net, shape[0], shape[1], shape[2], shape[3], &rng) &&
             fread(net->p, sizeof(float), net->nparam, f) == (size_t)net->nparam;
    fclose(f);
    if (!ok && net->p) {
        cnn_free(net);
    }
    return ok;
}
