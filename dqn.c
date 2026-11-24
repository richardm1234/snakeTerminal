#include "dqn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int dqn_feature_count(FeatType f) {
    switch (f) {
        case FEAT_VIEW: return FEAT_VIEW_N;
        case FEAT_TAIL: return FEAT_TAIL_N;
        default:        return FEAT_SMALL_N;
    }
}

const char *dqn_feature_name(FeatType f) {
    switch (f) {
        case FEAT_VIEW: return "view";
        case FEAT_TAIL: return "tail";
        default:        return "small";
    }
}

static int small_features(const Game *g, float *x) {
    int s = ql_state(g);
    for (int i = 0; i < FEAT_SMALL_N; i++) {
        x[i] = (float)((s >> i) & 1);
    }
    return FEAT_SMALL_N;
}

// Is p ahead of / behind / left of / right of the head, relative to the
// heading? Writes 4 values (0 or 1) to out.
static void relative_direction(const Game *g, Point p, float *out) {
    int fx = DIR_DX[g->dir], fy = DIR_DY[g->dir];
    int rx = DIR_DX[(g->dir + 1) % 4], ry = DIR_DY[(g->dir + 1) % 4];
    int vx = p.x - g->body[0].x, vy = p.y - g->body[0].y;
    int ahead = vx * fx + vy * fy, side = vx * rx + vy * ry;
    out[0] = ahead > 0;
    out[1] = ahead < 0;
    out[2] = side < 0; // left
    out[3] = side > 0; // right
}

int dqn_view(const Game *g, int radius, float *x) {
    // Mark the cells that would kill the head (like game_is_deadly: the tail
    // moves away this step, so it is not marked). The head itself isn't either.
    static unsigned char body[HEIGHT][WIDTH];
    memset(body, 0, sizeof(body));
    for (int i = 1; i < g->len - 1; i++) {
        body[g->body[i].y][g->body[i].x] = 1;
    }

    Point head = g->body[0];
    int w = 2 * radius + 1;
    // Unit vectors for "forward" and "right" in board coordinates.
    int fx = DIR_DX[g->dir], fy = DIR_DY[g->dir];
    int rx = DIR_DX[(g->dir + 1) % 4], ry = DIR_DY[(g->dir + 1) % 4];

    float *obstacle = x, *food = x + w * w;
    // Window row 0 is the farthest row ahead, column 0 the farthest left.
    for (int row = 0; row < w; row++) {
        for (int col = 0; col < w; col++) {
            int fwd = radius - row, right = col - radius;
            int px = head.x + fwd * fx + right * rx;
            int py = head.y + fwd * fy + right * ry;
            int k = row * w + col;
            int wall = px <= 0 || px >= WIDTH - 1 || py <= 0 || py >= HEIGHT - 1;
            obstacle[k] = (wall || body[py][px]) ? 1.0f : 0.0f;
            food[k] = (px == g->food.x && py == g->food.y) ? 1.0f : 0.0f;
        }
    }

    // Food direction relative to the heading, so it also works when the food
    // is outside the window.
    float *extra = x + 2 * w * w;
    relative_direction(g, g->food, extra);
    extra[4] = (float)g->len / MAX_LEN;
    return DQN_VIEW_COUNT(radius);
}

static int view_features(const Game *g, float *x) {
    return dqn_view(g, VIEW_R, x);
}

// The view, plus where the tail is. The tail is the one part of the body that
// is always safe to follow (it moves away every step), and for a long snake it
// is usually outside the window.
static int tail_features(const Game *g, float *x) {
    view_features(g, x);
    float *tail = x + FEAT_VIEW_N;
    Point head = g->body[0], end = g->body[g->len - 1];
    relative_direction(g, end, tail);
    // Manhattan distance, scaled so the farthest two interior cells give 1.
    int dist = abs(end.x - head.x) + abs(end.y - head.y);
    tail[4] = (float)dist / (WIDTH - 3 + HEIGHT - 3);
    return FEAT_TAIL_N;
}

int dqn_features(FeatType f, const Game *g, float *x) {
    switch (f) {
        case FEAT_VIEW: return view_features(g, x);
        case FEAT_TAIL: return tail_features(g, x);
        default:        return small_features(g, x);
    }
}

int replay_init(Replay *rb, int cap, int nfeat) {
    memset(rb, 0, sizeof(*rb));
    rb->cap = cap;
    rb->nfeat = nfeat;
    rb->s = malloc((size_t)cap * nfeat);
    rb->s2 = malloc((size_t)cap * nfeat);
    rb->a = malloc(cap);
    rb->done = malloc(cap);
    rb->r = malloc(sizeof(float) * cap);
    if (!rb->s || !rb->s2 || !rb->a || !rb->done || !rb->r) {
        replay_free(rb);
        return 0;
    }
    return 1;
}

void replay_free(Replay *rb) {
    free(rb->s);
    free(rb->s2);
    free(rb->a);
    free(rb->done);
    free(rb->r);
    memset(rb, 0, sizeof(*rb));
}

static void encode(uint8_t *dst, const float *x, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = (uint8_t)(x[i] * 255.0f + 0.5f);
    }
}

static void decode(float *x, const uint8_t *src, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = src[i] / 255.0f;
    }
}

void replay_get(const Replay *rb, int i, float *s, float *s2) {
    decode(s, rb->s + (size_t)i * rb->nfeat, rb->nfeat);
    decode(s2, rb->s2 + (size_t)i * rb->nfeat, rb->nfeat);
}

void replay_push(Replay *rb, const float *s, int a, float r, const float *s2,
                 int done) {
    int i = rb->next;
    encode(rb->s + (size_t)i * rb->nfeat, s, rb->nfeat);
    encode(rb->s2 + (size_t)i * rb->nfeat, s2, rb->nfeat);
    rb->a[i] = (uint8_t)a;
    rb->r[i] = r;
    rb->done[i] = (uint8_t)done;
    rb->next = (i + 1) % rb->cap; // overwrite the oldest once full
    if (rb->size < rb->cap) {
        rb->size++;
    }
}

static int argmax3(const float *q) {
    int best = 0;
    for (int a = 1; a < QL_ACTIONS; a++) {
        if (q[a] > q[best]) {
            best = a;
        }
    }
    return best;
}

int dqn_best(const Net *net, const float *x) {
    NNCache c;
    nn_forward(net, x, &c);
    return argmax3(c.a[NN_LAYERS]);
}

int dqn_act(const Net *net, const float *x, float eps, uint64_t *rng) {
    float u = (float)game_rand(rng) / 4294967296.0f;
    if (u < eps) {
        return (int)(game_rand(rng) % QL_ACTIONS); // explore
    }
    return dqn_best(net, x);                       // exploit
}

float dqn_target(float r, int done, float gamma, const float *q_next) {
    if (done) {
        return r;
    }
    return r + gamma * q_next[argmax3(q_next)];
}

float dqn_train_batch(Net *online, const Net *target, const Replay *rb,
                      int batch, float gamma, float lr, uint64_t *rng,
                      float *mean_q) {
    static NNCache c, cn;
    float s[FEAT_MAX], s2[FEAT_MAX];
    float loss = 0.0f, qsum = 0.0f;

    nn_zero_grad(online);
    for (int k = 0; k < batch; k++) {
        int i = (int)(game_rand(rng) % (uint32_t)rb->size);
        replay_get(rb, i, s, s2);
        int a = rb->a[i];

        // The target comes from the *target* network and is treated as a
        // constant: no gradient flows through it.
        nn_forward(target, s2, &cn);
        float y = dqn_target(rb->r[i], rb->done[i], gamma, cn.a[NN_LAYERS]);

        nn_forward(online, s, &c);
        float *q = c.a[NN_LAYERS];
        qsum += q[argmax3(q)];

        // Huber loss: squared for small errors, linear for large ones, so a
        // single surprising transition can't produce a huge gradient.
        float d = q[a] - y;
        if (fabsf(d) <= 1.0f) {
            loss += 0.5f * d * d;
        } else {
            loss += fabsf(d) - 0.5f;
            d = d > 0.0f ? 1.0f : -1.0f; // the slope of the linear part
        }
        nn_backward(online, &c, a, d);
    }
    nn_adam_step(online, lr, batch);

    if (mean_q) {
        *mean_q = qsum / batch;
    }
    return loss / batch;
}

float dqn_nstep_return(const float *r, int n, float gamma) {
    float sum = 0.0f;
    for (int k = n - 1; k >= 0; k--) {
        sum = r[k] + gamma * sum;
    }
    return sum;
}
