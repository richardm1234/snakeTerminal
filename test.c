/*
 * Small sanity tests for the game rules, the Q-learning update, the DQN and the CNN.
 *   make test && ./test
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cnn.h"
#include "dqn.h"
#include "game.h"
#include "nn.h"
#include "qlearn.h"

static int failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
} while (0)

// Build a game from a list of body points (head first), heading and food.
static void setup(Game *g, const Point *body, int len, Dir dir, Point food) {
    game_reset(g, 42);
    memcpy(g->body, body, sizeof(Point) * len);
    g->len = len;
    g->dir = dir;
    g->food = food;
}

static void test_wall_death(void) {
    Game g;
    Point body[] = {{1, 5}};
    setup(&g, body, 1, DIR_LEFT, (Point){20, 10});
    CHECK(game_step_rel(&g, ACT_STRAIGHT) == EV_DIED);
    CHECK(!g.alive);
}

static void test_eat_grows(void) {
    Game g;
    Point body[] = {{5, 5}};
    setup(&g, body, 1, DIR_RIGHT, (Point){6, 5});
    CHECK(game_step_rel(&g, ACT_STRAIGHT) == EV_ATE);
    CHECK(g.len == 2);
    CHECK(g.score == 1);
    CHECK(g.body[0].x == 6 && g.body[0].y == 5);
    CHECK(!(g.food.x == 6 && g.food.y == 5)); // new food somewhere else
}

static void test_self_death(void) {
    Game g;
    // Head at (5,5) moving left, body curls below it.
    Point body[] = {{5, 5}, {6, 5}, {6, 6}, {5, 6}, {4, 6}};
    setup(&g, body, 5, DIR_LEFT, (Point){20, 10});
    // Turning left from LEFT means going DOWN onto (5,6): body.
    CHECK(game_step_rel(&g, ACT_LEFT) == EV_DIED);
}

static void test_tail_is_safe(void) {
    Game g;
    // (5,6) is the tail, which moves away in the same step.
    Point body[] = {{5, 5}, {6, 5}, {6, 6}, {5, 6}};
    setup(&g, body, 4, DIR_LEFT, (Point){20, 10});
    CHECK(game_step_rel(&g, ACT_LEFT) == EV_NONE);
    CHECK(g.alive);
}

static void test_relative_turns(void) {
    CHECK(game_turn(DIR_UP, ACT_LEFT) == DIR_LEFT);
    CHECK(game_turn(DIR_UP, ACT_RIGHT) == DIR_RIGHT);
    CHECK(game_turn(DIR_LEFT, ACT_LEFT) == DIR_DOWN);
    CHECK(game_turn(DIR_DOWN, ACT_STRAIGHT) == DIR_DOWN);
}

static void test_reverse_ignored(void) {
    Game g;
    Point body[] = {{5, 5}, {4, 5}};
    setup(&g, body, 2, DIR_RIGHT, (Point){20, 10});
    CHECK(game_step(&g, DIR_LEFT) == EV_NONE);
    CHECK(g.body[0].x == 6 && g.body[0].y == 5);
}

static void test_state_bits(void) {
    Game g;
    // Head against the left wall, heading up, food up and to the right.
    Point body[] = {{1, 5}};
    setup(&g, body, 1, DIR_UP, (Point){10, 2});
    int s = ql_state(&g);
    CHECK(!(s & (1 << 0)));    // N is free
    CHECK(!(s & (1 << 2)));    // E is free
    CHECK(!(s & (1 << 4)));    // S is free
    CHECK(s & (1 << 5));       // SW is the wall
    CHECK(s & (1 << 6));       // W is the wall
    CHECK(s & (1 << 7));       // NW is the wall
    CHECK(((s >> 8) & 3) == DIR_UP);
    CHECK(!(s & (1 << 10)));   // food not left
    CHECK(s & (1 << 11));      // food right
    CHECK(s & (1 << 12));      // food above
    CHECK(!(s & (1 << 13)));   // food not below

    // Body segments show up in the ring too (tail excluded, it moves away).
    Point snake[] = {{5, 5}, {5, 6}, {4, 6}, {4, 5}};
    setup(&g, snake, 4, DIR_UP, (Point){10, 2});
    s = ql_state(&g);
    CHECK(s & (1 << 4));       // S: neck
    CHECK(s & (1 << 5));       // SW: body
    CHECK(!(s & (1 << 6)));    // W: tail, safe
}

static void test_q_update(void) {
    static QTable t;
    memset(&t, 0, sizeof(t));
    t.q[1][0] = 1.0f;
    t.q[2][0] = 2.0f; t.q[2][1] = 5.0f; t.q[2][2] = 3.0f;

    // target = 1 + 0.9 * 5 = 5.5 ; Q = 1 + 0.5 * (5.5 - 1) = 3.25
    ql_update(&t, 1, 0, 1.0f, 2, 0, 0.5f, 0.9f);
    CHECK(fabsf(t.q[1][0] - 3.25f) < 1e-6f);

    // Terminal: no bootstrapping. target = 1 ; Q = 3.25 + 0.5 * (1 - 3.25)
    ql_update(&t, 1, 0, 1.0f, 2, 1, 0.5f, 0.9f);
    CHECK(fabsf(t.q[1][0] - 2.125f) < 1e-6f);

    CHECK(ql_best(&t, 2) == 1);
}

// Compare backprop with a numerical derivative: nudge one weight up and down
// and see how much the loss changes. If backprop is right, the two agree.
static float grad_check_loss(const Net *net, const float *x, int a, float y) {
    NNCache c;
    nn_forward(net, x, &c);
    float d = c.a[NN_LAYERS][a] - y;
    return 0.5f * d * d;
}

static void test_gradient_check(void) {
    uint64_t rng = 7;
    Net net;
    CHECK(nn_init(&net, 4, 6, 3, &rng));
    for (int l = 0; l < NN_LAYERS; l++) {
        for (int o = 0; o < net.n[l + 1]; o++) {
            net.b[l][o] = 0.1f; // keep ReLUs away from their kink at 0
        }
    }
    float x[4] = {0.5f, -1.0f, 0.25f, 1.0f};
    int a = 1;
    float y = 2.0f;

    NNCache c;
    nn_zero_grad(&net);
    nn_forward(&net, x, &c);
    nn_backward(&net, &c, a, c.a[NN_LAYERS][a] - y); // dL/dq for L = 0.5 (q - y)^2

    const float h = 1e-2f;
    float worst = 0.0f;
    for (int l = 0; l < NN_LAYERS; l++) {
        for (int i = 0; i < net.n[l + 1] * net.n[l]; i++) {
            float keep = net.w[l][i];
            net.w[l][i] = keep + h;
            float up = grad_check_loss(&net, x, a, y);
            net.w[l][i] = keep - h;
            float down = grad_check_loss(&net, x, a, y);
            net.w[l][i] = keep;
            float diff = fabsf((up - down) / (2 * h) - net.gw[l][i]);
            if (diff > worst) worst = diff;
        }
    }
    CHECK(worst < 1e-2f);
    nn_free(&net);
}

static void test_priority_sampling(void) {
    Replay rb;
    float s[1] = {0.0f};
    CHECK(replay_init(&rb, 5, 1));
    CHECK(replay_enable_priority(&rb, 1.0f)); // alpha 1: proportional
    for (int i = 0; i < 4; i++) replay_push(&rb, s, 0, 0.0f, s, 0);
    // Priorities |td| + 0.01: 0.99 + 0.01 = 1, 2, 0, 1 (index 2 nearly never).
    replay_set_priority(&rb, 0, 0.99f);
    replay_set_priority(&rb, 1, 1.99f);
    replay_set_priority(&rb, 2, -0.0f);
    replay_set_priority(&rb, 3, 0.99f);
    CHECK(fabs(rb.tree[1] - 4.01) < 1e-6); // the root holds the total
    int count[4] = {0};
    for (int k = 0; k < 4000; k++) count[replay_sample(&rb, (k + 0.5) / 4000)]++;
    CHECK(abs(count[0] - 998) <= 2 && abs(count[1] - 1995) <= 2);
    CHECK(count[2] <= 11 && abs(count[3] - 998) <= 2);
    replay_free(&rb);
}

static void test_dqn_target(void) {
    float q_next[3] = {1.0f, 4.0f, 2.0f};
    CHECK(fabsf(dqn_target(1.0f, 0, 0.5f, q_next) - 3.0f) < 1e-6f); // 1 + 0.5 * 4
    CHECK(fabsf(dqn_target(-10.0f, 1, 0.5f, q_next) + 10.0f) < 1e-6f); // done: no bootstrap
}

// Window cell `fwd` steps ahead and `right` steps to the right of the head.
static int view_cell(int fwd, int right) {
    return (VIEW_R - fwd) * VIEW_W + (VIEW_R + right);
}

static void test_view_is_egocentric(void) {
    Game g;
    float x[FEAT_MAX];
    float *obstacle = x, *food = x + VIEW_W * VIEW_W, *extra = x + 2 * VIEW_W * VIEW_W;

    // Food two cells ahead: same window cell whichever way the snake faces.
    Point body[] = {{10, 10}};
    setup(&g, body, 1, DIR_UP, (Point){10, 8});
    dqn_features(FEAT_VIEW, &g, x);
    CHECK(food[view_cell(2, 0)] == 1.0f);
    CHECK(extra[0] == 1.0f && extra[1] == 0.0f); // food ahead, not behind

    setup(&g, body, 1, DIR_RIGHT, (Point){12, 10});
    dqn_features(FEAT_VIEW, &g, x);
    CHECK(food[view_cell(2, 0)] == 1.0f);

    // Left wall: on the left when heading up, on the right when heading down.
    Point edge[] = {{1, 5}};
    setup(&g, edge, 1, DIR_UP, (Point){20, 10});
    dqn_features(FEAT_VIEW, &g, x);
    CHECK(obstacle[view_cell(0, -1)] == 1.0f);
    CHECK(obstacle[view_cell(0, 1)] == 0.0f);
    setup(&g, edge, 1, DIR_DOWN, (Point){20, 10});
    dqn_features(FEAT_VIEW, &g, x);
    CHECK(obstacle[view_cell(0, 1)] == 1.0f);
    CHECK(obstacle[view_cell(0, -1)] == 0.0f);
    CHECK(extra[2] == 1.0f); // food (to the east) is on the left when heading down
}

static void test_tail_features(void) {
    Game g;
    float x[FEAT_MAX];
    float *tail = x + FEAT_VIEW_N;

    // Heading up, body bends right: the tail is behind and to the right.
    Point body[] = {{10, 10}, {10, 11}, {11, 11}, {12, 11}};
    setup(&g, body, 4, DIR_UP, (Point){20, 5});
    CHECK(dqn_features(FEAT_TAIL, &g, x) == FEAT_TAIL_N);
    CHECK(tail[0] == 0.0f && tail[1] == 1.0f); // behind
    CHECK(tail[2] == 0.0f && tail[3] == 1.0f); // right
    CHECK(fabsf(tail[4] - 3.0f / (WIDTH - 3 + HEIGHT - 3)) < 1e-6f);

    // The view part is the same as FEAT_VIEW.
    float v[FEAT_MAX];
    dqn_features(FEAT_VIEW, &g, v);
    CHECK(memcmp(x, v, sizeof(float) * FEAT_VIEW_N) == 0);
}

// Same idea as test_gradient_check, for the convolutional network: every
// parameter (filters, biases, dense weights) is nudged and compared.
static float cnn_check_loss(const CNN *net, const float *x, CNNCache *c, int a,
                            float y) {
    cnn_forward(net, x, c);
    float d = c->q[a] - y;
    return 0.5f * d * d;
}

static void test_cnn_gradient_check(void) {
    uint64_t rng = 11;
    CNN net;
    CNNCache c;
    // Radius 3: 7x7 input -> conv 5x5 -> stride-2 conv 2x2. Small enough to
    // check every parameter, big enough that patches overlap.
    CHECK(cnn_init(&net, 3, 3, 4, 5, &rng));
    CHECK(cnn_cache_init(&c, &net));
    for (int l = 0; l < CNN_LAYERS; l++) {
        int end = l + 1 < CNN_LAYERS ? net.w_off[l + 1] : net.nparam;
        for (int o = 0; o < end - net.b_off[l]; o++) {
            net.p[net.b_off[l] + o] = 0.1f; // keep ReLUs away from their kink
        }
    }
    int nin = cnn_input_count(&net);
    float x[DQN_VIEW_COUNT(3)];
    for (int i = 0; i < nin; i++) {
        x[i] = (float)(game_rand(&rng) % 1000) / 1000.0f;
    }
    int a = 2;
    float y = 1.5f;

    static float grad[4096];
    CHECK(net.nparam <= 4096);
    memset(grad, 0, sizeof(grad));
    cnn_forward(&net, x, &c);
    cnn_backward(&net, &c, a, c.q[a] - y, grad);

    const float h = 1e-2f;
    float worst = 0.0f;
    for (int i = 0; i < net.nparam; i++) {
        float keep = net.p[i];
        net.p[i] = keep + h;
        float up = cnn_check_loss(&net, x, &c, a, y);
        net.p[i] = keep - h;
        float down = cnn_check_loss(&net, x, &c, a, y);
        net.p[i] = keep;
        float diff = fabsf((up - down) / (2 * h) - grad[i]);
        if (diff > worst) worst = diff;
    }
    CHECK(worst < 1e-2f);
    cnn_cache_free(&c);
    cnn_free(&net);
}

// A convolution uses the same filter everywhere, so moving the input one cell
// to the right moves conv1's output one cell to the right. That's what lets a
// CNN learn "body next to me" once instead of once per position.
static void test_conv_shift(void) {
    uint64_t rng = 5;
    CNN net;
    CNNCache c;
    CHECK(cnn_init(&net, 4, 4, 4, 8, &rng));
    CHECK(cnn_cache_init(&c, &net));
    int w = net.side[0], w1 = net.side[1];
    float x[DQN_VIEW_COUNT(4)], a1[4 * 7 * 7];

    memset(x, 0, sizeof(x));
    x[4 * w + 3] = 1.0f; // one obstacle at row 4, column 3
    cnn_forward(&net, x, &c);
    memcpy(a1, c.a1, sizeof(float) * net.ch[1] * w1 * w1);

    memset(x, 0, sizeof(x));
    x[4 * w + 4] = 1.0f; // the same obstacle one column to the right
    cnn_forward(&net, x, &c);

    float worst = 0.0f;
    for (int ch = 0; ch < net.ch[1]; ch++) {
        for (int yy = 0; yy < w1; yy++) {
            for (int xx = 0; xx + 1 < w1; xx++) {
                float before = a1[(ch * w1 + yy) * w1 + xx];
                float after = c.a1[(ch * w1 + yy) * w1 + xx + 1];
                if (fabsf(before - after) > worst) worst = fabsf(before - after);
            }
        }
    }
    CHECK(worst < 1e-6f);
    cnn_cache_free(&c);
    cnn_free(&net);
}

// Splitting the batch over threads must give the same update as one thread
// (up to rounding, since the sums are added in a different order).
static void test_cnn_threads_agree(void) {
    uint64_t rng = 3;
    CNN one, many, target;
    CHECK(cnn_init(&one, 3, 4, 4, 8, &rng));
    CHECK(cnn_init(&many, 3, 4, 4, 8, &rng));
    CHECK(cnn_init(&target, 3, 4, 4, 8, &rng));
    cnn_copy(&many, &one);

    int nin = cnn_input_count(&one);
    Replay rb;
    CHECK(replay_init(&rb, 64, nin));
    Game g;
    game_reset(&g, 9);
    float s[DQN_VIEW_COUNT(3)], s2[DQN_VIEW_COUNT(3)];
    dqn_view(&g, 3, s);
    for (int i = 0; i < 64; i++) {
        int a = (int)(game_rand(&rng) % 3);
        Event ev = game_step_rel(&g, (Action)a);
        dqn_view(&g, 3, s2);
        replay_push(&rb, s, a, ev == EV_DIED ? -10.0f : 0.0f, s2, ev == EV_DIED);
        if (!g.alive) game_reset(&g, game_rand(&rng) + 1);
        dqn_view(&g, 3, s);
    }

    CNNWork w1, w4;
    CHECK(cnn_work_init(&w1, &one, 16, 1));
    CHECK(cnn_work_init(&w4, &many, 16, 4));
    uint64_t r1 = 77, r4 = 77; // same batch for both
    cnn_train_batch(&one, &target, &rb, &w1, 0.9f, 1e-3f, &r1, NULL);
    cnn_train_batch(&many, &target, &rb, &w4, 0.9f, 1e-3f, &r4, NULL);

    float worst = 0.0f;
    for (int i = 0; i < one.nparam; i++) {
        float d = fabsf(one.p[i] - many.p[i]);
        if (d > worst) worst = d;
    }
    CHECK(worst < 1e-5f);
    cnn_work_free(&w1);
    cnn_work_free(&w4);
    replay_free(&rb);
    cnn_free(&one);
    cnn_free(&many);
    cnn_free(&target);
}

// cnn_train_batch must do exactly what the pieces do by hand: target from
// the target network on s2, Q(s, a) from the online network on s. (An early
// version accidentally trained on Q(s2, a); each piece was right, only the
// wiring was wrong, so only a test like this one catches it.)
static void test_cnn_train_step_by_hand(void) {
    uint64_t rng = 21;
    CNN net, ref, target;
    CHECK(cnn_init(&net, 3, 4, 4, 8, &rng));
    CHECK(cnn_init(&ref, 3, 4, 4, 8, &rng));
    CHECK(cnn_init(&target, 3, 4, 4, 8, &rng));
    cnn_copy(&ref, &net);
    int nin = cnn_input_count(&net);

    // One non-terminal transition whose s and s2 differ.
    Game g;
    Point body[] = {{10, 10}, {10, 11}, {10, 12}};
    setup(&g, body, 3, DIR_UP, (Point){11, 8});
    float s[DQN_VIEW_COUNT(3)], s2[DQN_VIEW_COUNT(3)];
    dqn_view(&g, 3, s);
    game_step_rel(&g, ACT_RIGHT);
    dqn_view(&g, 3, s2);
    Replay rb;
    CHECK(replay_init(&rb, 1, nin));
    replay_push(&rb, s, ACT_RIGHT, 0.5f, s2, 0);

    CNNWork w;
    CHECK(cnn_work_init(&w, &net, 1, 1));
    uint64_t r = 1;
    cnn_train_batch(&net, &target, &rb, &w, 0.9f, 1e-2f, &r, NULL);

    // The same step, by hand.
    CNNCache c;
    CHECK(cnn_cache_init(&c, &ref));
    replay_get(&rb, 0, s, s2);
    cnn_forward(&target, s2, &c);
    float y = dqn_target(0.5f, 0, 0.9f, c.q);
    cnn_forward(&ref, s, &c);
    float d = c.q[ACT_RIGHT] - y;
    if (d > 1.0f) d = 1.0f;
    if (d < -1.0f) d = -1.0f;
    static float grad[8192];
    CHECK(ref.nparam <= 8192);
    memset(grad, 0, sizeof(grad));
    cnn_backward(&ref, &c, ACT_RIGHT, d, grad);
    cnn_adam_step(&ref, grad, 1e-2f, 1);

    float worst = 0.0f;
    for (int i = 0; i < net.nparam; i++) {
        float diff = fabsf(net.p[i] - ref.p[i]);
        if (diff > worst) worst = diff;
    }
    CHECK(worst < 1e-6f);
    cnn_cache_free(&c);
    cnn_work_free(&w);
    replay_free(&rb);
    cnn_free(&net);
    cnn_free(&ref);
    cnn_free(&target);
}

int main(void) {
    test_wall_death();
    test_eat_grows();
    test_self_death();
    test_tail_is_safe();
    test_relative_turns();
    test_reverse_ignored();
    test_state_bits();
    test_q_update();
    test_gradient_check();
    test_priority_sampling();
    test_dqn_target();
    test_view_is_egocentric();
    test_tail_features();
    test_cnn_gradient_check();
    test_conv_shift();
    test_cnn_threads_agree();
    test_cnn_train_step_by_hand();

    if (failures) {
        printf("%d check(s) failed\n", failures);
        return 1;
    }
    printf("All tests passed\n");
    return 0;
}
