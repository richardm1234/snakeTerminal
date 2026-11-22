/*
 * Headless trainer for the convolutional Q-network (cnn.h).
 *
 *   ./train_cnn [--radius R] [--steps N] [--seed S] [--threads T] [--out snake_cnn.bin]
 *
 * The loop is the same as train_dqn.c; only the network is different.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cnn.h"
#include "dqn.h"
#include "game.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// Same rewards and DQN settings as train_dqn.c, so the comparison is fair.
#define REWARD_FOOD   10.0f
#define REWARD_DEATH -10.0f

#define CONV1 16          // filters in the first convolution
#define CONV2 32          // filters in the second convolution
#define HIDDEN 128        // neurons in the dense hidden layer
#define GAMMA 0.95f
#define LR 1e-3f
#define BATCH 32
#define REPLAY_CAP 50000
#define WARMUP 1000
#define TRAIN_EVERY 4
#define TARGET_SYNC 1000
#define EPS_MIN 0.01f
#define EPS_DECAY 0.5
#define LOG_EVERY 10000

static float reward(Event ev) {
    switch (ev) {
        case EV_ATE:
        case EV_WON:  return REWARD_FOOD;
        case EV_DIED: return REWARD_DEATH;
        default:      return 0.0f;
    }
}

static uint64_t new_seed(uint64_t *rng) {
    return ((uint64_t)game_rand(rng) << 32) | game_rand(rng);
}

static double now(void) { // wall-clock seconds (clock() adds up all threads)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int eval_episode(const CNN *net, CNNCache *c, float eps, uint64_t *rng) {
    Game g;
    game_reset(&g, new_seed(rng));
    while (g.alive && g.steps_since_food <= QL_STARVE_STEPS(g.len)) {
        dqn_view(&g, net->radius, c->x);
        game_step_rel(&g, (Action)cnn_act(net, c->x, c, eps, rng));
    }
    return g.score;
}

int main(int argc, char *argv[]) {
    long steps = 600000;
    uint64_t seed = (uint64_t)time(NULL);
    int radius = VIEW_R, threads = 4; // more rarely helps: see CNN_TUTORIAL.md
    const char *out = "snake_cnn.bin";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
            steps = atol(argv[++i]);
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--radius") && i + 1 < argc) {
            radius = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--radius R] [--steps N] [--seed S] "
                            "[--threads T] [--out FILE]\n", argv[0]);
            return 1;
        }
    }
    if (steps <= WARMUP) {
        fprintf(stderr, "steps must be more than %d\n", WARMUP);
        return 1;
    }
    if (radius < 2 || radius > CNN_MAX_RADIUS) {
        fprintf(stderr, "radius must be between 2 and %d\n", CNN_MAX_RADIUS);
        return 1;
    }

    uint64_t rng = seed ? seed : 1;
    CNN online, target;
    CNNCache play;
    CNNWork work;
    Replay rb;
    if (!cnn_init(&online, radius, CONV1, CONV2, HIDDEN, &rng) ||
        !cnn_init(&target, radius, CONV1, CONV2, HIDDEN, &rng) ||
        !cnn_cache_init(&play, &online) ||
        !cnn_work_init(&work, &online, BATCH, threads)) {
        fprintf(stderr, "not enough memory for the network\n");
        return 1;
    }
    int nfeat = cnn_input_count(&online);
    if (!replay_init(&rb, REPLAY_CAP, nfeat)) {
        fprintf(stderr, "not enough memory for the replay buffer\n");
        return 1;
    }
    cnn_copy(&target, &online);

    printf("Training %ld steps (window %dx%d, %d inputs, %d parameters, "
           "%d threads, seed %llu)\n", steps, online.side[0], online.side[0],
           nfeat, online.nparam, work.threads, (unsigned long long)seed);
    printf("Layers: %dx%dx%d -> conv %dx%dx%d -> conv %dx%dx%d -> %d -> %d -> %d\n",
           online.ch[0], online.side[0], online.side[0],
           online.ch[1], online.side[1], online.side[1],
           online.ch[2], online.side[2], online.side[2],
           online.flat, online.hidden, online.outputs);
    printf("%8s %8s %9s %6s %6s %8s %8s\n", "steps", "episodes", "avg(100)",
           "max", "eps", "loss", "max Q");

    int recent[100] = {0};
    int episodes = 0, best = 0;
    double loss_sum = 0.0, q_sum = 0.0;
    int loss_n = 0;
    double start = now();

    Game g;
    float *s = malloc(sizeof(float) * nfeat), *s2 = malloc(sizeof(float) * nfeat);
    if (!s || !s2) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    game_reset(&g, new_seed(&rng));
    dqn_view(&g, radius, s);

    for (long step = 1; step <= steps; step++) {
        float eps = 1.0f - (1.0f - EPS_MIN) * (float)step / (float)(EPS_DECAY * steps);
        if (eps < EPS_MIN) eps = EPS_MIN;

        // 1. act and observe, 2. remember
        int a = cnn_act(&online, s, &play, eps, &rng);
        Event ev = game_step_rel(&g, (Action)a);
        dqn_view(&g, radius, s2);
        int done = (ev == EV_DIED || ev == EV_WON);
        replay_push(&rb, s, a, reward(ev), s2, done);
        memcpy(s, s2, sizeof(float) * nfeat);

        if (done || g.steps_since_food > QL_STARVE_STEPS(g.len)) {
            recent[episodes % 100] = g.score;
            if (g.score > best) best = g.score;
            episodes++;
            game_reset(&g, new_seed(&rng));
            dqn_view(&g, radius, s);
        }

        // 3. learn from a random batch of past transitions
        if (step > WARMUP && step % TRAIN_EVERY == 0) {
            float mq;
            loss_sum += cnn_train_batch(&online, &target, &rb, &work, GAMMA, LR,
                                        &rng, &mq);
            q_sum += mq;
            loss_n++;
        }

        // 4. every so often, let the target network catch up
        if (step % TARGET_SYNC == 0) {
            cnn_copy(&target, &online);
        }

        if (step % LOG_EVERY == 0 || step == steps) {
            int n = episodes < 100 ? episodes : 100, sum = 0;
            for (int i = 0; i < n; i++) sum += recent[i];
            printf("%8ld %8d %9.2f %6d %6.3f %8.4f %8.2f\n", step, episodes,
                   n ? (double)sum / n : 0.0, best, eps,
                   loss_n ? loss_sum / loss_n : 0.0,
                   loss_n ? q_sum / loss_n : 0.0);
            fflush(stdout);
            loss_sum = q_sum = 0.0;
            loss_n = 0;
        }
    }
    printf("Training took %.1f s\n", now() - start);

    int sum_greedy = 0, max_greedy = 0, sum_random = 0;
    for (int i = 0; i < 100; i++) {
        int sg = eval_episode(&online, &play, 0.0f, &rng);
        sum_greedy += sg;
        if (sg > max_greedy) max_greedy = sg;
        sum_random += eval_episode(&online, &play, 1.0f, &rng);
    }
    printf("\nGreedy eval over 100 games: avg %.2f, max %d\n",
           sum_greedy / 100.0, max_greedy);
    printf("Random policy for comparison: avg %.2f\n", sum_random / 100.0);

    int ok = cnn_save(&online, out);
    if (ok) {
        printf("Saved network to %s\n", out);
    } else {
        perror(out);
    }
    free(s);
    free(s2);
    cnn_free(&online);
    cnn_free(&target);
    cnn_cache_free(&play);
    cnn_work_free(&work);
    replay_free(&rb);
    return ok ? 0 : 1;
}
