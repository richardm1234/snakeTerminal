/*
 * Headless Q-learning trainer.
 *
 *   ./train [--episodes N] [--seed S] [--shaping] [--out snake_q.bin]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "game.h"
#include "qlearn.h"

#define REWARD_FOOD   10.0f
#define REWARD_DEATH -10.0f
#define REWARD_CLOSER  1.0f // only with --shaping
#define REWARD_FARTHER -1.0f

#define ALPHA 0.1f
#define GAMMA 0.9f
#define EPS_MIN 0.01f

static int food_distance(const Game *g) {
    return abs(g->body[0].x - g->food.x) + abs(g->body[0].y - g->food.y);
}

static float reward(Event ev, int shaping, int dist_before, int dist_after) {
    switch (ev) {
        case EV_ATE:
        case EV_WON:  return REWARD_FOOD;
        case EV_DIED: return REWARD_DEATH;
        default:
            if (!shaping) return 0.0f;
            return dist_after < dist_before ? REWARD_CLOSER : REWARD_FARTHER;
    }
}

// Play one episode. If learn is set, update the table after every step.
static int run_episode(QTable *Q, float eps, int learn, int shaping,
                       uint64_t *rng) {
    Game g;
    game_reset(&g, ((uint64_t)game_rand(rng) << 32) | game_rand(rng));
    int s = ql_state(&g);

    while (g.alive) {
        int a = ql_act(Q, s, eps, rng);
        int d0 = food_distance(&g);
        Event ev = game_step_rel(&g, (Action)a);
        float r = reward(ev, shaping, d0, food_distance(&g));
        int s2 = ql_state(&g);
        int done = (ev == EV_DIED || ev == EV_WON); // a real terminal state

        if (learn) {
            ql_update(Q, s, a, r, s2, done, ALPHA, GAMMA);
        }
        s = s2;

        // Starvation is a time limit, not a real game over: we stop the
        // episode, but the update above still bootstrapped (done = 0),
        // so the agent is not taught that "this state ends the game".
        if (g.steps_since_food > QL_STARVE_STEPS(g.len)) {
            break;
        }
    }
    return g.score;
}

int main(int argc, char *argv[]) {
    int episodes = 5000;
    uint64_t seed = (uint64_t)time(NULL);
    int shaping = 0;
    const char *out = "snake_q.bin";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--episodes") && i + 1 < argc) {
            episodes = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--shaping")) {
            shaping = 1;
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--episodes N] [--seed S] [--shaping] "
                            "[--out FILE]\n", argv[0]);
            return 1;
        }
    }
    if (episodes <= 0) {
        fprintf(stderr, "episodes must be positive\n");
        return 1;
    }

    QTable *Q = calloc(1, sizeof(QTable));
    if (!Q) {
        perror("calloc");
        return 1;
    }
    uint64_t rng = seed ? seed : 1;

    printf("Training %d episodes (seed %llu, shaping %s)\n", episodes,
           (unsigned long long)seed, shaping ? "on" : "off");
    printf("%8s %9s %6s %6s %8s\n", "episode", "avg(100)", "max", "eps", "states");

    int recent[100] = {0};
    int best = 0;
    int decay_episodes = (int)(0.7 * episodes);
    if (decay_episodes < 1) decay_episodes = 1;

    for (int ep = 0; ep < episodes; ep++) {
        // Linear decay from 1.0 to EPS_MIN over the first 70% of training.
        float eps = 1.0f - (1.0f - EPS_MIN) * (float)ep / decay_episodes;
        if (eps < EPS_MIN) eps = EPS_MIN;

        int score = run_episode(Q, eps, 1, shaping, &rng);
        recent[ep % 100] = score;
        if (score > best) best = score;

        if ((ep + 1) % 500 == 0 || ep + 1 == episodes) {
            int n = ep + 1 < 100 ? ep + 1 : 100, sum = 0;
            for (int i = 0; i < n; i++) sum += recent[i];
            printf("%8d %9.2f %6d %6.3f %8d\n", ep + 1, (double)sum / n, best,
                   eps, ql_visited(Q));
        }
    }

    // Greedy evaluation (eps = 0, no learning) vs. a random policy.
    int sum_greedy = 0, max_greedy = 0, sum_random = 0;
    for (int i = 0; i < 100; i++) {
        int sg = run_episode(Q, 0.0f, 0, shaping, &rng);
        sum_greedy += sg;
        if (sg > max_greedy) max_greedy = sg;
        sum_random += run_episode(Q, 1.0f, 0, shaping, &rng);
    }
    printf("\nGreedy eval over 100 games: avg %.2f, max %d\n",
           sum_greedy / 100.0, max_greedy);
    printf("Random policy for comparison: avg %.2f\n", sum_random / 100.0);

    if (!ql_save(Q, out)) {
        perror(out);
        free(Q);
        return 1;
    }
    printf("Saved Q-table to %s\n", out);
    free(Q);
    return 0;
}
