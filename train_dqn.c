/*
 * Headless DQN trainer.
 *
 *   ./train_dqn [--features small|view|tail|hunger] [--steps N] [--seed S]
 *               [--double] [--nstep N] [--gamma G] [--dueling] [--per]
 *               [--eval GAMES] [--out snake_dqn.bin]
 *
 * The upgrades (all off by default, see DQN_UPGRADES.md):
 *   --double   Double DQN targets
 *   --nstep N  store N-step returns instead of 1-step rewards
 *   --gamma G  discount (default 0.95)
 *   --dueling  a dueling output layer, Q = V + A - mean(A)
 *   --per      prioritized experience replay
 * --eval GAMES sets how many games the final evaluation plays (default 100).
 * --starve-penalty P: starving (no food for 100 * length steps) ends the
 *   episode like a death, with reward -P. The default 0 keeps the old rule:
 *   starving is a time limit, with no penalty and no "done".
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dqn.h"
#include "game.h"
#include "nn.h"

// Same rewards as train.c, so the two agents chase the same goal.
#define REWARD_FOOD   10.0f
#define REWARD_DEATH -10.0f

#define HIDDEN 128        // neurons per hidden layer
#define GAMMA 0.95f       // default; --gamma changes it
#define NSTEP_MAX 32
#define LR 1e-3f          // Adam learning rate
#define BATCH 32
#define REPLAY_CAP 50000  // transitions kept in the replay buffer
#define WARMUP 1000       // steps of play before learning starts
#define TRAIN_EVERY 4     // one gradient step every N environment steps
#define TARGET_SYNC 1000  // copy online -> target every N steps
#define EPS_MIN 0.01f
#define EPS_DECAY 0.5     // fraction of training over which eps goes 1 -> EPS_MIN
#define LOG_EVERY 10000
#define PER_ALPHA 0.6f    // how strongly priorities skew sampling
#define PER_BETA0 0.4f    // importance correction at the start, annealed to 1

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

// One evaluation game without learning. eps = 0 is greedy, eps = 1 random.
static int eval_episode(const Net *net, FeatType f, float eps, uint64_t *rng) {
    Game g;
    float x[FEAT_MAX];
    game_reset(&g, new_seed(rng));
    while (g.alive && g.steps_since_food <= QL_STARVE_STEPS(g.len)) {
        dqn_features(f, &g, x);
        game_step_rel(&g, (Action)dqn_act(net, x, eps, rng));
    }
    return g.score;
}

int main(int argc, char *argv[]) {
    long steps = 600000;
    uint64_t seed = (uint64_t)time(NULL);
    FeatType feat = FEAT_SMALL;
    const char *out = "snake_dqn.bin";
    int double_dqn = 0;
    int nstep = 1;
    float gamma = GAMMA;
    int dueling = 0;
    int per = 0;
    int eval_games = 100;
    float starve_penalty = 0.0f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
            steps = atol(argv[++i]);
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--features") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "small")) {
                feat = FEAT_SMALL;
            } else if (!strcmp(argv[i], "view")) {
                feat = FEAT_VIEW;
            } else if (!strcmp(argv[i], "tail")) {
                feat = FEAT_TAIL;
            } else if (!strcmp(argv[i], "hunger")) {
                feat = FEAT_HUNGER;
            } else {
                fprintf(stderr, "features must be small, view, tail or hunger\n");
                return 1;
            }
        } else if (!strcmp(argv[i], "--double")) {
            double_dqn = 1;
        } else if (!strcmp(argv[i], "--nstep") && i + 1 < argc) {
            nstep = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--gamma") && i + 1 < argc) {
            gamma = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--dueling")) {
            dueling = 1;
        } else if (!strcmp(argv[i], "--per")) {
            per = 1;
        } else if (!strcmp(argv[i], "--starve-penalty") && i + 1 < argc) {
            starve_penalty = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--eval") && i + 1 < argc) {
            eval_games = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--features small|view|tail|hunger] [--steps N] "
                            "[--seed S] [--double] [--nstep N] [--gamma G] "
                            "[--dueling] [--per] [--starve-penalty P] "
                            "[--eval GAMES] [--out FILE]\n",
                    argv[0]);
            return 1;
        }
    }
    if (eval_games < 1) {
        fprintf(stderr, "eval must be at least 1 game\n");
        return 1;
    }
    if (nstep < 1 || nstep > NSTEP_MAX) {
        fprintf(stderr, "nstep must be between 1 and %d\n", NSTEP_MAX);
        return 1;
    }
    if (gamma <= 0.0f || gamma >= 1.0f) {
        fprintf(stderr, "gamma must be between 0 and 1\n");
        return 1;
    }
    float gamma_n = powf(gamma, (float)nstep); // discount for the bootstrap
    if (steps <= WARMUP) {
        fprintf(stderr, "steps must be more than %d\n", WARMUP);
        return 1;
    }

    uint64_t rng = seed ? seed : 1;
    int nfeat = dqn_feature_count(feat);
    Net online, target;
    Replay rb;
    int (*init)(Net *, int, int, int, uint64_t *) = dueling ? nn_init_dueling : nn_init;
    if (!init(&online, nfeat, HIDDEN, QL_ACTIONS, &rng) ||
        !init(&target, nfeat, HIDDEN, QL_ACTIONS, &rng)) {
        fprintf(stderr, "could not create the network (%d inputs, %d hidden; "
                        "NN_MAX is %d)\n", nfeat, HIDDEN, NN_MAX);
        return 1;
    }
    if (!replay_init(&rb, REPLAY_CAP, nfeat) ||
        (per && !replay_enable_priority(&rb, PER_ALPHA))) {
        fprintf(stderr, "not enough memory for the replay buffer\n");
        return 1;
    }
    nn_copy(&target, &online);

    printf("Training %ld steps (features %s: %d inputs, seed %llu, "
           "%d-step, gamma %.3f%s%s%s, starve penalty %g)\n", steps,
           dqn_feature_name(feat), nfeat, (unsigned long long)seed, nstep,
           gamma, double_dqn ? ", double DQN" : "", dueling ? ", dueling" : "",
           per ? ", prioritized replay" : "", starve_penalty);
    printf("%8s %8s %9s %6s %6s %8s %8s\n", "steps", "episodes", "avg(100)",
           "max", "eps", "loss", "max Q");

    int recent[100] = {0};
    int episodes = 0, best = 0;
    double loss_sum = 0.0, q_sum = 0.0;
    int loss_n = 0;
    clock_t start = clock();

    Game g;
    float s[FEAT_MAX], s2[FEAT_MAX];
    // The last `nstep` (state, action, reward) triples that are not in the
    // replay buffer yet. pend_n of them, oldest first at index 0.
    static float pend_s[NSTEP_MAX][FEAT_MAX];
    int pend_a[NSTEP_MAX], pend_n = 0;
    float pend_r[NSTEP_MAX];
    game_reset(&g, new_seed(&rng));
    dqn_features(feat, &g, s);

    for (long step = 1; step <= steps; step++) {
        // Linear decay from 1.0 to EPS_MIN over the first EPS_DECAY of training.
        float eps = 1.0f - (1.0f - EPS_MIN) * (float)step / (float)(EPS_DECAY * steps);
        if (eps < EPS_MIN) eps = EPS_MIN;

        // 1. act and observe, 2. remember
        int a = dqn_act(&online, s, eps, &rng);
        Event ev = game_step_rel(&g, (Action)a);
        dqn_features(feat, &g, s2);
        int done = (ev == EV_DIED || ev == EV_WON);
        int starved = !done && g.steps_since_food > QL_STARVE_STEPS(g.len);
        float r = reward(ev);
        if (starved && starve_penalty != 0.0f) {
            r = -starve_penalty; // starving now counts as a real ending
            done = 1;
        }
        memcpy(pend_s[pend_n], s, sizeof(float) * nfeat);
        pend_a[pend_n] = a;
        pend_r[pend_n] = r;
        pend_n++;
        if (done) {
            // The episode ended: every waiting transition gets the rewards up
            // to the end (fewer than nstep of them) and no bootstrap.
            for (int k = 0; k < pend_n; k++) {
                replay_push(&rb, pend_s[k], pend_a[k],
                            dqn_nstep_return(pend_r + k, pend_n - k, gamma),
                            s2, 1);
            }
            pend_n = 0;
        } else if (pend_n == nstep) {
            // The oldest one now has nstep rewards and a state nstep later.
            replay_push(&rb, pend_s[0], pend_a[0],
                        dqn_nstep_return(pend_r, nstep, gamma), s2, 0);
            pend_n--;
            memmove(pend_s[0], pend_s[1], sizeof(pend_s[0]) * pend_n);
            memmove(pend_a, pend_a + 1, sizeof(int) * pend_n);
            memmove(pend_r, pend_r + 1, sizeof(float) * pend_n);
        }
        memcpy(s, s2, sizeof(float) * nfeat);

        // Without --starve-penalty, starvation is a time limit, not a real
        // game over (see train.c), so it ends the episode without being
        // stored as done. Transitions still waiting for their nstep rewards
        // are dropped.
        if (done || starved) {
            pend_n = 0;
            recent[episodes % 100] = g.score;
            if (g.score > best) best = g.score;
            episodes++;
            game_reset(&g, new_seed(&rng));
            dqn_features(feat, &g, s);
        }

        // 3. learn from a random batch of past transitions
        if (step > WARMUP && step % TRAIN_EVERY == 0) {
            float mq;
            rb.beta = PER_BETA0 + (1.0f - PER_BETA0) * (float)step / (float)steps;
            loss_sum += dqn_train_batch(&online, &target, &rb, BATCH, gamma_n,
                                        LR, double_dqn, &rng, &mq);
            q_sum += mq;
            loss_n++;
        }

        // 4. every so often, let the target network catch up
        if (step % TARGET_SYNC == 0) {
            nn_copy(&target, &online);
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
    printf("Training took %.1f s\n", (double)(clock() - start) / CLOCKS_PER_SEC);

    int sum_greedy = 0, max_greedy = 0, sum_random = 0;
    for (int i = 0; i < eval_games; i++) {
        int sg = eval_episode(&online, feat, 0.0f, &rng);
        sum_greedy += sg;
        if (sg > max_greedy) max_greedy = sg;
        sum_random += eval_episode(&online, feat, 1.0f, &rng);
    }
    printf("\nGreedy eval over %d games: avg %.2f, max %d\n", eval_games,
           (double)sum_greedy / eval_games, max_greedy);
    printf("Random policy for comparison: avg %.2f\n",
           (double)sum_random / eval_games);

    int ok = nn_save(&online, (int)feat, out);
    if (ok) {
        printf("Saved network to %s\n", out);
    } else {
        perror(out);
    }
    nn_free(&online);
    nn_free(&target);
    replay_free(&rb);
    return ok ? 0 : 1;
}
