#include "qlearn.h"

#include <stdio.h>
#include <string.h>

static const char QL_MAGIC[8] = "SNAKEQ2";

// The 3x3 block around the head, clockwise from the cell above it.
static const int RING_DX[8] = {0, 1, 1, 1, 0, -1, -1, -1};
static const int RING_DY[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

int ql_state(const Game *g) {
    Point head = g->body[0];
    int s = 0;

    // Which of the 8 neighbouring cells hold a wall or body?
    for (int i = 0; i < 8; i++) {
        Point p = {head.x + RING_DX[i], head.y + RING_DY[i]};
        if (game_is_deadly(g, p)) {
            s |= 1 << i;
        }
    }

    // Heading, as a 2-bit number.
    s |= g->dir << 8;

    // Where is the food?
    if (g->food.x < head.x) s |= 1 << 10;
    if (g->food.x > head.x) s |= 1 << 11;
    if (g->food.y < head.y) s |= 1 << 12;
    if (g->food.y > head.y) s |= 1 << 13;

    return s;
}

int ql_best(const QTable *t, int s) {
    int best = 0;
    for (int a = 1; a < QL_ACTIONS; a++) {
        if (t->q[s][a] > t->q[s][best]) {
            best = a;
        }
    }
    return best;
}

int ql_act(const QTable *t, int s, float eps, uint64_t *rng) {
    float u = (float)game_rand(rng) / 4294967296.0f; // uniform in [0, 1)
    if (u < eps) {
        return (int)(game_rand(rng) % QL_ACTIONS); // explore
    }
    return ql_best(t, s);                          // exploit
}

void ql_update(QTable *t, int s, int a, float r, int s2, int done,
               float alpha, float gamma) {
    float target = r;
    if (!done) {
        target += gamma * t->q[s2][ql_best(t, s2)];
    }
    t->q[s][a] += alpha * (target - t->q[s][a]);
}

int ql_visited(const QTable *t) {
    int n = 0;
    for (int s = 0; s < QL_STATES; s++) {
        for (int a = 0; a < QL_ACTIONS; a++) {
            if (t->q[s][a] != 0.0f) {
                n++;
                break;
            }
        }
    }
    return n;
}

int ql_save(const QTable *t, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        return 0;
    }
    int ok = fwrite(QL_MAGIC, sizeof(QL_MAGIC), 1, f) == 1 &&
             fwrite(t->q, sizeof(t->q), 1, f) == 1;
    return fclose(f) == 0 && ok;
}

int ql_load(QTable *t, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return 0;
    }
    char magic[sizeof(QL_MAGIC)];
    int ok = fread(magic, sizeof(magic), 1, f) == 1 &&
             memcmp(magic, QL_MAGIC, sizeof(magic)) == 0 &&
             fread(t->q, sizeof(t->q), 1, f) == 1;
    fclose(f);
    return ok;
}
