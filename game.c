#include "game.h"

#include <string.h>

const int DIR_DX[4] = {0, 1, 0, -1};
const int DIR_DY[4] = {-1, 0, 1, 0};

uint32_t game_rand(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return (uint32_t)((x * 0x2545F4914F6CDD1DULL) >> 32);
}

static int is_wall(Point p) {
    return p.x <= 0 || p.x >= WIDTH - 1 || p.y <= 0 || p.y >= HEIGHT - 1;
}

int game_spawn_food(Game *g) {
    static unsigned char occupied[HEIGHT][WIDTH];
    memset(occupied, 0, sizeof(occupied));
    for (int i = 0; i < g->len; i++) {
        occupied[g->body[i].y][g->body[i].x] = 1;
    }

    int free_cells = MAX_LEN - g->len;
    if (free_cells <= 0) {
        return 0;
    }

    // Pick the k-th free interior cell, so this always terminates.
    int k = (int)(game_rand(&g->rng) % (uint32_t)free_cells);
    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1; x++) {
            if (!occupied[y][x] && k-- == 0) {
                g->food.x = x;
                g->food.y = y;
                return 1;
            }
        }
    }
    return 0; // unreachable if len is consistent
}

void game_reset(Game *g, uint64_t seed) {
    memset(g, 0, sizeof(*g));
    g->rng = seed ? seed : 0x9E3779B97F4A7C15ULL; // xorshift must not start at 0
    g->len = 1;
    g->body[0].x = 1 + (int)(game_rand(&g->rng) % (WIDTH - 2));
    g->body[0].y = 1 + (int)(game_rand(&g->rng) % (HEIGHT - 2));
    g->dir = (Dir)(game_rand(&g->rng) % 4);
    g->alive = 1;
    game_spawn_food(g);
}

Dir game_turn(Dir d, Action a) {
    switch (a) {
        case ACT_LEFT:  return (Dir)((d + 3) % 4);
        case ACT_RIGHT: return (Dir)((d + 1) % 4);
        default:        return d;
    }
}

int game_is_deadly(const Game *g, Point p) {
    if (is_wall(p)) {
        return 1;
    }
    // The tail (body[len-1]) moves out of the way in the same step,
    // so only body[0 .. len-2] can be hit.
    for (int i = 0; i < g->len - 1; i++) {
        if (g->body[i].x == p.x && g->body[i].y == p.y) {
            return 1;
        }
    }
    return 0;
}

Event game_step(Game *g, Dir dir) {
    if (!g->alive) {
        return EV_DIED;
    }

    // Ignore a 180° turn when there is a body to crash into.
    if (g->len > 1 && dir == (Dir)((g->dir + 2) % 4)) {
        dir = g->dir;
    }
    g->dir = dir;

    Point next = {g->body[0].x + DIR_DX[dir], g->body[0].y + DIR_DY[dir]};
    if (game_is_deadly(g, next)) {
        g->alive = 0;
        return EV_DIED;
    }

    // Move: shift the body back by one and put the head in front.
    for (int i = g->len - 1; i > 0; i--) {
        g->body[i] = g->body[i - 1];
    }
    g->body[0] = next;
    g->steps_since_food++;

    if (next.x == g->food.x && next.y == g->food.y) {
        g->body[g->len] = g->body[g->len - 1]; // new segment appears at the tail
        g->len++;
        g->score++;
        g->steps_since_food = 0;
        if (!game_spawn_food(g)) {
            g->alive = 0; // board is full: nothing left to eat
            return EV_WON;
        }
        return EV_ATE;
    }
    return EV_NONE;
}

Event game_step_rel(Game *g, Action a) {
    return game_step(g, game_turn(g->dir, a));
}
