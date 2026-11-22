#ifndef GAME_H
#define GAME_H

#include <stdint.h>

/*
 * Pure snake rules: no ncurses, no sleeping, no global state.
 * Used by the human game (snake.c), the trainer (train.c) and the tests.
 */

#define WIDTH 40  // x, including the border columns
#define HEIGHT 20 // y, including the border rows
#define MAX_LEN ((WIDTH - 2) * (HEIGHT - 2)) // every interior cell

typedef struct {
    int x, y;
} Point;

// Clockwise order, so "turn right" is dir+1 and "turn left" is dir+3 (mod 4).
typedef enum { DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT } Dir;

// Actions relative to the current heading (what the agent chooses).
typedef enum { ACT_STRAIGHT, ACT_LEFT, ACT_RIGHT } Action;

typedef enum { EV_NONE, EV_ATE, EV_DIED, EV_WON } Event;

typedef struct {
    Point body[MAX_LEN]; // body[0] is the head
    int len;
    Dir dir;
    Point food;
    int score;
    int alive;
    int steps_since_food;
    uint64_t rng;
} Game;

extern const int DIR_DX[4];
extern const int DIR_DY[4];

void game_reset(Game *g, uint64_t seed);

// Move one step in an absolute direction. Reversing into the body is ignored.
Event game_step(Game *g, Dir dir);

// Move one step relative to the current heading.
Event game_step_rel(Game *g, Action a);

// Heading after taking action a.
Dir game_turn(Dir d, Action a);

// Would the head die if it moved onto p this step? (wall or body, tail excluded
// because it moves away in the same step)
int game_is_deadly(const Game *g, Point p);

// Spawn food on a random free interior cell. Returns 0 if the board is full.
int game_spawn_food(Game *g);

// xorshift64* PRNG, so training is reproducible and independent of rand().
uint32_t game_rand(uint64_t *state);

#endif
