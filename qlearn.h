#ifndef QLEARN_H
#define QLEARN_H

#include <stdint.h>

#include "game.h"

/*
 * Tabular Q-learning for snake.
 *
 * State: 14 bits
 *   bit 0..7   the 8 cells around the head are deadly (wall or body),
 *              clockwise from the cell above: N NE E SE S SW W NW
 *   bit 8..9   heading (0 up, 1 right, 2 down, 3 left)
 *   bit 10..13 food is left / right / above / below the head
 */

#define QL_STATE_BITS 14
#define QL_STATES (1 << QL_STATE_BITS)
#define QL_ACTIONS 3

// An episode is cut off if the snake goes this long without eating
// (otherwise a looping policy would run forever).
#define QL_STARVE_STEPS(len) (100 * (len))

typedef struct {
    float q[QL_STATES][QL_ACTIONS];
} QTable;

int ql_state(const Game *g);
int ql_best(const QTable *t, int s);
int ql_act(const QTable *t, int s, float eps, uint64_t *rng);
void ql_update(QTable *t, int s, int a, float r, int s2, int done,
               float alpha, float gamma);
int ql_visited(const QTable *t);

int ql_save(const QTable *t, const char *path);
int ql_load(QTable *t, const char *path);

#endif
