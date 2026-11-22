#include <ncurses.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "cnn.h"
#include "dqn.h"
#include "game.h"
#include "nn.h"
#include "qlearn.h"

#define Q_TABLE_PATH "snake_q.bin"
#define DQN_PATH "snake_dqn.bin"
#define CNN_PATH "snake_cnn.bin"

// Who plays when the AI is watched.
enum Player { HUMAN, AI_TABLE, AI_DQN, AI_CNN };

enum Difficulty {
    LIGHT = 100000,
    MEDIUM = 75000,
    HARD = 50000
};

void gameOver(int score, int len) {
    erase();
    mvprintw(HEIGHT / 2, WIDTH / 4, "You got a score of %d and your snake was %d units long\n", score, len);
    refresh();
}

int leaderboard(int score, const char *path) {
    FILE *scoring = fopen(path, "a");
    if (!scoring) {
        perror( "File couldn't be opened\n");
        return EXIT_FAILURE;
    }
    char name[30];
    char line[128];
    printf("Enter name:\n");
    if (!fgets(name, sizeof(name), stdin)) {
        fprintf(stderr, "Input error\n");
        fclose(scoring);
        return EXIT_FAILURE;
    }
    
    name[strcspn(name, "\n")] = '\0';
    while (name[0] == '\0') {
        printf("Enter a real name:\n");
        if (!fgets(name, sizeof(name), stdin)) {
            fprintf(stderr, "Input error\n");
            fclose(scoring);
            return EXIT_FAILURE;
        }
        name[strcspn(name, "\n")] = '\0';
    }
    
    snprintf(line, sizeof(line), "%s %d\n", name, score);

    if (fputs(line, scoring) == EOF) {
        perror("Error writing to file");
    }
    fclose(scoring);
    return EXIT_SUCCESS;
}


int setDifficulty(bool* funky, enum Player* player) {
    nodelay(stdscr, FALSE);
    int ch = getch();
    switch (ch) {
        case '1':
            erase();
            mvprintw(HEIGHT / 2, 0, "Selected Light");
            refresh();
            usleep(1000000);
            return LIGHT;
        case '2':
            erase();
            mvprintw(HEIGHT / 2, 0, "Selected Medium");
            refresh();
            usleep(1000000);
            return MEDIUM;
        case '3':
            erase();
            mvprintw(HEIGHT / 2, 0, "Selected Hard");
            refresh();
            usleep(1000000);
            return HARD;
        case '4':
            erase();
            mvprintw(HEIGHT / 2, 0, "Selected Funky");
            refresh();
            usleep(1000000);
            uint32_t val = ((uint32_t) time(NULL) % 3);
            *funky = true;
            switch (val) {
                case 0:
                    return LIGHT;
                case 1:
                    return MEDIUM;
                case 2:
                    return HARD;
                default:
                    return MEDIUM; 
            }
        case '5':
            erase();
            mvprintw(HEIGHT / 2, 0, "Selected Watch AI");
            refresh();
            usleep(1000000);
            *player = AI_TABLE;
            return MEDIUM;
        case '6':
            erase();
            mvprintw(HEIGHT / 2, 0, "Selected Watch DQN");
            refresh();
            usleep(1000000);
            *player = AI_DQN;
            return MEDIUM;
        case '7':
            erase();
            mvprintw(HEIGHT / 2, 0, "Selected Watch CNN");
            refresh();
            usleep(1000000);
            *player = AI_CNN;
            return MEDIUM;
        case 'q':
            erase();
            mvprintw(HEIGHT / 2, 0, "Exited game");
            refresh();
            usleep(1000000);
            return -1;
        default:
            erase();
            mvprintw(HEIGHT / 2, 0, "No selection, default: Light");
            refresh();
            usleep(1000000);
            return LIGHT;    
    }
}

int firstKeyPress() {
   // temporarily block until a key is pressed
    nodelay(stdscr, FALSE);
    int ch = getch();
    switch (ch) {
        case 'w':
        case KEY_UP: 
            return 'w';
        case 's':
        case KEY_DOWN: 
            return 's';
        case 'a':
        case KEY_LEFT: 
            return 'a';
        case 'd':    
        case KEY_RIGHT: 
            return 'd';
        case 'q': 
            return 'q';
        default:
            return 'd';
    }
}

// initialise color pairs
void initColors() {
    init_pair(1, COLOR_WHITE, COLOR_BLACK); // text
    init_pair(2, COLOR_BLACK, COLOR_GREEN); // snake head
    init_pair(3, COLOR_BLACK, COLOR_YELLOW); // snake body
    init_pair(4, COLOR_BLACK, COLOR_RED); // food
    init_pair(5, COLOR_WHITE, COLOR_WHITE); // border
    init_pair(6, COLOR_BLACK, COLOR_CYAN);
    init_pair(7, COLOR_BLACK, COLOR_MAGENTA);
    init_pair(8, COLOR_BLACK, COLOR_BLUE);
}


/*** DRAW FUNCTIONS ***/

void drawBlock(int y, int x, int colorPair) {
    attron(COLOR_PAIR(colorPair));
    mvaddch(y, x * 2, ' ');
    mvaddch(y, x * 2 + 1, ' ');
    attroff(COLOR_PAIR(colorPair));
}

void drawHead(int y, int x) {
    attron(COLOR_PAIR(2));
    mvaddch(y, x * 2, ':');
    mvaddch(y, x * 2 + 1, ':');
    attroff(COLOR_PAIR(2));
}

void drawMenu() {
    attron(COLOR_PAIR(1));
    mvprintw(0, 0, "####### #     #       # #     # #######");
    mvprintw(1, 0, "#       ##    #      ## #    #  #      ");
    mvprintw(2, 0, "#       # #   #     # # #   #   #      ");
    mvprintw(3, 0, "####### #  #  #    #  # ####    #######");
    mvprintw(4, 0, "      # #   # #   ##### #   #   #      ");
    mvprintw(5, 0, "      # #    ##  #    # #    #  #      ");
    mvprintw(6, 0, "####### #     # #     # #     # #######");
    attroff(COLOR_PAIR(1));
    mvprintw(8, 0, "WELCOME TO SNAKE");
    mvprintw(10, 0, "Select difficulty:");
    mvprintw(11, 0, "Light(1)");
    mvprintw(12, 0, "Medium(2)");
    mvprintw(13, 0, "Hard(3)");
    mvprintw(14, 0, "Funky(4)");
    mvprintw(15, 0, "Watch AI(5)");
    mvprintw(16, 0, "Watch DQN(6)");
    mvprintw(17, 0, "Watch CNN(7)");
    refresh();
        
}

void drawBorder(int score) {
    for (int x = 0; x < WIDTH; x++) {
        drawBlock(0, x, 5);
        drawBlock(HEIGHT-1, x, 5);
    }
    for (int y = 0; y < HEIGHT; y++) {
        drawBlock(y, 0, 5);
        drawBlock(y, WIDTH-1, 5);
    }
    
    mvprintw(0, 2*WIDTH + 1, "Score: %d", score);
    mvprintw(1, 2*WIDTH + 1, "WASD/ARROWS to move");
    mvprintw(2, 2*WIDTH + 1, "Press q to quit");
}

void drawStart(Point *snake, Point *food) {
    erase();
    drawBorder(0);
    drawBlock(snake[0].y, snake[0].x, 2);
    drawBlock(food->y, food->x, 4);
    refresh();     
}


void drawFunky(Point *snake, Point *food, int len, int score) {
    erase();
    drawBorder(score);
    drawHead(snake[0].y, snake[0].x);
    for (int i = 1; i < len; i++) {
        drawBlock(snake[i].y, snake[i].x, rand() % 8 + 1);
    }
    drawBlock(food->y, food->x, rand() % 8 + 1);
    refresh();
}

void draw(Point *snake, Point *food, int len, int score) {
    erase();
    drawBorder(score);
    drawHead(snake[0].y, snake[0].x);
    for (int i = 1; i < len; i++) {
        drawBlock(snake[i].y, snake[i].x, 2);
    }
    drawBlock(food->y, food->x, 4);
    refresh();
}


void render(Game *g, bool funky) {
    if (funky) {
        drawFunky(g->body, &g->food, g->len, g->score);
    } else {
        draw(g->body, &g->food, g->len, g->score);
    }
}

// A trained agent: picks the next action for the current game.
typedef Action (*Policy)(const Game *g, const void *agent);

static Action tablePolicy(const Game *g, const void *agent) {
    return (Action)ql_best((const QTable *)agent, ql_state(g)); // greedy: no exploration
}

typedef struct {
    Net net;
    FeatType feat;
} DQNAgent;

static Action dqnPolicy(const Game *g, const void *agent) {
    const DQNAgent *d = agent;
    float x[FEAT_MAX];
    dqn_features(d->feat, g, x);
    return (Action)dqn_best(&d->net, x);
}

typedef struct {
    CNN net;
    CNNCache cache; // cnn_forward's working memory
} CNNAgent;

static Action cnnPolicy(const Game *g, const void *agent) {
    CNNAgent *c = (CNNAgent *)agent; // the cache is scratch space, written each call
    dqn_view(g, c->net.radius, c->cache.x);
    return (Action)cnn_best(&c->net, c->cache.x, &c->cache);
}

// Let a trained agent play. Returns the final score.
int watchAI(Game *g, Policy policy, const void *agent, int delay) {
    drawStart(g->body, &g->food);
    nodelay(stdscr, TRUE);
    while (g->alive) {
        if (getch() == 'q') {
            break;
        }
        game_step_rel(g, policy(g, agent));
        if (!g->alive) {
            break;
        }
        if (g->steps_since_food > QL_STARVE_STEPS(g->len)) {
            break; // stuck in a loop
        }
        render(g, false);
        mvprintw(3, 2*WIDTH + 1, "AI is playing");
        refresh();
        usleep(delay);
    }
    return g->score;
}

int main(int argc, char *argv[]) {

    srand(time(NULL));
    initscr();
    noecho();
    curs_set(false);
    keypad(stdscr, TRUE);
    nodelay(stdscr, TRUE);

    start_color();
    initColors();

    // Set path to write to
    const char *path;
    if (argc == 2) {
        path = argv[1];
    } else {
        path = "leaderboard.txt";
    }

    Game g;
    game_reset(&g, (uint64_t)time(NULL));

    drawMenu();
    bool funky = false;
    enum Player player = HUMAN;
    int difficulty = setDifficulty(&funky, &player);
    if (difficulty == -1) {
        endwin();
        printf("Thanks for not playing\n");
        return 0;
    }

    if (player == AI_TABLE) {
        static QTable Q;
        if (!ql_load(&Q, Q_TABLE_PATH)) {
            endwin();
            printf("Could not load %s - run ./train first\n", Q_TABLE_PATH);
            return 1;
        }
        watchAI(&g, tablePolicy, &Q, difficulty);
        gameOver(g.score, g.len);
        usleep(1500000);
        endwin();
        printf("AI score: %d\n", g.score);
        return 0;
    }

    if (player == AI_DQN) {
        DQNAgent d;
        int feat;
        if (!nn_load(&d.net, &feat, DQN_PATH)) {
            endwin();
            printf("Could not load %s - run ./train_dqn first\n", DQN_PATH);
            return 1;
        }
        d.feat = (FeatType)feat;
        if (d.net.n[0] != dqn_feature_count(d.feat)) {
            endwin();
            printf("%s expects %d inputs, but the %s features have %d now "
                   "(VIEW_R changed?) - run ./train_dqn again\n", DQN_PATH,
                   d.net.n[0], dqn_feature_name(d.feat), dqn_feature_count(d.feat));
            nn_free(&d.net);
            return 1;
        }
        watchAI(&g, dqnPolicy, &d, difficulty);
        gameOver(g.score, g.len);
        usleep(1500000);
        endwin();
        nn_free(&d.net);
        printf("DQN score: %d\n", g.score);
        return 0;
    }

    if (player == AI_CNN) {
        CNNAgent c;
        if (!cnn_load(&c.net, CNN_PATH)) {
            endwin();
            printf("Could not load %s - run ./train_cnn first\n", CNN_PATH);
            return 1;
        }
        if (!cnn_cache_init(&c.cache, &c.net)) {
            endwin();
            printf("Out of memory\n");
            cnn_free(&c.net);
            return 1;
        }
        watchAI(&g, cnnPolicy, &c, difficulty);
        gameOver(g.score, g.len);
        usleep(1500000);
        endwin();
        cnn_cache_free(&c.cache);
        cnn_free(&c.net);
        printf("CNN score: %d\n", g.score);
        return 0;
    }

    drawStart(g.body, &g.food);
    int ch = firstKeyPress();

    nodelay(stdscr, TRUE);

    // Game loop
    Dir dir = g.dir;
    while (g.alive) {

        // Input
        int newCh = getch();
        if (newCh != ERR) {
            ch = newCh;
        }
        switch (ch) {
            case 'w':
            case KEY_UP:
                dir = DIR_UP;
                break;
            case 's':
            case KEY_DOWN:
                dir = DIR_DOWN;
                break;
            case 'a':
            case KEY_LEFT:
                dir = DIR_LEFT;
                break;
            case 'd':
            case KEY_RIGHT:
                dir = DIR_RIGHT;
                break;
            case 'q':
                endwin();
                printf("Thanks for playing\n");
                return 0;
        }

        Event ev = game_step(&g, dir);
        if (ev == EV_DIED || ev == EV_WON) {
            break;
        }
        if (ev == EV_ATE) {
            beep();
        }
        dir = g.dir; // a rejected 180° turn keeps the old heading

        render(&g, funky);
        usleep(difficulty);
    }

    gameOver(g.score, g.len);
    nodelay(stdscr, false);
    usleep(500000);
    endwin();

    // Put score on leaderboard
    leaderboard(g.score, path);
    printf("Score : %d\n", g.score);
    printf("Thanks for playing\n");

    return 0;
}
