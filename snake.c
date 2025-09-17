#include <ncurses.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define WIDTH 40
#define HEIGHT 20

typedef struct {
    int x, y;
} Point;

const Point middle = {WIDTH / 2, HEIGHT / 2};

int badLocation(Point *snake, Point *food, int len) {
    for (int i = 0; i < len; i++) {
        if (food->x == snake[i].x && food->y == snake[i].y) {
            return 1;
        }
    }
    return 0;
}

void eat(Point *snake, Point *food, int *len, int *score) {
    if (snake[0].x == food->x && snake[0].y == food->y) {
        (*len)++;
        (*score)++;
        do {
            food->x = 1 + rand() % (WIDTH - 3);
            food->y = 1 + rand() % (HEIGHT - 3);
        } while (badLocation(snake, food, *len));
    }
}

void draw(Point *snake, Point *food, int len, int score) {
    erase();
    for (int x = 0; x < WIDTH; x++) {
        mvprintw(0, x, "#");
        mvprintw(HEIGHT-1, x, "#");
    }
    for (int y = 0; y < HEIGHT; y++) {
        mvprintw(y, 0, "#");
        mvprintw(y, WIDTH-1, "#");
    }
    mvprintw(0, WIDTH + 1, "Score: %d\n", score);
    mvprintw(1, WIDTH + 1, "WASD/Arrows to move\n");
    mvprintw(2, WIDTH + 1, "Press q to quit\n");
    for (int i = 0; i < len; i++) {
        if (i == 0) {
            mvaddch(snake[i].y, snake[i].x, '@');
        } else {
            mvaddch(snake[i].y, snake[i].x, 'O');
        }
    }
    mvaddch(food->y, food->x, 'F');
    refresh();
}

void gameOver(int score, int len) {
    erase();
    mvprintw(HEIGHT / 2, WIDTH / 4, "You got a score of %d and your snake was %d units long\n", score, len);
    mvprintw(HEIGHT - 1, 0, "Press q to quit\n");
    refresh();
}

int leaderboard(int score, const char *path) {
    FILE *scoring = fopen(path, "a");
    if (!scoring) {
        fprintf(stderr, "File couldn't be opened\n");
        return EXIT_FAILURE;
    }
    char name[30];
    char sc[30];
    char line[100];
    printf("Enter name:\n");
    fgets(name, sizeof(name), stdin);
    while (*name == '\0' || *name == '\n') {
        printf("Enter a real name:\n");
        fgets(name, sizeof(name), stdin); 
    }
    
    name[strcspn(name, "\n")] = '\0';

    snprintf(sc, sizeof(sc), "%d", score);
    snprintf(line, sizeof(line), "%s %s\n", name, sc);

    const size_t length = strlen(line);
    size_t write = fwrite(line, 1, length, scoring);
    if (write != length) {
        perror("Error writing into file\n");
    }
    fclose(scoring);
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    initscr();
    noecho();
    curs_set(false);
    keypad(stdscr, TRUE);
    nodelay(stdscr, TRUE);

    int score = 0;
    int collision = 0;
    const char *path;
    if (argc == 2) {
        path = argv[1];
    } else {
        path = "leaderboard.txt";
    }

    // Snake
    Point snake[100];
    int snakeLength = 1;
    snake[0].x = middle.x;
    snake[0].y = middle.y;
    int dx = 1, dy = 0;

    // Food
    Point food = { 1 + rand() % (WIDTH - 3), 1 + rand() % (HEIGHT - 3)};
    
    int ch;
    while (!collision) {
        // Input
        ch = getch();
        switch (ch) {
            case 'w':
            case KEY_UP: 
                dx = 0; dy = -1; 
                break;
            case 's':
            case KEY_DOWN: 
                dx = 0; dy = 1; 
                break;
            case 'a':
            case KEY_LEFT: 
                dx = -1; dy = 0; 
                break;
            case 'd':    
            case KEY_RIGHT: 
                dx = 1; dy = 0; 
                break;
            case 'q': 
                endwin(); 
                return 0;
        }

        //Move snake
        for (int i = snakeLength - 1; i > 0; i--) {
            snake[i] = snake[i - 1];
        }
        snake[0].x += dx;
        snake[0].y += dy;

        // Check collision
        if (snake[0].x <= 0 || snake[0].x >= WIDTH - 1|| snake[0].y <= 0 || snake[0].y >= HEIGHT - 1) {
            break;
        }
        for (int i = 1; i < snakeLength; i++) {
            if (snake[0].x == snake[i].x && snake[0].y == snake[i].y) collision = 1;
        }

        eat(snake, &food, &snakeLength, &score);
        draw(snake, &food, snakeLength, score);
        usleep(100000); // Control speed
    }
    
    gameOver(score, snakeLength);
    while (getch() != 'q');
    endwin();
    leaderboard(score, path);
    printf("Score : %d\n", score);
    return 0;
}
