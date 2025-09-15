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



void foodEat(Point *snake, Point *food, int *len, int *score) {
    if (snake[0].x == food->x && snake[0].y == food->y) {
            (*len)++;
            (*score)++;
            food->x = (rand() % (WIDTH - 3)) + 1;
            food->y = (rand() % (HEIGHT - 3)) + 1;

        }

}

void draw(Point *snake, Point *food, int len, int score, int *refresh) {

    clear();
    for (int x = 0; x < WIDTH; x++) {
        mvprintw(0, x, "#");
        mvprintw(HEIGHT-1, x, "#");
    }
    for (int y = 0; y < HEIGHT; y++) {
        mvprintw(y, 0, "#");
        mvprintw(y, WIDTH-1, "#");
    }
    mvprintw(0, WIDTH + 1, "Score: %d\n", score);
    for (int i = 0; i < len; i++) {
        if (i == 0) {
            mvaddch(snake[i].y, snake[i].x, '@');
        } else {
            mvaddch(snake[i].y, snake[i].x, 'O');
        }
    }
    mvaddch(food->y, food->x, 'F');
    if (refresh) {
        refresh();
    } else {
        (*refresh) = !(*refresh);
    }
}

void gameOver(int score, int len) {
    clear();
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

    const char *path;
    if (argc == 2) {
        path = argv[1];
    } else {
        path = "leaderboard.txt";
    }
    int refresh = 0;
    int score = 0;
    
    // Snake start
    Point snake[100];
    int snakeLength = 1;
    snake[0].x = middle.x;
    snake[0].y = middle.y;

    // Food
    Point food = { rand() % WIDTH, rand() % HEIGHT};

    // Snake moves right;
    int dx = 1, dy = 0;
    
    int ch;
    while (1) {
        // Input
        ch = getch();
        //timeout();
        switch (ch) {
            case KEY_UP: dx = 0; dy = -1; break;
            case KEY_DOWN: dx = 0; dy = 1; break;
            case KEY_LEFT: dx = -1; dy = 0; break;
            case KEY_RIGHT: dx = 1; dy = 0; break;
            case 'q': endwin(); return 0;
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
            if (snake[0].x == snake[i].x && snake[0].y == snake[i].y) goto game_is_over;
        }

        foodEat(snake, &food, &snakeLength, &score);
        draw(snake, &food, snakeLength, score, &refresh);
        

        usleep(100000); // Control speed
    }

    game_is_over:
    gameOver(score, snakeLength);
    while (getch() != 'q') {
        
    }
    endwin();
    
    leaderboard(score, path);
    printf("Endscore : %d\n", score);
    return 0;
}