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
        snake[*len] = snake[*len - 1]; // to avoid reading garbage
        (*len)++;
        (*score)++;
        do {
            food->x = 1 + rand() % (WIDTH - 3);
            food->y = 1 + rand() % (HEIGHT - 3);
        } while (badLocation(snake, food, *len));
    }
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

/* Draw */

void drawBorder(int score) {
    for (int x = 0; x < WIDTH; x++) {
        mvaddch(0, x, '#');
        mvaddch(HEIGHT-1, x, '#');
    }
    for (int y = 0; y < HEIGHT; y++) {
        mvaddch(y, 0, '#');
        mvaddch(y, WIDTH-1, '#');
    }
    mvprintw(0, WIDTH + 1, "Score: %d", score);
    mvprintw(1, WIDTH + 1, "WASD/ARROWS to move");
    mvprintw(2, WIDTH + 1, "Press q to quit");
}

int drawStart(Point *snake, Point *food) {
    erase();
    drawBorder(0);
    mvaddch(snake[0].y, snake[0].x, '@');
    mvaddch(food->y, food->x, 'F');
    mvprintw(HEIGHT + 1, 1, "WASD/ARROWS to start");
    mvprintw(HEIGHT + 2, 1, "Other key -> move right");
    refresh();

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

void draw(Point *snake, Point *food, int len, int score) {
    erase();
    drawBorder(score);
    mvaddch(snake[0].y, snake[0].x, '@');
    for (int i = 1; i < len; i++) {
        mvaddch(snake[i].y, snake[i].x, 'O');
    }
    mvaddch(food->y, food->x, 'F');
    refresh();
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
    snake[0].x = 1 + rand() % (WIDTH - 3);
    snake[0].y = 1 + rand() % (HEIGHT - 3);
    int dx = 0, dy = 0;

    // Food
    Point food = { 1 + rand() % (WIDTH - 3), 1 + rand() % (HEIGHT - 3)};
    
    int ch = drawStart(snake, &food);
    
    nodelay(stdscr, TRUE);

    while (!collision) {
        
        // Input
        int newCh = getch();
        if (newCh != ERR) {
            ch = newCh;
        }
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
    nodelay(stdscr, false);
    while (getch() != 'q') {

    }
    endwin();
    leaderboard(score, path);
    printf("Score : %d\n", score);
    return 0;
}
