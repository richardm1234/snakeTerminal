#include <ncurses.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define WIDTH 40 // x
#define HEIGHT 20 // y


typedef struct {
    int x, y;
} Point;

enum Difficulty {
    LIGHT = 100000,
    MEDIUM = 75000,
    HARD = 50000
};

int badLocation(Point *snake, Point *food, int len) {
    for (int i = 0; i < len; i++) {
        if (food->x == snake[i].x && food->y == snake[i].y) {
            return 1;
        }
    }
    if (food->x == 0 || food->x == WIDTH-1 || food->y == 0 || food->y == HEIGHT-1) {
        return 1;
    }
    return 0;
}

void eat(Point *snake, Point *food, int *len, int *score) {
    if (snake[0].x == food->x && snake[0].y == food->y) {
        snake[*len] = snake[*len - 1]; // to avoid reading garbage
        (*len)++;
        (*score)++;
        do {
            food->x =  rand() % WIDTH;
            food->y =  rand() % HEIGHT;
        } while (badLocation(snake, food, *len));
    }
}



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


int setDifficulty() {
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
        case 'q':
            erase();
            mvprintw(HEIGHT / 2, 0, "Exited game");
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
/* Draw */

void drawMenu() {
    mvprintw(0, 0, "####### #     #       # #     # #######");
    mvprintw(1, 0, "#       ##    #      ## #    #  #      ");
    mvprintw(2, 0, "#       # #   #     # # #   #   #      ");
    mvprintw(3, 0, "####### #  #  #    #  # ####    #######");
    mvprintw(4, 0, "      # #   # #   ##### #   #   #      ");
    mvprintw(5, 0, "      # #    ##  #    # #    #  #      ");
    mvprintw(6, 0, "####### #     # #     # #     # #######");
    mvprintw(8, 0, "WELCOME TO SNAKE");
    mvprintw(10, 0, "Select difficulty:");
    mvprintw(11, 0, "Light(1)");
    mvprintw(12, 0, "Medium(2)");
    mvprintw(13, 0, "Hard(3)");
    refresh();
        
}

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

void drawStart(Point *snake, Point *food) {
    erase();
    drawBorder(0);
    mvaddch(snake[0].y, snake[0].x, '@');
    mvaddch(food->y, food->x, 'F');
    refresh();     
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

    // game variables
    int difficulty = LIGHT;
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
    snake[0].x = 1 + rand() % (WIDTH - 2);
    snake[0].y = 1 + rand() % (HEIGHT - 2);
    int dx = 0, dy = 0;

    // Food
    Point food = {  rand() % WIDTH, rand() % HEIGHT};

    while (badLocation(snake, &food, snakeLength)) {
        food.x =  rand() % WIDTH;
        food.y =  rand() % HEIGHT;
    }

    drawMenu();
    difficulty = setDifficulty();
    if (difficulty == -1) {
        endwin();
        printf("Thanks for not playing\n");
        return 0;
    }
    drawStart(snake, &food);
    int ch = firstKeyPress(); 
    
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
                printf("Thanks for playing\n"); 
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
        usleep(difficulty);
    }
    
    gameOver(score, snakeLength);
    nodelay(stdscr, false);
    usleep(1500000);
    endwin();
    leaderboard(score, path);
    printf("Score : %d\n", score);
    printf("Thanks for playing\n");

    return 0; 
}
