# -fopenmp lets train_cnn split each batch over several CPU cores (cnn.c).
CFLAGS = -O2 -fopenmp -Wall -Wextra
LDLIBS = -lm
CORE = game.c qlearn.c nn.c dqn.c cnn.c
HEADERS = game.h qlearn.h nn.h dqn.h cnn.h

all: snake train train_dqn train_cnn test

snake: snake.c $(CORE) $(HEADERS)
	$(CC) $(CFLAGS) snake.c $(CORE) -o snake -lncurses $(LDLIBS)

train: train.c $(CORE) $(HEADERS)
	$(CC) $(CFLAGS) train.c $(CORE) -o train $(LDLIBS)

train_dqn: train_dqn.c $(CORE) $(HEADERS)
	$(CC) $(CFLAGS) train_dqn.c $(CORE) -o train_dqn $(LDLIBS)

train_cnn: train_cnn.c $(CORE) $(HEADERS)
	$(CC) $(CFLAGS) train_cnn.c $(CORE) -o train_cnn $(LDLIBS)

test: test.c $(CORE) $(HEADERS)
	$(CC) $(CFLAGS) test.c $(CORE) -o test $(LDLIBS)

check: test
	./test

clean:
	rm -f snake train train_dqn train_cnn test

.PHONY: all check clean
