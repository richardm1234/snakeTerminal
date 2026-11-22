# snakeTerminal

This is a simple snake game which is played in the terminal written in C.

To run it in your terminal:

``` Bash
make
./snake (creates leaderboard.txt and appends to it)
./snake name.txt (creates name.txt)
```

# AI player (Q-learning)

The game includes a reinforcement learning agent written in plain C.

``` Bash
make            # builds snake, train, train_dqn, train_cnn and test
./test          # sanity tests: game rules, Q-learning, DQN and CNN
./train         # trains in well under a second, writes snake_q.bin
./snake         # pick "Watch AI(5)" in the menu
```

New to reinforcement learning? Read [RL_TUTORIAL.md](RL_TUTORIAL.md), which explains
everything step by step using this code.

# AI player (Deep Q-Network)

A second agent replaces the Q-table with a small neural network, written from scratch in C
(no libraries).

``` Bash
./train_dqn --features view   # about 4 minutes on one core, writes snake_dqn.bin
./snake                       # pick "Watch DQN(6)" in the menu
```

`--features small` gives the network the same 14 inputs as the Q-table; `--features view`
gives it a 9×9 window around the head; `--features tail` adds where the tail is (an
experiment that didn't help). [DQN_TUTORIAL.md](DQN_TUTORIAL.md) explains how it works and
compares all three.

# AI player (convolutional network)

A third agent swaps the DQN's fully connected network for a small convolutional one
(also from scratch, in `cnn.c`). Because a convolution reuses the same 3×3 filters
everywhere, it can look at a much bigger window around the head.

``` Bash
./train_cnn               # 9x9 window, the same inputs as --features view
./train_cnn --radius 9    # 19x19 window
./snake                   # pick "Watch CNN(7)" in the menu
```

[CNN_TUTORIAL.md](CNN_TUTORIAL.md) explains convolutions, their backprop, and compares the
results with the DQN.
