Snake Game Reinforcement Learning (DQN)

This repository contains a Deep Q-Network (DQN) implementation that learns to play the classic Snake game using reinforcement learning. The project is structured into five main Python modules, each responsible for a different aspect of the environment, agent, training, and gameplay.

### Table of Contents

-Overview  
-Prerequisites  
-Installation  
-Usage  
-File Structure  
-Detailed File Descriptions  
-Hyperparameters and Configuration  
-Saving and Loading Models  
-Contributing  
-License  

## Overview

The agent interacts with a Snake game environment implemented using Pygame. It uses a DQN architecture with separate Q-network and target network, replay memory, and soft updates to learn an optimal policy for maximizing the game score.

### Prerequisites

-Python 3.7+  
-TensorFlow 2.x  
-Keras (integrated within TensorFlow 2.x)  
-Pygame  
-NumPy  

### Install dependencies via pip:  
```
pip install tensorflow pygame numpy  
```
# Installation

Clone this repository:
```
git clone https://github.com/MIPatejim/dqn-snake.git
cd snake-dqn
```

2. Verify that all required packages are installed.


## Usage

### Training the Agent

```
python train.py
```
Training runs for EPISODES (default 1000).

The agent is trained using experience replay and soft target updates.

Model weights are saved every SAVE_FREQUENCY episodes to model.weights.h5.

A tracker.csv file logs episode, reward, score, and epsilon decay.  

### Playing with a Trained Agent
```
python play.py
```
Loads the latest model.weights.h5 if available.

Runs the game in a loop, letting the agent play automatically.

### Manual Play
```
python snake_game.py
```
Launches the Snake game with keyboard controls:

Left arrow: turn left

Right arrow: turn right

Escape: exit

File Structure
```
├── agent.py        # DQNAgent, Q-network, and replay buffer
├── config.py       # Hyperparameters and game constants
├── play.py         # Script to load a trained agent and play
├── snake_game.py   # Pygame environment for Snake
└── train.py        # Training loop and CSV tracker
```
Detailed File Descriptions

## agent.py

### Classes: 

ReplayBuffer(buffer_size): Fixed-size deque for storing experiences and sampling minibatches.

Qnetwork(state_size, action_size, l_rate): Defines and compiles a feedforward neural network with three hidden layers.

DQNAgent(...): Combines Qnetwork, target network, replay buffer, and implements epsilon-greedy policy, training step, weight updates, and model save/load.

## config.py

Stores all hyperparameters and constants:

State & Action Sizes: STATE_SIZE = 11, ACTION_SIZE = 3

RL Hyperparameters: GAMMA, LEARNING_RATE, EPSYLON_START, EPSYLON_DECAY, EPSYLON_END, TAU.

Replay & Training: MEMORY_SIZE, BATCH_SIZE, EPISODES, MAX_ITERATIONS, SAVE_FREQUENCY.

Game Settings: WIDTH, HEIGHT, BLOCK_SIZE, SPEED, GAME_SPEED, color definitions.

## play.py

Initializes SnakeGameEnv and a DQNAgent.

Loads pre-trained weights if available; otherwise starts untrained.

Runs an infinite loop where the agent selects actions (act_play) and the environment is rendered at high speed.

## snake_game.py

### Implements SnakeGameEnv class:

Methods:

reset(): Sets initial snake position, length, score, and places food.

step(action): Applies action, updates direction, moves snake, checks collisions, computes reward, and returns next state.

render(): Draws the snake, food, and score using Pygame.

_get_state(): Encodes the current game state into an 11-dimensional binary feature vector.

_is_collision(pt): Checks for wall or self-collisions.

## train.py

### Main training loop:

Initializes environment, agent, and tracker CSV.

Loads previous weights and epsilon from files if they exist.

For each episode:

Runs until done or exceeding MAX_ITERATIONS.

Agent acts, environment steps, rewards collected.

Experiences stored and training step performed.

On save episodes, writes weights and logs metrics to tracker.csv.

Hyperparameters and Configuration

All key parameters are defined in config.py. You can easily adjust:

Exploration rate (EPSYLON_START, EPSYLON_DECAY, EPSYLON_END).

Network learning rate (LEARNING_RATE).

Replay memory size (MEMORY_SIZE).

Training episodes and frequency (EPISODES, SAVE_FREQUENCY).

Game dimensions and speed (WIDTH, HEIGHT, GAME_SPEED).

Saving and Loading Models

Weights file: model.weights.h5 stores neural network weights.

Tracker file: tracker.csv logs per-episode metrics and epsilon.

Use agent.save() and agent.load() in train.py and play.py.

# Contributing

Fork the repository.

Create a feature branch:
```
git checkout -b feature/MyFeature
```


3. Commit your changes:
```
git commit -m "Add new feature"
```
Push to your branch and open a pull request.

License

This project is released under the MIT License. Feel free to use and modify.
