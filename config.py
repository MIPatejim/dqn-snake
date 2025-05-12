# Neural Network and Training Constants
STATE_SIZE = 11  # Size of the state space: 4 for direction, 4 for food position, 3 for immediate danger
ACTION_SIZE = 3  # Number of possible actions: Forward, Left, Right
GAMMA = 0.97  # Discount factor for future rewards
LEARNING_RATE = 0.001  # Learning rate for the optimizer

# Replay Memory Constants
MEMORY_SIZE = 10000  # Maximum size of the replay memory
BATCH_SIZE = 64  # Number of samples per training batch

# Target Network Update
TAU = 0.001  # Soft update rate for the target network

# Exploration Parameters for ε-greedy Policy
EPSYLON_START = 1.0  # Initial exploration rate (fully random actions)
EPSYLON_END = 0.01  # Minimum exploration rate (almost fully greedy actions)
EPSYLON_DECAY = 0.995  # Decay rate for exploration rate per episode

# Game Environment Constants
BLOCK_SIZE = 20  # Size of each block in the game grid (pixels)
WIDTH = 30  # Number of blocks in the grid's width
HEIGHT = 30  # Number of blocks in the grid's height
SPEED = 2  # Speed for manual testing (blocks per second)

# Color Definitions (RGB format)
WHITE = (255, 255, 255)  # Color for the background
BLACK = (0, 0, 0)  # Color for the grid or snake's body
RED = (255, 0, 0)  # Color for the food
GREEN = (0, 255, 0)  # Color for the snake's head
GREEN2 = (0, 100, 0)  # Alternate green color (optional use)

# Training Parameters
MAX_ITERATIONS = 1000  # Maximum number of iterations per episode
EPISODES = 1000  # Total number of episodes for training

# Model Saving
SAVE_FREQUENCY = 1  # Save the model every X episodes

# Game Speed During Training
GAME_SPEED = 120  # Delay in milliseconds between game steps during training