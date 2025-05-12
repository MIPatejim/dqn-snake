
from config import *

import pygame 
import csv
import os

from agent import DQNAgent
from snake_game import SnakeGameEnv

MODEL_WEIGHTS_FILE = "model.weights.h5"
TRACKER_FILE = "tracker.csv"

ENABLE_RENDERING = True


def get_episode_and_epsilon(filename, tracker_columns):
    """
        Retrieves the last epsilon value from a CSV file.
        
        Args:
            filename (str): Path to the CSV file.
            tracker_columns (list): List of column names in the tracker file.
        
        Returns:
            tuple: Last episode (int) and epsilon (float) if the file exists and has data,
                otherwise returns (None, None).
        """
    if os.path.isfile(filename) and os.stat(filename).st_size > 0:
        try:
            with open(filename, "rb") as f:
                # Mover el puntero al final del archivo
                f.seek(0, os.SEEK_END)
                # Leer hacia atrás hasta encontrar una nueva línea
                while f.tell() > 0:
                    f.seek(-2, os.SEEK_CUR)
                    if f.read(1) == b"\n":
                        break
                # Leer la última línea
                last_line = f.readline().decode().strip()
            
            # Dividir la última línea en columnas
            columns = last_line.split(",")
            # Buscar el índice de la columna "Epsilon" y "Episode"
            episode_index = tracker_columns.index("Episode")
            epsilon_index = tracker_columns.index("Epsilon")
            # Devolver el valor de epsilon como float
            return int(columns[episode_index]), float(columns[epsilon_index])
        except Exception as e:
            return None, None
    else:
        return None, None


if __name__ == "__main__":
    # Define colors for the game
    colors = {
        "white": WHITE,
        "black": BLACK,
        "red": RED,
        "green": GREEN,
        "green2": GREEN2
    }

    # Initialize the game environment
    game = SnakeGameEnv(WIDTH, HEIGHT, BLOCK_SIZE, colors)

    # Initialize the DQN agent with the specified parameters
    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        memsize=MEMORY_SIZE,
        epsilon=EPSYLON_START,
        epsilon_decay=EPSYLON_DECAY,
        epsilon_min=EPSYLON_END,
        gamma=GAMMA,
        tau=TAU,
        learning_rate=LEARNING_RATE
    )

    # Define the columns for the tracker file
    tracker_columns = ["Episode", "Reward", "Score", "Epsilon"]

    # Create the tracker file with headers if it doesn't exist or is empty
    if not os.path.isfile(TRACKER_FILE) or os.stat(TRACKER_FILE).st_size == 0:
        with open(TRACKER_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=tracker_columns)
            writer.writeheader()

    # Load the model weights if the file exists
    if os.path.isfile(MODEL_WEIGHTS_FILE):
        agent.load(MODEL_WEIGHTS_FILE)
    else:
        # Create an empty weights file if it doesn't exist
        with open(MODEL_WEIGHTS_FILE, "w") as f:
            pass

    # Initialize the episode offset
    additive_episode = 0

    # Retrieve the last episode and epsilon value from the tracker file
    last_episode, last_epsilon = get_episode_and_epsilon(TRACKER_FILE, tracker_columns)
    if last_episode is not None:
        additive_episode = last_episode
        agent.epsilon = last_epsilon

    # Main training loop
    for episode in range(EPISODES):
        # Reset the game environment and initialize variables
        state = game.reset()
        done = False
        total_reward = 0

        # Determine if the current episode should be visualized
        visualize = (episode + 1) % SAVE_FREQUENCY == 0

        while not done:
            # Handle pygame events (e.g., quit event)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Save the model and exit the program
                    agent.save(MODEL_WEIGHTS_FILE)
                    pygame.quit()
                    exit()

            # Select an action using the agent's policy
            action = agent.act(state)

            # Perform the action in the game environment
            next_state, reward, done = game.step(action)

            # Store the experience in the agent's replay memory
            agent.remember(state, action, reward, next_state, done)

            # Train the agent using a batch of experiences
            agent.train_step(BATCH_SIZE)

            # Update the current state and accumulate the reward
            state = next_state
            total_reward += reward

            # Render the game if visualization is enabled
            if visualize and ENABLE_RENDERING:
                game.render()
                game.clock.tick(GAME_SPEED)

            # End the episode if the maximum number of iterations is reached
            if game.game_iteration > MAX_ITERATIONS:
                done = True

        # Print the episode summary
        print(f"Episode: {episode + 1 + additive_episode}/{EPISODES}, Score: {game.score}")

        # Save the model weights and tracker data periodically
        if (episode + 1) % SAVE_FREQUENCY == 0:
            agent.save(MODEL_WEIGHTS_FILE)
            with open(TRACKER_FILE, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=tracker_columns)
                writer.writerow({
                    "Episode": episode + 1 + additive_episode,
                    "Reward": total_reward,
                    "Score": game.score,
                    "Epsilon": agent.epsilon
                })
                f.flush()
