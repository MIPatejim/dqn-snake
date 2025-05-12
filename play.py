from config import *

import pygame 
import os

from agent import DQNAgent
from snake_game import SnakeGameEnv

MODEL_WEIGHTS_FILE = "model.weights.h5"

if __name__ == "__main__":
    
    colors = {
        "white": WHITE,
        "black": BLACK,
        "red": RED,
        "green": GREEN,
        "green2": GREEN2
    }
    # Initialize the game environment
    game = SnakeGameEnv(WIDTH, HEIGHT, BLOCK_SIZE, colors)
    
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
    
    if os.path.exists(MODEL_WEIGHTS_FILE):
        agent.load(MODEL_WEIGHTS_FILE)
    else:
        print(f"Model weights file '{MODEL_WEIGHTS_FILE}' not found. Starting with untrained agent.")
    
    while True:
        
        state = game.reset()
        done = False
        
        while not done:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                    
            
            
            action = agent.act_play(state)
            next_state, _, done = game.step(action)
            
            state = next_state
            
            game.render()
            game.clock.tick(500)

