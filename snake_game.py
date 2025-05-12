import pygame
import random
import numpy as np



class SnakeGameEnv:
    """
    A class representing the Snake Game environment for reinforcement learning.
    Attributes:
        w (int): Width of the game grid in blocks.
        h (int): Height of the game grid in blocks.
        block_size (int): Size of each block in pixels.
        colors (dict): Dictionary containing color mappings for the game elements.
        font (pygame.font.Font): Font used for rendering text.
        display (pygame.Surface): Pygame surface for the game window.
        clock (pygame.time.Clock): Pygame clock for controlling the game loop.
        directions (tuple): Tuple of direction vectors for movement.
        direction (int): Current direction index (0 = up, 1 = right, 2 = down, 3 = left).
        direction_vector (numpy.ndarray): Current direction vector (x, y).
        head (numpy.ndarray): Current position of the snake's head (x, y).
        snake (list): List of numpy arrays representing the snake's body segments.
        snake_length (int): Current length of the snake.
        score (int): Current score of the game.
        food (numpy.ndarray): Position of the food (x, y).
        game_iteration (int): Number of iterations since the game started.
    Methods:
        __init__(w, h, block_size, colors):
            Initializes the SnakeGameEnv instance with the given parameters.
        reset():
            Resets the game to its initial state and returns the initial state.
        _place_food():
            Places the food at a random position on the grid, avoiding the snake's body.
        step(action):
            Updates the game state based on the given action and returns the next state, reward, done flag, and info.
        _is_collision(pt):
            Checks if the given point collides with the snake's body or the walls.
        _get_state():
            Returns the current state of the game as a numpy array of length 11.
        render():
            Renders the game window with the current state of the game.
    """
    
    def __init__(self, w, h, block_size, colors):
        """
        Initializes the SnakeGame instance.
        Args:
            w (int): The width of the game grid in blocks.
            h (int): The height of the game grid in blocks.
            block_size (int): The size of each block in pixels.
            colors (dict): A dictionary containing color mappings for the game.
        Attributes:
            w (int): The width of the game grid in blocks.
            h (int): The height of the game grid in blocks.
            block_size (int): The size of each block in pixels.
            colors (dict): A dictionary containing color mappings for the game.
            font (pygame.font.Font): The font used for rendering text in the game.
            display (pygame.Surface): The game window surface.
            clock (pygame.time.Clock): The clock object to control the game loop's frame rate.
        Initializes the Pygame library, sets up the game window, and resets the game state.
        """
        self.w = w
        self.h = h 
        self.block_size = block_size
        self.colors = colors
        
        pygame.init()
        self.font = pygame.font.SysFont("arial", 25) 
        display_width_px = self.block_size * self.w #int
        display_height_px = self.block_size * self.h #int
        self.display = pygame.display.set_mode((display_width_px, display_height_px)) #Surface
        pygame.display.set_caption("Snake Game with RL") #Set window title
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        """
        Resets the game to its initial state.
        Returns:
            numpy.ndarray: The initial state of the game.
        """
        # Initialize directions and starting direction
        self.directions = ((0, -1), (1, 0), (0, 1), (-1, 0))  # (up, right, down, left)
        self.direction = 3  # Initial direction (left)
        self.direction_vector = np.array(self.directions[self.direction])

        # Set initial position of the snake
        self.head = np.array([self.w // 2, self.h // 2])  # Start at the center of the grid
        self.snake = [self.head, self.head - self.direction_vector]  # Snake starts with 2 segments
        self.snake_length = 2

        # Initialize score and food
        self.score = 0
        self.food = None
        self._place_food()  # Place the first food on the grid

        # Reset game iteration counter
        self.game_iteration = 0

        # Return the initial state
        return self._get_state()
        
    def _place_food(self):
        """
        Places the food at a random position on the grid, ensuring it does not overlap with the snake's body.
        """
        while True:
            # Generate random coordinates for the food
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            self.food = np.array([x, y])
            
            # Ensure the food does not overlap with the snake's body
            if not any((self.food == segment).all() for segment in self.snake):
                break
            
    def step(self, action):
        """
        Updates the game state based on the given action.
        Args:
            action (int): Action to take (0 = nothing, 1 = turn left, 2 = turn right).
        Returns:
            tuple: (next_state, reward, done, info)
                - next_state (numpy.ndarray): The next state of the game.
                - reward (float): Reward for the current step.
                - done (bool): Whether the game is over.
        """
        self.game_iteration += 1
        done = False
        reward = 0

        # Update direction based on action
        if action == 1:  # Turn left
            self.direction = (self.direction - 1) % 4
            self.direction_vector = np.array(self.directions[self.direction])
        elif action == 2:  # Turn right
            self.direction = (self.direction + 1) % 4
            self.direction_vector = np.array(self.directions[self.direction])

        # Move the snake's head
        self.head = self.snake[0] + self.direction_vector

        # Check for collisions
        if self._is_collision(self.head):
            reward = -10  # Penalty for collision
            done = True
        else:
            # Check if the snake eats the food
            if (self.head == self.food).all():
                self.score += 1
                reward = 10  # Reward for eating food
                self.snake.insert(0, self.head)  # Grow the snake
                self.snake_length += 1
                self._place_food()  # Place new food
            else:
                # Move the snake forward
                for i in range(self.snake_length - 1, 0, -1):
                    self.snake[i] = self.snake[i - 1]
                self.snake[0] = self.head
                reward = -0.1  # Small penalty for moving

        # Get the next state
        state = self._get_state()

        return state, reward, done
    
    def _is_collision(self, pt):
        """
        Check if a given point results in a collision.

        This method determines whether the specified point `pt` causes a collision
        with the boundaries of the game area or with the snake's own body.

        Args:
            pt (tuple): A tuple representing the (x, y) coordinates of the point to check.

        Returns:
            bool: True if the point results in a collision (either with the walls or the snake's body),
                False otherwise.
        """
        #Check if snake has collided with itself or the walls
        x = pt[0]
        y = pt[1]
        if (x < 0 or x >= self.w or
            y < 0 or y >= self.h or
            any((pt == segment).all() for segment in self.snake[1:])):
            return True
        return False
    
    def _get_state(self):
        """
        Generate the current state representation of the snake game.
        The state is represented as an 11-dimensional numpy array where:
        - Indices 0-3 represent the current direction of the snake:
            - 0: Moving up
            - 1: Moving right
            - 2: Moving down
            - 3: Moving left
        - Index 4 indicates if there is a collision in the current direction.
        - Index 5 indicates if there is a collision when turning left.
        - Index 6 indicates if there is a collision when turning right.
        - Indices 7-10 represent the relative position of the food:
            - 7: Food is above the snake's head.
            - 8: Food is below the snake's head.
            - 9: Food is to the left of the snake's head.
            - 10: Food is to the right of the snake's head.
        Returns:
            np.ndarray: An 11-dimensional array representing the current state of the game.
        """
        state = np.zeros((11,), dtype=np.int8)
        if self.direction == 0:
            state[0] = 1
        elif self.direction == 1:
            state[1] = 1
        elif self.direction == 2:
            state[2] = 1
        elif self.direction == 3:
            state[3] = 1
            
        if self._is_collision(self.head + self.direction_vector):
            state[4] = 1
            
        left_turn = np.array(self.directions[(self.direction - 1) % 4])
        right_turn = np.array(self.directions[(self.direction + 1) % 4])
        
        if self._is_collision(self.head + left_turn):
            state[5] = 1
        if self._is_collision(self.head + right_turn):
            state[6] = 1
            
        #Food relative position
        if self.food[0] < self.head[0]:
            state[7] = 1
        elif self.food[0] > self.head[0]:
            state[8] = 1
        if self.food[1] < self.head[1]:
            state[9] = 1
        elif self.food[1] > self.head[1]:
            state[10] = 1
        
        return state
    
    def render(self):
        #Render the game
        self.display.fill(self.colors["black"])
        
        for i, segment in enumerate(self.snake):
            
            if i == 0:
                COLOR = self.colors["green"]
            else:
                COLOR = self.colors["green2"]
            pygame.draw.rect(
                            self.display,
                            COLOR,
                            pygame.Rect(
                                segment[0] * self.block_size,
                                segment[1] * self.block_size,
                                self.block_size,
                                self.block_size
                                )
                            )
        pygame.draw.rect(
                        self.display,
                        self.colors["red"],
                        pygame.Rect(
                            self.food[0] * self.block_size,
                            self.food[1] * self.block_size,
                            self.block_size,
                            self.block_size
                            )
                        )
        
        score_text = self.font.render(f"Score: {self.score}", True, self.colors["white"])
        self.display.blit(score_text, (10, 10))
        
        pygame.display.flip()
    
    
    

if __name__ == "__main__":
    import pygame 
    import random
    import numpy as np
    
    import config as cfg
    from config import BLOCK_SIZE, WIDTH, HEIGHT, SPEED, WHITE, BLACK, RED, GREEN, GREEN2
    
    colors = {
        "white": WHITE,
        "black": BLACK,
        "red": RED,
        "green": GREEN,
        "green2": GREEN2
    }
    
    game = SnakeGameEnv(WIDTH, HEIGHT, BLOCK_SIZE, colors)
    
    running = True
    
    while running:
        action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 1
                elif event.key == pygame.K_RIGHT:
                    action = 2
                elif event.key == pygame.K_ESCAPE:
                    running = False
                
        game.render()
        
        #Step the game
        state, reward, done = game.step(action)
        
        print(f"Direccion: {state[0:4]} | Colision: {state[4]} | Left: {state[5]} | Right: {state[6]} | Food: {state[7:11]} | Score: {game.score} | Iteration: {game.game_iteration}")
        
        if done:
            print("Game Over")
            game.reset()
        
        game.clock.tick(10)
        
    pygame.quit()