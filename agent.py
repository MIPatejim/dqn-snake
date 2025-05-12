import tensorflow as tf
import keras
from keras import layers, models, Optimizer , losses
import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    A class used to store and sample experiences for reinforcement learning.
    Attributes:
    ----------
    buffer : collections.deque
        A deque that holds the experiences with a fixed maximum size.
    Methods:
    -------
    __init__(buffer_size):
        Initializes the ReplayBuffer with a specified maximum buffer size.
    add(experience):
        Adds a new experience to the buffer. If the buffer is full, the oldest experience is removed.
    sample(batch_size):
        Samples a random batch of experiences from the buffer. The batch size will be the smaller of the requested size or the current buffer size.
    __len__():
        Returns the current number of experiences stored in the buffer.
    """
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        actual_batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, actual_batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
class Qnetwork:
    """
    A class representing a Q-network for reinforcement learning.
    Attributes:
        state_size (int): The size of the state space.
        action_size (int): The size of the action space.
        l_rate (float): The learning rate for the optimizer.
        model (keras.Model): The neural network model used for Q-value predictions.
    Methods:
        __init__(state_size, action_size, l_rate):
            Initializes the Qnetwork with the given state size, action size, and learning rate.
        _build_model():
            Builds and compiles the neural network model for Q-value predictions.
        predict(state):
            Predicts Q-values for a given state using the trained model.
    """
    def __init__(self, state_size, action_size, l_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.l_rate = l_rate
        
        self.model = self._build_model()
        
    def _build_model(self):
        model = models.Sequential(
            [   
                layers.Input(shape=(self.state_size,)),
                layers.Dense(64, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(self.action_size, activation='linear')
            ]
        )
                
        model.compile(loss=losses.MeanSquaredError(),  
                    optimizer=keras.optimizers.Adam(learning_rate=self.l_rate), 
                    )
        
        return model
    
    def predict(self, state):
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        return q_values[0]

    


class DQNAgent:
    """
    DQNAgent is a Deep Q-Network (DQN) agent implementation for reinforcement learning tasks.
    Attributes:
        state_size (int): The size of the state space.
        action_size (int): The size of the action space.
        memory_size (int): The maximum size of the replay buffer.
        epsilon (float): The initial exploration rate for the epsilon-greedy policy.
        epsilon_decay (float): The decay rate for epsilon after each training step.
        epsilon_min (float): The minimum value for epsilon.
        gamma (float): The discount factor for future rewards.
        tau (float): The soft update parameter for updating the target network.
        learning_rate (float): The learning rate for the Q-network.
        memory (ReplayBuffer): The replay buffer for storing experiences.
        q_network (Qnetwork): The main Q-network used for action-value estimation.
        target_network (Qnetwork): The target Q-network used for stable training.
    Methods:
        __init__(state_size, action_size, memsize, epsilon, epsilon_decay, epsilon_min, gamma, tau, learning_rate):
            Initializes the DQNAgent with the given parameters.
        act(state):
            Selects an action using an epsilon-greedy policy based on the current state.
        act_play(state):
            Selects an action greedily (without exploration) based on the current state.
        train_step(batch_size):
            Performs a single training step by sampling a minibatch from the replay buffer,
            updating the Q-network, and performing a soft update on the target network.
        remember(state, action, reward, next_state, done):
            Stores an experience tuple in the replay buffer.
        save(weights_filename):
            Saves the weights of the Q-network to a file.
        load(weights_filename, epsilon=False, epsilon_filename=None):
            Loads the weights of the Q-network from a file and synchronizes the target network.
            Optionally loads the epsilon value.
    """
    def __init__(self, state_size, action_size, memsize, epsilon, epsilon_decay, epsilon_min, gamma, tau, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memsize
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        
        
        self.memory = ReplayBuffer(self.memory_size)
        
        self.q_network = Qnetwork(state_size, action_size, self.learning_rate)
        self.target_network = Qnetwork(state_size, action_size, self.learning_rate)
        self.target_network.model.set_weights(self.q_network.model.get_weights())
        
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.q_network.predict(state) 
        return np.argmax(q_values)
    
    def act_play(self, state):
        q_values = self.q_network.predict(state) 
        return np.argmax(q_values)
    
    def train_step(self, batch_size):
        
        if len(self.memory) < batch_size:
            return
        
        minibatch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        y_hat = rewards + (1 - dones) * self.gamma * np.amax(self.target_network.model.predict(next_states, verbose=0), axis=1)
        
        pred = self.q_network.model.predict(states, verbose=0)
        
        pred[np.arange(batch_size), actions] = y_hat
            
        self.q_network.model.fit(states, pred, epochs=1, verbose=0)
        
        
        Qnetwork_weights = self.q_network.model.get_weights()
        target_network_weights = self.target_network.model.get_weights()
        
        updated_weights = [
            self.tau * q_weight + (1 - self.tau) * target_weight
            for q_weight, target_weight in zip(Qnetwork_weights, target_network_weights)
            ]
        self.target_network.model.set_weights(updated_weights)
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))
    
    def save(self, weights_filename):
        try:
            self.q_network.model.save_weights(weights_filename)
            
            print(f"Model weights saved to {weights_filename}")
        except Exception as e:
            print(f"Error saving weights: {e}")

    def load(self, weights_filename, epsilon=False, epsilon_filename=None):
        try:
            self.q_network.model.load_weights(weights_filename)
            # Sincronizar la target_network con la q_network
            self.target_network.model.set_weights(self.q_network.model.get_weights())
            if epsilon and epsilon_filename:
                with open(epsilon_filename, 'r') as f:
                    self.epsilon = float(f.read())
            print(f"Model weights loaded from {weights_filename} and target network synchronized.")
        except Exception as e:
            print(f"Error loading weights: {e}. Starting with initial weights.")