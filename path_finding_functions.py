import time
import random
import numpy as np

from base_resources import *


class MetodoBasadoEnValor(Agent):
    def __init__(self, world, initial_state, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(world, initial_state)
        
        self.actions = actions
        self.num_actions = len(actions)
        
        self.Q = np.zeros((self.world.size[0], 
                                self.world.size[1], 
                                self.num_actions), dtype=float)
        
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Balance exploration and explotation factor

    def updateValue(self, current_state, current_action, current_reward, next_state, next_action):
        raise Exception("updateValue() method not implemented.")

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            # Exploration: choose random action
            return random.randint(0, self.num_actions - 1)
        
        else:
            # Exploit: choose the best known action in 'state'
            values = self.Q[state[0]][state[1]]
            max_value = np.max(values)

            best_actions = [i for i, value in enumerate(values) if value == max_value ]

            return random.choice(best_actions)

    def train(self, num_episodes=1000, max_iter=99999):
        # Initialize Q
        self.Q = np.zeros((self.world.size[0], 
                    self.world.size[1], 
                    len(self.actions)), dtype=float)

        # Start training
        print(f"Starting training for {num_episodes} episodes.")
        start_time = time.time()

        for episode in range(num_episodes):
            # Initialize the model
            self.state = self.initial_state

            # Choose an action
            current_action = self.chooseAction(self.state)
            
            i = 0
            while self.world.map[(self.state[0], self.state[1])] != 1 and i < max_iter:
                # Execute action
                current_state = self.state
                next_state, current_reward = self.executeAction(self.actions[current_action])

                # Choose next action
                next_action = self.chooseAction(next_state)

                # Update Q values
                self.updateValue(current_state, current_action, current_reward, next_state, next_action)
                
                # Set current action
                current_action = next_action

                i += 1

            # Print progress
            if (episode + 1) % (num_episodes // 10) == 0: 
                print(f"  Episode {episode + 1}/{num_episodes} completed.")

        # Finish training
        end_time = time.time()
        print(f"Training finished in {end_time - start_time:.2f} seconds.")

    def getPolicy(self):
        policy = np.zeros((self.world.size[0], 
                                self.world.size[1]), dtype=int)

        for r in range(self.world.size[0]):
            for c in range(self.world.size[1]):
                policy[r][c] = np.argmax(self.Q[r][c])
                   
        return policy

    
class SARSA(MetodoBasadoEnValor):
    def __init__(self, world, initial_state, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(world, initial_state, actions, alpha, gamma, epsilon)

    def updateValue(self, current_state, current_action, current_reward, next_state, next_action):
        current_Q = self.Q[current_state[0], current_state[1], current_action]
        next_Q = self.Q[next_state[0], next_state[1], next_action]
        
        new_Q = current_Q + self.alpha * (current_reward + self.gamma * next_Q - current_Q)

        self.Q[current_state[0], current_state[1], current_action] = new_Q


class QLearning(MetodoBasadoEnValor):
    def __init__(self, world, initial_state, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(world, initial_state, actions, alpha, gamma, epsilon)

    def updateValue(self, current_state, current_action, current_reward, next_state, next_action):
        current_Q = self.Q[current_state[0], current_state[1], current_action]
        next_Q = np.max(self.Q[next_state[0], next_state[1]])
        
        new_Q = current_Q + self.alpha * (current_reward + self.gamma * next_Q - current_Q)

        self.Q[current_state[0], current_state[1], current_action] = new_Q
