import numpy as np
import random
from collections import defaultdict
import time # Optional: To time the training

def printMap(world):
  # Visualiza el mapa de GridWorld
  m = ""
  for i in range(world.size[0]):
    for j in range(world.size[1]):
      if world.map.get((i, j), 0) == 0: # Use .get for safety
        m += " O "
      elif world.map.get((i, j), 0) == -1:
        m += " X "
      elif world.map.get((i, j), 0) == 1:
        m += " F "
      elif world.map.get((i, j), 0) == 2:
        m += " T "
      else: # Handle potential unexpected values
        m += " ? "
    m += "\n"
  print(m)

def printPolicy(world, policy):
  # Visualiza la política con flechas
  p = ""
  action_arrows = {
      0: " ^ ", # Up
      1: " v ", # Down
      2: " < ", # Left
      3: " > ", # Right
      -1: " # " # Default for obstacles/terminals if needed
  }
  for i in range(world.size[0]):
    for j in range(world.size[1]):
      state_type = world.map.get((i, j), 0)
      color = "\033[0m" # Default color reset
      char = " ? "

      if state_type == -1:
        char = " # " # Obstacle
      elif state_type == 1:
        char = " @ " # Final/Goal
      else:
        if state_type == 2:
            color = "\033[94m" # Blue for wormhole

        # Get action index from policy, default to -1 if not found or invalid
        action_idx = policy.get((i,j), -1)
        char = action_arrows.get(action_idx, " ? ") # Use dictionary lookup

      p += color + char + "\033[0m" # Apply color and reset
    p += "\n"
  print(p)

class World:
  def __init__(self, size, terminal, obstacle, hole):
    # Crea un mundo
    self.size = size
    self.map = {}
    # Initialize all cells as free space first
    for i in range(size[0]):
      for j in range(size[1]):
        self.map[(i, j)] = 0

    # Set terminal states
    for t in terminal:
      if 0 <= t[0] < size[0] and 0 <= t[1] < size[1]:
        self.map[(t[0], t[1])] = 1

    # Set obstacle states
    for o in obstacle:
      if 0 <= o[0] < size[0] and 0 <= o[1] < size[1]:
        self.map[(o[0], o[1])] = -1

    # Set wormhole states
    self.holes = [] # Store hole locations
    for h in hole:
      if 0 <= h[0] < size[0] and 0 <= h[1] < size[1]:
         # Make sure it's not overwriting a terminal or obstacle
         if self.map[(h[0], h[1])] == 0:
            self.map[(h[0], h[1])] = 2
            self.holes.append(tuple(h)) # Store as tuple

    # Ensure there are exactly 0 or 2 holes for teleportation logic
    if len(self.holes) != 0 and len(self.holes) != 2:
        print(f"Warning: Found {len(self.holes)} wormholes. Teleportation requires 0 or 2.")
        # Decide how to handle this - perhaps disable teleportation?
        # For now, the agent's move logic might behave unpredictably.


  def get_state_type(self, state_tuple):
      """Gets the type of a state tuple (i, j). Returns 0 if out of bounds."""
      if 0 <= state_tuple[0] < self.size[0] and 0 <= state_tuple[1] < self.size[1]:
          return self.map.get(state_tuple, 0)
      return 0 # Treat out of bounds like free space for reward, but agent won't go there

  def is_goal(self, state_tuple):
    return self.get_state_type(state_tuple) == 1

  def is_obstacle(self, state_tuple):
    return self.get_state_type(state_tuple) == -1

  def is_wormhole(self, state_tuple):
      return self.get_state_type(state_tuple) == 2

class Agent:
  # Actions: 0: Up, 1: Down, 2: Left, 3: Right
  ACTION_VECTORS = [
      np.array([-1, 0]), # 0: Up
      np.array([1, 0]),  # 1: Down
      np.array([0, -1]), # 2: Left
      np.array([0, 1])   # 3: Right
  ]
  NUM_ACTIONS = len(ACTION_VECTORS)

  def __init__(self, world, initialState):
    # Crea un agente
    self.world = world
    self.initialState = tuple(initialState) # Store initial state as tuple
    self.state = tuple(initialState) # Current state as tuple

  def reset(self):
    """Resets the agent to its initial state."""
    self.state = self.initialState

  def _apply_boundaries(self, state_array):
      """Clamps the state array within world boundaries."""
      state_array[0] = np.clip(state_array[0], 0, self.world.size[0] - 1)
      state_array[1] = np.clip(state_array[1], 0, self.world.size[1] - 1)
      return state_array

  def move(self, current_state_tuple, action_idx):
    # Calculates the next state based on current state and action
    current_state_array = np.array(current_state_tuple)
    action_vector = self.ACTION_VECTORS[action_idx]

    next_state_array = current_state_array + action_vector
    next_state_array = self._apply_boundaries(next_state_array)
    next_state_tuple = tuple(next_state_array)

    # Handle Obstacles: If intended move is into an obstacle, stay put.
    if self.world.is_obstacle(next_state_tuple):
        return current_state_tuple # Stay in the current state

    # Handle Wormholes: If landing on a wormhole, teleport
    if self.world.is_wormhole(next_state_tuple) and len(self.world.holes) == 2:
        # Find the *other* hole
        if next_state_tuple == self.world.holes[0]:
            return self.world.holes[1]
        else:
            return self.world.holes[0]

    # Otherwise, the calculated next state is valid
    return next_state_tuple

  def reward(self, result_state_tuple):
    # Calculates the reward for *arriving* at result_state_tuple
    state_type = self.world.get_state_type(result_state_tuple)

    if state_type == -1:
      # This case should ideally not happen if move() prevents moving *into* obstacles
      # But if it does (e.g., starting in one?), penalize.
      # Or, perhaps the penalty comes from *trying* to move into one?
      # Let's adjust the reward logic: Reward is based on the state entered.
      # The penalty for hitting a wall comes from not moving + maybe a small step cost.
      # Reworking reward based on the description:
      return -1 # Original code: Reward when *agent attempts* to move to obstacle
       # This is tricky with the current move logic. Let's follow the description's intent:
       # Reward is determined *after* the move attempt.
       # If move() resulted in staying put because of an obstacle, the state hasn't changed.
       # If move() resulted in entering a goal, reward is +1.
       # Otherwise, reward is 0.
       # The original code description is slightly ambiguous. Let's simplify:
       # +1 for reaching Goal, -1 for hitting Obstacle (implicitly by staying put + step cost perhaps?), 0 otherwise.
       # A common approach is a small negative reward for *each step* to encourage efficiency.
       # Let's stick to the provided rewards for now:
      pass # This state type should be handled by the goal check


    if state_type == 1:
      # Refuerzo cuando el agente se mueve a una celda terminal
      return 1
    # We need a penalty for hitting an obstacle boundary in `executeAction`
    # For now, let's return 0 for free space and wormholes
    elif state_type == 0 or state_type == 2:
       return 0
    else: # Should not happen if map is correct
       return 0


  def executeAction(self, action_idx):
    """
    Calculates the next state and reward for taking an action.
    Updates the agent's internal state.
    Returns (new_state_tuple, reward).
    """
    current_state_tuple = self.state
    intended_next_state_array = np.array(current_state_tuple) + self.ACTION_VECTORS[action_idx]
    intended_next_state_array = self._apply_boundaries(intended_next_state_array)
    intended_next_state_tuple = tuple(intended_next_state_array)

    # Check if the *intended* move is into an obstacle
    tried_obstacle = self.world.is_obstacle(intended_next_state_tuple)

    # Determine the actual resulting state using the move logic
    next_state_tuple = self.move(current_state_tuple, action_idx)

    # Update agent's internal state
    self.state = next_state_tuple

    # Determine reward based on the description's logic
    if tried_obstacle:
        reward = -1 # Penalty for hitting obstacle
    elif self.world.is_goal(next_state_tuple):
        reward = 1  # Reward for reaching goal
    else:
        reward = 0  # Reward for moving to free space/wormhole

    return self.state, reward


  def is_terminal(self, state_tuple):
    """Checks if the given state tuple is a terminal state (Goal)."""
    # Obstacles are not terminal states in the sense that the episode ends,
    # the agent just gets stuck or penalized. The episode ends when the goal is reached.
    return self.world.is_goal(state_tuple)

# --- Learning Algorithms ---

def choose_action_epsilon_greedy(Q, state, epsilon):
    """Chooses an action using epsilon-greedy strategy."""
    if random.random() < epsilon:
        # Explore: choose a random action
        return random.randint(0, Agent.NUM_ACTIONS - 1)
    else:
        # Exploit: choose the best known action
        q_values = Q[state] # Assumes state exists in Q (defaultdict handles this)
        # Find the index of the maximum value. Handle ties by choosing randomly among maxima.
        max_q = np.max(q_values)
        # Get indices of all actions with the max Q-value
        best_actions = [idx for idx, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

# == SARSA Update Function ==
def sarsa_step(Q, state, action, reward, next_state, next_action, alpha, gamma):
    """Performs a single SARSA update."""
    current_q = Q[state][action]
    next_q = Q[next_state][next_action] # Q-value of the *actually chosen* next action
    target = reward + gamma * next_q
    new_q = current_q + alpha * (target - current_q)
    Q[state][action] = new_q

# == Q-Learning Update Function ==
def qlearning_step(Q, state, action, reward, next_state, alpha, gamma):
    """Performs a single Q-Learning update."""
    current_q = Q[state][action]
    best_next_q = np.max(Q[next_state]) # Q-value of the *best possible* next action
    target = reward + gamma * best_next_q
    new_q = current_q + alpha * (target - current_q)
    Q[state][action] = new_q

# == Training Loop ==
def train(agent, num_episodes, alpha, gamma, epsilon, step_function):
    """
    Trains the agent using the specified update function (SARSA or Q-Learning).

    Args:
        agent: The agent instance.
        num_episodes: Number of episodes to train for.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Exploration rate for epsilon-greedy.
        step_function: The update function to use (sarsa_step or qlearning_step).

    Returns:
        Q: The learned Q-table (a defaultdict).
    """
    # Initialize Q-table using defaultdict for convenience
    # It automatically assigns a default value (list of zeros) for new states
    Q = defaultdict(lambda: [0.0] * Agent.NUM_ACTIONS)
    world = agent.world

    print(f"Starting training for {num_episodes} episodes using {step_function.__name__}...")
    start_time = time.time()

    for episode in range(num_episodes):
        agent.reset()
        state = agent.state
        action = choose_action_epsilon_greedy(Q, state, epsilon)

        while not agent.is_terminal(state):
            # Execute the action in the environment
            next_state, reward = agent.executeAction(action)

            # Choose the next action based on the next state
            next_action = choose_action_epsilon_greedy(Q, next_state, epsilon)

            # Perform the update using the chosen step function
            if step_function == sarsa_step:
                 sarsa_step(Q, state, action, reward, next_state, next_action, alpha, gamma)
            elif step_function == qlearning_step:
                 qlearning_step(Q, state, action, reward, next_state, alpha, gamma)
            else:
                raise ValueError("Invalid step_function provided")

            # Move to the next state and action
            state = next_state
            action = next_action # Crucial for SARSA, Q-Learning recalculates anyway

            # Optional: Add a step limit per episode to prevent infinite loops in bad scenarios
            # if step_count > max_steps_per_episode: break

        # Optional: Decay epsilon over time
        # epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % (num_episodes // 10) == 0: # Print progress
             print(f"  Episode {episode + 1}/{num_episodes} completed.")

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    return Q

# == Policy Extraction ==
def extract_policy(Q, world):
    """Extracts the greedy policy from the Q-table."""
    policy = {}
    for r in range(world.size[0]):
        for c in range(world.size[1]):
            state = (r, c)
            state_type = world.get_state_type(state)
            if state_type == 1 or state_type == -1: # Goal or Obstacle
                 policy[state] = -1 # No action needed/possible
            else:
                 if state in Q:
                     # Choose the best action (index)
                     policy[state] = np.argmax(Q[state])
                 else:
                     # If state was never visited, choose a default (e.g., random or up)
                     policy[state] = 0 # Default to 'Up' if state unseen
                     # Or maybe keep it undefined? Let's default to Up.
                     # A better default might be random: random.randint(0, Agent.NUM_ACTIONS - 1)
    return policy


if __name__ == "__main__":

  # --- World Definitions ---
  # Mundo 1 pequeño: Laberinto fácil
  obstacles = []
  for j in range(0, 4): obstacles.append((j, 1))
  for j in range(1, 5): obstacles.append((j, 3))
  w1p = World((5, 5), [(4, 4)], obstacles, [])
  print("World 1 Small:")
  printMap(w1p)

  # Mundo 1 mediano: Laberinto fácil
  obstacles = []
  for i in [1, 5]:
    for j in range(0, 8): obstacles.append((j, i))
  for i in [3, 7]:
    for j in range(1, 9): obstacles.append((j, i))
  w1m = World((9, 9), [(8, 8)], obstacles, [])
  # print("World 1 Medium:")
  # printMap(w1m) # Keep output concise for example

  # Mundo 1 grande: Laberinto fácil
  obstacles = []
  for i in [1, 5, 9, 13, 17]:
    for j in range(0, 20): obstacles.append((j, i))
  for i in [3, 7, 11, 15, 19]:
    for j in range(1, 21): obstacles.append((j, i))
  w1g = World((21, 21), [(20, 20)], obstacles, [])
  # print("World 1 Large:")
  # printMap(w1g) # Keep output concise for example

  # Mundo 2 pequeño: Obstáculos aleatorios, teletransporte útil
  obstacles = []
  np.random.seed(42) # for reproducibility
  for _ in range(3):
    # Avoid placing obstacles on start (0,0), goal (4,4), or holes
    while True:
        o = (np.random.randint(0, 5), np.random.randint(0, 5))
        if o != (0,0) and o != (4,4) and o != (2,0) and o != (4,2):
            obstacles.append(o)
            break
  w2p = World((5, 5), [(4, 4)], obstacles, [(2, 0), (4, 2)])
  print("\nWorld 2 Small (Useful Teleport):")
  printMap(w2p)

  # Mundo 3 pequeño: Obstáculos aleatorios, teletransporte inútil
  obstacles = []
  np.random.seed(43) # different seed
  for _ in range(3):
     while True:
        o = (np.random.randint(0, 5), np.random.randint(0, 5))
        if o != (0,0) and o != (4,4) and o != (4,0) and o != (0,4):
            obstacles.append(o)
            break
  w3p = World((5, 5), [(4, 4)], obstacles, [(4, 0), (0, 4)])
  print("\nWorld 3 Small (Useless Teleport):")
  printMap(w3p)

  # Mundo 4: Laberinto difícil
  obstacles4 = [(0,1),(0,3),(0,9),(0,15),(0,16),(0,17),(0,19),
               (1,1),(1,3),(1,4),(1,5),(1,6),(1,7),(1,9),(1,10),(1,11),(1,12),(1,13),(1,17),(1,19),
               (2,1),(2,9),(2,13),(2,15),(2,16),(2,17),(2,19),
               (3,1),(3,3),(3,5),(3,7),(3,9),(3,11),(3,16),(3,19),
               (4,3),(4,5),(4,7),(4,8),(4,9),(4,10),(4,11),(4,12),(4,13),(4,14),(4,16),(4,18),(4,19),
               (5,0),(5,1),(5,2),(5,3),(5,5),(5,9),(5,16),
               (6,5),(6,6),(6,7),(6,9),(6,10),(6,11),(6,12),(6,13),(6,14),(6,16),(6,17),(6,19),
               (7,0),(7,1),(7,2),(7,3),(7,5),(7,7),(7,9),(7,19),
               (8,3),(8,7),(8,8),(8,9),(8,12),(8,13),(8,14),(8,15),(8,16),(8,17),(8,18),(8,19),
               (9,1),(9,3),(9,5),(9,7),(9,11),(9,12),(9,19),(9,20),
               (10,1),(10,3),(10,5),(10,6),(10,7),(10,9),(10,11),(10,14),(10,15),(10,16),(10,17),
               (11,1),(11,3),(11,5),(11,9),(11,11),(11,13),(11,14),(11,17),(11,18),(11,19),
               (12,1),(12,5),(12,6),(12,8),(12,9),(12,11),(12,13),(12,19),
               (13,1),(13,2),(13,3),(13,4),(13,5),(13,8),(13,15),(13,16),(13,17),(13,19),
               (14,4),(14,7),(14,8),(14,10),(14,12),(14,13),(14,15),(14,19),
               (15,0),(15,1),(15,2),(15,6),(15,7),(15,10),(15,13),(15,14),(15,15),(15,17),(15,18),(15,19),(15,20),
               (16,2),(16,3),(16,5),(16,6),(16,7),(16,8),(16,9),(16,10),(16,11),(16,15),(16,17),
               (17,0),(17,3),(17,5),(17,9),(17,13),(17,14),(17,15),(17,17),(17,19),
               (18,0),(18,1),(18,5),(18,6),(18,7),(18,9),(18,10),(18,11),(18,15),(18,19),
               (19,1),(19,2),(19,4),(19,5),(19,11),(19,13),(19,14),(19,15),(19,16),(19,17),(19,18),(19,19),
               (20,7),(20,8),(20,9),(20,11),(20,19)]
  w4 = World((21, 21), [(20, 20)], obstacles4, [])
  # print("\nWorld 4 (Difficult Maze):")
  # printMap(w4) # Keep output concise for example


  # --- Training ---
  # Hyperparameters (can be tuned)
  ALPHA = 0.1       # Learning rate
  GAMMA = 0.9       # Discount factor
  EPSILON = 0.1     # Exploration rate
  NUM_EPISODES = 1000 # Number of training episodes

  # Let's train on the small worlds (w1p, w2p, w3p) for demonstration

  # == SARSA Training ==
  print("\n--- Training with SARSA ---")
  agent_sarsa_w1 = Agent(w1p, initialState=(0, 0))
  Q_sarsa_w1 = train(agent_sarsa_w1, NUM_EPISODES, ALPHA, GAMMA, EPSILON, sarsa_step)
  policy_sarsa_w1 = extract_policy(Q_sarsa_w1, w1p)
  print("\nSARSA Policy for World 1 Small:")
  printPolicy(w1p, policy_sarsa_w1)

  agent_sarsa_w2 = Agent(w2p, initialState=(0, 0))
  Q_sarsa_w2 = train(agent_sarsa_w2, NUM_EPISODES, ALPHA, GAMMA, EPSILON, sarsa_step)
  policy_sarsa_w2 = extract_policy(Q_sarsa_w2, w2p)
  print("\nSARSA Policy for World 2 Small:")
  printMap(w2p) # Print map again for reference
  printPolicy(w2p, policy_sarsa_w2)

  agent_sarsa_w3 = Agent(w3p, initialState=(0, 0))
  Q_sarsa_w3 = train(agent_sarsa_w3, NUM_EPISODES, ALPHA, GAMMA, EPSILON, sarsa_step)
  policy_sarsa_w3 = extract_policy(Q_sarsa_w3, w3p)
  print("\nSARSA Policy for World 3 Small:")
  printMap(w3p) # Print map again for reference
  printPolicy(w3p, policy_sarsa_w3)

  # == Q-Learning Training ==
  print("\n--- Training with Q-Learning ---")
  agent_q_w1 = Agent(w1p, initialState=(0, 0))
  Q_q_w1 = train(agent_q_w1, NUM_EPISODES, ALPHA, GAMMA, EPSILON, qlearning_step)
  policy_q_w1 = extract_policy(Q_q_w1, w1p)
  print("\nQ-Learning Policy for World 1 Small:")
  printPolicy(w1p, policy_q_w1) # Compare with SARSA

  agent_q_w2 = Agent(w2p, initialState=(0, 0))
  Q_q_w2 = train(agent_q_w2, NUM_EPISODES, ALPHA, GAMMA, EPSILON, qlearning_step)
  policy_q_w2 = extract_policy(Q_q_w2, w2p)
  print("\nQ-Learning Policy for World 2 Small:")
  printMap(w2p) # Print map again for reference
  printPolicy(w2p, policy_q_w2) # Compare with SARSA

  agent_q_w3 = Agent(w3p, initialState=(0, 0))
  Q_q_w3 = train(agent_q_w3, NUM_EPISODES, ALPHA, GAMMA, EPSILON, qlearning_step)
  policy_q_w3 = extract_policy(Q_q_w3, w3p)
  print("\nQ-Learning Policy for World 3 Small:")
  printMap(w3p) # Print map again for reference
  printPolicy(w3p, policy_q_w3) # Compare with SARSA



  # You can add training for larger/more complex worlds (w1m, w1g, w4) here
  # Note: They will require significantly more episodes and potentially tuning
  # For example, training w4:
  print("\n--- Training World 4 with Q-Learning (might take time) ---")
  agent_q_w4 = Agent(w4, initialState=(0, 0))
  # Increase episodes significantly for complex maze
  Q_q_w4 = train(agent_q_w4, 20000, ALPHA, GAMMA, EPSILON, qlearning_step)
  policy_q_w4 = extract_policy(Q_q_w4, w4)
  print("\nQ-Learning Policy for World 4:")
  printMap(w4)
  printPolicy(w4, policy_q_w4)