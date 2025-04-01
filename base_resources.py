import time
import math
import numpy as np

# Funciones auxiliares para visualizar información:

def printMap(world):
    # Visualiza el mapa de GridWorld
    m = ""
    for i in range(world.size[0]):
        for j in range(world.size[1]):
            if world.map[(i, j)] == 0:
                m += " O "
            elif world.map[(i, j)] == -1:
                m += " X "
            elif world.map[(i, j)] == 1:
                m += " F "
            elif world.map[(i, j)] == 2:
                m += " T "
            elif world.map[(i, j)] == 3:
                m += " C "
        if i == world.size[0] - 1:
            m += "\n"
        else:
            m += "\n"
    print(m)

def printPolicy(world, policy):
    # Visualiza la política con flechas
    p = ""
    for i in range(world.size[0]):
        for j in range(world.size[1]):
            if policy[i][j] == 0:
                p += " ^ "
            elif policy[i][j] == 1:
                p += " V "
            elif policy[i][j] == 2:
                p += " < "
            elif policy[i][j] == 3:
                p += " > "
            else:
                p += " x "
        if i == world.size[0] - 1:
            p += "\n"
        else:
            p += "\n"
    print(p)


## Definición de la clase world

class World:

    def __init__(self, size, terminal, obstacle, hole, catapult):
        # Crea un mundo
        self.size = size
        self.map = {}
        for i in range(size[0]):
            for j in range(size[1]):
                # Estados libres
                self.map[(i, j)] = 0
                # Estados terminales
                for t in terminal:
                    if i==t[0] and j==t[1]:
                        self.map[(i, j)] = 1
                # Estados con obstáculos
                for o in obstacle:
                    if i==o[0] and j==o[1]:
                        self.map[(i, j)] = -1
                for h in hole:
                    if i==h[0] and j==h[1]:
                        self.map[(i, j)] = 2
                for c in catapult:
                    if i==c[0] and j==c[1]:
                        self.map[(i, j)] = 3



## Definición de la clase Agent

class Agent:

    def __init__(self, world, initialState):
        # Crea un agente
        self.world = world
        self.initial_state = np.array(initialState)
        self.state = np.array(initialState)
        
    def reset(self):
        # Resets agent to its initial state
        self.state = np.array(self.initial_state)

    def move(self, state, action):
        # Gestiona las transiciones de estados
        nextState = state + np.array(action)
        if nextState[0] < 0:
            nextState[0] = 0
        elif nextState[0] >= self.world.size[0]:
            nextState[0] = self.world.size[0] - 1
        if nextState[1] < 0:
            nextState[1] = 0
        elif nextState[1] >= self.world.size[1]:
            nextState[1] = self.world.size[1] - 1
        if self.world.map[(nextState[0], nextState[1])] == 2:
            aux = nextState
            for i in range(self.world.size[0]):
                for j in range(self.world.size[1]):
                    if self.world.map[(i, j)] == 2 and (nextState[0] != i and nextState[1] != j):
                        aux = np.array([i, j])
                        nextState = aux
        if self.world.map[(nextState[0], nextState[1])] == 3:
            if action == (1, 0):
                nextState = np.array([np.random.randint(nextState[0], self.world.size[0]-1), nextState[1]])
            elif action == (-1, 0):
                nextState = np.array([np.random.randint(0, nextState[0]), nextState[1]])
            elif action == (0, 1):
                nextState = np.array([nextState[0], np.random.randint(nextState[1], self.world.size[1]-1)])
            elif action == (0, -1):
                nextState = np.array([nextState[0], np.random.randint(0, nextState[1])])
        return nextState

    def reward(self, nextState):
        # Gestiona los refuerzos
        if self.world.map[(nextState[0], nextState[1])] == -1:
            # Refuerzo cuando el agente intenta moverse a un obstáculo
            reward = -1 # ** Prueba varios valores **
        elif self.world.map[(nextState[0], nextState[1])] == 1:
            # Refuerzo cuando el agente se mueve a una celda terminal
            reward = 1 # ** Prueba varios valores **
        else:
            # Refuerzo cuando el agente se mueve a una celda libre
            reward = 0 # ** Prueba varios valores **
        return reward

    def checkAction(self, state, action):
        # Planifica una acción
        nextState = self.move(state, action)
        if self.world.map[(state[0], state[1])] == -1:
            nextState = state
        reward = self.reward(nextState)
        return nextState, reward

    def executeAction(self, action):
        # Planifica y ejecuta una acción
        nextState = self.move(self.state, action)
        if self.world.map[(self.state[0], self.state[1])] == -1:
            nextState = self.state
        else:
            self.state = nextState
        reward = self.reward(nextState)
        return self.state, reward
    

## Definición de los mundos:


# Mundo 1 pequeño: Laberinto fácil
obstacles = []
for j in range(0, 4):
    obstacles.append((j, 1))
for j in range(1, 5):
    obstacles.append((j, 3))
w1p = World((5, 5), [(4, 4)], obstacles, [], [])

# Mundo 1 mediano: Laberinto fácil
obstacles = []
for i in [1, 5]:
    for j in range(0, 8):
        obstacles.append((j, i))
for i in [3, 7]:
    for j in range(1, 9):
        obstacles.append((j, i))
w1m = World((9, 9), [(8, 8)], obstacles, [], [])

# Mundo 1 grande: Laberinto fácil
obstacles = []
for i in [1, 5, 9, 13, 17]:
    for j in range(0, 20):
        obstacles.append((j, i))
for i in [3, 7, 11, 15, 19]:
    for j in range(1, 21):
        obstacles.append((j, i))
w1g = World((21, 21), [(20, 20)], obstacles, [], [])


# Mundo 2 pequeño: Obstáculos aleatorios, teletransporte útil
obstacles = []
for i in range(3):
    obstacles.append((np.random.randint(1, 4), np.random.randint(1, 4)))
w2p = World((5, 5), [(4, 4)], obstacles, [(2, 0), (4, 2)], [])

# Mundo 2 mediano: Obstáculos aleatorios, teletransporte útil
obstacles = []
for i in range(10):
    obstacles.append((np.random.randint(1, 9), np.random.randint(1, 9)))
w2m = World((10, 10), [(9, 9)], obstacles, [(3, 1), (8, 6)], [])

# Mundo 2 grande: Obstáculos aleatorios, teletransporte útil
obstacles = []
for i in range(50):
    obstacles.append((np.random.randint(1, 19), np.random.randint(1, 19)))
w2g = World((21, 21), [(20, 20)], obstacles, [(6, 2), (18, 14)], [])

# Mundo 3 pequeño: Obstáculos aleatorios, teletransporte inútil
obstacles = []
for i in range(3):
    obstacles.append((np.random.randint(1, 4), np.random.randint(1, 4)))
w3p = World((5, 5), [(4, 4)], obstacles, [(4, 0), (0, 4)], [])

# Mundo 3 mediano: Obstáculos aleatorios, teletransporte inútil
obstacles = []
for i in range(10):
    obstacles.append((np.random.randint(1, 9), np.random.randint(1, 9)))
w3m = World((10, 10), [(9, 9)], obstacles, [(8, 1), (1, 8)], [])

# Mundo 3 grande: Obstáculos aleatorios, teletransporte inútil
obstacles = []
for i in range(50):
    obstacles.append((np.random.randint(1, 19), np.random.randint(1, 19)))
w3g = World((21, 21), [(20, 20)], obstacles, [(18, 2), (2, 18)], [])

# Mundo 4 pequeño: Obstáculos aleatorios, teletransporte inútil
obstacles = []
for i in range(3):
    obstacles.append((np.random.randint(1, 4), np.random.randint(1, 4)))
w4p = World((5, 5), [(4, 4)], obstacles, [], [(1, 1)])

# Mundo 4 mediano: Obstáculos aleatorios, teletransporte inútil
obstacles = []
for i in range(10):
    obstacles.append((np.random.randint(1, 9), np.random.randint(1, 9)))
w4m = World((10, 10), [(9, 9)], obstacles, [], [(2, 2), (5, 5)])

# Mundo 4 grande: Obstáculos aleatorios, teletransporte inútil
obstacles = []
for i in range(50):
    obstacles.append((np.random.randint(1, 19), np.random.randint(1, 19)))
w4g = World((21, 21), [(20, 20)], obstacles, [], [(2, 2), (5, 5), (12, 12), (15, 15)])


# Mundo 5: Laberinto difícil
obstacles = [(0,1),(0,3),(0,9),(0,15),(0,16),(0,17),(0,19),
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

w5 = World((21, 21), [(20, 20)], obstacles, [], [])