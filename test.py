import numpy as np
import matplotlib.pyplot as plt

from path_finding_functions import SARSA, QLearning  # :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
from base_resources import *           # :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}

# ------------------------------------------------------------
# 1. Parámetros generales
# ------------------------------------------------------------
alpha         = 0.1
gamma         = 0.9
initial_state = (0, 0)
actions       = [(-1,0), (1,0), (0,-1), (0,1)]
episodes      = 500

# Mundo de ejemplo (teletransporte útil)
np.random.seed(42)
obstacles = [(np.random.randint(1,4), np.random.randint(1,4)) for _ in range(3)]
world = w2p

printMap(world)

epsilons = [0.01, 0.1, 0.3, 0.5]

# ------------------------------------------------------------
# 2. Función que colecciona pasos por episodio
# ------------------------------------------------------------
def run_lengths(AgentClass, ε):
    agent = AgentClass(world, initial_state, actions, alpha, gamma, ε)
    agent.Q[:] = 0.0

    lengths = np.zeros(episodes, dtype=int)
    for ep in range(episodes):
        agent.state = agent.initial_state.copy()
        # elige la primera acción
        action = agent.chooseAction(tuple(agent.state))

        step_count = 0
        # hasta llegar al terminal
        while world.map[tuple(agent.state)] != 1:
            state = agent.state.copy()
            next_state, _ = agent.executeAction(actions[action])
            step_count += 1

            if AgentClass is SARSA:
                next_action = agent.chooseAction(tuple(next_state))
                agent.updateValue(state, action, 0, next_state, next_action)
            else:
                agent.updateValue(state, action, 0, next_state, None)
                next_action = agent.chooseAction(tuple(next_state))

            action = next_action

        lengths[ep] = step_count

    return lengths

# ------------------------------------------------------------
# 3. Ejecutar experimentos
# ------------------------------------------------------------
results = {'SARSA':{}, 'QLearning':{}}
for ε in epsilons:
    results['SARSA'][ε]     = run_lengths(SARSA,     ε)
    results['QLearning'][ε] = run_lengths(QLearning, ε)

# ------------------------------------------------------------
# 4. Graficar pasos medios en bloques de 50 episodios
# ------------------------------------------------------------
block = 50
x = np.arange(block, episodes+1, block)

plt.figure(figsize=(10,6))
for algo, color in zip(['SARSA','QLearning'], ['C0','C1']):
    for ε, ls in zip(epsilons, ['-','--',':','-.']):
        # promedio de pasos en cada bloque
        avg_steps = [
            results[algo][ε][i-block:i].mean()
            for i in x
        ]
        plt.plot(x, avg_steps,
                 linestyle=ls,
                 color=color,
                 label=f"{algo}, ε={ε}")

plt.xlabel("Episodio")
plt.ylabel("Pasos medios (últimos 50 eps.)")
plt.title("Convergencia: número de pasos hasta la meta")
plt.legend()
plt.grid(True)
plt.show()
