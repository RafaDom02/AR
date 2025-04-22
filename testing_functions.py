import numpy as np

from base_resources import *
from path_finding_functions import SARSA, QLearning


def run_lengths(AgentClass, world, initial_state, actions, alpha, gamma, epsilon, episodes):
    agent = AgentClass(world, initial_state, actions, alpha, gamma, epsilon)
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
