"""
Implements agents for finding optimal policies for MDPs.

Matthew Alger
2015
"""

from itertools import product
import random

import numpy as np
import matplotlib.pyplot as plt

def show_grid_policy(policy, states):
    """
    Display a grid policy.

    policy: Function mapping (x, y) state pairs to (dx, dy) action pairs.
    states: Set of (x, y) pairs.
    """

    actions = np.array([policy(s) for s in states])
    states = np.array(states)
    plt.quiver(states[:, 0], states[:, 1], actions[:, 0], actions[:, 1])
    plt.axis((min(states)[0], max(states)[0],
        min(states, key=lambda s: s[1])[1], max(states, key=lambda s: s[1])[1]))
    plt.show()

def sarsa(mdp, learning_rate, epsilon, episodes):
    """
    Finds a policy for the given MDP using SARSA.

    mdp: An MDP.
    learning_rate: Learning rate in range [0, 1).
    epsilon: Chance of making a random move in range [0, 1).
    episodes: How many episodes to train for.
    -> Policy function mapping states to actions.
    """

    q = {state_action: 0
         for state_action in product(range(mdp.states), mdp.actions)}

    episode_lengths = []

    for i in range(episodes):
        episode = mdp.perform_episode_online()
        length = 0

        state, reward = next(episode)
        action = max(mdp.actions, key=lambda a: q[state, a])
        while True:
            try:
                length += 1

                new_state, reward = episode.send(action)
                new_action = max(mdp.actions, key=lambda a: q[new_state, a])
                if random.random() < epsilon:
                    new_action = random.choice(list(mdp.actions))

                q[state, action] += learning_rate * (reward +
                    mdp.discount * q[new_state, new_action] - q[state, action])

                state = new_state
                action = new_action

            except StopIteration:
                break

        episode_lengths.append(length)

    def policy(state):
        return max(mdp.actions, key=lambda a: q[state, a])

    return policy
