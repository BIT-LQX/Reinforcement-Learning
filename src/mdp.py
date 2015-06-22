"""
A generic MDP framework and some specific MDPs.

Matthew Alger
2015
"""

from itertools import product
import random

import numpy as np

class FiniteMDP(object):

    """
    A generic finite-state MDP.
    """

    def __init__(self, states, actions, transitions, discount, rewards,
                       start, goal):
        """
        states: Number of states. Each state will be represented by a unique
            integer.
        actions: Set of actions.
        transitions: Dictionary of matrices, mapping actions to matrices where
            the (i, j)th element of the matrix is the probability of transition
            from state i to state j after taking action a.
        discount: Discount factor. Float in [0, 1).
        rewards: Vector of rewards for each state.
        start: Start state.
        goal: Goal state.
        -> FiniteMDP
        """

        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.discount = discount
        self.rewards = rewards
        self.start = start
        self.goal = goal

    def step(self, state, action):
        """
        Returns the state an agent will be in after taking an action from some
        state.

        state: State agent is in.
        action: Action agent takes.
        -> State agent is in after taking the action.
        """

        probabilities = self.transitions[action][state, :]
        new_state = np.random.multinomial(1, probabilities).argmax()
        return new_state

    def reward(self, state):
        """
        Returns the reward given in the given state.

        state: State the agent is in.
        -> Reward.
        """

        return self.rewards[state]

    def perform_episode_offline(self, policy):
        """
        Send an agent through the MDP following a given policy, and get the
        trajectory and reward.

        policy: Function mapping states to actions.
        -> ([state], reward)
        """

        state = self.start
        trajectory = [state]
        total_reward = 0
        while state != self.goal:
            new_state = self.step(state, policy(state))
            reward = self.reward(new_state)

            trajectory.append(new_state)
            total_reward += reward

            state = new_state

        return trajectory, total_reward

    def perform_episode_online(self):
        """
        Send an agent through the MDP. Yields (state, reward) tuples and
        receives actions.

        -> (new state, reward)
        """

        state = self.start
        reward = 0
        while state != self.goal:
            action = yield (state, reward)
            new_state = self.step(state, action)
            reward = self.reward(new_state)

            state = new_state

        yield(state, reward)

        raise StopIteration("Reached goal.")


class WindyGridWorld(FiniteMDP):

    """
    The windy gridworld, where an agent must move around a grid and has a
    certain chance of an action moving them at random instead of in the desired
    direction.
    """

    def __init__(self, width, height, noise, discount, start, goal):
        """
        width: Width of grid.
        height: Height of grid.
        noise: Chance of moving at random in [0, 1).
        discount: Discount factor in [0, 1).
        start: Start position.
        goal: Goal position.
        -> WindyGridWorld
        """

        self.width = width
        self.height = height
        self.noise = noise

        states = width*height
        actions = {(1, 0), (0, 1), (-1, 0), (0, -1)} # Right, up, left, down

        start = self.position_to_state(start)
        goal = self.position_to_state(goal)

        p = {} # oldstate -> {action: newstate}
        for x, y in product(range(width), range(height)):
            n = {}
            for dx, dy in actions:
                nx = x + dx
                ny = y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    nx, ny = x, y
                n[dx, dy] = self.position_to_state((nx, ny))
            p[self.position_to_state((x, y))] = n

        transitions = {}
        for action in actions:
            matrix = np.zeros((states, states))
            for x, y in product(range(width), range(height)):
                state = self.position_to_state((x, y))
                intended_state = p[state][action]
                matrix[state, intended_state] = 1 - noise
                for unintended_state in p[state].values():
                    matrix[state, unintended_state] += noise/4
            transitions[action] = matrix

        rewards = np.zeros((states,))
        rewards[goal] = 1

        super().__init__(states, actions, transitions, discount, rewards,
                         start, goal)

    def position_to_state(self, position):
        """
        Get the state of a given position.

        position: (x, y)
        -> State integer
        """

        return position[0] + self.width * position[1]

    def state_to_position(self, state):
        """
        Get the position of a given state.

        state: State integer
        -> Position (x, y)
        """

        return state % self.width, state // self.width
