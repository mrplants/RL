from .base import Environment, Memory, Policy, ObservationType, StateType, ActionType
from .utils import value_iteration

from typing import Collection, Any, Optional, Callable
from collections import deque
import numpy as np
import random

class DiscreteMemory(Memory[Any, Any]):
    """RL Memory that stores MDPs with discretizable states and actions.

    Attributes:
        discretize_observation: Function to discretize an ObservationType
        discretize_action: Function to discretize an ActionType
        inverse_discretize_action: Function to convert an int to an ActionType
        observations: List of initial observations
        actions: List of actions
        rewards: List of rewards
        next_observations: List of subsequent observations
    """
    def __init__(self,
        num_observations: int,
        num_actions: int,
        memory_limit: Optional[int] = None,
        discretize_observation: Callable[[ObservationType], int] = lambda x: x,
        discretize_action: Callable[[ActionType], int] = lambda x: x,
        inverse_discretize_action: Callable[[int], ActionType] = lambda x: x):
        """Initialize the memory instance variables.
        """
        num_observations += 1 # Add a terminal state
        self.discretize_observation = discretize_observation
        self.discretize_action = discretize_action
        self.inverse_discretize_action = inverse_discretize_action
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.observations = deque(maxlen=memory_limit)
        self.actions = deque(maxlen=memory_limit)
        self.rewards = deque(maxlen=memory_limit)
        self.next_observations = deque(maxlen=memory_limit)
        self._T = np.zeros((self.num_observations,
                            self.num_actions,
                            self.num_observations), dtype=int)
        self._R_sum = np.zeros(self.num_observations)

    def remember(self,
                 observation: ObservationType,
                 action: ActionType,
                 reward: float,
                 next_observation: ObservationType,
                 is_terminal: bool) -> None:
        """Stores the arg Markov tuple.
        """
        # Store the raw transition (deque maxlen handles popping if len>maxlen)
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
        # Cache the discretized step
        self._T[self.discretize_observation(observation),
                self.discretize_actions(actions),
                self.discretize_observation(observation)] += 1
        self._R_sum[self.discretize_observation(next_observation)] += reward
        # If the steps ended in a terminal state, remember that.
        # In a terminal state, a step in any direction would loop back to the
        # terminal state.
        if is_terminal:
            for a in range(self.num_actions):
                action = self.inverse_discretize_action(a)
                self.remember(next_observation,
                              action,
                              0,
                              self.num_observations-1,
                              False)
                self.remember(self.num_observations-1,
                              action,
                              0,
                              self.num_observations-1,
                              False)
    
    @property
    def T(self) -> np.ndarray:
        """ Convenience getter for transition_counts.
        """
        return self._T
    
    @property
    def P(self) -> np.ndarray:
        """ Convenience getter for transition probabilities.
        """
        T = self.T
        out = np.full(T.shape, 1/self.num_actions)
        return np.divide(T,
                         T.sum(axis=2, keepdims=True),
                         out = out,
                         where = T.sum(axis=2, keepdims=True) != 0)
    
    @property
    def R(self) -> np.ndarray:
        """ Convenience getter for state rewards
        """
        # Use average reward, dependent only on result state
        return np.divide(self._R_sum,
                         T.sum(axis=1).sum(axis=0),
                         out = np.zeros(self.num_observations),
                         where = T.sum(axis=1).sum(axis=0) != 0)

class DiscretePolicy(Policy[Any, Any]):
    """RL Policy for discretizable observations and actions

    Attributes:
        discretize_observation: Function to discretize an ObservationType
        discretize_action: Function to discretize an ActionType
        inverse_discretize_action: Function to convert an int to an ActionType
        _Q: (np.ndarray, dtype=float, shape=(|O|,|A|)) State-action value
            function.  
    """

    def __init__(self,
        num_observations: int,
        num_actions: int,
        discretize_observation: Callable[[ObservationType], int] = lambda x: x,
        discretize_action: Callable[[ActionType], int] = lambda x: x,
        inverse_discretize_action: Callable[[int], ActionType] = lambda x: x):
        """Initialize the memory instance variables.
        """
        num_observations += 1 # Add a terminal state
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.discretize_observation = discretize_observation
        self.discretize_action = discretize_action
        self.inverse_discretize_action = inverse_discretize_action
        self._Q = np.zeros((num_observations, num_actions))

    def __call__(self,
                 observation: ObservationType,
                 memory: Memory) -> ActionType:
        """Returns the chosen action for the current observable state.
        """
        # When there is a tie, need to choose randomly among the best actions.
        action_values = self.Q[self.discretize_observation(observation)]
        best_actions = np.argwhere(action_values==np.amax(action_values))
        best_actions = best_actions.flatten()
        action = np.random.choice(best_actions)
        return self.inverse_discretize_action(action)

    @property
    def Q(self) -> np.ndarray:
        """Convenience getter for the state-action value function.
        """
        return self._Q
    
    def train(self, memory: Memory) -> None:
        """Trains the policy.
        """
        _, self._Q = value_iteration(memory.P, memory.R)

class RandomDiscretePolicy(DiscretePolicy):
    """RL Policy that chooses randomly from a discrete set of actions
    """
    def __call__(self, *args):
        """Chooses an action randomly.
        """
        n_actions = self._Q.shape[1]
        return self.inverse_discretize_action(np.random.randint(n_actions))
