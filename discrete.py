from base import Environmnet, Memory, Policy

from typing import Collection
from collections import deque
import numpy as np
import random

class DiscreteMemory(Memory[Any, Any]):
    """RL Memory that stores MDPs with discretizable states and actions.

    Attributes:
        discretize_observation: Function to discretize an ObservationType
        discretize_action: Function to discretize an ActionType
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
        discretize_action: Callable[[ActionType], int] = lambda x: x):
        """Initialize the memory instance variables.
        """
        self.discretize_observation = discretize_observation
        self.discretize_action = discretize_action
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.observations = deque(maxlen=memory_limit)
        self.actions = deque(maxlen=memory_limit)
        self.rewards = deque(maxlen=memory_limit)
        self.next_observations = deque(maxlen=memory_limit)

    def remember(self,
                 observation: ObservationType,
                 action: ActionType,
                 reward: float,
                 next_observation: ObservationType) -> None:
        """Stores the arg Markov tuple.
        """
        # Store the raw transition (deque maxlen handles popping if len>maxlen)
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
    
    @property
    def T(self) -> np.ndarray:
        """ Convenience getter for transition_counts.
        """
        T = np.zeros((self.num_observations,
                      self.num_actions,
                      self.num_observations), dtype=int)
        for o, a, o_prime in zip(self.observations,
                                 self.actions,
                                 self.next_observations):
            T[self.discretize_observation(o),
              self.discretize_action(a),
              self.discretize_observation(o_prime)] += 1
        return T
    
    @property
    def P(self) -> np.ndarray:
        """ Convenience getter for transition probabilities.
        """
        T = self.T
        return T / T.sum(axis=2, keepdims=True)
    
    @property
    def R(self) -> np.ndarray:
        """ Convenience getter for state rewards
        """
        # Use average reward, dependent only on result state
        R = np.zeros(self.num_observations)
        T = self.T
        for r, o_prime in zip(self.rewards, self.next_observations):
            R[self.discretize_observation] += r
        return R / T.sum(axis=1).sum(axis=0)

class RandomDiscretePolicy(Policy[Any, Any]):
    """RL Policy that chooses randomly from a discrete set of actions

    Attributes:
        all_actions: List of all actions
    """
    def __init__(self, all_actions: Collection[ActionType]):
        self.all_actions = all_actions
    
    def __call__(self, observation: ObservationType) -> ActionType:
        return random.choice(tuple(self.all_actions))
    
    def train(self, memory: Memory) -> None:
        """Trains the policy.
        """
        pass

class DiscretePolicy(Policy[Any, Any]):
    """RL Policy for discretizable observations and actions

    Attributes:
        discretize_observation: Function to discretize an ObservationType
        discretize_action: Function to discretize an ActionType
        inverse_discretize_action: Function to convert an int to an ActionType
        Q: (np.ndarray, dtype=float, shape=(|O|,|A|)) State-action value
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
        self.discretize_observation = discretize_observation
        self.discretize_action = discretize_action
        self.inverse_discretize_action = inverse_discretize_action
        self.Q = np.random.normal((num_observations, num_actions))

    def __call__(self, observation: ObservationType) -> ActionType:
        """Returns the chosen action for the current observable state.
        """
        action = np.argmax(Q[self.discretize_observation(observation)])
        return self.inverse_discretize_action(action)
    
    def train(self, memory: Memory) -> None:
        """Trains the policy.
        """
        _, self.Q = value_iteration(memory.P, memory.R)
