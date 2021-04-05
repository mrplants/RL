import numpy as np
from typing import Callable, Collection
from RL.base import Memory, ObservationType, ActionType
from RL.discrete import DiscretePolicy

class MCTSExplorePolicy(DiscretePolicy):
    """Policy that searches based on optimizing under uncertainty.

    This policy runs a Monte Carlo Tree Search using the following exploration
    rewards:

    (s, a, s', i, n), where s -a-> s' for the ith time on step n from the root.

    explore_weight * np.sqrt(np.log(T[s].sum()) / T[s,a].sum())

    Transitions that we visit less are weighted higher.  This is the formula for
    UCT described in https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=E125C28A59928B81F1A86886B44678AF?doi=10.1.1.102.1296&rep=rep1&type=pdf

    Attributes:
        search_depth:  The maximum number of steps to search.
        num_rollouts:  The number of rollouts to perform for each next action.
        explore_weight:  The weight applied to the search term.
    """
    def __init__(self,
                 search_depth: int = 10,
                 num_rollouts: int = 1,
                 explore_weight = 1,
                 *args, **kwargs):
        """Initializes class attributes.
        """
        super().__init__(*args, **kwargs)
        self.search_depth = search_depth
        self.num_rollouts = num_rollouts
        self.explore_weight = explore_weight
        self._Q_search = np.zeros_like(self._Q)
    
    def __call__(self,
                 observation: ObservationType,
                 memory: Memory) -> ActionType:
        """Performs exploration rollouts.
        """
        # When there is a tie, need to choose randomly among the best actions.
        action_values = (self.rollout(self.discretize_observation(observation),
                                      memory.T,
                                      memory.P) +
                         self.Q[self.discretize_observation(observation)])
        best_actions = np.argwhere(action_values==np.amax(action_values))
        best_actions = best_actions.flatten()
        action = np.random.choice(best_actions)
        return self.inverse_discretize_action(action)
        
    def rollout(self,
                observation: ObservationType,
                T: np.ndarray,
                P: np.ndarray) -> np.ndarray:
        """Performs a Monte Carlo Tree Search rollout with UCT as rewards.

        Args:
            observation:  The root observation from which to perform rollouts
            T:  The transition count matrix
            P:  The transition probability matrix
        
        Returns:
            Numpy array of exploration values for each action.
        """
        action_values = np.zeros(self.num_actions)
        for _ in range(self.num_rollouts):
            for action in range(self.num_actions):
                T_rollout = T.copy() + 1
                current_observation = observation
                next_action = action
                for _ in range(self.search_depth):
                    next_observation = np.random.choice(np.arange(self.num_observations),
                                                        p = P[current_observation, next_action])
                    action_values[action] += self.get_explore_value(current_observation,
                                                    next_action,
                                                    next_observation,
                                                    T_rollout)
                    T_rollout[current_observation, next_action, next_observation] += 1
                    current_observation = next_observation
                    next_action = np.random.randint(self.num_actions)
        return action_values / self.search_depth / self.num_rollouts
 
    def get_explore_value(self,
                          state: int,
                          action: int,
                          next_state: int,
                          T: np.ndarray) -> float:
        """Returns the search value for performing this transition.

        state, action, next_state:  The transition to analyze.
        T: The transition counts (including simulated counts).
        """
        return (self.explore_weight *
                np.sqrt(np.log(T[state].sum()) / T[state, action].sum()))
