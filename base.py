import abc
from typing import TypeVar, Generic, Optional, Callable, Any

StateType = TypeVar('StateType')
ActionType = TypeVar('ActionType')
ObservationType = TypeVar('ObservationType')

class Environment(Generic[StateType, ActionType, ObservationType], abc.ABC):
    """RL environment abstract base class.

    Environment's must maintain Markov state separate from environment logic
    (instances of this class).  This does not include static instance variables,
    such as grid size (for grid world), etc.  Instance variables should not
    change in difference states of an Environment.  If they do, store them in
    the state.
    """
    @property
    @abc.abstractmethod
    def initial_state(self) -> StateType:
        """Returns the first state in an episode of this environment
        """
        pass
    
    @abc.abstractmethod
    def is_terminal(self, state: StateType) -> bool:
        """Indicates whether or not the arg state is terminal.
        """
        pass

    @abc.abstractmethod
    def step(self, state: StateType, action: ActionType) -> (StateType, float):
        """ Takes a single step in the environment.

        Note that the reward must be independent of the action.  Rewards only
        depend on the resulting state.  The state must not be a terminal state.

        Args:
            state: The environment state prior to the step.
            action: The action taken during the step.
        
        Returns:
            The state resulting from the step.
            The reward for entering the next state. (independent of the action)
        """
        pass

    @abc.abstractmethod
    def state_to_observation(self, state: StateType) -> ObservationType:
        """Converts a state to an observation.
        """
        pass

class Memory(Generic[ObservationType, ActionType], abc.ABC):
    """RL Memory abstract base class.
    """

    @abc.abstractmethod
    def remember(observation: ObservationType,
                 action: ActionType,
                 reward: float,
                 next_observation: ObservationType) -> None:
        """Stores the arg Markov tuple.
        """
        pass
        
class Policy(Generic[ObservationType, ActionType], abc.ABC):
    """RL Policy abstract base class.
    """

    @abc.abstractmethod
    def __call__(self, observation: ObservationType) -> ActionType:
        """Returns the action for the current observable state:
        """
        pass
    
    @abc.abstractmethod
    def train(self, memory: Memory) -> None:
        """Trains the policy.
        """
        pass
