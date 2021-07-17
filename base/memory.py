import abc
from typing import TypeVar, Generic, Optional, Callable, Any

StateType = TypeVar('StateType')
ActionType = TypeVar('ActionType')
ObservationType = TypeVar('ObservationType')

class Memory(Generic[ObservationType, ActionType], abc.ABC):
    """RL Memory abstract base class.
    """

    @abc.abstractmethod
    def remember(observation: ObservationType,
                 action: ActionType,
                 reward: float,
                 next_observation: ObservationType,
                 is_terminal:bool) -> None:
        """Stores the arg Markov tuple.

        Args:
            (o, a, r, o'): the Markov Decision tuple
            is_terminal: indicates whether or not the step ended in a terminal
                state.
        """
        pass