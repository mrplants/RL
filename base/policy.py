import abc
from typing import TypeVar, Generic, Optional, Callable, Any

StateType = TypeVar('StateType')
ActionType = TypeVar('ActionType')
ObservationType = TypeVar('ObservationType')
        
class Policy(Generic[ObservationType, ActionType], abc.ABC):
    """RL Policy abstract base class.
    """

    @abc.abstractmethod
    def __call__(self,
                 observation: ObservationType,
                 memory: Memory) -> ActionType:
        """Returns the action for the current observable state:
        """
        pass
    
    @abc.abstractmethod
    def train(self, memory: Memory) -> None:
        """Trains the policy.
        """
        pass

