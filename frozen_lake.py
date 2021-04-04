from .base import Environment, StateType, ObservationType, ActionType
import numpy as np
from typing import Tuple
from enum import Enum

class Action(Enum):
    NORTH = 'NORTH'
    EAST = 'EAST'
    SOUTH = 'SOUTH'
    WEST = 'WEST'

ALL_ACTIONS = [a for a in Action]

class FrozenLake(Environment[str, Action, int]):
    """Dynamic FrozenLake environment.

    The FrozenLake environment is one where the entire world is a grid.  Each
    grid cell can have two states: ice or hole.  A whole cell is a terminal
    state.  There is at least one hole cell which contains a +1 reward.  The
    agent may take steps in any of the cardinal directions.  The resulting
    location of the agent from that step will be one cell in one of the cardinal
    directions, indicated by a probability distribution over the directions
    conditioned on the chosen direction.  For example, by choosing North the
    agent may instead slip East or West (depending on the probability
    distribution).

    ## STATE
    The state of this environment will be a string arranged in a grid, with '\n'
    characters at the end of each row.  An example might look like this when
    printed to the terminal (new lines are hidden):
        
        SFFFFFFF
        FFFFFFFF
        FFFHFFFF
        FFFFFHFF
        FFFHFFFF
        FHHFFFHF
        FHFFHFHF
        FFFHFFFG

    The grid is arranged in row-major format.  This specific grid and its
    nomenclature are borrowed from OpenAI gym at https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py

    S:  This is the current position of the agent
    F:  This is a frozen surface (ice)
    H:  This is a hole in the ice
    G:  This is a +1 reward hole

    ## OBSERVATION
    The agent will be able to observe only the flattened (row-first) index of
    their location in the grid.

    Attributes:
        initial_world_state: String describing the initial world state.
        slip_probs: np.ndarray indicating the probability distribution over the
            true steps taken, conditioned on the step chosen
    """

    no_slip_probs = np.identity(len(ALL_ACTIONS))
    yes_slip_probs = np.array([[0.50, 0.25, 0.00, 0.25],
                               [0.25, 0.50, 0.25, 0.00],
                               [0.00, 0.25, 0.50, 0.25],
                               [0.25, 0.00, 0.25, 0.50]])

    lake4x4 = (
        'SFFF\n'
        'FHFH\n'
        'FFFH\n'
        'HFFG\n'
    )

    lake8x8 = (
        'SFFFFFFF\n'
        'FFFFFFFF\n'
        'FFFHFFFF\n'
        'FFFFFHFF\n'
        'FFFHFFFF\n'
        'FHHFFFHF\n'
        'FHFFHFHF\n'
        'FFFHFFFG\n'
    )

    def __init__(self, initial_world_state: str,
                 slip_probs: np.ndarray = no_slip_probs):
        self.initial_world_state = initial_world_state
        self.slip_probs = slip_probs
    
    @property
    def initial_state(self) -> StateType:
        return self.initial_world_state
    
    def is_terminal(self, state: StateType) -> bool:
        """Indicates whether or not the arg state is terminal.
        """
        index = state.find('S')
        return self.initial_world_state[index] == 'H'

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
        position = state.replace('\n','').find('S')
        num_cols = state.find('\n')
        num_rows = len(state.replace('\n','')) // num_cols
        row = position // num_cols
        col = position % num_cols

        slip_probs = self.slip_probs[ALL_ACTIONS.index(action)]
        true_action = ALL_ACTIONS[np.random.choice(np.arange(len(ALL_ACTIONS)),
                                                   p=slip_probs)]

        if true_action == Action.NORTH:
            row -= 1 if row > 0 else 0
        elif true_action == Action.EAST:
            col += 1 if col < num_cols-1 else 0
        elif true_action == Action.SOUTH:
            row += 1 if row < num_rows-1 else 0
        elif true_action == Action.WEST:
            col -= 1 if col > 0 else 0
        
        new_position = row * (num_cols + 1) + col
        new_state = state[:state.find('S')] + 'F' + state[state.find('S')+1:]
        new_state = new_state[:new_position] + 'S' + new_state[new_position+1:]

        if self.initial_world_state[new_position] == 'G':
            reward = 1
        else:
            reward = 0
        return new_state, reward

    def state_to_observation(self, state: StateType) -> ObservationType:
        """Converts a state to an observation.

        This allows the environment class to remain 
        Raises:
            NotImplementedError: This is an abstract base class.
        """
        return state.replace('\n','').find('S')
    
    @staticmethod
    def discretize_action(action: ActionType) -> int:
        return ALL_ACTIONS.index(action)
    
    @staticmethod
    def inverse_discretize_action(index: int) -> ActionType:
        return ALL_ACTIONS[index]
    
    @property
    def num_observations(self) -> int:
        return len(self.initial_world_state.replace('\n', ''))
    
    @property
    def num_actions(self) -> int:
        return len(ALL_ACTIONS)

class FrozenLake4x4(FrozenLake):
    def __init__(self, slippery: bool = False):
        super().__init__(
            FrozenLake.lake4x4,
            slip_probs = FrozenLake.yes_slip_probs if slippery else self.no_slip_probs)

class FrozenLake8x8(FrozenLake):
    def __init__(self, slippery: bool = False):
        super().__init__(
            FrozenLake.lake8x8,
            slip_probs = FrozenLake.yes_slip_probs if slippery else self.no_slip_probs)