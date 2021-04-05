from RL.base import Environment, StateType, ObservationType, ActionType
import numpy as np
from typing import Tuple
from enum import Enum

class Action(Enum):
    LEFT = -1
    NEUTRAL = 0
    RIGHT = 1

ALL_ACTIONS = [a for a in Action]
K_VELOCITY = 0
K_POSITION = 1

class MountainCar(Environment[Tuple[float, float],
                              Action,
                              Tuple[float, float]]):
    """MountainCar Environment.

    The Mountain Car environment is one where a car is placed in the valley
    between two sinusoidal shaped slopes.  The car is underpowered and cannot
    drive up either slope without some initial velocity.  The goal is to get to
    the top of the right slope. The top of the left slope has an inelastic wall.

    ## STATE & OBSERVATION
    The state of this environment is the same as its observation space.  It is
    simply a tuple containing the car's velocity [-0.07, 0.07] and position
    [-1.2, 0.6]

    A description of the environment and its history can be found at https://en.wikipedia.org/wiki/Mountain_car_problem.
    """
    num_actions = len(ALL_ACTIONS)

    @property
    def initial_state(self) -> StateType:
        return (0.0, np.random.uniform(-0.6, -0.4))
    
    def is_terminal(self, state: StateType) -> bool:
        position = state[K_POSITION]
        return position >= 0.6
    
    def step(self, state: StateType, action: ActionType) -> (StateType, float):
        velocity = state[K_VELOCITY]
        position = state[K_POSITION]
        velocity = velocity + action.value * 0.001 + np.cos(3 * position) * -0.0025
        position = position + velocity
        if position < -1.2:
            position = -1.2
            velocity = max(0.0, velocity)
        velocity = np.clip(velocity, -0.07, 0.07)
        new_state = (velocity, position)
        reward = 1 if self.is_terminal(new_state) else 0
        return new_state, reward
    
    def state_to_observation(self, state: StateType) -> ObservationType:
        return state
    
    @staticmethod
    def discretize_action(action: ActionType) -> int:
        return ALL_ACTIONS.index(action)
    
    @staticmethod
    def inverse_discretize_action(index: int) -> ActionType:
        return ALL_ACTIONS[index]
    
    @property
    def num_actions(self) -> int:
        return len(ALL_ACTIONS)