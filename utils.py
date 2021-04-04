from .base import Environment, Policy, Memory
from typing import Optional, Callable

def run_episode(
    environment: Environment,
    policy: Policy,
    memory: Memory,
    max_steps: Optional[None] = None,
    step_callback: Optional[Callable[[ObservationType,
                                      int,
                                      Policy,
                                      Memory], None]]=None) -> int:
    """Runs one episode.

    Traverses the environment with the policy, saving the results to memory.

    Args:
        policy:  A policy to traverse the environment.
        environment:  An environment to traverse.
        memory:  The memory to save Markov tuples.
        max_steps: Optional arg for indicating the episode's maximum length.
        step_callback: Optional arg for executing custom code after each step in
            the episode.  This can be used to retrain the policy or visualize
            progress along the episode. 
    
    Returns:
        The number of steps taken in the episode.
    """
    state = environment.initial_state
    observation = environment.state_to_observation(state)
    step_number = 0
    while (not environment.is_terminal(state) and
           (max_steps is None or step_number < max_steps)):
        # Take one step in the episode
        action = policy(observation)
        next_state, reward = environment.step(state, action)
        next_observation = environment.state_to_observation(next_state)
        # Remember the step and perform the callback
        memory.remember(observation,
                        action,
                        reward,
                        next_observation,
                        environment.is_terminal(next_state))
        if step_callback: step_callback(next_observation,
                                        step_number,
                                        policy,
                                        memory)
        # Setup for next step
        state = next_state
        observation = next_observation
        step_number += 1
    return step_number
