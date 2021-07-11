from RL.base import Memory, ObservationType, StateType, ActionType
from typing import Any, Sequence, Generic, Tuple, Optional, Callable
import abc
import numpy as np
import scipy.special

class TransitionModel(Generic[StateType], abc.ABC):
    """ Transition Model base class.
    """
    @abc.abstractmethod
    def __call__(self, s: StateType, a: int) -> StateType:
        """ Predict a Markov Decision Process transition.

        Args:
            s: The current state.
            a: The index of the current action.

        Returns:
            s: A point in state space where the transition model predicts the
                agent will arrive after taking action a in state s.
        """
        pass
    
    @abc.abstractmethod
    def train(self, x: Sequence[Tuple[StateType, int]], y: Sequence[StateType]) -> None:
        """ Trains the model with sample input/output data

        Args:
            x: Input data to the model. A sequence of tuples containing the initial
                state and action index.
            
            y: Output data for the model. A sequence of resulting states when the
                respective state/action pair was executed.
        """
        pass

    @abc.abstractmethod
    def initialize(self, seed: Optional[float]) -> None:
        """ Resets the model to some initial state.  May be stochastic.

        Args:
            seed: The random seed for maintaining consistency across stochastic
                initialization (if necessary).
        """
        pass

class LeastSquaresTransitionModel(TransitionModel[np.ndarray]):
    """ Predicts MDP transitions using a Least Squares model.

    Attributes:
        n_actions: The number of actions possible in the environment.
        m: The weights of the least squares model.
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
    
    def initialize(self, seed: Optional[float] = None) -> None:
        """ No initialization is required for the least squares model.

        Least squares model minimization is convex.  It always convergs to the
        same optimal weights and therefore requires no initialization.
        """
        pass

    def train(self, x: Sequence[Tuple[StateType, int]], y: Sequence[StateType]) -> None:
        """ Perform least squares optimization using the provided inputs/outputs

        The model input is structure like this:
        [one-hot-action, state-representation, 1]
        """
        d_state_space = x[0][0].shape[0]
        n_samples = len(x)
        in_states = list(zip(*x))[0]
        actions = np.array(list(zip(*x))[1], dtype=int)

        ins = np.zeros((n_samples, d_state_space + self.n_actions + 1))
        ins[np.arange(n_samples),actions] = 1
        ins[:,self.n_actions:-1] = in_states
        ins[:,-1] = 1
        outs = np.array(y)

        self.m = np.linalg.lstsq(ins, outs, rcond=None)[0]

    def __call__(self, s: StateType, a: int) -> StateType:
        """ Predict the output state using the least squares weights.

        Must train beforehand.
        """
        d_state_space = s.shape[0]
        x = np.zeros(self.n_actions + d_state_space + 1)
        x[a] = 1
        x[self.n_actions:-1] = s
        x[-1] = 1
        return x @ self.m

# TODO: use the more complex (but optimal) algorithm for reservoir sampling.

class PredictiveTransitionMemory(Memory[Any, Any]):
    """RL memory that can predict transition probabilities.

    Attributes:
        num_state_centroids: This is the number of centroids in the state space,
            used for exploration rollouts and value iteration to predict future
            rewards.
        num_actions: This is the number of different actions the agent can take.
        o2s: This function transforms an observation into the state space.
        c2s: This function returns the state representation of the arg centroid.
        s2c: This function calculates and returns a probability distribution
            indicating the probability that the arg state belongs to each
            centroid. 
        T_model: This is used to model and generalize the
            (state, action) -> next_state transition.
        reservoir_max: Max number of transitions to remember for each
            (state, action) pair.  Uses reservoir sampling over the stream of
            observed transitions.
        discretize_action: Function to discretize an ActionType
        inverse_discretize_action: Function to convert an int to an ActionType
        recalculate_P: boolean indicating the cache status of P (stale or fresh)
        _R_sum: Sum of the rewards encountered when transitioning to each state.
        _K_sum: Number of times each state was the destination during a terminal
            transition.
        _T: Count of all transitions observed.
        reservoir_memory: Bins of recorded transitions associated with all
            (centroid, action) pairs.
        reservoir_memory_count: Count of the stream length for all
            (centroid, action) pairs. Required to implement reservoir sampling.
    """

    def __init__(self,
        num_state_centroids: int,
        num_actions: int,
        o2s: Callable[[ObservationType], StateType],
        c2s: Callable[[int], StateType],
        s2c: Callable[[StateType], np.ndarray],
        T_model: TransitionModel[StateType],
        reservoir_max: int = 1,
        discretize_action: Callable[[ActionType], int] = lambda x: x,
        inverse_discretize_action: Callable[[int], ActionType] = lambda x: x):
        """ Initialize instance variables and basic MDP setup.
        """

        # Declare arg instance variables
        self.num_state_centroids = num_state_centroids
        self.num_actions = num_actions
        self.o2s = o2s
        self.s2c = s2c
        self.c2s = c2s
        self.T_model = T_model
        self.discretize_action = discretize_action
        self.inverse_discretize_action = inverse_discretize_action
        self.reservoir_max = reservoir_max

        self.recalculate_P = True # Used to cache P
        # Instance variables for tracking the MDP
        self._R_sum = np.zeros(self.num_state_centroids)
        self._K_sum = np.zeros(self.num_state_centroids)
        self._T = np.zeros((self.num_state_centroids,
                            self.num_actions,
                            self.num_state_centroids))
        all_centroids, all_actions = np.meshgrid(np.arange(self.num_state_centroids),
                                                 np.arange(self.num_actions))
        all_centroids = all_centroids.flatten()
        all_actions = all_actions.flatten()
        self.reservoir_memory = {centroid_action: [] for centroid_action in zip(all_centroids, all_actions)}
        self.reservoir_memory_count = {centroid_action: 0 for centroid_action in zip(all_centroids, all_actions)}

    def remember(self,
                 observation: ObservationType,
                 action: ActionType,
                 reward: float,
                 next_observation: ObservationType,
                 is_terminal:bool) -> None:
        """Stores the arg Markov tuple.

        Args:
            (o, a, r, o'): the Markov Decision tuple (renamed for readability)
            is_terminal: indicates whether or not the step ended in a terminal
                state.
        """
        self.recalculate_P = True
        # Transform transition observations
        c = np.argmax(self.s2c(self.o2s(observation)))
        c_prime = np.argmax(self.s2c(self.o2s(next_observation)))
        a = self.discretize_action(action)
        # Update counters
        self._R_sum[c_prime] += reward
        self._K_sum[c_prime] += is_terminal
        self._T[c, a, c_prime] += 1
        # Perform reservoir sampling
        self.reservoir_memory_count[(c, a)] += 1
        reservoir_count = self.reservoir_memory_count[(c, a)]
        if reservoir_count <= self.reservoir_max:
            # The bin isn't full, so remember the transition
            self.reservoir_memory[(c, a)].append((observation,
                                                  action,
                                                  next_observation))
        else:
            # The bin is full, so perform reservoir sampling
            sample = np.random.randint(reservoir_count)
            if sample < self.reservoir_max:
                self.reservoir_memory[(c, a)][sample] = (observation,
                                                         action,
                                                         next_observation)

    @property
    def T(self) -> np.ndarray:
        """ Convenience getter for transition_counts.
        """
        return self._T
    
    @property
    def P(self) -> np.ndarray:
        """ Convenience getter for transition probabilities.

        Train the transition model and then calculate the transition
        probabilities.  Cache the result, since training is expensive.
        """
        if self.recalculate_P:
            # Train the transition model using the recorded transitions.
            all_transitions = [tx for bin in self.reservoir_memory.values() for tx in bin]
            if not all_transitions:
                return np.zeros_like(self.T) + 1 / self.num_state_centroids
            s, a, s_prime = zip(*[(self.o2s(o),
                                self.discretize_action(a),
                                self.o2s(o_prime)) for o, a, o_prime in all_transitions])
            self.T_model.initialize()
            self.T_model.train(list(zip(s, a)), s_prime)
            # Create a transition table using the transition model
            self._P = np.zeros_like(self.T)
            all_c, all_a = zip(*self.reservoir_memory.keys())
            self._P[all_c, all_a] = [self.s2c(s_prime) for s_prime in [self.T_model(self.c2s(c), a) for c, a in zip(all_c, all_a)]]
        return self._P

    @property
    def K(self) -> np.ndarray:
        """ Convenience getter for termination probabilities
        """
        # Use average number of times the state was a termination state
        return np.divide(self._K_sum,
                         self.T.sum(axis=1).sum(axis=0),
                         out = np.zeros(self.num_observations),
                         where = self.T.sum(axis=1).sum(axis=0) != 0)
    
    @property
    def R(self) -> np.ndarray:
        """ Convenience getter for state rewards
        """
        # Use average reward, dependent only on result state
        return np.divide(self._R_sum,
                         self.T.sum(axis=1).sum(axis=0),
                         out = np.zeros(self.num_observations),
                         where = self.T.sum(axis=1).sum(axis=0) != 0)
    
    @property
    def num_observations(self) -> int:
        return self.num_state_centroids