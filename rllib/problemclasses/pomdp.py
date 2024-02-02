from typing import Sequence, Hashable, TypeVar, Generic, Container
import random
import dataclasses
from random import Random

from ..distributions import Distribution, DiscreteDistribution, Uniform, Gaussian

State = TypeVar('State', bound=Hashable)
Action = TypeVar('Action', bound=Hashable)
Observation = TypeVar('Observation', bound=Hashable)
AgentState = TypeVar('AgentState', bound=Hashable)

class POMDP(Generic[State, Action, Observation]):
    discount_rate : float
    action_space : Sequence[Action]
    state_space : Sequence[State]
    observation_space : Sequence[Observation]
    def initial_state_dist(self) -> Distribution:
        raise NotImplementedError
    def initial_state_sample(self) -> Distribution:
        return self.initial_state_dist().sample()
    def next_state_dist(self, s : State, a : Action) -> Distribution:
        raise NotImplementedError
    def next_state_sample(self, s : State, a : Action, rng : Random = random) -> State:
        return self.next_state_dist(s, a).sample(rng)
    def reward(self, s : State, a : Action) -> float:
        raise NotImplementedError
    def is_absorbing(self, s : State) -> bool:
        raise NotImplementedError
    def observation_dist(self, a : Action, ns : State) -> Distribution:
        raise NotImplementedError
    def observation_sample(self, a : Action, ns : State, rng : Random = random) -> Observation:
        return self.observation_dist(a, ns).sample(rng)
    def observations(self, a : Action) -> Sequence[Observation]:
        observations = set()
        for ns in self.state_space:
            observations.update(self.observation_dist(a, ns).support)
        return tuple(sorted(observations))

class POMDPPolicy(Generic[AgentState, Action, Observation]):
    def action_dist(self) -> Distribution:
        raise NotImplementedError
    def action_sample(self, rng : Random = random) -> Action:
        raise NotImplementedError
    def agent_state(self) -> AgentState:
        raise NotImplementedError
    def update(self, o : State):
        raise NotImplementedError
    def end_episode(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

@dataclasses.dataclass(frozen=True, order=True)
class BeliefState:
    states : tuple
    probs : tuple

    def __post_init__(self):
        assert len(self.states) == len(self.probs), 'states and probs must have the same length'
        assert abs(sum(self.probs) - 1) < 1e-6, 'probs must be a probability distribution'

    def fromdict(d : dict) -> 'BeliefState':
        states, probs = zip(*sorted(d.items()))
        return BeliefState(states, probs)
    
    def items(self):
        return zip(self.states, self.probs)
    
    def __getitem__(self, state):
        if state in self.states:
            return self.probs[self.states.index(state)]
        else:
            return 0
