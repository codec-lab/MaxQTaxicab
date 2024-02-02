from typing import Sequence, Hashable, TypeVar, Generic, Container
from dataclasses import dataclass
import random
from random import Random

from ..distributions import Distribution, DiscreteDistribution, Uniform, Gaussian

State = TypeVar('State', bound=Hashable)
Action = TypeVar('Action', bound=Hashable)
@dataclass(frozen=True, eq=True, repr=True)
class Step:
    state : State
    action : Action
    reward : float
    next_state : State
Trajectory = Sequence[Step]

class MarkovDecisionProcess(Generic[State, Action]):
    discount_rate : float
    action_space : Sequence[Action]
    state_space : Sequence[State]
    state_action_space : Sequence[tuple[State, Action]]
    def initial_state_dist(self) -> Distribution[State]:
        raise NotImplementedError
    def actions(self, s : State) -> Sequence[Action]: 
        raise NotImplementedError
    def next_state_dist(self, s : State, a : Action) -> Distribution[State]:
        raise NotImplementedError
    def next_state_sample(self, s : State, a : Action, rng : Random = random) -> State:
        raise NotImplementedError
    def reward(self, s : State, a : Action, ns : State) -> float:
        raise NotImplementedError
    def is_absorbing(self, s : State) -> bool:
        raise NotImplementedError

class MDPPolicy(Generic[State, Action]):
    discount_rate : float
    def action_dist(self, s : State) -> Distribution[Action]:
        raise NotImplementedError
    def action_sample(self, s : State, rng : Random = random) -> Action:
        raise NotImplementedError
    def state_value(self, s : State) -> float:
        raise NotImplementedError
    def update(self, s : State, a : Action, r : float, ns : State) -> None:
        raise NotImplementedError
    def end_episode(self) -> None:
        raise NotImplementedError
    def reset(self) -> None:
        raise NotImplementedError
