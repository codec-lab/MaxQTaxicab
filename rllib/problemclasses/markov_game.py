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
ActionSpace = Sequence[Action]
JointAction = Sequence[Action]

class MarkovGame(Generic[State, Action]):
    discount_rate : float
    joint_action_space : Sequence[ActionSpace]
    state_space : Sequence[State]
    state_action_space : Sequence[tuple[State, Action]]
    def initial_state_dist(self) -> Distribution[State]:
        raise NotImplementedError
    def joint_actions(self, s : State) -> Sequence[JointAction]: 
        raise NotImplementedError
    def next_state_dist(self, s : State, ja : JointAction) -> Distribution[State]:
        raise NotImplementedError
    def next_state_sample(self, s : State, ja : JointAction, rng : Random = random) -> State:
        raise NotImplementedError
    def joint_reward(self, s : State, ja : JointAction, ns : State) -> Sequence[float]:
        raise NotImplementedError
    def is_absorbing(self, s : State) -> bool:
        raise NotImplementedError

class JointPolicy(Generic[State, Action]):
    def action_dist(self, s : State) -> Distribution[JointAction]:
        raise NotImplementedError
    def action_sample(self, s : State, rng : Random = random) -> JointAction:
        raise NotImplementedError
