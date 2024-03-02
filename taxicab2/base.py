from gorgo import flip, infer, draw_from, Categorical
from itertools import product
from dataclasses import dataclass
from typing import Tuple, Set, List, Hashable, Generic, TypeVar, Optional, Protocol, Dict

from abc import ABC, abstractmethod
from functools import cached_property

# Base classes for MDPs and policies

class HasDone(Hashable, Protocol):
    done: bool
State = TypeVar("State", bound=HasDone)
Action = TypeVar("Action", bound=Hashable)
Reward = TypeVar("Reward", bound=float)

TransitionTable = Dict[Tuple[State, Action], Categorical[State]]
RewardTable = Dict[Tuple[State, Action], float]

class Task(ABC, Generic[State, Action]):
    @abstractmethod
    def init_state_dist(self) -> Categorical[State]:
        pass

    @abstractmethod
    def next_state_reward(self, state: State, action: Action) -> Categorical[Tuple[State, Reward]]:
        pass

    @property
    @abstractmethod
    def actions(self) -> List[Action]:
        pass

    def reachable_states(self, max_states=1000) -> Set[State]:
        frontier: List[State] = list(self.init_state_dist().support)
        states: Set[State] = set()
        while frontier:
            if len(states) >= max_states:
                print("Warning: reached maximum number of states enumerated")
                break
            s = frontier.pop()
            states.add(s)
            if s.done:
                continue
            for a in self.actions:
                ns_r_dist = self.next_state_reward(s, a)
                for ns, _ in ns_r_dist.support:
                    if ns not in states:
                        frontier.append(ns)
        return states
    
    @cached_property
    def states(self) -> Tuple[State, ...]:
        return tuple(self.reachable_states())
    
    @cached_property
    def _transition_reward_tables(self) -> Tuple[TransitionTable, RewardTable]:
        transition_probabilities = {}
        rewards = {}
        for s in self.states:
            for a in self.actions:
                ns_r_dist = self.next_state_reward(s, a)
                ns_dist = ns_r_dist.marginalize(lambda ns_r: ns_r[0])
                transition_probabilities[(s, a)] = ns_dist
                rewards[(s, a)] = ns_r_dist.expected_value(lambda ns_r: ns_r[1])
        return transition_probabilities, rewards
    
    @property
    def transition_table(self) -> TransitionTable:
        return self._transition_reward_tables[0]
    
    @property
    def reward_table(self) -> RewardTable:
        return self._transition_reward_tables[1]

class Policy(ABC, Generic[State, Action]):
    @abstractmethod
    def action_dist(self, state: State) -> Categorical[Action]:
        pass