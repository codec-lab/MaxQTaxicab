from typing import Sequence, Hashable, TypeVar, Generic, Container, Union
from functools import lru_cache
import random
from collections import defaultdict
from random import Random
from itertools import count

from ..distributions import Distribution, DiscreteDistribution, Uniform, Gaussian
from .mdp import MarkovDecisionProcess, State, Action, Trajectory, Step

class Option(Generic[State, Action]):
    def action_dist(self, s : State) -> Distribution:
        raise NotImplementedError
    def action_sample(self, s : State, rng : Random = random) -> Action:
        raise NotImplementedError
    def is_terminal(self, s : State) -> bool:
        raise NotImplementedError
    def is_initial(self, s : State) -> bool:
        raise NotImplementedError

class ActionOption(Option):
    def __init__(self, action : Action):
        self.action = action
    def action_dist(self, s : State) -> Distribution:
        return DiscreteDistribution([self.action], [1])
    def action_sample(self, s : State, rng : Random = random) -> Action:
        return self.action
    def is_terminal(self, s : State) -> bool:
        return True
    def is_initial(self, s : State) -> bool:
        return True

class MultiTimeModel(dict):
    def prob(self, x):
        return self[x]

class SemiMDP(Generic[State, Action]):
    pass

class OptionSemiMDP(SemiMDP):
    def __init__(
        self,
        mdp : MarkovDecisionProcess,
        option_space : Sequence[Option],
        include_mdp_actions : bool = True
    ):
        self.mdp = mdp
        self.option_space = option_space
        if include_mdp_actions:
            self.option_space = [ActionOption(a) for a in self.mdp.action_space] + option_space

    def initial_state_dist(self) -> Distribution:
        return self.mdp.initial_state_dist()

    def options(self, s : State) -> Sequence[Hashable]:
        option_set = []
        for option in self.option_space:
            if option.is_initial(s):
                option_set.append(option)
        return option_set

    def discounted_next_state_model_reward(
        self,
        s : State,
        o : Option
    ) -> tuple[MultiTimeModel, float]:
        """
        Returns a tuple of:
        - a MultiTimeModel, which maps exit states to discounted probabilities
        - the discounted expected reward
        See Sutton, Precup, and Singh (1999), pg 189
        """
        @lru_cache(maxsize=None)
        def state_to_nextstate_reward(s) -> tuple[dict[State, float], float]:
            if o.is_terminal(s):
                return {}
            ns_dist = defaultdict(float)
            expected_reward = 0
            for a, a_prob in o.action_dist(s).items():
                for ns, ns_prob in self.mdp.next_state_dist(s, a).items():
                    r = self.mdp.reward(s, a, ns)
                    expected_reward += a_prob * ns_prob * r
                    ns_dist[ns] += a_prob * ns_prob
            return ns_dist, expected_reward

        def state_dist_to_nextstate_reward(s_dist) -> dict[State, float]:
            ns_dist = defaultdict(float)
            expected_reward = 0
            for s, s_prob in s_dist.items():
                ns_dist, r = state_to_nextstate_reward(s)
                expected_reward += s_prob * r
                for (ns, r), ns_prob in ns_dist.items():
                    ns_dist[ns] += s_prob * ns_prob
            return ns_dist, expected_reward
        
        terminal_states = set()
        hitting_prob = defaultdict(float)
        discounted_timestep_prob = defaultdict(float)
        discounted_return = 0
        state_prob = {s : 1}
        for k in count():
            for s_, prob in state_prob.items():
                if o.is_terminal(s_):
                    terminal_states.add(s_)
                    hitting_prob[s_] += prob
                    discounted_timestep_prob[s_] += prob * (self.mdp.discount_rate ** k)
            state_prob, expected_reward = state_dist_to_nextstate_reward(state_prob)
            discounted_return += (self.mdp.discount_rate ** k) * expected_reward
            if sum(state_prob.values()) < 1e-6:
                break
        
        next_state_model = MultiTimeModel({
            hitting_prob[s_]*discounted_timestep_prob[s_]
            for s_ in terminal_states
        })
        return next_state_model, discounted_return

    def next_state_sample(self, s : State, o : Option, rng : Random = random) -> State:
        trajectory = self.trajectory_sample(s, o, rng=rng)
        return trajectory[-1].next_state

    def trajectory_sample(self, s : State, o : Option, rng : Random = random) -> Trajectory:
        trajectory = []
        while not o.is_terminal(s):
            a = o.action_sample(s, rng=rng)
            ns = self.mdp.next_state_sample(s, a, rng=rng)
            r = self.mdp.reward(s, a, ns)
            trajectory.append(Step(s, a, r, ns))
            s = ns
        return trajectory

    def is_absorbing(self, s : State) -> bool:
        return self.mdp.is_absorbing(s)

class SemiMDPPolicy(Generic[State]):
    def option_dist(self, s : State) -> Distribution:
        raise NotImplementedError
    def option_sample(self, s : State, rng : Random = random) -> Action:
        raise NotImplementedError
    def end_episode(self):
        raise NotImplementedError
