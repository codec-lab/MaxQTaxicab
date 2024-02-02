import random
from typing import Union, Any

import numpy as np
import gymnasium as gym

from .mdp import MarkovDecisionProcess

ObsType = Union[int, np.ndarray, dict[str, Any]]
ActType = Union[int, np.ndarray, dict[str, Any]]

class GymWrapper(gym.Env):
    def __init__(self, mdp : MarkovDecisionProcess):
        self.mdp = mdp
        self.current_state = None
        self.rng : random.Random = random
        self.action_space = gym.spaces.Discrete(len(mdp.action_space))
        assert not isinstance(next(iter(mdp.action_space)), int), \
            "To avoid ambiguity with gym action encoding, action space must not be an integer"
        self.observation_space = gym.spaces.Discrete(len(mdp.state_space))
        self.reward_range = (float('-inf'), float('inf'))

    def step(self, action : ActType) -> tuple[ObsType, float, bool, dict[str, Any]]:
        # returns: obs, reward, terminated, info
        assert isinstance(action, int), "Input is the action index"
        action = self.mdp.action_space[action]
        next_state = self.mdp.next_state_sample(self.current_state, action, rng=self.rng)
        next_state_idx = self.mdp.state_space.index(next_state)
        reward = self.mdp.reward(self.current_state, action, next_state)
        terminated = self.mdp.is_absorbing(next_state)
        self.current_state = next_state
        info = dict(
            state=self.current_state,
            action=action,
            next_state=next_state,
            reward=reward,
        )
        return next_state_idx, reward, terminated, info

    def reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.rng = random.Random(seed)
        self.current_state = self.mdp.initial_state_dist().sample(rng=self.rng)
        return self.mdp.state_space.index(self.current_state), dict(state=self.current_state)

    def render(self) -> None:
        raise NotImplementedError
        assert self.render_mode in [
            None, "human", "rgb_array", "ansi", "rgb_array_list", "ansi_list"
        ]