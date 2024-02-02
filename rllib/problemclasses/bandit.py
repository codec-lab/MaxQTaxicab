from typing import Sequence, Hashable, TypeVar, Generic
import random
from random import Random

import seaborn as sns

from ..distributions import Distribution

Action = TypeVar('Action', bound=Hashable)

class MultiArmedBandit(Generic[Action]):
    action_space : Sequence[Action]
    def reward_dist(self, a : Action) -> Distribution:
        raise NotImplementedError
    def reward_sample(self, a : Action, rng : Random = random) -> float:
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError
    def expected_reward(self, a : Action) -> float:
        raise NotImplementedError
    # for plotting
    def colors(self):
        return sns.color_palette("husl", n_colors=len(self.action_space))

class BanditPolicy(Generic[Action]):
    def action_dist(self) -> Distribution:
        raise NotImplementedError
    def action_sample(self, rng : Random = random) -> Action:
        raise NotImplementedError
    def action_value(self, a : Action) -> float:
        raise NotImplementedError
    def update(self, a : Action, r : float):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError