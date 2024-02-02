from collections import namedtuple, defaultdict
from typing import Sequence, Tuple, Dict
from itertools import product
import random

import numpy as np

from .problemclasses.mdp import MarkovDecisionProcess
from .distributions import Distribution, DiscreteDistribution, Uniform

Location = namedtuple('Location', ['x', 'y'])
Direction = namedtuple('Direction', ['dx', 'dy'])

move_east = Direction(1, 0)
move_west = Direction(-1, 0)
move_north = Direction(0, 1)
move_south = Direction(0, -1)

class GridWorld(MarkovDecisionProcess[Location, Direction]):
    INIT_STATE_FEATURE = '@'
    GOAL_STATE_FEATURE = '$'
    LAVA_STATE_FEATURE = 'x'
    OBSTACLE_STATE_FEATURE = '#'
    SLIP_STATE_FEATURE = '~'
    PUDDLE_STATE_FEATURE = 'o'
    NORMAL_STATE_FEATURE = '.'
    SLIP_PROB = 0.2
    LAVA_COST = -100
    GOAL_REWARD = 0
    STEP_COST = -1
    PUDDLE_COST = -10
    feature_list = tuple("@$x#~o.")

    def __init__(self, *, layout_string, discount_rate):
        assert isinstance(layout_string, str)
        layout_string = layout_string.strip().split('\n')
        layout = np.array([list(row.strip()) for row in layout_string[::-1]])
        self.discount_rate = discount_rate

        # set up gridworld state types
        self.height, self.width = layout.shape
        self.state_space : Sequence[Location] = tuple(
            Location(x, y) for (x, y) in product(range(self.width), range(self.height))
        )
        self.states_to_features : Dict[Location,str] = {
            s: layout[s.y, s.x] for s in self.state_space
        }
        self.features_to_states : Dict[str,Tuple[Location]] = defaultdict(tuple)
        for s, feature in self.states_to_features.items():
            self.features_to_states[feature] += (s,)

        # set up gridworld action types
        self.action_space = tuple([move_east, move_west, move_north, move_south])
        self.state_action_space = tuple(product(self.state_space, self.action_space))

    def initial_state_dist(self):
        initial_states = self.locations_with(self.INIT_STATE_FEATURE)
        return Uniform(initial_states)

    def actions(self, s):
        return self.action_space
    
    def next_state_dist(self, s : Location, a : Direction) -> Distribution:
        intended_ns = Location(s.x + a.dx, s.y + a.dy)
        if self.is_slippery(s):
            if a.dx == 0: # up or down
                slip1_ns = Location(s.x - 1, s.y)
                slip2_ns = Location(s.x + 1, s.y)
            else: # left or right
                slip1_ns = Location(s.x, s.y - 1)
                slip2_ns = Location(s.x, s.y + 1)
            slip1_ns = slip1_ns if self.is_valid(slip1_ns) else s
            slip2_ns = slip2_ns if self.is_valid(slip2_ns) else s
            intended_ns = intended_ns if self.is_valid(intended_ns) else s
            return DiscreteDistribution(
                support=[intended_ns, slip1_ns, slip2_ns],
                probs=[1 - self.SLIP_PROB, self.SLIP_PROB / 2, self.SLIP_PROB / 2]
            )
        else:
            intended_ns = intended_ns if self.is_valid(intended_ns) else s
            return DiscreteDistribution([intended_ns], [1])

    def next_state_sample(
        self,
        s : Location,
        a : Direction,
        rng : random.Random = random
    ) -> Location:
        return self.next_state_dist(s, a).sample(rng=rng)

    def reward(self, s : Location, a : Direction, ns : Location) -> float:
        reward = self.STEP_COST
        if self.is_lava(ns):
            reward += self.LAVA_COST
        elif self.is_puddle(ns):
            reward += self.PUDDLE_COST
        elif self.is_goal(ns):
            reward += self.GOAL_REWARD
        return reward

    def is_absorbing(self, s):
        return self.is_goal(s) or self.is_lava(s)

    # helper methods
    def is_slippery(self, s):
        return self.states_to_features[s] == self.SLIP_STATE_FEATURE
    
    def is_lava(self, s):
        return self.states_to_features[s] == self.LAVA_STATE_FEATURE
    
    def is_puddle(self, s):
        return self.states_to_features[s] == self.PUDDLE_STATE_FEATURE
    
    def is_obstacle(self, s):
        return self.states_to_features[s] == self.OBSTACLE_STATE_FEATURE
    
    def is_goal(self, s):
        return self.states_to_features[s] == self.GOAL_STATE_FEATURE
    
    def is_in_bounds(self, s):
        return 0 <= s.x < self.width and 0 <= s.y < self.height
    
    def is_valid(self, s):
        return self.is_in_bounds(s) and not self.is_obstacle(s)
    
    def locations_with(self, feature):
        return self.features_to_states[feature]

    def plot(self, ax=None):
        #avoid circular dependency
        import matplotlib.pyplot as plt
        from msdm.domains.gridmdp.plotting import GridMDPPlotter
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plotter = GridMDPPlotter(self, ax=ax)
        plotter.fill_features(
            feature_colors={
                self.INIT_STATE_FEATURE: 'w',
                self.GOAL_STATE_FEATURE: 'green',
                self.LAVA_STATE_FEATURE: 'r',
                self.OBSTACLE_STATE_FEATURE: 'k',
                self.SLIP_STATE_FEATURE: 'lightblue',
                self.PUDDLE_STATE_FEATURE: 'rosybrown',
                self.NORMAL_STATE_FEATURE: 'w'
            },
            default_color='w'
        )
        plotter.mark_features({
            self.INIT_STATE_FEATURE: 'o',
            self.LAVA_STATE_FEATURE: 'x',
            self.GOAL_STATE_FEATURE: '*',
        })
        plotter.plot_outer_box()
        return plotter