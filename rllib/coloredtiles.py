from collections import namedtuple, defaultdict
from functools import cached_property
from typing import Sequence, Tuple, Dict
from itertools import product
import random

import numpy as np
from matplotlib.colors import to_rgb, is_color_like

from .problemclasses.mdp import MarkovDecisionProcess
from .distributions import Distribution, DiscreteDistribution, Uniform

Location = namedtuple('Location', ['x', 'y'])
Direction = namedtuple('Direction', ['dx', 'dy'])

move_east = Direction(1, 0)
move_west = Direction(-1, 0)
move_north = Direction(0, 1)
move_south = Direction(0, -1)

px_per_tile = 3

class ColoredTiles(MarkovDecisionProcess[Location, Direction]):
    INIT_STATE_FEATURE = '@'
    GOAL_STATE_FEATURE = '$'
    feature_list = tuple("@$rbgycm.")

    def __init__(
        self, *,
        layout_string: str,
        discount_rate: float,
        color_rewards: Dict[str, float]
    ):
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
        self.color_rewards = color_rewards

        # set up gridworld action types
        self.action_space = tuple([move_east, move_west, move_north, move_south])
        self.state_action_space = tuple(product(self.state_space, self.action_space))

    def initial_state_dist(self):
        initial_states = self.locations_with(self.INIT_STATE_FEATURE)
        return Uniform(initial_states)

    def actions(self, s):
        return self.action_space
    
    def next_state_dist(self, s : Location, a : Direction) -> Distribution:
        ns = Location(s.x + a.dx, s.y + a.dy)
        if self.is_valid(ns):
            return DiscreteDistribution([ns], [1])
        return DiscreteDistribution([s], [1])

    def next_state_sample(
        self,
        s : Location,
        a : Direction,
        rng : random.Random = random
    ) -> Location:
        return self.next_state_dist(s, a).sample(rng=rng)

    def reward(self, s : Location, a : Direction, ns : Location) -> float:
        return self.color_rewards.get(self.states_to_features[s], 0)

    def is_absorbing(self, s):
        return self.is_goal(s)

    # helper methods
    def is_goal(self, s):
        return self.states_to_features[s] == self.GOAL_STATE_FEATURE
    
    def is_in_bounds(self, s):
        return 0 <= s.x < self.width and 0 <= s.y < self.height
    
    def is_valid(self, s):
        return self.is_in_bounds(s)
    
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
                **{c: c for c in self.color_rewards.keys() if c in "rbgycm"}
            },
            default_color='w'
        )
        plotter.mark_features({
            self.INIT_STATE_FEATURE: 'o',
            self.GOAL_STATE_FEATURE: '*',
        })
        plotter.plot_outer_box()
        return plotter
    
    def get_state_feature_matrix(self):
        state_feature_matrix = np.zeros((len(self.state_space), len(self.feature_list)))
        for i, s in enumerate(self.state_space):
            state_feature_matrix[i, self.feature_list.index(self.states_to_features[s])] = 1
        return state_feature_matrix
    
    def get_reward_weights(self):
        w = [self.color_rewards.get(f, 0) for f in self.feature_list]
        w = np.array(w)
        return w
    
    def get_absorbing_state_mask(self):
        return np.array([self.is_absorbing(s) for s in self.state_space])
    
    def get_state_reward_vector(self):
        rf = self.get_state_feature_matrix() @ self.get_reward_weights()
        return rf
    
    def get_transition_matrix(mdp : MarkovDecisionProcess):
        tf = np.zeros((len(mdp.state_space), len(mdp.action_space), len(mdp.state_space)))
        for s in mdp.state_space:
            si = mdp.state_space.index(s)
            for a in mdp.actions(s):
                ai = mdp.action_space.index(a)
                for ns, prob in mdp.next_state_dist(s, a).items():
                    nsi = mdp.state_space.index(ns)
                    tf[si, ai, nsi] = prob
        assert np.allclose(tf.sum(axis=2), 1)

        # set probability of leaving absorbing states to 0
        for s in mdp.state_space:
            si = mdp.state_space.index(s)
            if mdp.is_absorbing(s):
                tf[si] = 0
        return tf

    
    @cached_property
    def tile_color_image(self):
        img = np.zeros((self.height*3, self.width*3, 3))
        for s in self.state_space:
            x, y = s.x, self.height - s.y - 1
            color = self.states_to_features.get(s, 'w')
            if color == "$":
                color = to_rgb('lime')
            elif color == "@":
                color = 'w'
            elif not is_color_like(color):
                color = 'w'
            xs = slice(x*px_per_tile, (x + 1)*px_per_tile)
            ys = slice(y*px_per_tile, (y + 1)*px_per_tile)
            img[ys, xs, :] = to_rgb(color)
        img.flags.writeable = False
        return img
    
    def render_state(self, s: Location):
        img = self.tile_color_image.copy()
        x, y = s.x, self.height - s.y - 1
        img[y*px_per_tile+ 1, x*px_per_tile+1, :] = to_rgb('k')
        return img

    # def get_state_images(self):
