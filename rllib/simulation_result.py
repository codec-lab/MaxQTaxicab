import matplotlib.pyplot as plt
import random
from matplotlib import patheffects
import numpy as np
import pandas as pd
from collections import Counter
from .bandit import MultiArmedBandit, BanditPolicy
from .mdp import MDPPolicy, MarkovDecisionProcess
from .gridworld import GridWorld

class BanditSimulationResult:
    def __init__(
        self,
        action_space,
        rewards,
        action_values,
        expected_rewards,
        bandit : MultiArmedBandit,
        policy : BanditPolicy
    ):
        assert isinstance(action_space, np.ndarray)
        assert isinstance(rewards, np.ndarray)
        assert isinstance(action_values, np.ndarray)
        assert action_space.shape == rewards.shape
        assert action_space.shape == action_values.shape[:-1]
        assert action_space.ndim == 2
        assert action_values.ndim == 3
        assert rewards.ndim == 2
        self.action_space = action_space
        self.rewards = rewards
        self.action_values = action_values
        self.expected_rewards = expected_rewards
        self.n_episodes, self.n_timesteps, self.n_actions = action_values.shape
        self.bandit = bandit
        self.policy = policy
    
    def summary(self):
        return dict(
            mean_reward_rate=self.rewards.mean(),
            std_reward_rat=self.rewards.mean(-1).std()
        )
    
    def plot_reward_rate_hist(self, ax: plt.Axes = None, **kw):
        if ax is None:
            fig, ax = plt.subplots()
        reward_rate = self.rewards.mean(-1)
        ax.hist(reward_rate, **kw)
        stats = f"T = {self.n_timesteps}; " + \
            f"Mean = {reward_rate.mean():.2f}; " + \
            f"S.E. = {reward_rate.std()/np.sqrt(self.n_episodes):.2f}"
        ax.set_title(f"Final reward rates\n({stats})")

    def plot_action_values(
        self,
        ax : plt.Axes = None,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        colors = self.bandit.colors()
        to_plot = sorted(list(range(self.n_episodes)), key=lambda i: random.random())
        for i in to_plot[:20]:
            ep_action_values = self.action_values[i]
        # for ep_action_values in self.action_values:
            for a in self.bandit.action_space:
                ax.plot(ep_action_values[:,a], color=colors[a], lw=0.5)
        mean_action_values = self.action_values.mean(0)
        for a in self.bandit.action_space:
            ax.plot(
                mean_action_values[:,a], color=colors[a], lw=4,
                path_effects=[patheffects.withStroke(linewidth=5, foreground="k")]
            )
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Estimated Action Value")
        ax.legend([f"$a_{i}$" for i in self.bandit.action_space])
        ax.set_title(f"Action value estimates")
        return ax
    
    def plot_action_ranks(
        self,
        ax : plt.Axes = None,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        action_ranks = np.array([-self.bandit.expected_reward(a) for a in self.bandit.action_space]).T.argsort()
        n_episodes = self.action_space.shape[0]
        n_timesteps = self.action_space.shape[1]
        action_props = [(self.action_space == a).sum(0)/n_episodes for a in action_ranks]
        colors = [self.bandit.colors()[a] for a in action_ranks]
        ax.stackplot(
            range(n_timesteps),
            action_props,
            labels=[f"$a_{i}$ $(\mathbb{{E}}[r_{i}] = {self.bandit.expected_reward(i):.2f})$" for i in action_ranks],
            colors=colors
        )
        ax.legend(loc='upper left', reverse=True)
        ax.set_ylabel("Action Proportion")
        ax.set_xlabel("Timestep")
        ax.set_title(f"Selected action proportions (by rank)")
        return ax
    
    def plot_reward(
        self,
        ax : plt.Axes = None
    ):
        if ax is None:
            fig, ax = plt.subplots()
        n_steps = self.action_space.shape[1]
        for ep_rewards in self.rewards:
            ax.plot(range(n_steps), ep_rewards, color='k', lw=.2)
        ax.plot(range(n_steps), self.rewards.mean(0), color='red', lw=3)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward")
        ax.set_title(f"Reward per Timestep")
        return ax
    
    def plot_reward_rate(
        self,
        ax : plt.Axes = None
    ):
        if ax is None:
            fig, ax = plt.subplots()
        n_steps = self.action_space.shape[1]
        for ep_rewards in self.rewards:
            ax.plot(range(n_steps), ep_rewards.cumsum() / np.arange(1, n_steps + 1), color='k', lw=.2)
        ax.plot(range(n_steps), self.rewards.mean(0).cumsum() / np.arange(1, n_steps + 1), color='red', lw=3)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward Rate")
        ax.set_title(f"Reward Rate over time")
        return ax
    
    def plot_true_max_rewards(
        self,
        ax : plt.Axes = None
    ):
        if ax is None:
            fig, ax = plt.subplots()
        n_steps = self.action_space.shape[1]
        for ep_expected_rewards in self.expected_rewards:
            ax.plot(range(n_steps), ep_expected_rewards.max(-1), color='k', lw=.2)
        ax.plot(range(n_steps), self.expected_rewards.max(-1).mean(0), color='red', lw=3)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Expected Reward")
        ax.set_title(f"Max Expected Reward per Timestep")
        return ax
    
    def plot(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        self.plot_reward_rate(axes[0][0])
        self.plot_reward_rate_hist(axes[0][1], bins=50)
        self.plot_action_ranks(axes[1][0])
        self.plot_action_values(axes[1][1])
        # self.plot_reward(axes[1][0])
        plt.tight_layout()

class TDLearningSimulationResult:
    def __init__(
        self,
        trajectory,
        state_values,
        policy : MDPPolicy,
        gw : GridWorld,
    ):
        self.trajectory = trajectory
        self.state_values = state_values
        self.policy = policy
        self.gw = gw
    
    def plot_timestep(self, timestep):
        timestep = timestep if timestep >= 0 else len(self.trajectory) + timestep
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        states_visited = Counter(
            [s for s, _, _, _, _ in self.trajectory[:timestep]]
        )
        gwp = self.gw.plot(ax=axes[0])
        gwp.plot_location_map(states_visited)
        gwp.ax.set_title(f"States Visitation Counts at Timestep {timestep}")
        gwp = self.gw.plot(ax=axes[1])
        gwp.plot_location_map(self.state_values[timestep])
        gwp.ax.set_title(f"State Values at Timestep {timestep}")
    
    def state_value_error(
        self,
        state_value_exact : dict,
        states = None
    ):
        if states is None:
            states = self.gw.state_space
        n_steps = len(self.state_values)
        n_states = len(states)
        errors = np.zeros((n_steps, n_states))
        for ep, est_state_value in enumerate(self.state_values):
            for si, s in enumerate(states):
                errors[ep, si] = abs(est_state_value[s] - state_value_exact[s])
        return errors
    
    def rewards(self):
        return np.array([r for _, _, r, _, _ in self.trajectory])

    def plot_state_value_error(
        self,
        state_value_exact : dict,
        states = None,
        ax=None,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        errors = self.state_value_error(state_value_exact, states)
        _ = ax.plot(errors, lw=.5, color="black")
        _ = ax.plot(errors.mean(-1), color="red")
        ax.set_title(f"Absolute Error in State Value Estimates (final mean = {errors[-1].mean():.2f})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Absolute Error")
    
    def plot_reward_rate(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=200)
        _ = ax.plot(self.rewards().cumsum() / np.arange(1, len(self.rewards()) + 1))
        ax.set_title("Reward Rate")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward Rate")
    
    def plot_stats(self, state_value_exact):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        self.plot_reward_rate(ax=axes[0])
        self.plot_state_value_error(state_value_exact, ax=axes[1])
        plt.tight_layout()
    
    def summary(self, state_value_exact : dict):
        errors = self.state_value_error(state_value_exact)
        return dict(
            final_mean_absolute_error=errors[-1].mean(),
            total_reward=self.rewards().sum(),
            total_reward_rate=self.rewards().sum() / len(self.trajectory),
        )
    
    def plot_control_results(self):
        fig, axes = plt.subplots(1, 3, figsize=(16, 3))
        gwp = self.gw.plot(ax=axes[0])
        _ = gwp.plot_location_map(self.state_values[-1])
        gwp.ax.set_title("Final State Values")

        gwp = self.gw.plot(ax=axes[1])
        optimal_policy = {}
        for s, qvals in self.policy.state_action_values.items():
            max_q = max(qvals.values())
            optimal_actions = [a for a, q in qvals.items() if q == max_q]
            optimal_policy[s] = {a: 1 / len(optimal_actions) for a in optimal_actions}
        _ = gwp.plot_location_action_map(optimal_policy)
        gwp.ax.set_title("Final Optimal Policy")

        episodic_rewards = []
        current_return = 0
        for step in self.trajectory:
            s, a, r, ns, done = step
            current_return += r
            if done:
                episodic_rewards.append(current_return)
                current_return = 0
        smoothed_rewards = pd.Series(episodic_rewards).rolling(50).mean()
        _ = axes[2].plot(smoothed_rewards)
        axes[2].set_title("Episodic Rewards (rolling mean of 50)")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Reward")
        axes[2].set_ylim(-100, 0)
        axes[2].set_xlim(0, len(episodic_rewards))
