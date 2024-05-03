from typing import Any, Tuple, Union, TypeVar

Action = TypeVar("Action")
State = TypeVar("State")
StateDist = list[Tuple[State, float]]
StateRewardDist = list[Tuple[Tuple[State, float], float]]

class MDP:
    state_list : list[State]
    action_space : list[Action]
    def actions(self, state: State) -> list[Action]: pass
    def next_state_reward_dist(self, state: State, action: Action) -> StateRewardDist: pass #1
    def is_terminal(self, state: State) -> bool: pass

class SubTask(MDP):
    mdp: MDP
    child_subtasks: list[Union["SubTask", Action]]
    def continuation_prob(self, state: State) -> float: pass #2, just for whole mdp?
    def exit_reward(self, state: State) -> float: pass #3 eq 7
    def exit_distribution(self, state: State) -> StateDist: pass #eq 8 #4
