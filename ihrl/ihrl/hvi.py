import functools
from ihrl.base import SubTask, State
from ihrl.taxicab import Root, taxi_state,taxi_put_state, Nav, Get
import random

@functools.lru_cache(maxsize=None)
def horizon_value(subtask: SubTask, s: State,horizon):
    if horizon == 0:
        return 0
    #after using continue_prob needed to make sure that at least the mdp exits out if terminal to prevent infinite loop
    if subtask.mdp.is_terminal(s) == True:
        return 0

    continue_prob = 0 if subtask.is_terminal(s) else 1 # subtask.continuation_prob(s) #for now, same as just if terminal. bc first if will exit out this can equal 1 for now
    max_qval = float("-inf")
    max_action = None #just for debugging it
    for a in subtask.child_subtasks:
        # Get next-state/reward distribution for semi-MDP
        if a in subtask.mdp.actions(s):  # if isinstance(a, Action):
            ns_r_prob = subtask.mdp.next_state_reward_dist(s, a)
        else: #will polish later but this if/else is same functionality as designating each a as action class # elif isinstance(a, SubTask):
            ns_r_prob = []
            for ns, prob in a.exit_distribution(s):
                v = horizon_value(a, s, horizon)
                ns_r_prob.append(((ns, v),(prob)))
        # Calculate expected value of action Q(subtask, s, a)
        qval = 0
        for ns_r, prob in ns_r_prob:
            ns, r = ns_r
            v_prime = horizon_value(subtask, ns,horizon-1)
            qval += prob * (r + continue_prob * v_prime)
            if qval > max_qval:
                max_action = a
            max_qval = max(max_qval, qval)
    return max_qval

@functools.lru_cache(maxsize=None)
def horizon_qvalue(subtask: SubTask, s: State,action,  horizon):

    continue_prob = 0 if subtask.is_terminal(s) else 1 #for now instead of subtask.continuation_prob(s) since prob is 0 or 1
    # Get next-state/reward distribution for semi-MDP
    if action in subtask.mdp.actions(s):  # if isinstance(a, Action):
        #check if mdp terminated
        ns_r_prob = subtask.mdp.next_state_reward_dist(s, action)
    else: #will polish later but this if/else is same functionality as designating each a as action class # elif isinstance(a, SubTask):
        ns_r_prob = []
        for ns, prob in action.exit_distribution(s):
            v = horizon_value(action, s, horizon) #horizon_value or horizon_qvalue...
            ns_r_prob.append(((ns, v),(prob)))
    # Calculate expected value of action Q(subtask, s, a)
    qval = 0
    for ns_r, prob in ns_r_prob:
        ns, r = ns_r
        if subtask.mdp.is_terminal(ns):
            v = 0
        else:
            v = horizon_value(subtask, ns,horizon-1)
        qval += prob * (r + continue_prob * v)
    return qval

def val_true(subtask: SubTask, s: State, max_horizon=100, epsilon=0.01):
    # TODO: modify so that you can specify the cache size
    for i in range(1, max_horizon):
        v1 = horizon_value(subtask, s, i-1)
        v2 = horizon_value(subtask, s, i)
        if abs(v1 - v2) < epsilon:
            return v2
    return v2

def greedy_policy(task,init_state,deterministic=False):
    state = init_state
    traj = []
    while True:
        max_q_value = -float('inf')
        max_action = None
        next_state = None
        max_actions = []
        #Step 1, get the best action using its qval
        #if multiple best actions choose randomly
        for action in task.child_subtasks:
            q_value = horizon_qvalue(task,state,action,10)
            if q_value > max_q_value:
                max_q_value = q_value
                max_actions = [action]
            elif q_value == max_q_value:
                max_actions.append(action)
        if deterministic: #for testing behavioral cloning
            max_action = max_actions[0]
        else:
            max_action = random.choice(max_actions)
        ###########################################
        #Step 2, we have best action now
        #if primitive take it and add it to trajectory
        if max_action in task.mdp.actions(state): # if primitive action
            next_state = task.mdp.next_state_sample(state,max_action)
            traj.append((state,max_action))
        #if it's not primitive, enter into the subtask
        else:
            #get sub_traj of whatever happened in the subtask and the state the subtask ended up in
            sub_traj, next_state = greedy_policy(max_action,state,deterministic=deterministic)
            traj += sub_traj
        #Step 3 base case: check if the task is terminal
        if task.is_terminal(next_state):
            state = next_state
            return traj,state
        state = next_state
        
def get_trajectories(root_mdp,num_traj,deterministic=False):
    trajectories = []
    for i in range(num_traj):
        trajectories.append(greedy_policy(root_mdp,root_mdp.mdp.initial_state_sample(),deterministic)[0]) #0 bc its traj,next_state
    return trajectories