import functools
from ihrl.base import SubTask, State

@functools.lru_cache(maxsize=None)
def horizon_value(subtask: SubTask, s: State,horizon):
    if horizon == 0:
        return 0
    if subtask.is_terminal(s) == True:
        return 0

    continue_prob = 1  # subtask.continuation_prob(s) #for now, same as just if terminal. bc first if will exit out this can equal 1 for now
    max_qval = float("-inf")
    max_action = None #just for debugging it
    for a in subtask.child_subtasks:
        # Get next-state/reward distribution for semi-MDP
        if a in subtask.mdp.actions(s):  # if isinstance(a, Action):
            ns_r_prob = subtask.mdp.next_state_reward_dist(s, a)
        else: #will polish later but this if/else is same functionality as designating each a as action class # elif isinstance(a, SubTask):
            ns_r_prob = []
            for ns, prob in a.exit_distribution(s):
                if prob == 0:
                    continue
                #if (a,s,horizon) in horizon_cache:
                #    v = horizon_cache[(a,s,horizon)]   
                #else:
                v = horizon_value(a, s, horizon)
                ns_r_prob.append(((ns, v),(prob)))
        # Calculate expected value of action Q(subtask, s, a)

        qval = 0
        for ns_r, prob in ns_r_prob:
            if prob == 0:
                continue
            ns, r = ns_r
            v = horizon_value(subtask, ns,horizon-1)
            qval += prob * (r + continue_prob * v)
            if qval > max_qval:
                max_action = a
            max_qval = max(max_qval, qval)
    return max_qval

def val_true(subtask: SubTask, s: State, max_horizon=100, epsilon=0.01):
    # TODO: modify so that you can specify the cache size
    for i in range(1, max_horizon):
        v1 = horizon_value(subtask, s, i-1)
        v2 = horizon_value(subtask, s, i)
        if abs(v1 - v2) < epsilon:
            return v2
    return v2