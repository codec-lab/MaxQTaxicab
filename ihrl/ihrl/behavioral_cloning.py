from ihrl.hvi import val_true, horizon_value, horizon_qvalue, get_trajectories
from ihrl.taxicab import TaxiMDP, Root,Get, Nav, Put,taxi_state, taxi_put_state

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


layout_str = """
A--B
----
----
C--D 
"""
mdp = TaxiMDP(layout_str)
test_root = Root(mdp)
init_state = taxi_state(0,0,0,0)

# def state_to_vec(state): 
#     taxi_x_pos = state.taxi.location.x
#     taxi_y_pos = state.taxi.location.y
#     if state.taxi.passenger is None:
#         pass_dest_x = -1
#         pass_dest_y = -1
#     else:
#         pass_dest_x = state.taxi.passenger.destination.x
#         pass_dest_y = state.taxi.passenger.destination.y
#     if len(state.waiting_passengers) == 0:
#         wait_dest_x = -1
#         wait_dest_y = -1
#     else:
#         wait_dest_x = state.waiting_passengers[0].location.x
#         wait_dest_y = state.waiting_passengers[0].location.y
#     return [taxi_x_pos,taxi_y_pos,pass_dest_x,pass_dest_y,wait_dest_x,wait_dest_y]
# def action_to_vec(action):
#     dx = action.direction.dx
#     dy = action.direction.dy
#     pickup = 1 if action.pickup else 0
#     dropoff = 1 if action.dropoff else 0
#     return [dx,dy,pickup,dropoff]
# def action_to_one_hot(action): 
#     y = mdp.actions(init_state).index(action)
#     return [1 if i == y else 0 for i in range(len(mdp.actions(init_state)))]
# def preprocess_trajectories_tensor(traj_list):
#     preprocessed_traj = []
#     for traj in traj_list:
#         for state, action in traj:
#             state_vec = torch.tensor(state_to_vec(state), dtype=torch.float)
#             action = torch.tensor(mdp.actions(init_state).index(action))
#             preprocessed_traj.append((state_vec, action))
#     return preprocessed_traj

def state_to_vec(state): 
    taxi_x_pos = state.taxi.location.x
    taxi_y_pos = state.taxi.location.y
    if state.taxi.passenger is None:
        pass_dest_x = -1
        pass_dest_y = -1
    else:
        pass_dest_x = state.taxi.passenger.destination.x
        pass_dest_y = state.taxi.passenger.destination.y
    if len(state.waiting_passengers) == 0:
        wait_dest_x = -1
        wait_dest_y = -1
    else:
        wait_dest_x = state.waiting_passengers[0].location.x
        wait_dest_y = state.waiting_passengers[0].location.y
    return [taxi_x_pos,taxi_y_pos,pass_dest_x,pass_dest_y,wait_dest_x,wait_dest_y]
def action_to_vec(action):
    dx = action.direction.dx
    dy = action.direction.dy
    pickup = 1 if action.pickup else 0
    dropoff = 1 if action.dropoff else 0
    return [dx,dy,pickup,dropoff]
def action_to_one_hot(action): 
    y = mdp.actions(init_state).index(action)
    return [1 if i == y else 0 for i in range(len(mdp.actions(init_state)))]
def preprocess_trajectories_numpy(traj_list):
    preprocessed_traj = []
    for traj in traj_list:
        for state, action in traj:
            state_vec = state_to_vec(state)
            action = mdp.actions(init_state).index(action)
            state_vec.append(action)
            preprocessed_traj.append(state_vec)
    return preprocessed_traj


