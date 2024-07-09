import random
from dataclasses import dataclass, replace
from typing import Tuple, Any, Tuple, Union, Literal, TypeVar, List
import numpy as np
import functools

from ihrl.base import MDP, SubTask, State

@dataclass(frozen=True, eq=True, repr=True)
class Location:
    x: int
    y: int

@dataclass(frozen=True, eq=True, repr=True)
class TaxiStand:
    location: Location

@dataclass(frozen=True, eq=True, repr=True)
class Direction:
    dx: Literal[-1, 0, 1]
    dy: Literal[-1, 0, 1]

@dataclass(frozen=True, eq=True, repr=True)
class Taxi:
    location: Location
    passenger: Union[None, "Passenger"]

@dataclass(frozen=True, eq=True, repr=True)
class Passenger:
    location: Location
    destination : Location

@dataclass(frozen=True, eq=True, repr=True)
class Wall:
    location: Location

@dataclass(frozen=True, eq=True, repr=True)
class TaxiCabState:
    taxi: Taxi
    waiting_passengers: Tuple[Passenger, ...]

@dataclass(frozen=True, eq=True, repr=True)
class TaxiCabAction:
    direction: Direction
    pickup: bool
    dropoff: bool








# taxicab = TaxiCab(layout_str)

# east,west, north, south, pickup, putdown = taxicab.actions()
east = TaxiCabAction(Direction(1,0),False,False)
west = TaxiCabAction(Direction(-1,0),False,False)
north = TaxiCabAction(Direction(0,1),False,False)
south = TaxiCabAction(Direction(0,-1),False,False)
pickup = TaxiCabAction(Direction(0,0),True,False)
putdown = TaxiCabAction(Direction(0,0),False,True)

def taxi_state(taxi_x,taxi_y,waiting_passenger_x,waiting_passenger_y,passenger_x = None, passenger_y = None):
    taxi = Taxi(Location(taxi_x,taxi_y),None)
    waiting_passenger = Passenger(Location(waiting_passenger_x,waiting_passenger_y),None)
    return TaxiCabState(taxi, (waiting_passenger,))

def taxi_put_state(taxi_x,taxi_y,dest_x,dest_y):
    passenger = Passenger(None,Location(dest_x,dest_y))
    taxi = Taxi(Location(taxi_x,taxi_y),passenger)
    return TaxiCabState(taxi, ())


#all the code below will be polished up later

##these are needed bc next_state_sample can only work as p(s') if p(s') = 1
def pickup_transition(state,next_state):
    if len(state.waiting_passengers) == 0:
        return False
    if state.taxi.passenger is None and state.waiting_passengers[0].location == state.taxi.location:
        if next_state.taxi.location == state.taxi.location and next_state.taxi.passenger is not None:
            return True
    return False

def putdown_transition(state,next_state):
    if state.taxi.passenger is not None and state.taxi.passenger.destination == state.taxi.location:
        if next_state.taxi.location == state.taxi.passenger.destination and next_state.taxi.passenger is None and len(next_state.waiting_passengers) == 0:
            return True
    return False

class TaxiMDP(MDP):
    def __init__(self, layout_str: str):
        layout_str = layout_str.strip()
        self.layout = [list(row.strip()) for row in layout_str.split("\n")]
        self.height = len(self.layout)
        self.width = len(self.layout[0])
        self.walls : List[Wall] = []
        self.taxi_stands : List[TaxiStand] = []
        self.action_length = 6

        self.legal_locations : List[Location] = []
        for y, row in enumerate(self.layout[::-1]): #add walls and passenger locations from layouts 
            for x, char in enumerate(row):
                if char == "#":
                    self.walls.append(Wall(Location(x, y)))
                else:
                    self.legal_locations.append(Location(x,y))
                    if char in "ABCDEFGH":
                        self.taxi_stands.append(TaxiStand(Location(x, y)))
        self.walls = tuple(self.walls)
        self.taxi_stands = tuple(self.taxi_stands)
        self.discount_rate = 0.9
        self.action_space = (east,west, north, south, pickup, putdown)

    def list_all_possible_states(self):
        all_states = []
        # Iterate through all possible taxi locations
        for taxi_x in range(self.width):
            for taxi_y in range(self.height):
                taxi_location = Location(taxi_x, taxi_y)
                if taxi_location not in [wall.location for wall in self.walls]:
                    # Include states where the passenger is in the taxi
                    for passenger_in_taxi in [True, False]:
                        # Iterate through all possible passenger locations (only if not in the taxi)
                        passenger_locations = self.taxi_stands if not passenger_in_taxi else [None]
                        for passenger_location in passenger_locations:
                            # Iterate through all possible destinations
                            for destination_stand in self.taxi_stands:
                                # If passenger is in the taxi, set the current passenger location to None
                                if passenger_in_taxi:
                                    passenger_current_location = None
                                    destination_stand = destination_stand.location
                                else:
                                    passenger_current_location = passenger_location.location
                                    destination_stand = None

                                # Construct the detailed state representation
                                passenger = Passenger(
                                    location = passenger_current_location,
                                    destination = destination_stand)
                                if passenger_in_taxi:
                                    taxi = Taxi(taxi_location,passenger)
                                    state = TaxiCabState(taxi,())
                                else: #passenger not in taxi
                                    taxi = Taxi(taxi_location,None)
                                    state = TaxiCabState(taxi,(passenger,))
                                #removing ilegal states
                                if state.taxi.passenger is None and len(state.waiting_passengers) == 0:
                                    continue
                                else:
                                    all_states.append(state)
        #at terminal states
        for destination_stand in self.taxi_stands:
            taxi = Taxi(destination_stand.location, None)
            state = TaxiCabState(taxi,())
            all_states.append(state)
        #remove dupicates
        all_states = list(set(all_states))
        return all_states

    def start_states(self):
        return [s for s in self.list_all_possible_states() if s.taxi.passenger is None and len(s.waiting_passengers) > 0]

    def actions(self, state=None):
        return self.action_space

    def new_passenger_at_stand(self, s: TaxiCabState, rng: random.Random = random) -> TaxiCabState:
        if s.waiting_passengers is None and s.taxi.passenger is None:
            new_loc = rng.choice(self.taxi_stands).location
            passenger = Passenger(
                location=new_loc,
                destination=None
            )   
            passengers = s.waiting_passengers + (passenger,)
            return TaxiCabState(taxi=s.taxi, waiting_passengers=passengers)
        return s
    
    def move_taxi(self, s: TaxiCabState, a: TaxiCabAction) -> TaxiCabState:
        dx, dy = a.direction.dx, a.direction.dy
        x, y = s.taxi.location.x, s.taxi.location.y
        new_x, new_y = x + dx, y + dy
        if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
            return s
        if Location(new_x, new_y) in [wall.location for wall in self.walls]:
            return s
        taxi = replace(s.taxi, location=Location(new_x, new_y))

        return replace(s, taxi=taxi)
    
    def dropoff_passenger(self, s: TaxiCabState, a: TaxiCabAction) -> TaxiCabState:
        if s.taxi.passenger is None:
            return s
        if s.taxi.passenger.destination != s.taxi.location:
            return s
        taxi = replace(s.taxi, passenger=None)
        return replace(s, taxi=taxi)
    
    def pickup_passenger(self, s: TaxiCabState, a: TaxiCabAction) -> TaxiCabState:
        if s.taxi.passenger is not None or s.waiting_passengers is None:
            return s
        for passenger in s.waiting_passengers:
            if passenger.location == s.taxi.location:
                taxi = replace(s.taxi, passenger=passenger)
                #####
                new_passenger = Passenger(location=None, 
                                      destination=random.choice([dest.location for dest in self.taxi_stands]))
                taxi = replace(taxi, passenger=new_passenger)

                #####
                passengers = tuple(p for p in s.waiting_passengers if p != passenger)
                return replace(s, taxi=taxi, waiting_passengers=passengers) #was this error?
        return s
    
    def initial_state_sample(self, rng: random.Random = random) -> TaxiCabState:
        taxi = Taxi(rng.choice(self.legal_locations), None)
        passenger = Passenger(rng.choice(self.taxi_stands).location, None)
        return TaxiCabState(taxi, (passenger,))
    
    def next_state_sample(self, s, a, rng: random.Random = random) -> TaxiCabState:
        s = self.move_taxi(s, a)
        if a.dropoff:
            s = self.dropoff_passenger(s, a)
        if a.pickup:
            s = self.pickup_passenger(s, a) #added ifs
        s = self.new_passenger_at_stand(s, rng)
        return s
    
    def state_action_reward(self,state,action):
        reward = -1
        next_state = self.next_state_sample(state,action)
        if action == pickup:
            reward = 15 if next_state.taxi.passenger is not None else -10
        elif action == putdown:
            reward = 15 if next_state.taxi.passenger is None else -10
        return reward
    

    def get_transition_matrix(self):
        transition_matrix = []
        for state in self.list_all_possible_states():
            action_next_state_prob = []
            for action in self.actions(state): 
                next_state_dist = []
                possibilities = 0
                for next_state in self.list_all_possible_states():
                    if action == pickup:
                        if pickup_transition(state,next_state):
                            possibilities += 1
                            probability = 1
                        elif next_state == state and pickup_transition(state,next_state) == False:
                            possibilities += 1
                            probability = 1
                        else:
                            probability = 0
                    elif action == putdown:
                        if putdown_transition(state,next_state):
                            possibilities += 1
                            probability = 1
                        elif next_state == state and putdown_transition(state,next_state) == False:
                            possibilities += 1
                            probability = 1
                        else:
                            probability = 0
                    else:
                        if next_state == self.next_state_sample(state,action):
                            possibilities += 1
                            probability = 1
                        else:
                            probability = 0
                    next_state_dist.append(probability)
                if possibilities > 0:
                    #normalize probabilities
                    for i in range(len(next_state_dist)):
                        next_state_dist[i] = next_state_dist[i]/possibilities
                action_next_state_prob.append(next_state_dist)
            transition_matrix.append(action_next_state_prob)
        return transition_matrix

    def get_reward_matrix(self):
        reward_weights_true = []
        for state in self.list_all_possible_states():
            state_rewards = []
            for action in self.actions(state):
                if action == pickup:
                    reward = 15 if (self.next_state_sample(state,action).taxi.passenger is not None 
                    and state.taxi.passenger is None) else -10
                elif action == putdown:
                    reward = 15 if (self.next_state_sample(state,action).taxi.passenger is None 
                    and state.taxi.passenger is not None) else -10
                else:
                    reward = -1
                state_rewards.append(reward)
            reward_weights_true.append(state_rewards)
        return reward_weights_true
    
    
    def next_state_reward_dist(self,state,action): #state,action,probability
    #only for prim actions
        state_reward_dist = []
        possibilities = 0
        if action == pickup:
            #this method rewards failed pickups if there's already a passenger
            for next_state in self.list_all_possible_states():
                reward = 15 if self.next_state_sample(state,action).taxi.passenger is not None else -10
                #probability starts at 1 and then is normalized at the end                
                probability = 1 if pickup_transition(state,next_state) else 0
                if probability > 0:
                    possibilities += 1
                    state_reward_dist.append(((next_state,reward),probability))
                #no need to include and loop through if prob is 0
        elif action == putdown:
            for next_state in self.list_all_possible_states():
                reward = 15 if self.next_state_sample(state,action).taxi.passenger is None else -10
                #p is 1 since if successful since dropoff results in no new waiting passengers
                probability = 1 if putdown_transition(state,next_state) else 0
                if probability > 0:
                    possibilities += 1
                    state_reward_dist.append(((next_state,reward),probability))
        else:
            for next_state in self.list_all_possible_states():
                reward = -1 #always -1 for nav
                #this is for primitive actions, not nav itself
                probability = 1 if next_state == self.next_state_sample(state,action) else 0
                if probability > 0:
                    possibilities += 1
                    #simpler computaion
                    state_reward_dist.append(((next_state,reward),probability))
        #normalize probabilities
        if possibilities > 0:
            for i in range(len(state_reward_dist)):
                state_reward_dist[i] = (state_reward_dist[i][0],state_reward_dist[i][1]/possibilities)
        return state_reward_dist
    def is_terminal(self, state):
        if state.taxi.passenger is None and len(state.waiting_passengers) == 0:
            return True
        return False
    
    def get_transition_reward_obj(self):
        class TransitionReward:
            def __init__(self,transition_matrix,reward_weights):
                self.transition_matrix = np.array(transition_matrix)
                self.reward_weights = np.array(reward_weights)
        transition_matrix = self.get_transition_matrix()
        reward_weights = self.get_reward_matrix()
        return TransitionReward(transition_matrix,reward_weights)
    
    @functools.lru_cache(maxsize=None)
    def value_iteration(
        self,
        init_state_index : int,
        horizon: int,
        transition_reward_obj,
        deterministic = False
    ):
        state = self.list_all_possible_states()[init_state_index]
        if self.is_terminal(state) or horizon == 0: 
            return 0, None
        max_a_val = float('-inf')
        max_actions = []
        for a in self.actions(): 
            a_index = self.actions().index(a)
            p_s_primes = transition_reward_obj.transition_matrix[init_state_index,a_index,:]
            if len(p_s_primes) == 0:
                raise ValueError("No transition probabilities for this state-action pair")
            for p_sprime_index in range(len(p_s_primes)): #for each next state 0-132
                if p_s_primes[p_sprime_index] == 0:
                    continue
                p_sprime = p_s_primes[p_sprime_index] #get probability of next state (usually 0) 
                v_next, _ = self.value_iteration(p_sprime_index,horizon-1,transition_reward_obj) 
                #want max_a [p(s') * (r(s,a) + gamma * v(s'))]
                val = p_sprime * (transition_reward_obj.reward_weights[p_sprime_index,a_index] + 0.9 * v_next)
                if val > max_a_val:
                    max_a_val = val
                    max_actions = [a]
                elif val == max_a_val:
                    max_actions.append(a)
        if deterministic:
            max_action = max_actions[0]
        else:
            max_action = random.choice(max_actions)

        return max_a_val, max_action
    
    def gen_trajectory(self, init_state_index : int, max_t_length=110, transition_reward_obj=None,deterministic=False):
        state = self.list_all_possible_states()[init_state_index]
        state_index = init_state_index
        trajectory = []
        for i in range(81,max_t_length):
            if self.is_terminal(state):
                break
            _, max_a = self.value_iteration(state_index, i, transition_reward_obj=transition_reward_obj,deterministic=deterministic)
            if max_a is None:
                raise ValueError("No action found")
            state_prime = self.next_state_sample(state,max_a) #stochasticity
            trajectory.append((state,max_a))
            state = state_prime
            state_index = self.list_all_possible_states().index(state)
        assert len(trajectory) < 25
        return trajectory
    
    def get_state_action_indices(self, total_trajectories, max_t_length=110, transition_reward_obj=None,deterministic=False):
        #step 1 get trajectories
        trajectories = []
        for i in range(total_trajectories):
            init_state_index = self.list_all_possible_states().index(self.initial_state_sample())
            trajectory = self.gen_trajectory(init_state_index,max_t_length,transition_reward_obj,deterministic)
            trajectories.append(trajectory)
        #step 2 get state action indices
        state_indices = []
        action_indices = []
        for trajectory in trajectories:
            assert len(trajectory) < 50 #will have to change if map is larger but for now this is fine
            for state,action in trajectory:
                state_index = self.list_all_possible_states().index(state)
                action_index = self.actions().index(action)
                state_indices.append(state_index)
                action_indices.append(action_index)
        return state_indices, action_indices

def get_terminal(s):
    if s.taxi.passenger is not None:
        return True
    return False
def put_terminal(s):
    if s.taxi.passenger is None:
        return True
    return False
root_terminal = put_terminal
#only loop thorugh exits w non zero probability
class Root(SubTask,MDP):
    def __init__(self,MDP):
        self.mdp = MDP
        c_nav = Nav(self.mdp) #keeps same value of nav in memory for both get and put
        self.child_subtasks = [Get(self.mdp,c_nav),Put(self.mdp,c_nav)]
        self.continuation_prob = None
        self.exit_reward = None 
    def is_terminal(self,s):
        if s.taxi.passenger is None and len(s.waiting_passengers) == 0:
            return True
        return False 
    def exit_distribution(self, s: State):
        distribution = []
        possibilities = 0
        for state in self.mdp.list_all_possible_states():
            if self.is_terminal(state) and len(s.waiting_passengers) == 0 and s.passenger is None and s.taxi.location == state.taxi.location: #
                distribution.append((state,1))
                possibilities += 1
            else:
                pass
        if possibilities > 0:
            for i in range(len(distribution)):
                distribution[i] = (distribution[i][0],distribution[i][1]/possibilities)
        return distribution
class Get(SubTask,MDP):
    def __init__(self,MDP,child_nav = None):
        self.mdp = MDP
        self.child_nav = child_nav
        if child_nav is None:
            self.child_subtasks = [Nav(self.mdp),pickup]
        else:
            self.child_subtasks = [self.child_nav,pickup]
        self.continuation_prob = None
        self.exit_reward = None 
    def is_terminal(self,s):
        if s.taxi.passenger is not None:
            return True
        return False
    def exit_distribution(self, s: State):
        distribution = []
        possibilities = 0
        for state in self.mdp.list_all_possible_states():
            if self.is_terminal(state)  and len(s.waiting_passengers) == 1:
                if s.waiting_passengers[0].location == state.taxi.location:
                    if len(state.waiting_passengers) == 0:
                        if state.taxi.passenger is not None: #
                            distribution.append((state,1))
                            possibilities += 1
            else:
                pass
        if possibilities > 0:
            for i in range(len(distribution)):
                distribution[i] = (distribution[i][0],distribution[i][1]/possibilities)
        return distribution
        
class Put(SubTask,MDP):
    def __init__(self,MDP,child_nav =None):
        self.mdp = MDP
        self.child_nav = child_nav
        if child_nav is None:
            self.child_subtasks = [Nav(self.mdp),putdown]
        else:
            self.child_subtasks = [self.child_nav,putdown]
        self.continuation_prob = None
        self.exit_reward = None
    def is_terminal(self,s):
        if s.taxi.passenger is None and len(s.waiting_passengers) == 0:
            return True
        return False 
    def exit_distribution(self, s):
        distribution = []
        possibilities = 0
        for state in self.mdp.list_all_possible_states():
            if self.is_terminal(state) and len(state.waiting_passengers) == 0 and state.taxi.passenger is None and s.taxi.location == state.taxi.location: #
                distribution.append((state,1))
                possibilities += 1
            else:
                pass
        if possibilities > 0:
            for i in range(len(distribution)):
                distribution[i] = (distribution[i][0],distribution[i][1]/possibilities)
        return distribution
class Nav(SubTask,MDP):
    def __init__(self,MDP):
        self.mdp = MDP
        self.child_subtasks = [east,west,north,south]

        self.continuation_prob = None
        self.exit_reward = None

    def is_terminal(self,s): #return true if at pass dest or waiting pass location 
        if s.taxi.passenger is not None:
            if s.taxi.location == s.taxi.passenger.destination:
                return True
            else:
                return False
        elif len(s.waiting_passengers) == 0:
            return False #hack for now
        elif s.taxi.location == s.waiting_passengers[0].location:
            return True
        return False 
    def exit_distribution(self, s: State) -> list[Tuple[Any, float]]:
        distribution = []
        possibilities = 0
        for state in self.mdp.list_all_possible_states():
            if s.taxi.passenger is not None and state.taxi.passenger is not None:
                if s.taxi.passenger.destination != state.taxi.passenger.destination:
                    continue
            if self.is_terminal(state) and s.waiting_passengers == state.waiting_passengers and (s.taxi.passenger is None) == (state.taxi.passenger is None):
                #need to also not include if starting state dest != end state dest
                distribution.append((state,1))
                possibilities += 1
            else:
                pass
                #distribution.append((state,0))
        if possibilities > 0:
            for i in range(len(distribution)):
                distribution[i] = (distribution[i][0],distribution[i][1]/possibilities)
        return distribution