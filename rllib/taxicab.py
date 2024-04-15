import random
from itertools import product
from typing import Tuple, Dict, Iterable, Iterator, Optional, Union, Literal, TypeVar, List
from dataclasses import dataclass, replace

import numpy as np
import matplotlib.pyplot as plt

from rllib.distributions import Distribution, DiscreteDistribution

from rllib.problemclasses.mdp import MarkovDecisionProcess

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

class TaxiCab(MarkovDecisionProcess[TaxiCabState, TaxiCabAction]):
    passenger_appear_prob: float = 0.1

    def __init__(self, layout_str: str):
        layout_str = layout_str.strip()
        self.layout = [list(row.strip()) for row in layout_str.split("\n")]
        self.height = len(self.layout)
        self.width = len(self.layout[0])
        self.walls : List[Wall] = []
        self.taxi_stands : List[TaxiStand] = []
            
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
        actions = []
#         for dx, dy, pickup, dropoff in product(
#             [-1, 0, 1], [-1, 0, 1], [True, False], [True, False]
#         ):
#             if dx == dy == 0:
#                 continue
#             actions.append(TaxiCabAction(Direction(dx, dy), pickup, dropoff))
    #######################
        actions.append(TaxiCabAction(Direction(1,0),False,False))
        actions.append(TaxiCabAction(Direction(-1,0),False,False))
        actions.append(TaxiCabAction(Direction(0,1),False,False))
        actions.append(TaxiCabAction(Direction(0,-1),False,False))
        actions.append(TaxiCabAction(Direction(0,0),True,False))
        actions.append(TaxiCabAction(Direction(0,0),False,True))


        ##############################
        self.action_space = tuple(actions)
    
    def initial_state_sample(self, rng: random.Random = random) -> TaxiCabState:
        while True:
            loc = Location(rng.randint(0, self.width -1), rng.randint(0, self.height-1)) #added -1
            new_loc = rng.choice(self.taxi_stands).location
            passenger = Passenger(
                location=new_loc,
                destination=None
            )   
            passengers = (passenger,)
            if loc not in [wall.location for wall in self.walls]:
                taxi = Taxi(loc, None)
                break

        
        return TaxiCabState(taxi=taxi, waiting_passengers=passengers)

    def actions(self): 
        return self.action_space
    
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
                                #
 
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
#                                 state = {
#                                     'taxi_location': taxi_location,
#                                     'passenger_in_taxi': passenger_in_taxi,
#                                     'passenger_location': passenger_current_location,
#                                     'destination': destination_stand.location
#                                 }
                                all_states.append(state)
        #at terminal states
        for destination_stand in self.taxi_stands:
            taxi = Taxi(destination_stand.location, None)
            state = TaxiCabState(taxi,())
            all_states.append(state)
        #remove dupicates
        all_states = list(set(all_states))
        return all_states


         
    
#     def new_passenger_at_stand(self, s: TaxiCabState, rng: random.Random = random) -> TaxiCabState:
#         if len(s.waiting_passengers) == len(self.taxi_stands):
#             return s
#         passenger_locations = [p.location for p in s.waiting_passengers]
#         for stand in self.taxi_stands:
#             if stand.location not in passenger_locations:
#                 if rng.random() < self.passenger_appear_prob:
#                     passenger = Passenger(
#                         location=stand.location,
#                         destination=rng.choice([
#                             dest.location for dest in self.taxi_stands if dest != stand
#                         ])
#                     )
#                     passengers = s.waiting_passengers + (passenger,)
#                     return TaxiCabState(taxi=s.taxi, waiting_passengers=passengers)
#         return s

    def new_passenger_at_stand(self, s: TaxiCabState, rng: random.Random = random) -> TaxiCabState:
        if s.waiting_passengers is  None and s.taxi.passenger is None:
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
        
#         if s.taxi.passenger is not None:
#             passenger = replace(s.taxi.passenger, location=Location(new_x, new_y))
#         else:
#             passenger = None
#        taxi = replace(s.taxi, location=Location(new_x, new_y), passenger=passenger)
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
    
    def next_state_sample(self, s, a, rng: random.Random = random) -> TaxiCabState:
        s = self.move_taxi(s, a)
        if a.dropoff:
            s = self.dropoff_passenger(s, a)
        if a.pickup:
            s = self.pickup_passenger(s, a) #added ifs

        s = self.new_passenger_at_stand(s, rng)
        return s

    def reward(self, s, a, ns):
        reward = -1  # for just taking an action
       #SWITCHING to POSITIVE 1... bc maybe its the traction CV needs
#         lx = s.taxi.location.x
#         ly = s.taxi.location.y
#         if s.taxi.passenger is not None:
#             px = s.taxi.passenger.destination.x
#             py = s.taxi.passenger.destination.y

#             reward -= np.abs(px-lx) + np.abs(py-ly)
#         else:
#             px = s.waiting_passengers[0].location.x
#             py = s.waiting_passengers[0].location.y
#             reward -= np.abs(px-lx) + np.abs(py-ly)
            
        # Check if illegal pickup
        if a.pickup:
            if not any(passenger.location == s.taxi.location for passenger in s.waiting_passengers) or (s.taxi.passenger is not None): 
                #bc you can't pickup if not at location or someone in car already
                reward -= 10  # Penalty for illegal pickup

               # print('successful pickup')
            else:
                reward += 1 #for debugging

        # Check if illegal dropoff
        if a.dropoff:
            # If there's no passenger in the taxi or the passenger's destination doesn't match the taxi's location
            if (s.taxi.passenger is not None) and (ns.taxi.passenger is None):
                reward += 20
                #    print('successful dropoff')
            else:
                reward -= 10  # Reward for successful dropoff
        

        return reward


        
        #return -len(s.waiting_passengers)
    
    def is_absorbing(self, s):
        False
    
class TaxiCabRenderer:
    def __init__(self, taxi_cab: TaxiCab):
        self.taxi_cab = taxi_cab
    
    def as_image(self, s: TaxiCabState, pps: int = 10) -> np.ndarray:
        pix_range = lambda i: slice(i * pps, (i + 1) * pps)
        layout = np.ones((self.taxi_cab.height*pps, self.taxi_cab.width*pps, 3))
        for wall in self.taxi_cab.walls:
            layout[pix_range(wall.location.y), pix_range(wall.location.x), :] = 0
        for stand in self.taxi_cab.taxi_stands:
            layout[pix_range(stand.location.y), pix_range(stand.location.x), :] = [.8, .8, .8]
        for passenger in s.waiting_passengers:
            layout[pix_range(passenger.location.y), pix_range(passenger.location.x), :] = [0, 1, 0]
        taxi_color = [1, 0, 1] if s.taxi.passenger is not None else [.5, 0, .5]
        layout[pix_range(s.taxi.location.y), pix_range(s.taxi.location.x), :] = taxi_color
        return layout
    
    def render(self, s: TaxiCabState, pps: int = 10, ax: Optional[plt.Axes] = None):
        if ax is None:
            fig, ax = plt.subplots()
        layout = self.as_image(s, pps)
        ax.imshow(layout[::-1])
        ax.axis('off')
        ax.set_aspect('equal')