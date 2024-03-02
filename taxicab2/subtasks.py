from typing import Tuple, Union
from itertools import product

from gorgo import Categorical, infer, draw_from
from base import Task
from taxicab import TaxiCab, TaxiCabAction, TaxiCabState

class Navigate(TaxiCab):
    def __init__(self, destination):
        assert destination in self.stands
        self.destination = destination

    @infer
    def init_state_dist(self) -> TaxiCabState:
        return TaxiCabState(
            tloc=draw_from(product(range(5), range(5))),
            done=False
        )
    
    def next_state_reward(
        self,
        state: TaxiCabState,
        action: TaxiCabAction
    ) -> Categorical[Tuple[TaxiCabState, float]]:
        ntloc = self.move_taxi(state.tloc, action)
        reached_destination = ntloc == self.destination
        reward = self.move_reward(action)
        next_state = TaxiCabState(
            tloc=ntloc,
            done=reached_destination
        )
        return Categorical.from_dict({(next_state, reward): 1})


class Get(TaxiCab):
    actions = \
        TaxiCab.actions + \
        [Navigate(d) for d in TaxiCab.stands]
    
    @infer
    def init_state_dist(self) -> TaxiCabState:
        # do we need to know the taxi's location?
        return TaxiCabState(
            ploc=draw_from(self.stands),
            done=False
        )

    def next_state_reward(
        self,
        state: TaxiCabState,
        action: Union[TaxiCabAction, Task]
    ) -> Categorical[Tuple[TaxiCabState, float]]:
        if isinstance(action, Task):
            # need to compute the resulting semi-MDP transition
            raise NotImplementedError
        nploc = self.interact_with_passenger(state.ploc, state.tloc, action)
        passenger_picked_up = state.ploc == "in_taxi"
        # is this a pseudo-reward?
        reward = -1 
        next_state = TaxiCabState(
            ploc=nploc,
            done=passenger_picked_up
        )
        return Categorical.from_dict({(next_state, reward): 1})
    
class Drop(TaxiCab):
    actions = \
        TaxiCab.actions + \
        [Navigate(d) for d in TaxiCab.stands]
    
    @infer
    def init_state_dist(self) -> TaxiCabState:
        raise NotImplementedError

    def next_state_reward(
        self,
        state: TaxiCabState,
        action: Union[TaxiCabAction, Task]
    ) -> Categorical[Tuple[TaxiCabState, float]]:
        raise NotImplementedError

class Root(TaxiCab):
    actions = \
        TaxiCab.actions + \
        [Get(), Drop()]
    
    @infer
    def init_state_dist(self) -> TaxiCabState:
        raise NotImplementedError
    
    def next_state_reward(
        self,
        state: TaxiCabState,
        action: Union[TaxiCabAction, Task]
    ) -> Categorical[Tuple[TaxiCabState, float]]:
        raise NotImplementedError