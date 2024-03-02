
from gorgo import flip, infer, draw_from, Categorical
from itertools import product
from dataclasses import dataclass
from typing import Tuple, Set, List, Hashable, Generic, TypeVar, Optional, Protocol, Dict

from base import Task

@dataclass(frozen=True, order=True)
class TaxiCabState:
    tloc: Tuple[int, int] = None
    ploc: str = None
    pdest: str = None
    done: bool = None

@dataclass(frozen=True)
class TaxiCabAction:
    move: str = None
    pickup: bool = None
    dropoff: bool = None

class TaxiCab(Task):
    stands = [(2, 0), (0, 4)]
    actions = [
        *[{"move": move} for move in ["north", "south", "east", "west"]],
        *[{"pickup": pickup} for pickup in [True, False]],
        *[{"dropoff": dropoff} for dropoff in [True, False]]
    ]
    actions = [TaxiCabAction(**a) for a in actions]

    def move_taxi(self, tloc, action: TaxiCabAction):
        ntx, nty = tloc
        if action.move == "north":
            nty += 1
        elif action.move == "south":
            nty -= 1
        elif action.move == "east":
            ntx += 1
        elif action.move == "west":
            ntx -= 1
        if 0 <= ntx < 5 and 0 <= nty < 5:
            return (ntx, nty)
        return tloc

    def interact_with_passenger(self, ploc, tloc, action: TaxiCabAction):
        in_taxi = ploc == "in_taxi"
        at_stand = tloc in self.stands
        if in_taxi:
            if at_stand and action.dropoff:
                return tloc
        if not in_taxi:
            if tloc == ploc and action.pickup:
                return "in_taxi"
        return ploc

    def move_reward(self, action: TaxiCabAction):
        if action.move in ["north", "south", "east", "west"]:
            return -1
        return 0

    def dropoff_reward(self, ploc, nploc, pdest, action: TaxiCabAction):
        if ploc == "in_taxi" and nploc == pdest and action.dropoff:
            return 20
        return 0

    @infer
    def init_state_dist(self) -> TaxiCabState:
        return TaxiCabState(
            tloc=draw_from(product(range(5), range(5))),
            ploc=draw_from(self.stands),
            pdest=draw_from(self.stands),
            done=False
        )
    
    def next_state_reward(
            self,
            state: TaxiCabState,
            action: TaxiCabAction
        ) -> Categorical[Tuple[TaxiCabState, float]]:
        ntloc = self.move_taxi(state.tloc, action)
        nploc = self.interact_with_passenger(state.ploc, state.tloc, action)
        npdest = state.pdest
        passenger_dropped_off = state.ploc == "in_taxi" and nploc == state.pdest
        reward = \
            self.move_reward(action) + \
            self.dropoff_reward(state.ploc, nploc, state.pdest, action)
        next_state = TaxiCabState(
            tloc=ntloc,
            ploc=nploc,
            pdest=npdest,
            done=passenger_dropped_off
        )
        return Categorical.from_dict({(next_state, reward): 1})