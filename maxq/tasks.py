#v can take into account whole state or just partial state
#task_list = [root(),put(),get(),navigate(),pickup(),putdown(),north(),south(),east(),west()]
import random
class task:
    def __init__(self):
        self.state = None
        self.active_state = None
        self.is_primitive = False
        self.t = None
        self.action_counts = {}
        self.cv = {}
        
        self.ct = {}
        
        self.alpha = 0.95
        self.epsilon = 0.1
        self.epsilon_decay = 0.9999
        
        self.reward_value = 10
        self.penalty_value = -1
        
    def update_state(self,state):
        self.state = state
        
    def init_av(self,state):
        if state not in self.action_counts.keys():
            self.action_counts[state] = {}
        for action in self.available_actions:
            if action not in self.action_counts[state].keys():
                self.action_counts[state][action] = 0

    def egreedy_policy(self, state,ct_on = False):

        if state.taxi.passenger is not None:
            self.t = state.taxi.passenger.destination
        else:
            self.t = state.waiting_passengers[0].location
        # If the state is not in self.cv or it's a new state with no actions, return a random action
        if state not in self.cv.keys() or random.random() < self.epsilon:
            act = random.choice(self.available_actions)
            return act,self.t,0 #change to substate for abstraction

        else:
            if ct_on:
                max_value = max(self.ct[state].values())
                max_actions = [action for action, value in self.ct[state].items() if value == max_value]
            else:
                max_value = max(self.cv[state].values())
                max_actions = [action for action, value in self.cv[state].items() if value == max_value]
            act = random.choice(max_actions) #HUGE DIFFERENCE IF RANDOM OR IN ORDER
            return act,self.t, len(max_actions)
            
    def init_cv(self, state, action):


        # Check if the state key exists in self.cv; if not, create a new dictionary for it.
        if state not in self.cv.keys():
            self.cv[state] = {}
            self.ct[state] = {}
            #self.action_counts[state] = {}
            if self.available_actions is not None and self.is_primitive == False:
                for a in self.available_actions:
                    #self.action_counts[state][a] = 0
                    self.cv[state][a] = 0
                    self.ct[state][a] = 0

        # Init state[action] for prims
        if action not in self.cv[state]:
            self.cv[state][action] = 0 #or random
            self.ct[state][action] = 0
    def termination_condition(self,state,next_state):
        pass
            
    def reward(self,state,next_state):
        if self.termination_condition(state,next_state):
            return self.reward_value
        else:
            return self.penalty_value #or mabee -1?
        #else can return some sort of fail value
            
            
class root(task):
    def __init__(self):
        super().__init__()
        self.available_actions = [1,2] #put,get
    def update_state(self,state):
        self.state = state
        self.active_state = [state]
    def termination_condition(self,state,next_state):
        if state.taxi.passenger is not None and next_state.taxi.passenger is None:
            return True #!
        else:
            return False
    
    def random_policy(self,state): 
        self.t = None
        if state.taxi.passenger is None:
            return 2,self.t
        else:
            return 1,self.t
    def egreedy_policy(self, state,ct_on=False):
        self.t = None
        if state.taxi.passenger is None:
            return 2,self.t, 0
        else:
            return 1,self.t,0
        
    
    
class get(task): #needs to know where taxi is, if waiting passengers, where waiting passengers are
    def __init__(self):
        super().__init__()
        self.reward_value = 15
        self.available_actions = [3,4] #navigate,pickup
    def update_state(self,state):
        self.state = state
        waiting_passengers = None
        if len(state.waiting_passengers) > 0:
            waiting_passengers = [passenger.location for passenger in state.waiting_passengers]
        self.active_state = [state.taxi.location, waiting_passengers] #just locations
    def termination_condition(self,state,next_state):
        if next_state.taxi.passenger is not None:
            #print('get Terminated')
            return True
        else:
            return False


class put(task): #needs to know where taxi is, passengers in taxi, destinations
    def __init__(self):
        super().__init__()
        self.reward_value = 15
        self.available_actions = [3,5]#navigate, putdown    

    def update_state(self,state):
        self.state = state
        destination = None
        if state.taxi.passenger is not None:
            destination = state.taxi.passenger.destination
        self.active_state = [state.taxi.location,destination] #just destination
        
    def termination_condition(self,state,next_state):
        if state.taxi.passenger is not None and next_state.taxi.passenger is None:
            return True
        else:
            return False        
    
class navigate(task): #just needs to know where taxi is and where destination is
    def __init__(self):
        super().__init__()
        self.available_actions = [6,7,8,9]#north,south,east,west
        self.t = None
        self.reward_value = -1
        self.penalty_value = -1
        
    def update_state(self,state,t):
        self.state = state
        self.t = t
        self.active_state = [state.taxi.location,t]
    def termination_condition(self,state,next_state):
        if next_state.taxi.location == self.t:
            return True
        else:
            return False
class primitive_task(task):
    def __init__self(self):
        super().__init__()
        self.reward_value = -1
        self.penalty_value = -1
    def egreedy_policy(self,state,ct_on = False):
        print('inheritance working')
        t = random.choice(taxicab.legal_locations)
        return self.available_actions[0],t
        
class north(primitive_task):
    def __init__(self):
        super().__init__()
        self.available_actions = [12]
        self.is_primitive = True
class south(primitive_task):
    def __init__(self):
        super().__init__()
        self.available_actions = [13]
        self.is_primitive = True
class east(primitive_task):
    def __init__(self):
        super().__init__()
        self.available_actions = [10]
        self.is_primitive = True
class west(primitive_task):
    def __init__(self):
        super().__init__()
        self.available_actions = [11]
        self.is_primitive = True
class pickup(primitive_task):
    def __init__(self):
        super().__init__()
        self.available_actions = [14]
        self.is_primitive = True
        self.reward_value = -1
        self.penalty_value = -1
    def termination_condition(self,state,next_state):
        if next_state.taxi.passenger is not None:
            return True
        else:
            return False
class putdown(primitive_task):
    def __init__(self):
        super().__init__()
        self.available_actions = [15]
        self.is_primitive = True
        self.reward_value = -1
        self.penalty_value = -1
    def termination_condition(self,state,next_state):
        if state.taxi.passenger is not None and next_state.taxi.passenger is None:
            return True
        else:
            return False
    

