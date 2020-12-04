import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Shot():
    def __init__(self, state):
        self.current_state = state
    
    def update(state):
        self.current_state = state
        

class State():
    def __init__(self,):
        pass
    
    # def get_all_states():
    #     return np.arange(3)

    # def generate_next_state(self,):
        # next_state = np.random.choice(self.next_states) #choose a next state as random, from the list of possible next states
        # transition = self.possible_transitions[next_state]
        # transition_len = np.random.randint(1, 10)
        # 
        # return next_state_obj, (transition, transition_len)
    
# class Transition():
#     transition_dictionary = {'0':0, 'LH':1, 'HL':2, 'DL':3, 'LD':4, 'HD':5, 'DH':6}
#     def __init__(self,):
#         pass
#     
#     # def possible_transitions():
#     #     possible_transitions = {1: ['LH', 'LD'], 2: ['DL', 'DH'], 3: ['HL', 'HD'],}
#     def get_trans_code(trans_name):
#         return Transition.transition_dictionary[trans_name]
    
    
class Low(State):
    def __init__(self,):
        # self.next_states = [2, 3]
        self.possible_transitions = {2: 'LD', 3: 'LH'}
        
    def generate_next_state(self,):
        next_state = np.random.choice([Dither(), High()]) #choose a next state as random, from the list of possible next states
        transition = self.possible_transitions[int(next_state)]
        transition_len = np.random.randint(1, 10)
        return next_state, (transition, transition_len)
    
    def generate_data(self, trans, prev_data):
        if trans in ['0', 'DL']:
            return prev_data[0]
        else:
            return 1.3*prev_data[0]
    
    def __int__(self,):
        return 1
    
    def __str__(self,):
        return 'L'
    
class Dither(State):
    
    def __init__(self,):
        self.next_states = [1, 3]
        self.possible_transitions = {1: 'DL', 3: 'DH'}
    
    def generate_next_state(self,):
        next_state = np.random.choice([High(), Low()]) #choose a next state as random, from the list of possible next states
        transition = self.possible_transitions[int(next_state)]
        transition_len = np.random.randint(1, 10)
        return next_state, (transition, transition_len)

    def generate_data(self, trans, prev_data):
        # if trans in ['LD', 'HD']:
        #     return prev_data[0]
        return np.sin(2)/np.sin(1)*np.sin(prev_data[0])-np.sin(prev_data[1])

    def __int__(self,):
        return 2
    
    def __str__(self,):
        return 'D'
    
class High(State):
    def __init__(self,):
        self.next_states = [2, 1]
        self.possible_transitions = {1: 'HL', 2: 'HD'}

    def generate_next_state(self,):
        next_state = np.random.choice([Low(), Dither()]) #choose a next state as random, from the list of possible next states
        transition = self.possible_transitions[int(next_state)]
        transition_len = np.random.randint(1, 10)
        return next_state, (transition, transition_len)
    
    def generate_data(self, trans, prev_data):
        if trans in ['0', 'DH']:
            return prev_data[0]
        else:
            return .8*prev_data[0]
    
    def __int__(self,):
        return 3
    
    def __str__(self,):
        return 'H'
    
#dummy initial state, never used
class NoState(State):
    def __init__(self,):
        self.state_int_code = 0
        self.state_str_code = 'None'
        self.next_states = [1, 2, 3]
        # self.possible_transitions = {1: 'HL', 2: 'HD'}

    # def __int__(self,):
    #     return 0
    # 
shot_size = 20000
initial_state = 1 # Low
all_states = np.arange(3)
print('All states:', all_states)
# since_last = 0
current_state = initial_state
states = []

# while len(states) < shot_size:
#     state_len = np.random.randint(100, 2000) #minimal length of 100, maximum of 1000
#     states.extend(np.ones(state_len) * current_state)
#     possible_next = np.delete(all_states, current_state) #next state cannot be the same as the one that just ended
#     current_state = np.random.choice(possible_next) #choose a next state as random, from the list of possible next states
#     
# states = np.asarray(states)
# print(states.shape)
# plt.plot(range(len(states)), states)
# plt.show()


transitions_seq = []
states_seq = []
# possible_transitions = {1: ['LH', 'LD'], 2: ['DL', 'DH'], 3: ['HL', 'HD'],} #'0',
# states_list = [NoState(), Low(), Dither(), High()]
initial_state = Low()
state = initial_state
# print(int(state))
# exit(0)
while len(states_seq) < shot_size:
    steps_to_trans = np.random.randint(100, 2000)
    # next_trans = np.random.choice(possible_transitions[initial_state])
    # print(state)
    next_state, (transition, transition_len) = state.generate_next_state()
    # print(state, next_state)
    # if next_trans
    
    states_seq.extend(np.asarray([state] * steps_to_trans))
    transitions_seq.extend(['0'] * steps_to_trans)
    
    states_seq.extend(np.asarray([next_state] * transition_len))
    transitions_seq.extend([transition] * transition_len)
    
    state = next_state
    
states_seq = np.asarray(states_seq)
# transitions = np.asarray()
# print(states_seq.astype(int))
# exit(0)
# plt.plot(range(len(states_seq)), states_seq.astype(int))
# plt.yticks([1,2,3])
# plt.show()

assert len(states_seq) == len(transitions_seq)
initial_data_value = 1
data_vals = [initial_data_value, initial_data_value]
for k in range(len(states_seq)):
    state, trans = states_seq[k], transitions_seq[k]
    
    # if state is dither, sinusoid of last value
    
    # if state is low, check: did I come from H, from D, or from L?
    
    new_data_p = state.generate_data(trans, (data_vals[-1], data_vals[-2]))
    # print(new_data_p)
    data_vals.append(new_data_p)
data_vals = data_vals[2:]
data_vals = np.asarray(data_vals)

times = np.arange(len(data_vals))
plt.plot(times, data_vals)
plt.fill_between(times, -2., 2., where = states_seq.astype(int) == 1, facecolor='g', alpha=0.2)
plt.fill_between(times, -2., 2., where = states_seq.astype(int) == 3, facecolor='r', alpha=0.2)
plt.fill_between(times, -2., 2., where = states_seq.astype(int) == 2, facecolor='y', alpha=0.2)
plt.show()
