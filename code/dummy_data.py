import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd


class Shot():
    def __init__(self, state):
        self.current_state = state
    
    def update(state):
        self.current_state = state
        

class State():
    def __init__(self,):
        pass
    
    
class Low(State):
    def __init__(self,):
        self.possible_transitions = {2: 'LD', 3: 'LH'}
        
    def generate_next_state(self,):
        next_state = np.random.choice([Dither(), High()]) #choose a next state as random, from the list of possible next states
        transition = self.possible_transitions[int(next_state)]
        transition_len = np.random.randint(1, 10)
        return next_state, (transition, transition_len)
    
    def generate_data(self, prev_state, trans, prev_data, states_stacked, args):
        return_args = ()
        if trans == '0' or (trans == 'DL' and states_stacked[-3] == 1):
            # print('here1', prev_data, return_args)
            return prev_data, return_args
        elif trans == 'HL' or (trans == 'DL' and states_stacked[-3] == 3):
            # print('here2')
            return 1.1*prev_data, return_args
    
        print('I should not be here L')
        
    
    
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

    def generate_data(self, prev_state, trans, prev_data, states_stacked, args):
        if int(prev_state) != int(self):
            sin_coord = 0
            y_offset = prev_data
            
        else:
            sin_coord = args[0] + 1
            y_offset = args[1]
        
        return y_offset + .5*np.sin((np.pi*(1/10))*sin_coord), (sin_coord, y_offset) #the larger the sine denominator, the larger the period

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
    
    def generate_data(self, prev_state, trans, prev_data, states_stacked, args):
        return_args = ()
        # print(trans, states_stacked[-2])
        if trans == '0' or (trans == 'DH' and states_stacked[-3] == 3):
            # print('here4', prev_data, return_args)
            return prev_data, return_args
        elif trans == 'LH' or (trans == 'DH' and states_stacked[-3] == 1):
            # print('here5')
            return .9*prev_data, return_args
        
        print('I should not be here H')
        # return 0, 1
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
        
    def __int__(self,):
        return 0
    
    def __str__(self,):
        return 'N'

def generate_shot(initial_data_value, max_shot_size=20000, plot=False):
    transitions_seq = []
    states_seq = []
    initial_state = Low()
    state = initial_state
    while len(states_seq) < max_shot_size:
        steps_to_trans = np.random.randint(20, 2000)
        next_state, (transition, transition_len) = state.generate_next_state()
        
        states_seq.extend(np.asarray([state] * steps_to_trans))
        transitions_seq.extend(['0'] * steps_to_trans)
        
        states_seq.extend(np.asarray([next_state] * transition_len))
        transitions_seq.extend([transition] * transition_len)
        
        state = next_state
        
    states_seq = np.asarray(states_seq)
    
    assert len(states_seq) == len(transitions_seq)
    data_vals = [initial_data_value]
    state = NoState()
    args = ()
    states_stacked = [int(state)]
    for k in range(len(states_seq)):
        prev_state = state
        state, trans = states_seq[k], transitions_seq[k]
        if int(state) != int(states_stacked[-1]):
            states_stacked.append(int(state))
        
        # if state is low, check: did I come from H, from D, or from L?
        # print(states_stacked)
        ret = state.generate_data(prev_state, trans, data_vals[-1], states_stacked, args)
        # print(ret)
        new_data_p, args = ret
        
        # print(new_data_p)
        data_vals.append(new_data_p)
    
        
    data_vals = data_vals[1:]
    data_vals = np.asarray(data_vals)
    
    times = np.arange(len(data_vals))
    data_min, data_max = np.min(data_vals), np.max(data_vals)
    # print(states_stacked)
    if plot:
        plt.plot(times, data_vals)
        plt.fill_between(times, data_min, data_max, where = states_seq.astype(int) == 1, facecolor='g', alpha=0.2)
        plt.fill_between(times, data_min, data_max, where = states_seq.astype(int) == 3, facecolor='r', alpha=0.2)
        plt.fill_between(times, data_min, data_max, where = states_seq.astype(int) == 2, facecolor='y', alpha=0.2)
        plt.show()
    
    return data_vals, states_seq.astype(int)


def dummy_data(args):
    # print('args')
    num_shots_to_simulate = int(args[0])
    
    print('Generating', num_shots_to_simulate, 'shots.')
    max_shot_size = 20000
    
    plot = False
    for s in range(num_shots_to_simulate):
        d = np.random.randint(3,10)
        # print(s, plot)
        # print(s)
        if s == num_shots_to_simulate - 1:
            plot = True
        signal, labels = generate_shot(initial_data_value = d, max_shot_size=max_shot_size, plot=plot)
        if not (signal > 0).all():
            # print('skipped one')
            s -= 1
            continue
        df = pd.DataFrame({'PD': signal,
                           'IP': signal,
                           'FIR': signal,
                           'DML': signal,
                           'time': np.arange(len(signal))/10000,
                           'LHD_label': labels})
        df.to_csv('../data/DUMMY_MACHINE/dummy/DUMMY_MACHINE_' + str(s).zfill(5) + '_dummy_labeled.csv')

if __name__ == '__main__':
    dummy_data(sys.argv[1:])