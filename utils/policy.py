import numpy as np

def greedy_sel(Q_action_value_table):
    return np.argmax(Q_action_value_table)
    
def e_greedy_policy(Q_action_value_table, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(Q_action_value_table))
    return np.argmax(Q_action_value_table)
        
def behavior_policy(action_space):
    return np.random.choice(action_space)