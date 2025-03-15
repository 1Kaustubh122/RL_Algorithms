import numpy as np

class PolicySelection:
    def __init__(self, action_space, epsilon =0.9):
        self.action_space= action_space
        self.epsilon = epsilon
        
    def greedy_sel(self, Q_action_value_table):
        """ Returns the action with the highest Q Values (greedy selection)"""
        max_value = np.max(Q_action_value_table)
        max_actions = np.where(Q_action_value_table == max_value)[0]
        return np.random.choice(max_actions)
        
    def e_greedy_policy(self, Q_action_value_table):
        """ Returns an action using e-greedy selection. """
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(Q_action_value_table))
        return np.argmax(Q_action_value_table)
            
    def behavior_policy(self):
        """ Returns an action randomly from thr action space"""
        return np.random.choice(self.action_space)
    
    def target_policy(self):
        """ Returns the greedy action, used as the target policy in off-policy learning"""
        probab = np.ones(len(self.action_space)) / len(self.action_space)
        return np.random.choice(self.action_space, p=probab)



