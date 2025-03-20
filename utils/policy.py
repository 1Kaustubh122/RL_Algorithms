import numpy as np

class PolicySelection:
    def __init__(self, action_space, epsilon =0.9):
        self.action_space= action_space
        self.epsilon = epsilon
        self.epsilon_policy = {}
        
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
            
    # def behavior_policy(self):
    #     """ Returns an action randomly from thr action space"""
    #     return np.random.choice(self.action_space)
    
    def behavior_policy(self, Q_action_value_table):
        """ Update behavior Policy """
        if np.random.rand() < 0.5:
            return np.random.choice(self.action_space)
        return np.argmax(Q_action_value_table)
            
    
    def target_policy(self, Q_action_value_table):
        """ Returns the greedy action, used as the target policy in off-policy learning"""
        return np.argmax(Q_action_value_table)
    
    def epsilon_greedy_probab(self, Q_value):
        num_actions = len(self.action_space)
        best_action = np.argmax(Q_value)

        action_probabs = np.ones(num_actions) * (self.epsilon/ num_actions)
        action_probabs[best_action] += (1-self.epsilon)

        return action_probabs


