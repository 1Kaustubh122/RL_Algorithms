import numpy as np

class PolicySelection:
    def __init__(self, action_space, policy_type, epsilon =0.9):
        self.action_space= action_space
        self.epsilon = epsilon
        self.policy_type = policy_type
        # self.epsilon_policy = {}
    
    def select_action(self, state, policy_type, Q=None):
        if policy_type == "random":
            return np.random.choice(self.action_space)
        
        elif policy_type == "fixed_action":
            return 0
        
        elif Q is not None:
            if policy_type == "e-greedy":
                return self.e_greedy_policy(Q[state])
            
            elif policy_type == "off_policy_behavior":
                return self.behavior_policy()
            
            elif policy_type == "greedy":
                return self.greedy_sel(Q[state])
            
            elif policy_type == "off_policy_target":
                return self.target_policy(Q[state])
            
            else:
                raise ValueError("Policy not Found")
                
        
        else:
            raise ValueError("Either Q-Value is None or policy not Found")
            
    def get_policy_probab_e_greedy(self, state, action, policy_type, Q=None):
        num_actions = len(self.action_space)
        
        best_action = np.argmax(Q[state])
        if action == best_action:
            return 1 - self.epsilon
        else:
            return self.epsilon/ num_actions


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
    
    # def behavior_policy(self, Q_action_value_table):
    #     """ Update behavior Policy """
    #     if np.random.rand() < 0.5:
    #         return np.random.choice(self.action_space)
    #     return np.argmax(Q_action_value_table)
            
    
    def target_policy(self, Q_action_value_table):
        """ Returns the greedy action, used as the target policy in off-policy learning"""
        return np.argmax(Q_action_value_table)
    


