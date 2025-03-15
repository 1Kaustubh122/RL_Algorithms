import numpy as np

class Enviornment:
    """ Creates a state space, action space, and reward structure and returns the same"""
    def __init__(self, size=5):
        self.size = size
        self.state_space = [(row, col) for row in range(size) for col in range(size)]
        self.action_space = ["up", "down", "right", "left"]
        self.rewards = {state: -1 for state in self.state_space}
        self.rewards[(size-1,size-1)] = 1                                       ## Default -1 per step
        self.state = (0, 0)

    def reset(self):
        """ resets the agent position to start from 0,0 """
        self.state = (0, 0)
        return self.state
        
    def step(self, action_idx):
        """
        Takes an action, and moves the agent according to it
        """
        next_state = self.state
        row, col = self.state
        
        new_row, new_col = row, col
        
        action = self.action_space[action_idx]


        if action not in self.action_space:
            raise ValueError(f"{type(action)}({action}) : Action not found")
        elif action == "up" and row > 0:
            # next_state = (row - 1, col) if row - 1 > 0 else (row, col)
            new_row -= 1
        elif action == "down" and row < self.size -1:
            # next_state = (row + 1, col) if row + 1 <= self.size else (row, col)
            new_row += 1
        elif action == "right" and col < self.size - 1:
            # next_state = (row, col + 1) if col + 1 <= self.size else (row, col)
            new_col += 1
        elif action == "left" and col > 0:
            # next_state = (row, col - 1) if col -1 > 0 else (row, col)
            new_col -= 1
        

        
        next_state = (new_row, new_col)        
        self.state = next_state
        reward = self.rewards.get(next_state, -1)
        
        done = self.is_terminal(self.state)
        return self.state, reward, done
    
    def render(self, verbose=True):
        """ Visualization of the grid world with the agent position 
        S = Start, 
        G = Goal, 
        A = Agent's current position
        """
        if not verbose:
            return
        
        for row in range(self.size):
            line = ""
            for col in range(self.size):
                if (row, col) == self.state:
                    line += " A "
                elif (row, col) == (0, 0):
                    line += " S "
                elif (row, col) == (self.size - 1, self.size -1):
                    line += " G "
                else:
                    line += " . "
            print(line)
        print()
            
    def is_terminal(self, state):
        """ Check if agent reaches the goal state """
        return state == (self.size -1, self.size -1)


