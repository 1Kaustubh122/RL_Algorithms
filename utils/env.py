import numpy as np

def GridWorld_5x5():
    """ Creates a state space, action space, and reward structure and returns the same"""
    state_space = [(row, col) for row in range(5) for col in range(5)]
    action = ["up", "down", "right", "left"]
    rewards = {(4,4) : 1}                                       ## Default -1 per step
    return state_space, action, rewards

def reset():
    """ resets the agent position to start from 0,0 """
    return (0,0)

def step(state, action):
    """
    Takes an action, and moves the agent according to it
    """
    next_state = state
    row, col = state
    
    if action == "up":
        # next_state = (row - 1, col) if row - 1 > 0 else (row, col)
        next_state = (max(row - 1, 0), col)
        
    if action == "down":
        # next_state = (row + 1, col) if row + 1 <= 4 else (row, col)
        next_state = (min(row + 1, 4), col)
        
    if action == "right":
        # next_state = (row, col + 1) if col + 1 <= 4 else (row, col)
        next_state = (row, min(col+1, 4))
        status = True
        
    if action == "left":
        # next_state = (row, col - 1) if col -1 > 0 else (row, col)
        next_state = (row, max(col -1, 0))
        status = True
        
    _, _, rewards = GridWorld_5x5()
    reward = rewards.get(next_state, 0)
    
    done = is_terminal(next_state)
    
    return next_state, reward, done
    
def render(curr_state):
    """ Visualization of the grid world with the agen position S = Start, G = Goal, A = Agent's current position"""
    state, _, _ = GridWorld_5x5()
    grid_display = []

    for i, cell in enumerate(state):
        if cell == curr_state:
            grid_display.append(" A ")
        elif cell == (0, 0):
            grid_display.append(" S ")
        elif cell == (4, 4):
            grid_display.append(" G ")
        else:
            grid_display.append(" . ")
            
        if (i + 1) % 5 == 0:
            print("".join(grid_display))
            grid_display = []
        
def is_terminal(state):
    """ Check if agent reaches the goal state """
    return state == (4,4)