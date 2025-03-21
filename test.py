# import numpy as np

# state_space = [(i, j) for i in range(3) for j in range(3)]
# val_pi_s = {}
# rewards = {(0,2): 1, (2,2):-1}
# episode_lst = []
# num_visits = {}

# print(state_space)
# for liner_st in state_space:
#     for state in liner_st:
#         num_visits[state] = 0
#         val_pi_s[state] = 0
        
#         if state in rewards.keys():
#             val_pi_s[state] = rewards.get(state)
            
            
# print(num_visits, val_pi_s)


# state = [(row, col) for row in range(5) for col in range(5) ]
# curr_state = (0,1)

# grid_display = []

# for i, cell in enumerate(state):
#     if cell == curr_state:
#         grid_display.append(" A ")
#     elif cell == (0, 0):
#         grid_display.append(" S ")
#     elif cell == (4, 4):
#         grid_display.append(" G ")
#     else:
#         grid_display.append(" . ")
        
#     if (i + 1) % 5 == 0:
#         print("".join(grid_display))
#         grid_display = []



"""
# = Done

To Do:
# 1. K armed bandit
# 2. MDP
3. Monte Carlo
    - First Visit, Every Visit MC, Off Policy MC Predection
    - On Policy, Off Policy MC Control
4. Temporal Difference (TD)
"""