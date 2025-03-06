import numpy as np

state_space = [(i, j) for i in range(3) for j in range(3)]
val_pi_s = {}
rewards = {(0,2): 1, (2,2):-1}
episode_lst = []
num_visits = {}

print(state_space)
for liner_st in state_space:
    for state in liner_st:
        num_visits[state] = 0
        val_pi_s[state] = 0
        
        if state in rewards.keys():
            val_pi_s[state] = rewards.get(state)
            
            
print(num_visits, val_pi_s)