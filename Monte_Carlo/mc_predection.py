import os
import random
import sys
import numpy as np
import matplotlib.pyplot as plt


"""
Implementation of First-Visit Monte Carlo and Every-Visit Monte Carlo

First-Visit:
    This method estimates value function using sample returns from the full episodes.
    It considers only first time a state 's' is visitited in an episode for updating the value function

    For each state, it computes G (sum of discounted rewards from that point onwards)
    It updates V(s) by taking average of all first visit returns.

    Drawbacks: Ignore extra info when states are visited multiple times,
            Requires full episodes.

Every-Visit:

"""

os.makedirs("Monte_Carlo/Results", exist_ok=True)

