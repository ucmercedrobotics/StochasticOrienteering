#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:56:40 2022

@author: shamano
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

B=2 
P=0.05

MCTS = [ 2.5749,3.2060,6.0735, 7.9928]
CMDP= [1.7893,2.3127,  2.4282,5.6137]

N = 4

ind = np.arange(N)
# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, MCTS , width, label='MCTS')
plt.bar(ind + width, CMDP, width, label='CMDP')

plt.xlabel('Number of vertices')
plt.ylabel('Average Reward')
plt.title('Budget = {}, $P_f$={}'.format(B,P))

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, (10,20,30,40))

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()

B=2 
P=0.1

MCTS = [ 2.7383,3.6336,6.3150, 8.2123 ]
CMDP= [2.3346,2.9767 ,   5.1459,6.6671]

N = 4

ind = np.arange(N)
# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, MCTS , width, label='MCTS')
plt.bar(ind + width, CMDP, width, label='CMDP')

plt.xlabel('Number of vertices')
plt.ylabel('Average Reward')
plt.title('Budget = {}, $P_f$={}'.format(B,P))

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, (10,20,30,40))

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()

B=3 
P=0.05

MCTS = [ 3.0872, 4.9909, 7.9702,10.5365 ]
CMDP= [3.0685,4.0719,  7.3309,8.7362 ]

N = 4

ind = np.arange(N)
# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, MCTS , width, label='MCTS')
plt.bar(ind + width, CMDP, width, label='CMDP')

plt.xlabel('Number of vertices')
plt.ylabel('Average Reward')
plt.title('Budget = {}, $P_f$={}'.format(B,P))

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, (10,20,30,40))

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()

B=3 
P=0.1

MCTS = [ 3.0887, 5.5328 ,  8.0259, 10.4461 ]
CMDP= [ 3.0878, 4.8006 ,  8.0893 , 9.6842]

N = 4

ind = np.arange(N)
# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, MCTS , width, label='MCTS')
plt.bar(ind + width, CMDP, width, label='CMDP')

plt.xlabel('Number of vertices')
plt.ylabel('Average Reward')
plt.title('Budget = {}, $P_f$={}'.format(B,P))

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, (10,20,30,40))

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()