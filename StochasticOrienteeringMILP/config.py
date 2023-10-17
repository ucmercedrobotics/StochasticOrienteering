#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:06:40 2022
@author: shamano
"""

## General config values that can be imported between all MCTS Applications
## Specific configs can be found in respective config.txt files.

Q_VALUE = 100
NVERTICES=20
M_CONST = 1000
N_SIM = 1000
ALPHA =0.1
FAILURE_PROBABILITY = 0.1
SOLVER='Gurobi'
BUDGET=2
REPEATS=10