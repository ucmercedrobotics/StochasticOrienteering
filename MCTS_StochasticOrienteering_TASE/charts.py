#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:41:21 2021

@author: shamano
"""

import numpy as np
import matplotlib.pyplot as plt

t = np.arange(1,10000,1)

bperc = 0.1
cperc = 0.2

bfail = 0.15
cfail = 0.08


bs = bperc * t
bf = bfail * bs

cs = cperc * t
cf = cfail * cs

c1 = 2
c2 = 2

bseries = c1 * np.sqrt(np.divide(np.log(t),bs)) + c2 * np.sqrt(np.divide(np.log(t),(bf+1)))
cseries = c1 * np.sqrt(np.divide(np.log(t),cs)) + c2 * np.sqrt(np.divide(np.log(t),(cf+1)))

plt.plot(t,bseries,'b',t,cseries,'r')