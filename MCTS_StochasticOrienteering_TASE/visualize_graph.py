#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:42:44 2022

@author: shamano
"""



import sys

sys.path.append('../OrienteeringGraph')

import graph
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from progress.bar import Bar
import configparser
import argparse
import time
import os.path
import os
import shutil


VERTICES_LIST = [10,20,30,40]

for NVERTICES in VERTICES_LIST:

    og = graph.OrienteeringGraph('graph_test_{}.mat'.format(NVERTICES))
 

    x = [og.vertices[i].x for i in og.vertices.keys()]
    y = [og.vertices[i].y for i in og.vertices.keys()]

    fig = plt.figure(figsize = (5,5))
   # for i  in range(len(x)):
    plt.plot(x,y,'o',markersize=10)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    #plt.axis('equal')

    plt.grid()
    plt.plot(x[0],y[0],'ro')
    plt.plot(x[NVERTICES-1],y[NVERTICES-1],'ko')
    
    fig.savefig('graph_{}.pdf'.format(NVERTICES))
#fig.tight_layout()

#ax = fig.gca()
#for i in range(len(x)):
#    ax.text(x[i],y[i],'{}'.format(i))
