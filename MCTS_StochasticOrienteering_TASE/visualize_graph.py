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
import glob


file_list = glob.glob('../datasets/*.mat')

for fname in file_list:

    og = graph.OrienteeringGraph(fname)
    bn = os.path.basename(fname)
 
    NVERTICES = og.get_n_vertices()
    x = [og.vertices[i].x for i in og.vertices.keys()]
    y = [og.vertices[i].y for i in og.vertices.keys()]

  #  fig = plt.figure(figsize = (5,5))
    fig = plt.figure()
    ax = plt.gca()
   # for i  in range(len(x)):
    plt.plot(x,y,'o',markersize=10)
    # if min(x) > 0:
    #     plt.xlim([0.9*min(x),1.1*max(x)])
    # else:
    #     plt.xlim([1.1*min(x),1.1*max(x)])
        
    # if min(y) > 0:
    #     plt.ylim([0.9*min(y),1.1*max(y)])
    # else:
    #     plt.ylim([1.1*min(y),1.1*max(y)])    
        
    #ax.set_aspect('equal','box')
        
    #plt.ylim([0.95*min(y),1.05*max(y)])
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    plt.axis('equal')

    plt.grid()
    plt.title(bn.rsplit( ".", 1 )[ 0 ] )
    plt.plot(x[0],y[0],'ro')
    plt.plot(x[NVERTICES-1],y[NVERTICES-1],'ko')
    
    
    
    fig.savefig(bn.rsplit( ".", 1 )[ 0 ] +'.pdf')
#fig.tight_layout()

#ax = fig.gca()
#for i in range(len(x)):
#    ax.text(x[i],y[i],'{}'.format(i))
