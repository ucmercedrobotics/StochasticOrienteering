#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 17:41:14 2021

@author: shamano
"""

# Produces the figure in the paper for the time dependency

import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import configparser



PARTIAL = False

FAILURE_RATE_CONSTRAINTS = 0.100001

EPSMIN = 0
EPSN = 0
EPSINC = 0
ITERMIN = 0
ITERINC = 0
ITERN = 0
NTRIALS = 0

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = '15'



def read_configuration(fname):
    config = configparser.ConfigParser()
    print("Reading configuration file ",fname)
    config.read(fname)
    global NVERTICES,EPSMIN,EPSINC,EPSN,ITERMIN,ITERINC,ITERN,NTRIALS
         
    if config['MAIN']['NVERTICES'] is None:
        print('Missing configuration parameter ',NVERTICES)
    else:
         NVERTICES = int(config['MAIN']['NVERTICES'])
                  
    if config['MAIN']['EPSMIN'] is None:
        print('Missing configuration parameter ',EPSMIN)
    else:
         EPSMIN = float(config['MAIN']['EPSMIN']) 
         
    if config['MAIN']['EPSN'] is None:
        print('Missing configuration parameter ',EPSN)
    else:
         EPSN = int(config['MAIN']['EPSN']) 
        
    if config['MAIN']['EPSINC'] is None:
        print('Missing configuration parameter ',EPSINC)
    else:
         EPSINC = float(config['MAIN']['EPSINC']) 
         
    if config['MAIN']['ITERMIN'] is None:
        print('Missing configuration parameter ',ITERMIN)
    else:
         ITERMIN = int(config['MAIN']['ITERMIN']) 
         
    if config['MAIN']['ITERINC'] is None:
        print('Missing configuration parameter ',ITERINC)
    else:
         ITERINC = int(config['MAIN']['ITERINC']) 
        
    if config['MAIN']['ITERN'] is None:
        print('Missing configuration parameter ',ITERN)
    else:
         ITERN = int(config['MAIN']['ITERN']) 
         
    if config['MAIN']['NTRIALS'] is None:
        print('Missing configuration parameter ',NTRIALS)
    else:
         NTRIALS = int(config['MAIN']['NTRIALS'])
         
    print('Done reading configuration')



if __name__ == "__main__":
    
    # read first data series
    # Old code
   # NVERTICES = 20
   # PREFIX= 6

   # read_configuration("Data/Graph{}/DataSet{}/config.txt".format(NVERTICES,PREFIX))
   # iterations_list =  [(ITERMIN + i*ITERINC) for i in range(ITERN)]
   # complete_time = pickle.load(open("Data/Graph{}/DataSet{}/complete_time_series.dat".format(NVERTICES,PREFIX),"rb"))
   
   
    NVERTICES = 10
    PREFIX = 19
   # read_configuration("Revised/{}/{}/config.txt".format(NVERTICES,PREFIX))
    read_configuration("TASE/Fig4/config.txt")
    iterations_list =  [(ITERMIN + i*ITERINC) for i in range(ITERN)]
    complete_time = pickle.load(open("TASE/Fig4/complete_time_series.dat","rb"))
    
    time_series = np.zeros((len(iterations_list),))
    time_deviation = np.zeros((len(iterations_list),))
    time_min = np.zeros((len(iterations_list),))
    time_max = np.zeros((len(iterations_list),))
    start = 0
    for i in range(len(iterations_list)):
        aslice = complete_time[start:start+NTRIALS]
        start += NTRIALS
        time_series[i] = np.mean(aslice)
        time_deviation[i] = np.std(aslice)
        time_min[i] = np.min(aslice)
        time_max[i] = np.max(aslice)
        
   # iterations_list = iterations_list[0:76]
   # time_series = time_series[0:76]
   # time_deviation = time_deviation[0:76]
        
    figtime, axtime = plt.subplots()
    color = 'tab:red'
    colormin = 'tab:blue'
    colormax = 'tab:green'
    axtime.set_ylabel('time ($s$)')
    axtime.set_xlabel('iterations ($K$)')
    iterations_a = np.array(iterations_list)
    #axtime.errorbar(iterations_a,time_series,yerr=time_deviation,color=color,linewidth=4)
    axtime.plot(iterations_a,time_series,color=color,linewidth=2)
    
    axtime.fill_between(iterations_a, time_series-time_deviation, time_series+time_deviation , alpha=0.3,facecolor=colormax)
    
    #axtime.plot(iterations_a,time_series - time_deviation,color=colormin,linewidth=2)
    #axtime.plot(iterations_a,time_series + time_deviation,color=colormin,linewidth=2)
    axtime.tick_params(axis='y')
    figtime.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(b=True,which='major',axis='both')
    plt.ylim(0,1.05*np.max(time_series + time_deviation))
    plt.show()
    figtime.savefig('time_dependency_new.pdf')