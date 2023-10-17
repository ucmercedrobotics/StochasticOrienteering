#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:53:18 2022

@author: shamano
"""

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
matplotlib.rcParams['font.size'] = '16'

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
    # Settings for original submission
    #NVERTICES = 20
    #PREFIX= 8
    #read_configuration("Data/Graph{}/DataSet{}/config.txt".format(NVERTICES,PREFIX))
    
    # New Settings
    NVERTICES = 20
    
   # read_configuration("Revised/{}/{}/config.txt".format(NVERTICES,PREFIX))
    read_configuration("TASE/Iterations/{}/config.txt".format(NVERTICES))
    
    iterations_list1 =  [(ITERMIN + i*ITERINC) for i in range(ITERN)]
    complete_reward1 = pickle.load(open("TASE/Iterations/{}/complete_reward_series.dat".format(NVERTICES),"rb"))
    reward_series1 = np.zeros((len(iterations_list1),))
    reward_deviation1 = np.zeros((len(iterations_list1),))
    reward_min1 = np.zeros((len(iterations_list1),))
    reward_max1 = np.zeros((len(iterations_list1),))
    start = 0
    for i in range(len(iterations_list1)):
        aslice = complete_reward1[start:start+NTRIALS]
        start += NTRIALS
        reward_series1[i] = np.mean(aslice)
        reward_min1[i] = np.min(aslice)
        reward_max1[i] = np.max(aslice)
        reward_deviation1[i] = np.std(aslice)
    
    # read second data series
    # Settings for original submission
   # NVERTICES = 40
    #PREFIX= 2
    #read_configuration("Data/Graph{}/DataSet{}/config.txt".format(NVERTICES,PREFIX))
    
    # New Settings
    NVERTICES = 30
    
    read_configuration("TASE/Iterations/{}/config.txt".format(NVERTICES))
    
    iterations_list2 =  [(ITERMIN + i*ITERINC) for i in range(ITERN)]
    complete_reward2 = pickle.load(open("TASE/Iterations/{}/complete_reward_series.dat".format(NVERTICES),"rb"))
    reward_series2 = np.zeros((len(iterations_list2),))
    reward_deviation2 = np.zeros((len(iterations_list2),))
    reward_min2 = np.zeros((len(iterations_list2),))
    reward_max2 = np.zeros((len(iterations_list2),))
    start = 0
    for i in range(len(iterations_list2)):
        aslice = complete_reward2[start:start+NTRIALS]
        start += NTRIALS
        reward_series2[i] = np.mean(aslice)
        reward_min2[i] = np.min(aslice)
        reward_max2[i] = np.max(aslice)
        reward_deviation2[i] = np.std(aslice)
        
        
     # New Settings
    NVERTICES = 40
     
    # read_configuration("Revised/{}/{}/config.txt".format(NVERTICES,PREFIX))
    read_configuration("TASE/Iterations/{}/config.txt".format(NVERTICES))
     
    iterations_list3 =  [(ITERMIN + i*ITERINC) for i in range(ITERN)]
    complete_reward3 = pickle.load(open("TASE/Iterations/{}/complete_reward_series.dat".format(NVERTICES),"rb"))
    reward_series3 = np.zeros((len(iterations_list3),))
    reward_deviation3 = np.zeros((len(iterations_list3),))
    reward_min3 = np.zeros((len(iterations_list3),))
    reward_max3 = np.zeros((len(iterations_list3),))
    start = 0
    for i in range(len(iterations_list3)):
         aslice = complete_reward3[start:start+NTRIALS]
         start += NTRIALS
         reward_series3[i] = np.mean(aslice)
         reward_min3[i] = np.min(aslice)
         reward_max3[i] = np.max(aslice)
         reward_deviation3[i] = np.std(aslice)    
        
        
    figreward, axreward = plt.subplots()
    color1 = 'tab:red'
    color2 = 'tab:blue'
    color3 = 'tab:green'
    color4 = 'tab:orange'
    axreward.set_ylabel('reward',fontsize=15)
    axreward.set_xlabel('iterations ($K$)',fontsize=15)
    iterations_a = np.array(iterations_list1)
    iterations_b = np.array(iterations_list2)
    iterations_c = np.array(iterations_list3)
    # axreward.errorbar(iterations_a,reward_series,yerr=reward_deviation,color=color,linewidth=4)
    axreward.plot(iterations_a,reward_series1,color=color1,linewidth=2) 
    axreward.fill_between(iterations_a, reward_series1-reward_deviation1, reward_series1+reward_deviation1 , alpha=0.3,facecolor=color3)
    
    axreward.plot(iterations_b,reward_series2,color=color2,linewidth=2) 
    axreward.fill_between(iterations_a, reward_series2-reward_deviation2, reward_series2+reward_deviation2 , alpha=0.3,facecolor=color3)
    
    axreward.plot(iterations_c,reward_series3,color=color4,linewidth=2) 
    axreward.fill_between(iterations_c, reward_series3-reward_deviation3, reward_series3+reward_deviation3 , alpha=0.3,facecolor=color3)
    axreward.tick_params(axis='y')
    figreward.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(b=True,which='both',axis='both')
    plt.ylim(0,1.05*np.max(reward_series3+reward_deviation2))
    plt.show()
    figreward.savefig('reward_iteration_dependency_TASE.pdf')
   