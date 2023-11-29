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


def read_series_and_plot(basepath):
    # read_configuration("Revised/{}/{}/config.txt".format(NVERTICES,PREFIX))
     if not basepath.endswith("/"):
         basepath = basepath + "/"
     read_configuration(basepath + "config.txt")
     iterations_list =  [(ITERMIN + i*ITERINC) for i in range(ITERN)]
     complete_time = pickle.load(open(basepath + "complete_time_series.dat","rb"))
     
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
     plt.grid(visible=True,which='major',axis='both')
     plt.ylim(0,1.05*np.max(time_series + time_deviation))
     plt.show()
     figtime.savefig('time_dependency_new.pdf')
     return iterations_a,time_series,time_deviation


if __name__ == "__main__":
    
    # read first data series
    # Old code
   # NVERTICES = 20
   # PREFIX= 6

   # read_configuration("Data/Graph{}/DataSet{}/config.txt".format(NVERTICES,PREFIX))
   # iterations_list =  [(ITERMIN + i*ITERINC) for i in range(ITERN)]
   # complete_time = pickle.load(open("Data/Graph{}/DataSet{}/complete_time_series.dat".format(NVERTICES,PREFIX),"rb"))
   
   it,first_t,first_d = read_series_and_plot("TASE/TimeStudio")
   it, second_t,second_d = read_series_and_plot("TASE/TimeStudio2")
   it, third_t,third_d = read_series_and_plot("TASE/TimeStudio3")
   it, fourth_t,fourth_d = read_series_and_plot("TASE/TimeStudio4")
   
   post_t = np.minimum(first_t,second_t)
   post_d = second_d
   post_d[np.where(first_t<second_t)] = first_d[np.where(first_t<second_t)]
   
   post_d[np.where(third_t<post_t)] = third_d[np.where(third_t<post_t)]
   post_t = np.minimum(post_t,third_t)
   post_d[np.where(fourth_t<post_t)] = fourth_d[np.where(fourth_t<post_t)]
   post_t = np.minimum(post_t,fourth_t)
   
   
   
   figtime, axtime = plt.subplots()
   color = 'tab:red'
   colormin = 'tab:blue'
   colormax = 'tab:green'
   axtime.set_ylabel('time ($s$)')
   axtime.set_xlabel('iterations ($K$)')
   #axtime.errorbar(iterations_a,time_series,yerr=time_deviation,color=color,linewidth=4)
   axtime.plot(it,post_t,color=color,linewidth=2)
   axtime.fill_between(it, post_t-post_d, post_t+post_d , alpha=0.3,facecolor=colormax)
   axtime.tick_params(axis='y')
   figtime.tight_layout()  # otherwise the right y-label is slightly clipped
   plt.grid(visible=True,which='major',axis='both')
   plt.ylim(0,1.05*np.max(post_t + post_d))
   plt.show()
   figtime.savefig('time_dependency_new.pdf')
  