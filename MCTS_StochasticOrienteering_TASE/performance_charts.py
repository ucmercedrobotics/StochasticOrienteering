#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:08:48 2021

@author: shamano
"""
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import configparser


NVERTICES = 20
PREFIX= 6

PARTIAL = False

FAILURE_RATE_CONSTRAINTS = 0.100001

EPSMIN = 0
EPSN = 0
EPSINC = 0
ITERMIN = 0
ITERINC = 0
ITERN = 0
NTRIALS = 0


#matplotlib.rcParams['text.usetex'] = True

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
    
    read_configuration("Data/Graph{}/DataSet{}/config.txt".format(NVERTICES,PREFIX))
    
    epsilon_vals = [(EPSMIN + i*EPSINC) for i in range(EPSN)]
    iterations_list =  [(ITERMIN + i*ITERINC) for i in range(ITERN)]

    if not PARTIAL:
        reward=pickle.load(open("Data/Graph{}/DataSet{}/reward.dat".format(NVERTICES,PREFIX),"rb"))
        failure = pickle.load(open("Data/Graph{}/DataSet{}/failure_rate.dat".format(NVERTICES,PREFIX),"rb"))
        residual_budget = pickle.load(open("Data/Graph{}/DataSet{}/residual_budget.dat".format(NVERTICES,PREFIX),"rb"))
        complete_time = pickle.load(open("Data/Graph{}/DataSet{}/complete_time_series.dat".format(NVERTICES,PREFIX),"rb"))
        complete_reward = pickle.load(open("Data/Graph{}/DataSet{}/complete_reward_series.dat".format(NVERTICES,PREFIX),"rb"))
        
        
        #p = pickle.load(open("p_factor.dat","rb"))    
        
        reward = np.array(reward)
        failure = np.array(failure)
        complete_time = np.array(complete_time)
        complete_reward = np.array(complete_reward)
        
        fig, ax1 = plt.subplots()
        
        color = 'tab:red'
        #ax1.set_xlabel('penalty')
        ax1.set_ylabel('reward', color=color)
        #ax1.plot(p, reward, color=color)
        ax1.plot(reward,color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        color = 'tab:blue'
        ax2.set_ylabel('residual', color=color)  # we already handled the x-label with ax1
        ax2.plot(residual_budget, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        
        fig2, ax3 = plt.subplots()
        color = 'tab:red'
        #ax1.set_xlabel('penalty')
        ax3.set_ylabel('failure', color=color)
        #ax1.plot(p, reward, color=color)
        ax3.plot(failure,color=color)
        ax3.tick_params(axis='y', labelcolor=color)
        fig2.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        
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
            
        reward_series = np.zeros((len(iterations_list),))
        reward_deviation = np.zeros((len(iterations_list),))
        reward_min = np.zeros((len(iterations_list),))
        reward_max = np.zeros((len(iterations_list),))
        start = 0
        for i in range(len(iterations_list)):
            aslice = complete_reward[start:start+NTRIALS]
            start += NTRIALS
            reward_series[i] = np.mean(aslice)
            reward_min[i] = np.min(aslice)
            reward_max[i] = np.max(aslice)
            reward_deviation[i] = np.std(aslice)
        
        
        figtime, axtime = plt.subplots()
        color = 'tab:red'
        colormin = 'tab:blue'
        colormax = 'tab:green'
        axtime.set_ylabel('time ($s$)')
        axtime.set_xlabel('iterations ($K$)')
        iterations_a = np.array(iterations_list)
        #axtime.errorbar(iterations_a,time_series,yerr=time_deviation,color=color,linewidth=4)
        axtime.plot(iterations_a,time_series,color=color,linewidth=2)
        axtime.plot(iterations_a,time_series - time_deviation,color=colormin,linewidth=2)
        axtime.plot(iterations_a,time_series + time_deviation,color=colormin,linewidth=2)
        axtime.tick_params(axis='y')
        figtime.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.grid(b=True,which='major',axis='both')
        plt.ylim(0,1.05*np.max(time_series + time_deviation))
        plt.show()
        
        figreward, axreward = plt.subplots()
        color = 'tab:red'
        colormin = 'tab:blue'
        colormax = 'tab:green'
        axreward.set_ylabel('reward')
        axreward.set_xlabel('iterations ($K$)')
        iterations_a = np.array(iterations_list)
       # axreward.errorbar(iterations_a,reward_series,yerr=reward_deviation,color=color,linewidth=4)
        axreward.plot(iterations_a,reward_series,color=color,linewidth=3) 
        axreward.plot(iterations_a,reward_series+reward_deviation,color=colormin,linewidth=3) 
        axreward.plot(iterations_a,reward_series-reward_deviation,color=colormin,linewidth=3) 
        
        
        
        axreward.tick_params(axis='y')
        figreward.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.grid(b=True,which='both',axis='both')
        plt.ylim(0,1.05*np.max(reward_max))
        plt.show()
        
     
        epsilon_series = np.zeros((len(epsilon_vals),))
        epsilon_deviation = np.zeros((len(epsilon_vals),))
        start = 0
        for i in range(len(epsilon_vals)):
            aslice = complete_reward[start:start+NTRIALS]
            start += NTRIALS
            epsilon_series[i] = np.mean(aslice)
            epsilon_deviation[i] = np.std(aslice)

        figepsilon, axepsilon = plt.subplots()
        color = 'tab:red'
        axepsilon.set_ylabel('reward')
     #   axepsilon.set_xlabel('$\\varepsilon$')
        epsilon_a = np.array(epsilon_vals)
        axepsilon.errorbar(epsilon_a,epsilon_series,yerr=epsilon_deviation,color=color,linewidth=2)
        axepsilon.tick_params(axis='y')
        figepsilon.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.grid(b=True,which='major',axis='both')
        plt.ylim(0,1.05*np.max(epsilon_series+epsilon_deviation))
        plt.show()
        
    
    

        
    
    best_reward = -1
    valid_solutions = 0
    
    for iterations in iterations_list:
        print('\n')
        for EPSILON in epsilon_vals:
            reward=pickle.load(open('Data/Graph{}/DataSet{}/rewards_EPSILON_{:.2f}_iterations_{}.dat'.format(NVERTICES,PREFIX,EPSILON,iterations),"rb"))
            budgets = pickle.load(open('Data/Graph{}/DataSet{}/budgets_EPSILON_{:.2f}_iterations_{}.dat'.format(NVERTICES,PREFIX,EPSILON,iterations),"rb"))
            reward = np.array(reward)
            budgets = np.array(budgets)
            size = budgets.size
            successes = np.flatnonzero(budgets >= 0)
            goodrewards = reward[successes]
            print('EPSILON = {:.2f} Iterations = {}'.format(EPSILON,iterations))
            print('  Failure rate: ', 1-successes.size/size)
            print('  Average reward: ',np.average(goodrewards))
            if (1-successes.size/size) < FAILURE_RATE_CONSTRAINTS:
                valid_solutions +=1
                print("====================>Found Valid solution")
                if np.average(goodrewards) > best_reward:
                    best_reward = np.average(goodrewards)
                    bestEPSILON = EPSILON
                    bestIterations = iterations
                
    if best_reward > 0:
        print("\n\nBest EPSILON=",bestEPSILON)
        print("Best Iterations=",bestIterations)
        print("Best Reward=",best_reward)
        print("Number of valid solutions: ",valid_solutions)    
    else:
        print("\n\nNo suitable parameter configuration found.")
        
        
            