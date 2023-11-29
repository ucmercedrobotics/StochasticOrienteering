#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:36:39 2023

@author: shamano
"""

# this script produces the figures comparing MCTS with MILP


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True

BASE_FOLDER_TASE = "TASE"
BASE_FOLDER_MILP = "../StochasticOrienteeringMILP/MILPTASEAPRIL2023"


VERTICES = [10,20,30,40]
BUDGET = [2,3]
PF = ["005", "01"]




def read_results_MCTS():
    results_map = dict()
    for v in VERTICES:
        for b in BUDGET:
            for p in PF:
                file_name = BASE_FOLDER_TASE+"/{}/{}/{}/results.txt".format(v,b,p)
                f = open(file_name,"r")
                results = []
                for x in f:
                    if x.startswith("Average reward"):
                        tmp = x.split(":")[1].strip(" \n[]")
                        tmp = tmp.split(".")[0] + "." + tmp.split(".")[1][0:4]
                        results.append(tmp)
                    if x.startswith("Failure rate"):
                        tmp = x.split(":")[1].strip(" \n[]")
                        tmp = tmp.split(".")[0] + "." + tmp.split(".")[1][0:4]
                        results.append(tmp)
                    if x.startswith("Average time"):
                        tmp = x.split(":")[1].strip(" \n[]")
                        tmp = tmp.split(".")[0] + "." + tmp.split(".")[1][0:4]
                        results.append(tmp)
                key = (v,b,p)
                results_map[key] = list(results)
                f.close()
    return results_map

def read_results_MILP():    
    results_map = dict()        
    for v in VERTICES:
        for b in BUDGET:
            for p in PF:
                file_name = BASE_FOLDER_MILP+"/{}/{}/{}/results.txt".format(v,b,p)
                f = open(file_name,"r")
                results = []
                for x in f:
                    if x.startswith("Average Reward"):
                        tmp = x.split(":")[1].strip(" \n")
                        tmp = tmp.split(".")[0] + "." + tmp.split(".")[1][0:4]
                        results.append(tmp)
                    if x.startswith("Average Actual"):
                        tmp = x.split(":")[1].strip(" \n")
                        tmp = tmp.split(".")[0] + "." + tmp.split(".")[1][0:4]
                        results.append(tmp)
                    if x.startswith("Average Solution time"):
                        tmp = x.split(":")[1].strip(" \n")
                        tmp = tmp.split(".")[0] + "." + tmp.split(".")[1][0:4]
                        results.append(tmp)
                    key = (v,b,p)
                    results_map[key] = list(results)
                f.close()
    return results_map


def charts_failure(tree,milp):
    
    labels = ['10','20','30','40']

    x = np.arange(len(labels))  
    width = 0.3  # the width of the bars
    
  #  figtime, axtime = plt.subplots()
    b = 2
    p = PF[0]
    failure_milp = [float(milp[(10,b,p)][2]),float(milp[(20,b,p)][2]),float(milp[(30,b,p)][2]),float(milp[(40,b,p)][2])]
    failure_mcts = [float(tree[(10,b,p)][2]),float(tree[(20,b,p)][2]),float(tree[(30,b,p)][2]),float(tree[(40,b,p)][2])]
    
    fig, ax = plt.subplots(layout='constrained')

    ax.bar(x-width/2,failure_milp ,width,label='MILP')
    ax.bar(x+width/2,failure_mcts,width,label='MCTS')
        
    ax.set_ylabel('Failure Rate')
    ax.set_xlabel('Number of Vertices')
    ax.set_title('Budget=2, $P_f = 0.05$')
    ax.set_xticks(x, labels)
    ax.legend(loc='upper center', ncols=3)
    plt.grid(visible=True,which='major',axis='both')
    plt.show()
    fig.savefig('failure_milp_mcts_budget2_pf005.pdf')
    
    b = 2
    p = PF[1]
    failure_milp = [float(milp[(10,b,p)][2]),float(milp[(20,b,p)][2]),float(milp[(30,b,p)][2]),float(milp[(40,b,p)][2])]
    failure_mcts = [float(tree[(10,b,p)][2]),float(tree[(20,b,p)][2]),float(tree[(30,b,p)][2]),float(tree[(40,b,p)][2])]
    
    fig, ax = plt.subplots(layout='constrained')

    ax.bar(x-width/2,failure_milp ,width,label='MILP')
    ax.bar(x+width/2,failure_mcts,width,label='MCTS')
        
    ax.set_ylabel('Failure Rate')
    ax.set_xlabel('Number of Vertices')
    ax.set_title('Budget=2, $P_f = 0.1$')
    ax.set_xticks(x, labels)
    ax.legend(loc='upper left', ncols=3)
    plt.grid(visible=True,which='major',axis='both')
    plt.show()
    fig.savefig('failure_milp_mcts_budget2_pf01.pdf')
    
    b = 3
    p = PF[0]
    failure_milp = [float(milp[(10,b,p)][2]),float(milp[(20,b,p)][2]),float(milp[(30,b,p)][2]),float(milp[(40,b,p)][2])]
    failure_mcts = [float(tree[(10,b,p)][2]),float(tree[(20,b,p)][2]),float(tree[(30,b,p)][2]),float(tree[(40,b,p)][2])]
    
    fig, ax = plt.subplots(layout='constrained')

    ax.bar(x-width/2,failure_milp ,width,label='MILP')
    ax.bar(x+width/2,failure_mcts,width,label='MCTS')
        
    ax.set_ylabel('Failure Rate')
    ax.set_xlabel('Number of Vertices')
    ax.set_title('Budget=3, $P_f = 0.05$')
    ax.set_xticks(x, labels)
    ax.legend(loc='upper left', ncols=3)
    plt.grid(visible=True,which='major',axis='both')
    plt.show()
    fig.savefig('failure_milp_mcts_budget3_pf005.pdf')
    
    b = 3
    p = PF[1]
    failure_milp = [float(milp[(10,b,p)][2]),float(milp[(20,b,p)][2]),float(milp[(30,b,p)][2]),float(milp[(40,b,p)][2])]
    failure_mcts = [float(tree[(10,b,p)][2]),float(tree[(20,b,p)][2]),float(tree[(30,b,p)][2]),float(tree[(40,b,p)][2])]
    
    fig, ax = plt.subplots(layout='constrained')

    ax.bar(x-width/2,failure_milp ,width,label='MILP')
    ax.bar(x+width/2,failure_mcts,width,label='MCTS')
        
    ax.set_ylabel('Failure Rate')
    ax.set_xlabel('Number of Vertices')
    ax.set_title('Budget=3, $P_f = 0.1$')
    ax.set_xticks(x, labels)
    ax.legend(loc='upper right', ncols=3)
    plt.grid(visible=True,which='major',axis='both')
    plt.show()
    fig.savefig('failure_milp_mcts_budget3_pf01.pdf')
    
  
   

def charts_reward(tree,milp):
  #  figtime, axtime = plt.subplots()
    b = 2
    p = PF[0]
    reward_milp = [float(milp[(10,b,p)][0]),float(milp[(20,b,p)][0]),float(milp[(30,b,p)][0]),float(milp[(40,b,p)][0])]
    reward_mcts = [float(tree[(10,b,p)][0]),float(tree[(20,b,p)][0]),float(tree[(30,b,p)][0]),float(tree[(40,b,p)][0])]
    ratio_2_005= tuple([reward_mcts[i]/reward_milp[i] for i in range(4)])
  
    b = 2
    p = PF[1]
    reward_milp = [float(milp[(10,b,p)][0]),float(milp[(20,b,p)][0]),float(milp[(30,b,p)][0]),float(milp[(40,b,p)][0])]
    reward_mcts = [float(tree[(10,b,p)][0]),float(tree[(20,b,p)][0]),float(tree[(30,b,p)][0]),float(tree[(40,b,p)][0])]
    ratio_2_01= tuple([reward_mcts[i]/reward_milp[i] for i in range(4)])
    
    labels = ['10','20','30','40']

    x = np.arange(len(labels))  
    width = 0.3  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    ax.bar(x-width/2,ratio_2_005,width,label=r'$P_f=0.05$')
    ax.bar(x+width/2,ratio_2_01,width,label=r'$P_f=0.1$')
        
    ax.set_ylabel('Ratio')
    ax.set_xlabel('Number of Vertices')
    ax.set_title('Budget=2')
    ax.set_xticks(x, labels)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1.3)
    plt.grid(visible=True,which='major',axis='both')
    plt.show()
    fig.savefig('reward_milp_mcts_budget2.pdf')
    
    b = 3
    p = PF[0]
    reward_milp = [float(milp[(10,b,p)][0]),float(milp[(20,b,p)][0]),float(milp[(30,b,p)][0]),float(milp[(40,b,p)][0])]
    reward_mcts = [float(tree[(10,b,p)][0]),float(tree[(20,b,p)][0]),float(tree[(30,b,p)][0]),float(tree[(40,b,p)][0])]
    ratio_2_005= tuple([reward_mcts[i]/reward_milp[i] for i in range(4)])
  
    b = 3
    p = PF[1]
    reward_milp = [float(milp[(10,b,p)][0]),float(milp[(20,b,p)][0]),float(milp[(30,b,p)][0]),float(milp[(40,b,p)][0])]
    reward_mcts = [float(tree[(10,b,p)][0]),float(tree[(20,b,p)][0]),float(tree[(30,b,p)][0]),float(tree[(40,b,p)][0])]
    ratio_2_01= tuple([reward_mcts[i]/reward_milp[i] for i in range(4)])
    
    labels = ['10','20','30','40']

    x = np.arange(len(labels))  
    width = 0.3  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    ax.bar(x-0.3/2,ratio_2_005,width,label=r'$P_f=0.05$')
    ax.bar(x+0.3/2,ratio_2_01,width,label=r'$P_f=0.1$')
        
    ax.set_ylabel('Ratio')
    ax.set_xlabel('Number of Vertices')
    ax.set_title('Budget=3')
    ax.set_xticks(x, labels)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1.3)
    plt.grid(visible=True,which='major',axis='both')
    plt.show()
    fig.savefig('reward_milp_mcts_budget3.pdf')
    

def charts_time(tree,milp):
    b = 2
    p = PF[0]
    time_milp = [float(milp[(10,b,p)][1]),float(milp[(20,b,p)][1]),float(milp[(30,b,p)][1]),float(milp[(40,b,p)][1])]
    time_mcts = [float(tree[(10,b,p)][1]),float(tree[(20,b,p)][1]),float(tree[(30,b,p)][1]),float(tree[(40,b,p)][1])]
    

    x = np.array([10,20,30,40])  
    fig, ax = plt.subplots(layout='constrained')

    ax.plot(x,time_milp,label='MILP')
    ax.plot(x,time_mcts,label='MCTS')
        
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Number of Vertices')
    ax.set_title('Budget=2, $P_f$=0.05')
    ax.legend(loc='upper left', ncols=3)
    
    plt.grid(visible=True,which='major',axis='both')
    plt.show()
    fig.savefig('time_milp_mcts_budget2_pf005.pdf')
    
    b = 2
    p = PF[1]
    time_milp = [float(milp[(10,b,p)][1]),float(milp[(20,b,p)][1]),float(milp[(30,b,p)][1]),float(milp[(40,b,p)][1])]
    time_mcts = [float(tree[(10,b,p)][1]),float(tree[(20,b,p)][1]),float(tree[(30,b,p)][1]),float(tree[(40,b,p)][1])]
    

    x = np.array([10,20,30,40])  
    fig, ax = plt.subplots(layout='constrained')

    ax.plot(x,time_milp,label='MILP')
    ax.plot(x,time_mcts,label='MCTS')
        
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Number of Vertices')
    ax.set_title('Budget=2, $P_f$=0.1')
    ax.legend(loc='upper left', ncols=3)
    
    plt.grid(visible=True,which='major',axis='both')
    plt.show()
    fig.savefig('time_milp_mcts_budget2_pf01.pdf')
    
    b = 3
    p = PF[0]
    time_milp = [float(milp[(10,b,p)][1]),float(milp[(20,b,p)][1]),float(milp[(30,b,p)][1]),float(milp[(40,b,p)][1])]
    time_mcts = [float(tree[(10,b,p)][1]),float(tree[(20,b,p)][1]),float(tree[(30,b,p)][1]),float(tree[(40,b,p)][1])]
    

    x = np.array([10,20,30,40])  
    fig, ax = plt.subplots(layout='constrained')

    ax.plot(x,time_milp,label='MILP')
    ax.plot(x,time_mcts,label='MCTS')
        
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Number of Vertices')
    ax.set_title('Budget=3, $P_f$=0.05')
    ax.legend(loc='upper left', ncols=3)
    
    plt.grid(visible=True,which='major',axis='both')
    plt.show()
    fig.savefig('time_milp_mcts_budget3_pf005.pdf')
    
    b = 3
    p = PF[1]
    time_milp = [float(milp[(10,b,p)][1]),float(milp[(20,b,p)][1]),float(milp[(30,b,p)][1]),float(milp[(40,b,p)][1])]
    time_mcts = [float(tree[(10,b,p)][1]),float(tree[(20,b,p)][1]),float(tree[(30,b,p)][1]),float(tree[(40,b,p)][1])]
    

    x = np.array([10,20,30,40])  
    fig, ax = plt.subplots(layout='constrained')

    ax.plot(x,time_milp,label='MILP')
    ax.plot(x,time_mcts,label='MCTS')
        
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Number of Vertices')
    ax.set_title('Budget=3, $P_f$=0.1')
    ax.legend(loc='upper left', ncols=3)
    
    plt.grid(visible=True,which='major',axis='both')
    plt.show()
    fig.savefig('time_milp_mcts_budget3_pf01.pdf')


if __name__ == "__main__":
    MCTS = read_results_MCTS()
    MILP = read_results_MILP()
    charts_reward(MCTS,MILP)
    charts_failure(MCTS,MILP)
    charts_time(MCTS,MILP)
    
    
            
# now format the results 

# first table 

# b = 2
# print("Budget = ",b)

# for v in VERTICES:
#   #  print("V ={}".format(v))
#     print("{} &".format(v),end=" ")
#     for p in PF:
#         print("{} & {} & {} & ".format(results_map[(v,b,p)][0],results_map[(v,b,p)][2],results_map[(v,b,p)][1]),end=" " )
#     print(r"\\")
        
      
# b = 3
# print("Budget = ",b)

# for v in VERTICES:
#     #print("V ={}".format(v))
#     print("{} &".format(v),end=" ")
#     for p in PF:
#         print("{} & {} & {} & ".format(results_map[(v,b,p)][0],results_map[(v,b,p)][2],results_map[(v,b,p)][1]),end=" " )



#     print(r"\\")