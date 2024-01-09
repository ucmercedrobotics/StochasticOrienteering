#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# Produces the figure in the paper for the sensitivity to Pr


import sys
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np



#BASE_FOLDER = "MILPTASEAPRIL2023"   # results original submission
BASE_FOLDER = "MCTS2024_01_08/PROB"

PARTIAL = False

FAILURE_RATE_CONSTRAINTS = 0.100001

EPSMIN = 0
EPSN = 0
EPSINC = 0
ITERMIN = 0
ITERINC = 0
ITERN = 0
NTRIALS = 0
SAMPLESMIN = 0
SAMPLESN = 0
SAMPLESINC = 0
FAILURE_PROB = 0


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = '16'


def analyze_and_plot(folder):
    TESTCASES = ['10','20','30','40']
    
    results_map = dict()
    
    for i in TESTCASES:
        folderlist = glob.glob(folder+"/"+i+'/*')
        folderlist.sort()
        returns_x = []
        returns_y = []
        for j in folderlist:
            filename =  j + "/results.txt"
            returns_x.append(float(filename[filename.find('Pr')+2:filename.find('/results.txt')]))
            f = open(filename,"r")
            for x in f:
                if x.startswith("Average reward"):
                    returns_y.append(float(x[x.find('[')+1:x.find(']')]))
            f.close()
        results_map[i] = (returns_x,returns_y)
    
    figtime, axtime = plt.subplots()
    axtime.set_ylabel('Average Normalized Return')
    axtime.set_xlabel('$P_R$')
    
    
    
    for i in results_map.keys():
        couple = results_map[i]
        x = np.array(couple[0])
        y = np.array(couple[1])
        y = y / y.max()
        
        
        axtime.plot(x,y,linewidth=2,label=i)
    

    
    axtime.legend()
   # axtime.tick_params(axis='both')
 #   axtime.pyplot.xticks(ticks)
   # figtime.tight_layout()  
    plt.xticks(x)
    plt.grid(visible=True,which='major',axis='both')
   
    plt.show()
    figtime.savefig('pr_dependency.pdf')
    
    
   





    
if __name__ == "__main__":
    if len(sys.argv) > 1 :
        BASE_FOLDER = sys.argv[1]
            
    print("Printing results from folder ",BASE_FOLDER)
    analyze_and_plot(BASE_FOLDER)
    
   
   