#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:58:58 2022

@author: shamano
"""

# Analyzes and prints to the console the results of the self-developed test cases.
# Results are created using script.sh
# Use: python process_data_MILP <folder_with_results>
# where <folder_with_results> is created by script.sh (see script for details
# and pay attention if you cross midnight).

import sys

#BASE_FOLDER = "MILPTASEAPRIL2023"   # results original submission
BASE_FOLDER = "MILP2023_11_30"


def analyze_and_print(folder):
    VERTICES = [10,20,30,40]
    BUDGET = [2,3]
    PF = ["005", "01"]
    
    results_map = dict()
    
    for v in VERTICES:
        for b in BUDGET:
            for p in PF:
                file_name = folder+"/{}/{}/{}/results.txt".format(v,b,p)
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
                
                
    # now format the results 
    
    # first table 
    
    b = 2
    print("Budget = ",b)
    
    for v in VERTICES:
      #  print("V ={}".format(v))
        print("{} &".format(v),end=" ")
        for p in PF:
            print("{} & {} & {} & ".format(results_map[(v,b,p)][0],results_map[(v,b,p)][2],results_map[(v,b,p)][1]),end=" " )
        print(r"\\")
            
          
    b = 3
    print("Budget = ",b)
    
    for v in VERTICES:
        #print("V ={}".format(v))
        print("{} &".format(v),end=" ")
        for p in PF:
            print("{} & {} & {} & ".format(results_map[(v,b,p)][0],results_map[(v,b,p)][2],results_map[(v,b,p)][1]),end=" " )
    
    
    
        print(r"\\")
    
    
if __name__ == "__main__":
    if len(sys.argv) == 1 :
        print("Base folder not provided; looking into folder ",BASE_FOLDER)
        analyze_and_print(BASE_FOLDER)
    else:
        analyze_and_print(sys.argv[1])
        
    