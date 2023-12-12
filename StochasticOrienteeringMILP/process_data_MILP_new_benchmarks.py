#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:58:58 2022

@author: shamano
"""

# Analyzes and prints to the console the results of the new benchmark problems.
# Results are created using script_new_cases.sh
# Use: python process_data_MILP_new_benchmarks <folder_with_results>
# where <folder_with_results> is created by script_new_cases.sh (see script for details
# and pay attention if you cross midnight).

import sys
import glob

#BASE_FOLDER = "MILPTASEAPRIL2023"   # results original submission
BASE_FOLDER = "MILP2023_11_30"


def analyze_and_print(folder):
    TESTCASES = ['ulysses','berlin','burma','att','st']
    
    results_map = dict()
    
    for i in TESTCASES:
        filelist = glob.glob(folder,i+'*')
        for j in filelist:
            filename = folder + "/" + j + "results.txt"
            tokens = j.split('_')
            budget = tokens[1]
            probability = tokens[2]
            vertices = tokens[0][-2:]
            f = open(filename,"r")
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
                key = (i,vertices,budget,probability)
                results_map[key] = list(results)
            f.close()
    
    
    
    
    for i in results_map.keys():
       values = results_map[i]
       print(f"{i[0]} & {i[1]} & {i[2]} & {values[0]} & {values[2]} & {values[1]}")
                
                
   
    
if __name__ == "__main__":
    if len(sys.argv) == 1 :
        print("Base folder not provided; looking into folder ",BASE_FOLDER)
        analyze_and_print(BASE_FOLDER)
    else:
        analyze_and_print(sys.argv[1])
        
    