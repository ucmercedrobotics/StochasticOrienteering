#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:58:58 2022

@author: shamano
"""



BASE_FOLDER = "MILPTASEAPRIL2023"


VERTICES = [10,20,30,40]
BUDGET = [2,3]
PF = ["005", "01"]

results_map = dict()

for v in VERTICES:
    for b in BUDGET:
        for p in PF:
            file_name = BASE_FOLDER+"/{}/{}/{}/results.txt".format(v,b,p)
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