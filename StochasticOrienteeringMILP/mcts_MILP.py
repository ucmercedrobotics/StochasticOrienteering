#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 09:49:59 2022

@author: shamano
"""

import sys
sys.path.append('../OrienteeringGraph')

import pulp
import graph
import config
import importlib
import os.path
import os
import configparser
import argparse
import shutil
from progress.bar import Bar
import numpy as np

importlib.reload(config)


#
#  If you redo the tests for the "CLASSIC SCOP", i.e., start vertex at 0, compare the version of
#  create_pulp_instance with create_pulp_instance_MCTS. This coudl affect the TASE paper
#  (though it probably does not change the results becasue vertex 0 always had value 0).
#  prob += pulp.lpSum([pi_vars(s)*og.get_reward(s[1]) for s in pi_tuples]) should probably be
#  prob += pulp.lpSum([pi_vars(s)*og.get_reward(s[0]) for s in pi_tuples])
#



# create a MILP instance using pulp and following the formulation in 
#https://link.springer.com/content/pdf/10.1007/978-3-642-41575-3_30.pdf
def create_pulp_instance(og,alpha):
    
    prob = pulp.LpProblem("SCOP", pulp.LpMaximize)  # setup maximization problem
    # create optimization variables
    
    # first generate r integer variables for contstraints (14), (15)
    n = og.number_of_nodes()
    
    r2s = lambda x: '{}'.format(x)
    r_indices = [r2s(i) for i in range(n)]
    r_vars_dict = pulp.LpVariable.dicts("r",indices=r_indices,lowBound=0,upBound=n-1,cat='Integer')
    
    # next generate binary variables for each edge
    ij2s = lambda i, j: '{},{}'.format(i, j)
    pi_indices = []
    pi_tuples = []
    for i in range(n):
        for j in range(n):
            if i!= j:
                pi_indices.append(ij2s(i,j))
                pi_tuples.append((i,j))
    pi_vars_dict = pulp.LpVariable.dicts("pi",indices=pi_indices,cat='Binary')
    
    pi_vars = lambda s: pi_vars_dict['{},{}'.format(s[0],s[1])]
    
    #setup objective function
    prob += pulp.lpSum([pi_vars(s)*og.get_reward(s[1]) for s in pi_tuples])
    
    # add constraint (10)
    all_ind = [i for i in range(n)]
    for i in range(n):
        array = [pi_vars((j,i)) for j in all_ind[:i]+all_ind[i+1:]]
        prob += pulp.lpSum(array) <= 1
    # add constraint (11)
    for i in range(n):
        array = [pi_vars((i,j)) for j in all_ind[:i]+all_ind[i+1:]]
        prob += pulp.lpSum(array) <= 1
        
    # add constraints (12)
    array = [pi_vars((0,j)) for j in all_ind[:0]+all_ind[0+1:]]
    prob += pulp.lpSum(array) == 1
    array = [pi_vars((j,n-1)) for j in all_ind[:n-1]]
    prob += pulp.lpSum(array) == 1
    
    
    # add constraint (13)
    # case i = 0
    i = 0
    array_plus = [pi_vars((i,j)) for j in all_ind[:i]+all_ind[i+1:]] 
    array_minus =  [pi_vars((j,i)) for j in all_ind[:i]+all_ind[i+1:]] 
    coeff_plus = [1]*len(array_plus)
    coeff_minus = [-1]*len(array_minus)
    array = array_plus+array_minus
    coeff = coeff_plus + coeff_minus
    prob += pulp.LpAffineExpression(zip(array,coeff)) == 1
    
    
    # 1 <= i < n-1
    for i in range(1,n-1):
       # for j in range(n):
        array_plus = [pi_vars((i,j)) for j in all_ind[:i]+all_ind[i+1:]] 
        array_minus =  [pi_vars((j,i)) for j in all_ind[:i]+all_ind[i+1:]] 
        coeff_plus = [1]*len(array_plus)
        coeff_minus = [-1]*len(array_minus)
        array = array_plus+array_minus
        coeff = coeff_plus + coeff_minus
        prob += pulp.LpAffineExpression(zip(array,coeff)) == 0
        
    
    # case i = n-1
    i = n-1
    array_plus = [pi_vars((i,j)) for j in all_ind[:i]+all_ind[i+1:]] 
    array_minus =  [pi_vars((j,i)) for j in all_ind[:i]+all_ind[i+1:]] 
    coeff_plus = [1]*len(array_plus)
    coeff_minus = [-1]*len(array_minus)
    array = array_plus+array_minus
    coeff = coeff_plus + coeff_minus
    prob += pulp.LpAffineExpression(zip(array,coeff)) == -1
    

    # add constraint (14)
    for i in range(n):
        for j in range(n):
            if i != j:
                prob += r_vars_dict[r_indices[i]] <= r_vars_dict[r_indices[j]] -1 + (1-pi_vars((i,j)))*config.M_CONST
    
    
    # add constraints (15)
    single = [r_vars_dict[r_indices[0]]]
    prob += pulp.lpSum(single) == 0
    single = [r_vars_dict[r_indices[n-1]]]
    prob += pulp.lpSum(single) == n-1
    
    # now add constraint for budget -- not necessary for stochastic case
    #setup objective function
    #prob += pulp.lpSum([pi_vars(s)*og.vertices[s[0]].edges[s[1]].d for s in pi_tuples]) <= og.budget
    
    H = og.budget
    
    # now add the z variables for the constraints (17-19)
    z2s = lambda x: '{}'.format(x)
    z_indices = [z2s(i) for i in range(config.Q_VALUE)]
    z_vars_dict = pulp.LpVariable.dicts("z",indices=z_indices,cat='Binary')
    # constraint (17)
    for z in z_indices:
        prob +=  H + H * z_vars_dict[z] >= pulp.lpSum([pi_vars(s)*og.vertices[s[0]].edges[s[1]].sample_cost()  for s in pi_tuples]) 
    
    # constraint (19)
    prob += pulp.lpSum([z_vars_dict[s] for s in z_indices]) <= config.Q_VALUE*alpha
    
    #prob.writeLP("test.lp")
    return prob

def create_pulp_instance_MCTS(og,alpha,visited,leaf,budget):
    
    start = leaf.node.id
    end_v = og.get_end_vertex()
    
    prob = pulp.LpProblem("SCOP", pulp.LpMaximize)  # setup maximization problem
    # create optimization variables
    
    
    # first generate r integer variables for contstraints (14), (15)
    n = og.number_of_nodes()
    r2s = lambda x: '{}'.format(x)
    
    to_remove = set(visited.keys()) - set ([start])
    
    r_indices_int = [i for i in (set(range(n))-to_remove)]
    r_indices = [r2s(i) for i in r_indices_int]
    r_vars_dict = pulp.LpVariable.dicts("r",indices=r_indices,lowBound=1,upBound=len(r_indices_int),cat='Integer')
    
    # next generate binary variables for each edge
    ij2s = lambda i, j: '{},{}'.format(i, j)
    pi_indices = []
    pi_tuples = []
    for i in r_indices_int:
        for j in r_indices_int:
            if i!= j:
                pi_indices.append(ij2s(i,j))
                pi_tuples.append((i,j))
    pi_vars_dict = pulp.LpVariable.dicts("pi",indices=pi_indices,cat='Binary')
    
    pi_vars = lambda s: pi_vars_dict['{},{}'.format(s[0],s[1])]
    
    #setup objective function (Eq. (8))
    # changed from prob += pulp.lpSum([pi_vars(s)*og.get_reward(s[1]) for s in pi_tuples])
    prob += pulp.lpSum([pi_vars(s)*og.get_reward(s[0]) for s in pi_tuples])
    
    # add constraints (10 and 11)
   
    for i in r_indices_int:
     #   array = [pi_vars((j,i)) for j in (set(r_indices_int)-set([i]))]
        array1 = [pi_vars((j,i)) for j in r_indices_int if j != i]
        array2 = [pi_vars((i,j)) for j in r_indices_int if j != i]
        prob += pulp.lpSum(array1) <= 1
        prob += pulp.lpSum(array2) <= 1
    # add constraint (11)
   # for i in r_indices_int:
   #     array = [pi_vars((i,j)) for j in (set(r_indices_int)-set([i]))]
     #   prob += pulp.lpSum(array) <= 1
        
    
    # add constraints (12)
    array = [pi_vars((start,j)) for j in r_indices_int if j != start]
    prob += pulp.lpSum(array) == 1
    array = [pi_vars((j,end_v)) for j in r_indices_int if j != end_v]
    prob += pulp.lpSum(array) == 1
    
    
    # add constraint (13)
    # case i = start
    i = start
    array_plus = [pi_vars((i,j)) for j in r_indices_int if j != start] 
    array_minus =  [pi_vars((j,i)) for j in r_indices_int if j != start] 
    coeff_plus = [1]*len(array_plus)
    coeff_minus = [-1]*len(array_minus)
    array = array_plus+array_minus
    coeff = coeff_plus + coeff_minus
    prob += pulp.LpAffineExpression(zip(array,coeff)) == 1
    
    
    # 1 <= i < n-1
    for i in (set(r_indices_int) - set([start,end_v])):
       # for j in range(n):
        array_plus = [pi_vars((i,j)) for j in r_indices_int if j != i] 
        array_minus =  [pi_vars((j,i)) for j in r_indices_int  if j != i] 
        coeff_plus = [1]*len(array_plus)
        coeff_minus = [-1]*len(array_minus)
        array = array_plus+array_minus
        coeff = coeff_plus + coeff_minus
        prob += pulp.LpAffineExpression(zip(array,coeff)) == 0
        
    
    # case i = n-1
    i = end_v
    array_plus = [pi_vars((i,j)) for j in r_indices_int if j != end_v] 
    array_minus =  [pi_vars((j,i)) for j in r_indices_int if j != end_v] 
    coeff_plus = [1]*len(array_plus)
    coeff_minus = [-1]*len(array_minus)
    array = array_plus+array_minus
    coeff = coeff_plus + coeff_minus
    prob += pulp.LpAffineExpression(zip(array,coeff)) == -1
    

    # add constraint (14)
    for i in range(len(r_indices_int)):
        for j in range(len(r_indices_int)):
            if i != j:
                prob += r_vars_dict[r_indices[i]] <= r_vars_dict[r_indices[j]] -1 + (1-pi_vars((r_indices_int[i],r_indices_int[j])))*config.M_CONST
    
    
    # add constraints (15)
    single = [r_vars_dict[(str(start))]]
    prob += pulp.lpSum(single) == 1
    single = [r_vars_dict[str(end_v)]]
    prob += pulp.lpSum(single) == len(r_indices_int)
    
    # now add constraint for budget -- not necessary for stochastic case
    #setup objective function
    #prob += pulp.lpSum([pi_vars(s)*og.vertices[s[0]].edges[s[1]].d for s in pi_tuples]) <= og.budget
    
    H = budget
    
    # now add the z variables for the constraints (17-19)
    z2s = lambda x: '{}'.format(x)
    z_indices = [z2s(i) for i in range(config.Q_VALUE)]
    z_vars_dict = pulp.LpVariable.dicts("z",indices=z_indices,cat='Binary')
    # constraint (17)
    for z in z_indices:
        prob +=  H + H * z_vars_dict[z] >= pulp.lpSum([pi_vars(s)*og.vertices[s[0]].vertex.edges[s[1]].sample_cost()  for s in pi_tuples]) 
    
    # constraint (19)
    prob += pulp.lpSum([z_vars_dict[s] for s in z_indices]) <= config.Q_VALUE*alpha
    
    #prob.writeLP("test.lp")
    return prob

def split_variable_name(v):
    tail = v[3:]
    ind = tail.split(',')
    return int(ind[0]),int(ind[1])

def compute_solution_length(prob,og):
    length = 0
    varsdict = {}
    for v in prob.variables():
        varsdict[v.name] = v.varValue
    for i in varsdict:
        if i.startswith('pi'):
            if varsdict[i] > 0:
               # print(i)
                i,j = split_variable_name(i)
                length += og.vertices[i].edges[j].d
    return length

def get_traversed_vertices(prob,og,leaf):
    
    varsdict = {}
    traversed = [leaf.node.id]
    tmp_list = []
    for v in prob.variables():
        varsdict[v.name] = v.varValue
        
    for i in varsdict:
        if i.startswith('pi'):
            if varsdict[i] > 0:
               # print(i)
               i,j = split_variable_name(i)
               tmp_list.append((int(varsdict['r_'+str(j)]),j)) 
    
    tmp_list.sort()
    for i in tmp_list:
        traversed.append(i[1])           
    return traversed
            

def simulate_solution_length(prob,og):
    length = 0
    varsdict = {}
    for v in prob.variables():
        varsdict[v.name] = v.varValue
    for i in varsdict:
        if i.startswith('pi'):
            if varsdict[i] > 0:
               # print(i)
                i,j = split_variable_name(i)
                length += og.vertices[i].edges[j].sample_cost()
    return length

# returns sequence of vertices traversed by the solution saved in prob
def extract_solution(prob):
    varsdict = {}
    edge_list = []
    for v in prob.variables():
        if v.varValue > 0:
            if v.name.startswith('pi'):
                edge_list.append(v.name)
    print(edge_list)


def create_nested_path(path):
    # first break down all intermediate folders
    l = []
    done = False
    while not done:
        a,b = os.path.split(path)
        l.insert(0,b)
        if len(a)==0:
            done = True
        else:
            path = a
    partial_path = ''
    for i in l:
        partial_path = os.path.join(partial_path,i)
        if not os.path.isdir(partial_path):
            os.mkdir(partial_path)
            


def read_configuration(fname):
    configfile = configparser.ConfigParser()
    print("Reading configuration file ",fname)
    if os.path.exists(fname):
        configfile.read(fname)
    else:
        raise Exception("Can't read configuration file {}".format(fname))

    if configfile['MAIN']['M_CONST'] is None:
        print('Missing configuration parameter M_CONST')
    else:
         config.M_CONST = int(configfile['MAIN']['M_CONST'])
         
    if configfile['MAIN']['Q_VALUE'] is None:
        print('Missing configuration parameter Q_VALUE')
    else:
         config.Q_VALUE = int(configfile['MAIN']['Q_VALUE'])
         
    if configfile['MAIN']['SOLVER'] is None:
        print('Missing configuration parameter SOLVER')
    else:
         config.SOLVER = configfile['MAIN']['SOLVER']
         
    if configfile['MAIN']['NVERTICES'] is None:
        print('Missing configuration parameter NVERTICES')
    else:
         config.NVERTICES = int(configfile['MAIN']['NVERTICES'])
         
    if configfile['MAIN']['BUDGET'] is None:
        print('Missing configuration parameter BUDGET')
    else:
         config.BUDGET = float(configfile['MAIN']['BUDGET'])
         
    if configfile['MAIN']['ALPHAMILP'] is None:
        print('Missing configuration parameter ALPHAMILP')
    else:
         config.ALPHAMILP = float(configfile['MAIN']['ALPHAMILP'])
         
    if configfile['MAIN']['FAILURE_PROBABILITY'] is None:
        print('Missing configuration parameter FAILURE_PROBABILITY')
    else:
         config.FAILURE_PROBABILITY = float(configfile['MAIN']['FAILURE_PROBABILITY'])
         
    if configfile['MAIN']['N_SIM'] is None:
        print('Missing configuration parameter N_SIM')
    else:
         config.N_SIM = int(configfile['MAIN']['N_SIM'])
         
    if configfile['MAIN']['REPEATS'] is None:
        print('Missing configuration parameter REPEATS')
    else:
         config.REPEATS = int(configfile['MAIN']['REPEATS'])
         
         
def solve_orienteering_problem_MILP(og):
    prob = create_pulp_instance(og,config.ALPHAMILP)
    if config.SOLVER == 'Gurobi':
           # print('Starting Gurobi...')
       path_to_gurobi = r'/usr/local/bin/gurobi_cl'
           # print('Creating solver...')
       solver = pulp.GUROBI_CMD(path=path_to_gurobi,msg=0,timeLimit=600)
           # print('Calling solver...')
       a = prob.solve(solver)
    else:
#            print('Starting CBC...')
       a = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return a,prob

def solve_orienteering_problem_MCTS_MILP(og,visited,leaf,budget):
    prob = create_pulp_instance_MCTS(og,config.ALPHAMILP,visited,leaf,budget) 
    if config.SOLVER == 'Gurobi':
           # print('Starting Gurobi...')
       path_to_gurobi = r'/usr/local/bin/gurobi_cl'
           # print('Creating solver...')
       solver = pulp.GUROBI_CMD(path=path_to_gurobi,msg=0,timeLimit=600)
           # print('Calling solver...')
       a = prob.solve(solver)
    else:
#            print('Starting CBC...')
       a = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return a,prob
         

if __name__ == "__main__":
    
    print('Starting...')
    
    parser = argparse.ArgumentParser(description='Process parameters.')
    parser.add_argument('--logdir', type = str, default = 'sandbox', help = 'Directory where data will be saved')
    parser.add_argument('--conf', type = str, default = 'config.txt', help = 'Config file to use')
    args = parser.parse_args()

    read_configuration(args.conf)   
    
    if not os.path.isdir(args.logdir):
        print("{} does not exist and will be created".format(args.logdir))
        # create all intermediate folders if needed
        create_nested_path(args.logdir)
    
    #backup configuration file and the version of the code used
    shutil.copyfile(args.conf,os.path.join(args.logdir,"config.txt"))
    shutil.copyfile("MCTS_MILP.py",os.path.join(args.logdir,"MCTS_MILP.py"))
    
    print('Processing graph with {} vertices'.format(config.NVERTICES))
    print('Budget is ',config.BUDGET)
    print('Failure probability is ',config.FAILURE_PROBABILITY)
    
    og = graph.OrienteeringGraph('../datasets/graph_test_{}.mat'.format(config.NVERTICES))
    og.budget = config.BUDGET
  #  print(sum([a.value for a in og.vertices.values()]))
    
   
  #  print(prob)
  
    if config.SOLVER == 'Gurobi':
        print("Solving using Gurobi...")
    else:
        print("Solving using CBC MILP Solver...")
        
    print("Solving the problem {} times".format(config.REPEATS))
    bar = Bar('Processing', max=config.REPEATS) 
    
    obj_list = [] 
    time_list = []
    failure_list = []
     
    for i in range(config.REPEATS):
  
        #prob = create_pulp_instance(og,config.ALPHAMILP)
        #if config.SOLVER == 'Gurobi':
           # print('Starting Gurobi...')
        #    path_to_gurobi = r'/usr/local/bin/gurobi_cl'
           # print('Creating solver...')
        #    solver = pulp.GUROBI_CMD(path=path_to_gurobi,msg=0,timeLimit=600)
           # print('Calling solver...')
        #    a = prob.solve(solver)
       # else:
#            print('Starting CBC...')
       #     a = prob.solve(pulp.PULP_CBC_CMD(msg=0))  

        a,prob = solve_orienteering_problem_MILP(og)

        if a == 1:
            #print("Problem correctly solved")
          #  print("Solution")
            #varsdict = {}
            #for v in prob.variables():
            #    varsdict[v.name] = v.varValue
            #print(varsdict)
            #extract_solution(prob)
            sol_length = compute_solution_length(prob,og)
            #print("Length of the solution: {}".format(sol_length))
            obj_value = prob.objective.value()
            obj_list.append(obj_value)
            time_list.append(prob.solutionTime)
            #print("Objective value: {}".format(obj_value))
            #print("Time spent: {}".format(prob.solutionTime))
            #print('Solution found. Now simulating it {} times'.format(config.N_SIM))
            failures = 0
            for _ in range(config.N_SIM):
                if simulate_solution_length(prob,og) > og.budget:
                    failures += 1
            #print("Nominal failure probability: {}".format(config.FAILURE_PROBABILITY))
            #print("Target failure probability: {}".format(config.ALPHA))
            #print("Effective failure probability: {}".format(failures/config.N_SIM))
            failure_list.append(failures/config.N_SIM)
        else:
            print("The solver did not solve the problem. Return code is {}".format(a))
            raise SystemExit(1)
            
        bar.next()
    bar.finish()

    print("Objective values:",obj_list)
    print("Time values:",time_list)
    print("Failure values:",failure_list)
   
    with open(os.path.join(args.logdir,'results.txt'),"w") as f:
        f.write("Comprehensive Results\n")
        f.write("Vertices:{}\n".format(config.NVERTICES))
        f.write("Budget:{}\n".format(config.BUDGET))
        f.write("Average Reward: {}\n".format(np.mean(obj_list)))
        f.write("Stddev Reward: {}\n".format(np.std(obj_list)))
        #f.write("Path length: {}\n".format(sol_length ))
        f.write("Average Solution time: {}\n".format(np.mean(time_list)))
        f.write("Stddev Solution time: {}\n".format(np.std(time_list)))
        f.write("Nominal Failure Probability:{}\n".format(config.FAILURE_PROBABILITY))
        f.write("Adjusted Failure Probability: {}\n".format(config.ALPHAMILP))
        f.write("Average Actual Failure Probability: {}\n".format(np.mean(failure_list)))
        f.write("Stddev Actual Failure Probability: {}\n".format(np.std(failure_list)))
    

        
    
        
