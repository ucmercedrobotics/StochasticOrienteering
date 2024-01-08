#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 21:27:57 2021

@author: shamano
    """


import sys

sys.path.append('../OrienteeringGraph')
sys.path.append('../StochasticOrienteeringMILP')

import graph
import mcts_MILP
import pulp
import math
import numpy as np
import pickle
from progress.bar import Bar
import configparser
import argparse
import time
import os.path
import os
import shutil
import importlib
import config

importlib.reload(config)


BUFFER_DISTANCES = {}
SUM_BUFFER_DISTANCE = {}
MEAN_DIST = {}
CHILDREN_MAP = {}
KNEAR = 11

class MCTS_node:
    def __init__(self,n,p=None):
        self.node = n
        self.parent = p
        self.children = []
        self.childrenmap = {}
        if p is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1
        self.Q = {}
        self.N = {}
        self.F = {}
     #   self.B = {}
      #  self.T = {}

    def is_leaf(self):
        return len(self.children)==0
    
    def has_child(self,n): # checks if the node has a child with a given id
        if self.N.get(n):
            return True
        else:
            return False

    def set_parent(self,p):
        self.parent = p

    def add_child(self,n):
        if n.node.id == 0:
            print('Tring to add child node with 0 id.')
        self.children.append(n)
        if n.node.id in self.childrenmap:
            print("Adding duplicate child node {} to {}".format(n.node.id,self.node.id))
        self.childrenmap[n.node.id] = n # useful to retrieve a child node by id
        n.set_parent(self)
        self.Q[n.node.id] = 0
        self.N[n.node.id] = 0
        self.F[n.node.id] = 0
        
    def get_child(self,id):
        return self.childrenmap.get(id)
                
    def get_ancestors(self):
        retval = []
        ct = self
        while ct is not None:
            retval.append(ct.node.id)
            ct = ct.parent
        if not ( 0 in retval ):
            pass
        return retval

class MCTS_rollout_data:
    def __init__(self,g,s,tn,b,bias=None):
        self.og = g    # orienteering graph
        self.leaf = s  # orienteering vertex
        self.treenode = tn  # MCTS node
        #if self.treenode.node.id != self.leaf.id:
        #    print("Here is wrong pointer")
        self.budget = b  # residual budget
        if bias is None:
            self.bias = config.BIAS
        else:
            self.bias = bias    
        
    def set_bias(self,bias):
        self.bias = bias

def get_key_with_max_val(d):   # CANDIDATE FOR BETTER IMPLEMENTATION --> should be faster now
    # bestValue = float('-inf')
    # bestKey = None
    # for i in d.keys():
    #     if d[i] > bestValue:
    #         bestValue = d[i]
    #         bestKey = i
    # if bestKey is None:
    #    # traceback.print_stack()
    #     print("Here I am")
    # return bestKey

    a = np.argmax(list(d.values()))
    keys = list(d.keys())
    return keys[a]


def get_key_with_min_val(d):
    bestValue = float('inf')
    bestKey = None
    for i in d.keys():
        if d[i] < bestValue:
            bestValue = d[i]
            bestKey = i
    if bestKey is None:
       # traceback.print_stack()
        print("Here I am")
    return bestKey



def MCTS_TASE_prefill_buffer(g):
    end_id =  g.get_end_vertex()
    for i in range(g.get_n_vertices()):
        for j in range(g.get_n_vertices()):
            if ( i!= j) and (i!= end_id):
                v = g.get_vertex_by_id(i)
                dist1 = v.get_edge_towards_vertex(j).sample_cost_parallel(SAMPLES)
                BUFFER_DISTANCES[(v.id,j)]= dist1
                
    for i in range(g.get_n_vertices()):
        if (i!= end_id):
            dist2 = g.vertices[i].get_edge_towards_vertex(end_id).sample_cost_parallel(SAMPLES)
            BUFFER_DISTANCES[(i,end_id)]= dist2

    for i in range(g.get_n_vertices()):
        for j in range(g.get_n_vertices()):
            if (i!=j) and (i!= end_id) and (j!=end_id):
                SUM_BUFFER_DISTANCE[(i,j,j,end_id)] = BUFFER_DISTANCES[(i,j)] + BUFFER_DISTANCES[(j,end_id)] 
                MEAN_DIST[(i,j,j,end_id)] = np.mean(SUM_BUFFER_DISTANCE[(i,j,j,end_id)])

    for i in g.vertices.keys():
        if i!= end_id:
            neighbors = []
            for j in g.vertices.keys():
                if (i != j):
                    neighbors.append((j,g.vertices[j].value/g.vertices[i].get_edge_towards_vertex(j).d))

            neighbors.sort(key=lambda y: y[1],reverse=True)
            subset = [y[0] for y in neighbors]
            children = list(subset[0:KNEAR])
            if len(children) < len(subset):
                subset = subset[KNEAR:2*KNEAR]
                for _ in range(min(4,len(subset))):
                    toadd = np.random.choice(subset)
                    children.append(toadd)
                    subset.remove(toadd)
           # subset = [y[0] for y in neighbors]
            #children = list(subset)
          #  subset = children[0:2*KNEAR]
            if end_id not in children:
                children.append(end_id)
            CHILDREN_MAP[i] = list(children[:])

def MCTS_get_new_vertex_greedy(g,current,children,budget,unusable=None):
    end_id = g.get_end_vertex()
    r = end_id
    max_reward = -1
    for i in children:
        if (i != end_id) and (i not in unusable):
            v = g.get_vertex_by_id(current)
    #        if not (v.id,i) in BUFFER_DISTANCES:
    #            dist1 = v.get_edge_towards_vertex(i).sample_cost_parallel(SAMPLES)
    #            BUFFER_DISTANCES[(v.id,i)]= dist1
    #        else:
  #          dist1 = BUFFER_DISTANCES[(v.id,i)]
     #       if not (i,end_id) in BUFFER_DISTANCES:
     #           dist2 = g.vertices[i].get_edge_towards_vertex(end_id).sample_cost_parallel(SAMPLES)
     #           BUFFER_DISTANCES[(i,end_id)]= dist2
     #       else:
   #         dist2 = BUFFER_DISTANCES[(i,end_id)]

     #       failure_prob = (((dist1+dist2)>budget).sum())/SAMPLES
            failure_prob = ((SUM_BUFFER_DISTANCE[(v.id,i,i,end_id)]>budget).sum())/SAMPLES
            if failure_prob < config.FAILURE_PROB :
                #dist = np.mean(dist1+dist2)
                dist = MEAN_DIST[(v.id,i,i,end_id)]
                value = g.vertices[i].value
                ratio = value / dist
                if  ratio > max_reward:
                    r = i
                    max_reward = ratio  
    return r
    

def MCTS_greedy_rollout_expansion(leaf,g,budget,vs):   # only called if leaf is a leaf different from the end vertex
    end_vertex_idx = g.get_end_vertex()

    children = [i for i in range(g.get_n_vertices())]
    bv = leaf
    # remove ancestors
    while bv is not None:
        children.remove(bv.node.id)
        bv = bv.parent
    # remove visited vertices
    for j in list(g.vertices.values()):
        if (j.get_visited()) and ( j.id in children ):
            children.remove(j.id)

    current = leaf.node.id
    done = False

    traversed = []
    while not done:
        if np.random.uniform() < config.GREEDY_THRESHOLD:
            new = MCTS_get_new_vertex_greedy(g,current,children,budget)
        else:
            new = np.random.choice(children)

        traversed.append(new)
        time_to_go = 0
        for _ in range(config.REPETITIONS):
            time_to_go += g.vertices[current].get_edge_towards_vertex(new).sample_cost()
        time_to_go /= config.REPETITIONS
        if time_to_go > budget:
            return traversed, True# failure
        if new == end_vertex_idx:
            done = True
        else:
            children.remove(new)
            budget -= time_to_go
        current = new

    return traversed, False


def MCTS_sample_traverse_cost_ids(vs,g,number=1):  # OK
    #CURRENT WORKING VERSION
    #value = 0
    #for source,dest in  vs:
    #    edge = g.vertices[source].get_edge_towards_vertex(dest)
    #    value += edge.sample_cost()
    #return value
    
    #values = np.zeros((number,))
    values = 0
    for source,dest in  vs:
        edge = g.vertices[source].get_edge_towards_vertex(dest)
        values += edge.sample_cost()
    return values
        


def MCTS_sample_traverse_cost(sequence,g):  # OK
    value = 0
    times = []
    for i in range(len(sequence)-1):
        source = sequence[i].node.id
        dest = sequence[i+1].node.id
        edge = g.vertices[source].get_edge_towards_vertex(dest)
        t = edge.sample_cost()
        times.append(t)
        value += t
    return value,times


def MCTS_TASE_pick_action_max(node,og):  # only called on a node that is not a leaf
    if node.is_leaf():
        print(node.node.id)
        print(CHILDREN_MAP[node.node.id])
        raise Exception("Can't do action selection on a childless node")
    totalN = sum([i for i in node.N.values()])
   # totalReward= og.get_total_reward()
    uct_values = {}
    for i in node.children:
        uct_values[i.node.id] = node.Q[i.node.id]*(1-node.F[i.node.id]) + 3* math.sqrt(math.log(totalN)/node.N[i.node.id])
    return get_key_with_max_val(uct_values)


def id_sequence_to_node_sequence(vs,root):
    cp = root
    sequence = [cp]
    for i in vs[1:]:
        sequence.append(cp.get_child(i))
        cp = cp.get_child(i)
    return sequence

# Implements tree policy. It will either:
# pick a child node that has not been tried yet (if it exists)
# descend into a child using a constrained UCTF and apply itself recursively from there
def MCTS_TASE_Traverse(current_vertex,g):

    bakcup_init_vertex = current_vertex

    visited = {}
    for v in g.vertices.values():
        visited[v.id] = v.visited
    visited[current_vertex.node.id] = True # can't go back to same vertex I'm coming from

    if visited[g.get_end_vertex()]:
        print("We have a problem in MCTS_traverse_random_explore")
        _ = input("Press Enter to continue.")

    traversed = []
    traversed.append(current_vertex.node.id)

    while True:
        # FIRST: search for an unvisited child, if it exists
        candidates = list(CHILDREN_MAP[current_vertex.node.id])
        for c in candidates[:]:
            if visited[c] or current_vertex.has_child(c):
                    candidates.remove(c)

        if len(candidates) > 0:   # Unvisited child exists; then pick one  and return
            down_vertex = np.random.choice(candidates)
            traversed.append(down_vertex)
            return traversed
        else:  # all children have been visited; pick one using constrained UCT and recurse down
            bestAction = MCTS_TASE_pick_action_max(current_vertex,g)
            traversed.append(bestAction)
           # return traversed ######## TO BE REMOVED
            if bestAction == g.get_end_vertex():
                return traversed
            else:
                current_vertex = current_vertex.get_child(bestAction)
                visited[current_vertex.node.id] = True  # this vertex can't be chosen anymore

def MCTS_TASE_greedy_rollout2(leaf,g,budget,vs,visited):
    end_id = g.get_end_vertex()

    current = leaf.node.id
    done = False

    traversed = []
    reward = 0

    unusable = dict(visited)

    while not done:

        children = set(CHILDREN_MAP[current]) - set(unusable.keys())
        children = list(children)

        new = np.random.choice(children)
        v = g.get_vertex_by_id(current)
        if new != end_id:
            failure_prob = ((SUM_BUFFER_DISTANCE[(v.id,new,new,end_id)]>budget).sum())/SAMPLES
            if failure_prob <= config.FAILURE_PROB :
                reward += g.vertices[new].value
                traversed.append(new)
                time_to_go = g.vertices[current].get_edge_towards_vertex(new).sample_cost()
                budget -= time_to_go
                current = new
            unusable[new] = True
            if budget < 0:
                return traversed,True,reward
        else:
            traversed.append(new)
            reward += g.vertices[new].value
            time_to_go = g.vertices[current].get_edge_towards_vertex(new).sample_cost()
            if time_to_go > budget:
                return traversed,True,reward
            else:
                return traversed,False,reward


def MCTS_TASE_greedy_rollout3(leaf,g,budget,vs,visited):
    end_id = g.get_end_vertex()

    current = leaf.node.id
    done = False

    traversed = []
    reward = 0

    unusable = dict(visited)

    while not done:

        children = set(CHILDREN_MAP[current]) - set(unusable.keys())
        children = list(children)

        if np.random.uniform() < config.PROBABILTY_RANDOM:
            new = np.random.choice(children)
        else:
            new = MCTS_get_new_vertex_greedy(g,current,children,budget,unusable)
        v = g.get_vertex_by_id(current)
        if new != end_id:
            failure_prob = ((SUM_BUFFER_DISTANCE[(v.id,new,new,end_id)]>budget).sum())/SAMPLES
            if failure_prob <= config.FAILURE_PROB :
                reward += g.vertices[new].value
                traversed.append(new)
                time_to_go = g.vertices[current].get_edge_towards_vertex(new).sample_cost()
                budget -= time_to_go
                current = new
            unusable[new] = True
            if budget < 0:
                return traversed,True,reward
        else:
            traversed.append(new)
            reward += g.vertices[new].value
            time_to_go = g.vertices[current].get_edge_towards_vertex(new).sample_cost()
            if time_to_go > budget:
                return traversed,True,reward
            else:
                return traversed,False,reward
            
def MCTS_TASE_greedy_rollout4(leaf,g,budget,vs,visited):
    
    local_g = g.clone()
    local_g.clear_visits()
    local_g.budget = budget
    to_keep = set(local_g.vertices.keys())-set(visited)
    local_g.shrink_to_subset_of_vertices(to_keep)
    
    current = leaf.node.id
    
    prob = mcts_MILP.create_pulp_instance(local_g,0.1)
    a = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    
    
    

    end_id = g.get_end_vertex()

    current = leaf.node.id
    done = False

    traversed = []
    reward = 0

    unusable = dict(visited)

    while not done:

        children = set(CHILDREN_MAP[current]) - set(unusable.keys())
        children = list(children)

        if np.random.uniform() < 0.5:
            new = np.random.choice(children)
        else:
            new = MCTS_get_new_vertex_greedy(g,current,children,budget,unusable)
        v = g.get_vertex_by_id(current)
        if new != end_id:
            failure_prob = ((SUM_BUFFER_DISTANCE[(v.id,new,new,end_id)]>budget).sum())/SAMPLES
            if failure_prob <= config.FAILURE_PROB :
                reward += g.vertices[new].value
                traversed.append(new)
                time_to_go = g.vertices[current].get_edge_towards_vertex(new).sample_cost()
                budget -= time_to_go
                current = new
            unusable[new] = True
            if budget < 0:
                return traversed,True,reward
        else:
            traversed.append(new)
            reward += g.vertices[new].value
            time_to_go = g.vertices[current].get_edge_towards_vertex(new).sample_cost()
            if time_to_go > budget:
                return traversed,True,reward
            else:
                return traversed,False,reward

def MCTS_TASE_greedy_rollout(leaf,g,budget,vs,visited):
    end_vertex_idx = g.get_end_vertex()
    
    current = leaf.node.id
    done = False

    traversed = []
    reward = 0
    
    all = [i for i in range(end_vertex_idx+1)]
    
    unusable = dict(visited)

    while not done:

        children = set(CHILDREN_MAP[current]) - set(unusable.keys())
        children = list(children)
        
        # depth limited rollout
        if len(traversed) < 1:
                new = np.random.choice(children)
        else: #len(traversed)<3:
            new = MCTS_get_new_vertex_greedy(g,current,children,budget,unusable)

        reward += g.vertices[new].value

        traversed.append(new)
        time_to_go = g.vertices[current].get_edge_towards_vertex(new).sample_cost()
        if time_to_go > budget:
            return traversed, True,reward# failure
        if new == end_vertex_idx:
            done = True
        else:
            unusable[new] = True
            budget -= time_to_go
        current = new
    return traversed, False,reward


def MCTS_TASE_Backup(toadd,reward,failure_rate):

    parent = toadd.parent
    nodebackup = toadd

    # this must be done in any case
    if parent.N[toadd.node.id] == 0: # adding a new child ?
        parent.Q[toadd.node.id] = reward
        parent.F[toadd.node.id] = failure_rate
    else:  # update statistics using incremental formulas
        parent.Q[toadd.node.id] = (parent.Q[toadd.node.id]*parent.N[toadd.node.id] + reward) / (parent.N[toadd.node.id]+1)
        parent.F[toadd.node.id] = (parent.F[toadd.node.id]*parent.N[toadd.node.id] + failure_rate) / (parent.N[toadd.node.id]+1)

    current = toadd

    # now propagate upwards
    while parent.parent:  # as long as there is a v_k
        
        if (parent.parent.F[parent.node.id] <= config.FAILURE_PROB): #v_i is feasible;
            if parent.F[current.node.id] <= config.FAILURE_PROB: # consider changes only if not introducing a violation
                if parent.parent.Q[parent.node.id] < parent.node.value + parent.Q[current.node.id]: # found path with better reward?
                    parent.parent.Q[parent.node.id] = parent.node.value + parent.Q[current.node.id]
                    parent.parent.F[parent.node.id] = parent.F[current.node.id]
        else:  # v_k violates the constraint; make changes only if F value comes down
            if (parent.parent.F[parent.node.id] > parent.F[current.node.id]):  # found child lowering failure rate
                parent.parent.F[parent.node.id] = parent.F[current.node.id]
                parent.parent.Q[parent.node.id] = parent.Q[current.node.id] + parent.node.value

        current = current.parent
        parent = parent.parent

    toadd = nodebackup
    parent = toadd.parent
    while parent:  # propagate count upwards
       parent.N[toadd.node.id] += 1
       parent = parent.parent
       toadd=toadd.parent
        
        
    return




def get_reward_sequence_from_id_sequence(s,g):
    retval = []
    for i in s:
        n = g.get_vertex_by_id(i)
        retval.append(n.get_value())
    return retval

def check_early_exit(root):
    s = [i for i in root.N.values()]
    if len(s) == 1:
        return True
    s.sort(reverse=True)
    if s[0]>2*s[1]:
        return True
    else:
        return False

    
def MCTS_TASE_pick_root_action(root,og):  # picks the best action on the root among those who satisfy the failure constraints
    candidates = {}
    for i in root.children:
        if root.F[i.node.id] <= config.FAILURE_PROB:
            candidates[i.node.id] = root.Q[i.node.id]
    if candidates:
        return get_key_with_max_val(candidates)
    else:
        return None


# THIS IS THE ONE USED FOR THE TASE SUBMISSION
def MCTS_search_TASE(g,start_vertex,budget,max_iterations):
    root = MCTS_node(g.get_vertices()[start_vertex])
    end_node_id = g.get_end_vertex()
    arm_pulls = 0

    for itc_counter in range(max_iterations):
        current_vertex = root
        vs = MCTS_TASE_Traverse(current_vertex,g)
        
        arm_pulls += 1

        sequence = id_sequence_to_node_sequence(vs,root)
        parent = sequence[-2]

        if not parent.has_child(vs[-1]):
            toadd = MCTS_node(g.get_vertices()[vs[-1]],parent)
            parent.add_child(toadd)
        else:
            toadd = parent.get_child(vs[-1])

        sequence = id_sequence_to_node_sequence(vs,root)
        leaf = sequence[-1]
        rewards_list = []
        fail_list = []

        unusable = {}
        bv = leaf
        # remove ancestors
        while bv is not None:
            unusable[bv.node.id]=True
            bv = bv.parent
        # remove visited vertices
        for j in list(g.vertices.values()):
            if (j.get_visited()) and (j.id not in unusable):
                unusable[j.id] = True

        if vs[-1] != end_node_id:
            couples = list(zip(vs, vs[1:]))
            #toggle = True
            for _ in range(int(SAMPLES)):
             #   if toggle:
                cost_to_leaf = MCTS_sample_traverse_cost_ids(couples,g)
              #      toggle = not toggle
                traversed,fail,reward = rollout_function(leaf,g,budget-cost_to_leaf,vs,unusable)
                fail_list.append(fail)
                if not fail:
                    #reward_sequence = get_reward_sequence_from_id_sequence(traversed,g)
                    #reward_sequence.insert(0,leaf.node.value)
                    reward += g.vertices[vs[-1]].value
                    rewards_list.append(reward)
            if not rewards_list:
                rewards_list.append(0)
        else:
            rewards_list = [g.get_vertex_by_id(end_node_id).get_value()]
            couples = list(zip(vs, vs[1:]))
            for _ in range(int(SAMPLES)):
                a = MCTS_sample_traverse_cost_ids(couples,g)
                fail_list.append(a>budget)

        reward = np.mean(rewards_list)
        failure_rate = sum(fail_list)/len(fail_list)

        MCTS_TASE_Backup(toadd,reward,failure_rate)
        if (itc_counter+1) % 10 == 0:
            if check_early_exit(root):
                break

    bestAction = MCTS_TASE_pick_root_action(root,g)

    if bestAction is not None:
        return root,root.node.get_edge_towards_vertex(bestAction),arm_pulls
    else:
        return root,root.node.get_edge_towards_vertex(end_node_id),arm_pulls



def MCTS_TASE_simulate(og,max_iterations,verbose=True):
    current_budget = og.get_budget()
    current_vertex = og.get_start_vertex()
    goal_vertex = og.get_end_vertex()
    og.vertices[current_vertex].set_visited()
    cumulative_reward = 0
    done = False
    traversed = [current_vertex]
    arm_pulls = 0

    while not done:

        # THIS IS USED FOR TASE
        tree,action,pulls = MCTS_search_TASE(og,current_vertex,current_budget,max_iterations)
        traversed.append(action.dest.id)
        arm_pulls += pulls

        if action is None:
            return 0,-1,None  # failure; could not find action
        else:
            current_vertex = action.dest.id   # execute action (i.e., move to vertex)
            cumulative_reward = cumulative_reward + og.vertices[current_vertex].get_value() # cash reward
            og.vertices[current_vertex].set_visited()  # mark vertex visited so it won't be considered
            current_budget = current_budget - action.sample_cost() # pay travel cost
            if (current_budget < 0) or (current_vertex == goal_vertex): # task ends if run out of budget or reached goal vertex
                done = True

    return cumulative_reward,current_budget,tree,traversed,arm_pulls




def read_configuration(fname):
    
    configfile = configparser.ConfigParser()
    print("Reading configuration file ",fname)
    if os.path.exists(fname):
        configfile.read(fname)
    else:
        raise Exception("Can't read configuration file {}".format(fname))
    
    if configfile['MAIN']['NTRIALS'] is None:
        print('Missing configuration parameter NTRIALS')
    else:
         config.NTRIALS = int(configfile['MAIN']['NTRIALS'])
            
    if configfile['MAIN']['REPETITIONS'] is None:
        print('Missing configuration parameter REPETITIONS')
    else:
         config.REPETITIONS = int(configfile['MAIN']['REPETITIONS']) 
         
    if configfile['MAIN']['EPSMIN'] is None:
        print('Missing configuration parameter EPSMIN')
    else:
         config.EPSMIN = float(configfile['MAIN']['EPSMIN']) 
         
    if configfile['MAIN']['EPSN'] is None:
        print('Missing configuration parameter EPSN')
    else:
         config.EPSN = int(configfile['MAIN']['EPSN']) 
        
    if configfile['MAIN']['EPSINC'] is None:
        print('Missing configuration parameter EPSINC')
    else:
         config.EPSINC = float(configfile['MAIN']['EPSINC']) 
         
    if configfile['MAIN']['ITERMIN'] is None:
        print('Missing configuration parameter ITERMIN')
    else:
         config.ITERMIN = int(configfile['MAIN']['ITERMIN']) 
         
    if configfile['MAIN']['ITERINC'] is None:
        print('Missing configuration parameter ITERINC')
    else:
         config.ITERINC = int(configfile['MAIN']['ITERINC']) 
        
    if configfile['MAIN']['ITERN'] is None:
        print('Missing configuration parameter ITERN')
    else:
         config.ITERN = int(configfile['MAIN']['ITERN']) 
         
    if configfile['MAIN']['SAMPLESMIN'] is None:
        print('Missing configuration parameter SAMPLESMIN')
    else:
         config.SAMPLESMIN = int(configfile['MAIN']['SAMPLESMIN'])   
         
    if configfile['MAIN']['SAMPLESINC'] is None:
        print('Missing configuration parameter SAMPLESINC')
    else:
         config.SAMPLESINC = int(configfile['MAIN']['SAMPLESINC'])   
         
    if configfile['MAIN']['SAMPLESN'] is None:
        print('Missing configuration parameter SAMPLESN')
    else:
        config.SAMPLESN = int(configfile['MAIN']['SAMPLESN'])   
         
    if configfile['MAIN']['BIAS'] is None:
        print('Missing configuration parameter BIAS')
    else:
         config.BIAS = float(configfile['MAIN']['BIAS'])         

    if configfile['MAIN']['VERBOSE'] is None:
        print('Missing configuration parameter VERBOSE')
    else:
         config.VERBOSE = (configfile['MAIN']['VERBOSE'] == "True")
         
    if configfile['MAIN']['BUDGET'] is None:
        print('Missing configuration parameter BUDGET')
    else:
         config.BUDGET = float(configfile['MAIN']['BUDGET'])  
         
    if configfile['MAIN']['FAILURE_PROB'] is None:
        print('Missing configuration parameter FAILURE_PROB')
    else:
         config.FAILURE_PROB = float(configfile['MAIN']['FAILURE_PROB']) 
         
    if not ('PROBABILITY_RANDOM' in configfile['MAIN']):
        print(f'Missing configuration parameter PROBABILTY_RANDOM. Using default value {config.PROBABILTY_RANDOM}')
    else:
        config.PROBABILTY_RANDOM = float(configfile['MAIN']['PROBABILITY_RANDOM'])
        print(f'Setting PROABABILTY_RANDOM to {config.PROBABILTY_RANDOM}')
         
    if configfile['MAIN']['GREEDY_THRESHOLD'] is None:
        print('Missing configuration parameter GREEDY_THRESHOLD')
    else:
         config.GREEDY_THRESHOLD = float(configfile['MAIN']['GREEDY_THRESHOLD'])
         
    if configfile['MAIN']['ROLLOUT'] is None:
        print('Missing configuration parameter ROLLOUT')
    else:
         config.ROLLOUT = int(configfile['MAIN']['ROLLOUT'])
         
    if configfile['MAIN']['FILENAME'] is None:
        print('Missing mandatory configuration parameter FILENAME. Aboriting')
        exit(1)
    else:
         config.FILENAME = configfile['MAIN']['FILENAME']     
         
    print('Done reading configuration')
    
    
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

if __name__ == "__main__":
    
    print('Starting...')
    
    parser = argparse.ArgumentParser(description='Process parameters.')
    parser.add_argument('--logdir', type = str, default = 'sandbox',help = 'Directory where data will be saved')
    parser.add_argument('--conf', type = str, default = 'config.txt',help = 'Config file to use')
  #  parser.add_argument('--conf', type = str, default = 'config_files/config_ulysses16_50_005.txt',help = 'Config file to use')
    
    args = parser.parse_args()
    
    read_configuration(args.conf)   
    
    if not os.path.isdir(args.logdir):
        print("{} does not exist and will be created".format(args.logdir))
        # create all intermediate folders if needed
        create_nested_path(args.logdir)
    
    #backup configuration file and the version of the code used
    shutil.copyfile(args.conf,os.path.join(args.logdir,"config.txt"))
    shutil.copyfile("MCTS.py",os.path.join(args.logdir,"MCTS.py"))
    
    
    og = graph.OrienteeringGraph('../datasets/'+config.FILENAME)
    config.NVERTICES = og.get_n_vertices()
    og.budget = config.BUDGET
    og.set_budget(config.BUDGET)
    
    print('Processsing file ../datasets/'+config.FILENAME)
    print('Number of vertices:',config.NVERTICES)
    print('Budget:',config.BUDGET)
    print('Total reward: ',og.get_total_reward())
    print("Rollout strategy:",config.ROLLOUT)
    
    if config.ROLLOUT == 1:
        rollout_function = MCTS_TASE_greedy_rollout
    elif config.ROLLOUT ==2:
        rollout_function = MCTS_TASE_greedy_rollout2
    elif config.ROLLOUT == 3:
        rollout_function = MCTS_TASE_greedy_rollout3
    elif config.ROLLOUT == 4:
        rollout_function = MCTS_TASE_greedy_rollout4
    else:
        raise ValueError("Unknown rollout function")
        
    print('Starting simulation')
    ntrials = config.NTRIALS
    
    #biases = np.linspace(0,1,11).tolist()
    biases = [0.2] 
    
    epsilon_vals = [(config.EPSMIN + i*config.EPSINC) for i in range(config.EPSN)]
    best_reward = -1
    k_bias = 5
    iterations_list =  [(config.ITERMIN + i*config.ITERINC) for i in range(config.ITERN)]
    samples_list = [(config.SAMPLESMIN + i*config.SAMPLESINC) for i in range(config.SAMPLESN)]
    
    best_residual_budget = []
    best_reward = []
    best_failure_rate = []
    best_bias = []
    best_iterations = []
    best_tree_leaves = []
    best_tree_depth = []
    best_p_factor = []
    best_folicy_failure =[]
    best_time = []
    best_pulls = []
    
    config.BIAS = 0.1
    
    absolute_best = -1
    found_absolute_best = False
    
    complete_time_series = []
    complete_reward_series = []
    complete_budget_series = []
    complete_pulls_series = []
    
    for SAMPLES in samples_list:
        
        for iterations in iterations_list:

            for EPSILON in epsilon_vals:

                rewards = []
                budgets = []
                times = []
                arm_pulls = []
                totalRewards = 0
                failures = 0
                policyFailure = 0

                print("SAMPLES=",SAMPLES)
                print("Iterations=",iterations)
                print("Epsilon=",EPSILON)
                print("Failure probability=",config.FAILURE_PROB)

                bar = Bar('Processing', max=ntrials)
                residual = 0

                for _ in range(ntrials):
                    CHILDREN_MAP = {}
                    MCTS_TASE_prefill_buffer(og)
                    start = time.time()
                    reward,budget,tree,traversed,pulls = MCTS_TASE_simulate(og,iterations,config.VERBOSE)   
                    end = time.time()
                    rewards.append(reward)
                    budgets.append(budget)
                    times.append(end-start)
                    arm_pulls.append(pulls)
                    complete_time_series.append(end-start)
                    complete_reward_series.append(reward)
                    complete_budget_series.append(budget)
                    complete_pulls_series.append(pulls)
                    if config.VERBOSE:
                        print("\nReward: ",reward)    
                        print("Budget: ",budget)
                        print("Time: ",end-start)
                        print("Pulls: ",pulls)
                        print("Path: ",traversed)
                    if tree is None:
                        policyFailure += 1
                    if budget < 0:
                        failures = failures + 1
                    else:
                        totalRewards += reward
                        residual += budget
                            # reset flags to start a new iteration
                    for i in og.vertices.keys():
                        og.vertices[i].clear_visited()
                    og.vertices[0].set_visited()
                    bar.next()
                bar.finish()
                if ntrials-failures > 0:
                    av_rev = totalRewards/(ntrials-failures)
                    av_time = sum(times)/ntrials
                    av_pulls = sum(arm_pulls)/ntrials
                    #print("Average reward: ",av_rev)
                    best_residual_budget.append(residual / (ntrials - failures))
                    best_reward.append(av_rev)
                    best_failure_rate.append(failures/ntrials)
                    best_folicy_failure.append(policyFailure/ntrials)
                    best_time.append(av_time)
                    best_pulls.append(av_pulls)
                    best_bias.append(config.BIAS)
                    best_iterations.append(iterations)
                  #  best_tree_leaves.append(MCTS_tree_leaf_number(tree))
                  #  best_tree_depth.append(MCTS_tree_depth(tree))
                  #  best_p_factor.append(P_FACTOR)
                    if (av_rev >= absolute_best) and (failures/ntrials <= config.FAILURE_PROB):
                        found_absolute_best = True
                        absolute_best = av_rev
                        best_iterations_val = iterations
                        best_epsilon_val = EPSILON
                        absolute_best_time = av_time
                        absolute_best_pulls = av_pulls
                #print("Saving run data to files....")
                pickle.dump(rewards,open(os.path.join(args.logdir,'rewards_EPSILON_{:.2f}_iterations_{}.dat'.format(EPSILON,iterations)),"wb"))
                #print('Saved rewards_EPSILON_{:.2f}_iterations_{}.dat'.format(EPSILON,iterations))
                pickle.dump(budgets,open(os.path.join(args.logdir,'budgets_EPSILON_{:.2f}_iterations_{}.dat'.format(EPSILON,iterations)),"wb"))
                #print('Saved budgets_EPSILON_{:.2f}_iterations_{}.dat'.format(EPSILON,iterations))
                pickle.dump(times,open(os.path.join(args.logdir,'times_EPSILON_{:.2f}_iterations_{}.dat'.format(EPSILON,iterations)),"wb"))
                #print('Saved times_EPSILON_{:.2f}_iterations_{}.dat'.format(EPSILON,iterations))
                pickle.dump(arm_pulls,open(os.path.join(args.logdir,'pulls_EPSILON_{:.2f}_iterations_{}.dat'.format(EPSILON,iterations)),"wb"))
                
            
    print("----------------")
    print("Comprehensive Results")
    #print("Best Iterations:",best_iterations)   
    #print("BIAS:",best_bias)
    print("Average reward:",best_reward)
    print("Average residual budget:",best_residual_budget)
    print("Average time:",best_time)
    #print("Number of leaves in the best tree:",best_tree_leaves)
    #print("Depth of the best tree:",best_tree_depth)
    print("Failure rate:",best_failure_rate)
    print("Policy failure rate:",best_folicy_failure)
    print("Average pulls:",best_pulls)
    
    print("----------------")
    print("Grid Search Results")
    if found_absolute_best:
        print("Best EPSILON:",best_epsilon_val)
        print("Best Iterations:",best_iterations_val)
        print('Best Reward:',absolute_best)
        print('Best Pulls :',absolute_best_pulls)
        print('Best time:',absolute_best_time)
    else:
        print("No optimal combination found")
        
        
    with open(os.path.join(args.logdir,'results.txt'),"w") as f:
        f.write("Comprehensive Results\n")
        f.write("Vertices:{}\n".format(config.NVERTICES))
        f.write("Budget:{}\n".format(config.BUDGET))
        f.write("Failure Probability:{}\n".format(config.FAILURE_PROB))
        f.write("Iterations:{}\n".format(config.ITERMIN))
        f.write("Independent Runs:{}\n".format(config.NTRIALS))
        f.write("Average reward: {}\n".format(best_reward))
        f.write("Average residual budget: {}\n".format(best_residual_budget))
        f.write("Average pulls: {}\n".format(best_pulls))
        f.write("Average time: {}\n".format(best_time))
        f.write("Failure rate: {}\n".format(best_failure_rate))


    print("Saving data to files....")
    pickle.dump(best_residual_budget,open(os.path.join(args.logdir,'residual_budget.dat'),"wb"))
    pickle.dump(best_reward,open(os.path.join(args.logdir,'reward.dat'),"wb"))
    pickle.dump(best_failure_rate,open(os.path.join(args.logdir,'failure_rate.dat'),"wb"))
    
    pickle.dump(complete_time_series,open(os.path.join(args.logdir,'complete_time_series.dat'),"wb"))
    pickle.dump(complete_reward_series,open(os.path.join(args.logdir,'complete_reward_series.dat'),"wb"))
    pickle.dump(complete_budget_series,open(os.path.join(args.logdir,'complete_budget_series.dat'),"wb"))
    pickle.dump(complete_pulls_series,open(os.path.join(args.logdir,'complete_pulls_series.dat'),"wb"))