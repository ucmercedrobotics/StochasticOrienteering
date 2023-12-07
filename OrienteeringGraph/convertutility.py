#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:30:33 2023

@author: shamano
"""

# converts tsp files to internal mat format

import tsplib95
import numpy as np
import graph
import sys
import scipy.io as sio


def load_graph(fname):
    problem = tsplib95.load(fname)
    G = problem.get_graph()
    return G
    
def convert_graph(G):
    nvertices = len(G.nodes)
    og = graph.OrienteeringGraph(nvertices)
    j = 0
    for i in G.nodes:
        nv = graph.OrienteeringVertex(G.nodes[i]['coord'][0],G.nodes[i]['coord'][1],1,j)
        og.vertices[j]= nv
        j = j+1
        
    og.set_start_vertex(0)
    og.vertices[0].value = 0 # no value in the start vertex
    og.end_vertex = og.nvertices - 1          
        
    for i in range(nvertices):
        for j in range(nvertices):
            if (i != j): #and ( i != self.end_vertex ):  # no loops and end vertex has no outgoing edges
                edge = graph.OrienteeringEdge(og.vertices[i], og.vertices[j], og.alpha)
                og.vertices[i].add_edge(edge)
            
    og.distance_matrix = np.zeros((nvertices,nvertices),dtype=np.float32)
    dist_total = 0
    for i in range(nvertices):
        for j in range(nvertices):
            if (i != j): #and ( i != self.end_vertex ): # do not set values for non-existing edges
                og.distance_matrix[i][j] = og.get_vertex_by_id(i).get_edge_towards_vertex(j).d
                dist_total = dist_total + og.distance_matrix[i][j]
    og.budget = dist_total / 10  # this is just a placeholder
        
    return og

def save_graph(og,fname):
    contents = {}
    coords = np.zeros((og.nvertices,2),dtype=np.float32)
    
    rewards = np.zeros((og.nvertices,1))
    for i in range(og.nvertices):
        coords[i][0] = og.vertices[i].x
        coords[i][1] = og.vertices[i].y
        rewards[i][0] = og.vertices[i].value
    rewards[0][0] = 0 # always no reward in the start location
    
    contents['xy'] = coords
    contents['rewards']= rewards
    edge_list = np.zeros((og.nvertices*(og.nvertices-1),3))
    k = 0
    for i in range(og.nvertices):
        for j in range(og.nvertices):
            if i != j:
                edge_list[k][0] = i
                edge_list[k][1] = j
                edge_list[k][2] = np.linalg.norm(coords[i][:]-coords[j][:])
                k = k+1
    contents['edge_list'] = edge_list
    al = np.zeros((1,1))
    al[0][0]= og.alpha
    contents['alpha'] = al
    b = np.zeros((1,1))
    b[0][0] = og.budget
    contents['t_max'] = b
    sio.savemat(fname,contents)

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usage: python convertutility inputfile.tsp outputfile.mat")
        exit(1)
        
    G = load_graph(sys.argv[1])
    graph.VERBOSE = False # be quiet
    orienteering_graph = convert_graph(G)
    save_graph(orienteering_graph ,sys.argv[2])
    
    og = graph.OrienteeringGraph(sys.argv[2])
    
    if graph.compare_graphs(orienteering_graph, og):
        print("Conversion successfully commpleted")
        print(f"Budget set to {og.budget}")
    else:
        print("Conversion unsuccessful")
    