          #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:11:11 2021

@author: shamano
"""

import sys

sys.path.append('../MCTS-GP-MultiRobot')
sys.path.append('.')

import math
import numpy as np
import scipy.io as sio
#from scipy.stats import expon
import copy

import os
import psutil

#from utils import track


BUFFER_SIZE=50000

VERTEX_MIN = 10
VERTEX_MAX = 50
MAX_REWARD = 1
MAX_X = 1
MAX_Y = 1
VERBOSE=True





class OrienteeringVertex:
    def __init__(self,x,y,value,idv):
        self.x = x
        self.y = y
        self.value = value
        #self.edges = list()
        self.edges = {}
        self.id = idv
        self.visited = False
        
    def get_coordinates(self):
        return self.x,self.y
        
    def get_value(self):
        return self.value

    def set_value(self,value):
        self.value = value
        
    def get_edges(self):
        #return self.edges
        return self.edges.values()
    
    def add_edge(self,edge):
        if edge.source != self:
            raise Exception("Edge does not originate from vertex")
        #self.edges.append(edge)
        self.edges[edge.dest.id] = edge
    
    def set_visited(self):
        self.visited = True
        
    def clear_visited(self):
        self.visited = False
    
    def get_visited(self):
        return self.visited
        
    def add_edge_to_vertex(self,to,alpha):
        if to == self:
            raise Exception("Can't add loop edges")
            
        # avoid adding duplicate edges    
        #for e in self.edges:
        #    if e.dest == to:
        #        return
        #newedge = OrienteeringEdge(self,to,alpha)
        #self.edges.add(newedge)
        if self.edges.has_key(to):
            return
        newedge = OrienteeringEdge(self,to,alpha)
        self.edges[to] = newedge
        
    def get_edge_towards_vertex(self,idx):  # returns the edge towards a vertex with a given id
        if idx not in self.edges:
            raise Exception("Vertex from {} to {} does not exist".format(self.id,idx))
        return self.edges[idx]
        

class OrienteeringEdge:
    def __init__(self,source,dest,alpha):
        self.source = source
        self.dest = dest
        self.alpha = alpha
        self.d = math.sqrt((self.source.x-self.dest.x)**2 + (self.source.y-self.dest.y)**2) 
        self.offset = self.alpha * self.d
        self.parameter = (1-self.alpha)*self.d
        self.buffer = self.offset + np.random.exponential(scale=self.parameter,size=BUFFER_SIZE)
        self.buffer_index = 0
        
    def __del__(self):
        del self.buffer


    # check this -- should be fine
    def sample_cost(self):
        if self.buffer_index == BUFFER_SIZE:
            self.regenerate_buffer()
        value = self.buffer[self.buffer_index]
        self.buffer_index += 1
        return value
    
    def regenerate_buffer(self):
        del self.buffer
        self.buffer = self.offset + np.random.exponential(scale=self.parameter,size=BUFFER_SIZE)
        self.buffer_index = 0
    
    def sample_cost_parallel(self,number=1):
       #return self.offset + np.random.exponential(self.parameter) 
        #3if self.buffer_index == BUFFER_SIZE:
         #   self.regenerate_buffer()
        
        value = np.zeros((number,))
        
        if self.buffer_index + number > BUFFER_SIZE:
            self.regenerate_buffer()
        value[0:number] = self.buffer[self.buffer_index:self.buffer_index+number]
        self.buffer_index += number

        return value
        
def set_verbose():
    global VERBOSE
    VERBOSE=True

def set_silent():
    global VERBOSE
    VERBOSE=False
    


def create_orienteering_graph(xmin,xmax,ymin,ymax,nvertices,alpha,budget,fname,start=None,end=None):
    """
    Creates a complete orienteering graph and saves it to a mat file in a format that can be
    loaded by the OrienteeringGraph class. All rewards are randomly set in the range [0,1].
    Start and end must be inside or on the boundary of the rectangle
    Tested: OK
    Arguments:
        xmin,xmax: float -- min and max x coordinates
        ymin,ymax: float -- min and max y coordinates
        nvertices: int32 -- number of vertices in the graph
        alpha: float -- alpha factor for edges (all equal)
        budget: float -- budget for the orienteering problem
        start: list -- coordinates of the start vertex
        end: list -- coordinates of the end vertex
        fname: string -- name of the file to save (.mat extension will not be added)
    Returns:
        nothing
    """
    
    contents = {}
    coords = np.zeros((nvertices,2),dtype=np.float32)
    
    rewards = np.zeros((nvertices,1))
    for i in range(nvertices):
        coords[i][0] = xmin + np.random.random()* (xmax - xmin)
        coords[i][1] = ymin + np.random.random()* (ymax - ymin)
        rewards[i][0] = np.random.random()
    rewards[0][0] = 0 # always no reward in the start location
    
    if start:
        coords[0][0] = start[0]
        coords[0][1] = start[1]
        
    if end:
        coords[nvertices-1][0] = end[0]
        coords[nvertices-1][1] = end[1]
    
    contents['xy'] = coords
    contents['rewards']= rewards
    edge_list = np.zeros((nvertices*(nvertices-1),3))
    k = 0
    for i in range(nvertices):
        for j in range(nvertices):
            if i != j:
                edge_list[k][0] = i
                edge_list[k][1] = j
                edge_list[k][2] = np.linalg.norm(coords[i][:]-coords[j][:])
                k = k+1
    contents['edge_list'] = edge_list
    al = np.zeros((1,1))
    al[0][0]= alpha
    contents['alpha'] = al
    b = np.zeros((1,1))
    b[0][0] = budget
    contents['t_max'] = b
    sio.savemat(fname,contents)
    

def load_orienteering_graph(fname):
    contents = sio.loadmat(fname)
    alpha = contents['alpha'][0][0]
    rewards = contents['rewards']
    coords = contents['xy']
    edge_list = contents['edge_list']
    
    npoints = coords.shape[0]
    vertices = dict()
    total_reward = -1
    
    for i in range(npoints):
        nv = OrienteeringVertex(coords[i,0],coords[i,1],rewards[i,0],i)
        vertices[i]= nv
                   
    nedges = edge_list.shape[0]
    offset = np.min(edge_list[:,0])  # accommodate for both 0-index and 1-index
    
    for i in range(nedges):
        edge = OrienteeringEdge(vertices[edge_list[i][0]-offset], vertices[edge_list[i][1]-offset], alpha)
        vertices[edge_list[i][0]-offset].add_edge(edge)

        
    
    budget = contents['t_max'][0][0]
    
    distance_matrix = np.zeros((npoints,npoints),dtype=np.float32)
    for i in range(npoints):
        for j in range(npoints):
            if i!= j:
                distance_matrix[i][j] = vertices[i].get_edge_towards_vertex(j).d
                
    return vertices,distance_matrix,budget
        
class OrienteeringGraph:
    #@track
    def __init__(self,fname=None):
        if type(fname) == str:
            if VERBOSE:
                print('Loading graph ',fname,' ....')

            
                
            contents = sio.loadmat(fname)
            alpha = contents['alpha'][0][0]
            rewards = contents['rewards']
            coords = contents['xy']
            edge_list = contents['edge_list']
            
            npoints = coords.shape[0]
            self.vertices = dict()
            self.total_reward = -1
            
            # necessary to accommodate both type of rewards from MCTS and GP papers
            if rewards.shape[1] == 1:
                offset = 0
            else:
                offset = 1
            
            for i in range(npoints):
                # Here you may need one or the other version depending on how the graphs were created
                # For DARS, use the first version, for CASE use the second version. This needs to be fixed and made
                # generic
                # Fixed on 9/21/2022
                nv = OrienteeringVertex(coords[i,0],coords[i,1],rewards[i,offset],i)
                #nv = OrienteeringVertex(coords[i,0],coords[i,1],rewards[i,1],i)
                self.vertices[i]= nv
                           
            nedges = edge_list.shape[0]
            offset = np.min(edge_list[:,0])  # accommodate for both 0-index and 1-index
            
            for i in range(nedges):
                edge = OrienteeringEdge(self.vertices[edge_list[i][0]-offset], self.vertices[edge_list[i][1]-offset], alpha)
                self.vertices[edge_list[i][0]-offset].add_edge(edge)

                
            self.set_start_vertex(0)
            self.end_vertex = npoints - 1
            self.nvertices = npoints
            self.budget = contents['t_max'][0][0]
            self.alpha = alpha
            
            self.distance_matrix = np.zeros((npoints,npoints),dtype=np.float32)
            for i in range(npoints):
                for j in range(npoints):
                    if i!= j:
                        self.distance_matrix[i][j] = self.get_vertex_by_id(i).get_edge_towards_vertex(j).d
            if VERBOSE:
                print('Done!')
            del contents
            
        else:
            if VERBOSE:
                print("Generating random orienteering graph...")
            if type(fname)== int:
                self.nvertices = fname
            else:
                self.nvertices = VERTEX_MIN + int(np.random.random()*(VERTEX_MAX-VERTEX_MIN))
            self.vertices = dict()
            
            self.alpha = 0.5
            
            nv = OrienteeringVertex(0,0,np.random.random()*MAX_REWARD,0)
            self.vertices[0] = nv
            for i in range(1,self.nvertices-1):
                nv = OrienteeringVertex(np.random.random()*MAX_X,np.random.random()*MAX_Y,np.random.random()*MAX_REWARD,i)
                self.vertices[i]= nv
                
            nv = OrienteeringVertex(0,0,np.random.random()*MAX_REWARD,self.nvertices-1)
            self.vertices[self.nvertices-1] = nv
                
            self.set_start_vertex(0)
            self.end_vertex = self.nvertices - 1          
                
            for i in range(self.nvertices):
                for j in range(self.nvertices):
                    if (i != j): #and ( i != self.end_vertex ):  # no loops and end vertex has no outgoing edges
                        edge = OrienteeringEdge(self.vertices[i], self.vertices[j], self.alpha)
                        self.vertices[i].add_edge(edge)
                    
            self.distance_matrix = np.zeros((self.nvertices,self.nvertices),dtype=np.float32)
            for i in range(self.nvertices):
                for j in range(self.nvertices):
                    if (i != j): #and ( i != self.end_vertex ): # do not set values for non-existing edges
                        self.distance_matrix[i][j] = self.get_vertex_by_id(i).get_edge_towards_vertex(j).d
            
            self.budget =  1 + np.sum(self.distance_matrix)/100
            #self.budget = 1 + 5*np.random.random()
            self.total_reward = -1
            
                 
            if VERBOSE:
                print("Done!")
                
    def clone(self):
        return copy.deepcopy(self)
    
    def reassign_vertices(self,newpoints):
        
        self.vertices = dict()
        self.total_reward = -1 ## needed?
        
        # create new vertices
        self.nvertices = newpoints.shape[0]
        for i in range(self.nvertices):
            nv = OrienteeringVertex(newpoints[i][0],newpoints[i][1],newpoints[i][2],i)
            self.vertices[i] = nv
        
        self.set_start_vertex(0)
        self.end_vertex = self.nvertices - 1          
            
        # create new edges
        for i in range(self.nvertices):
            for j in range(self.nvertices):
                if (i != j): #and ( i != self.end_vertex ):  # no loops and end vertex has no outgoing edges
                    edge = OrienteeringEdge(self.vertices[i], self.vertices[j], self.alpha)
                    self.vertices[i].add_edge(edge)
        # create new distances
        self.distance_matrix = np.zeros((self.nvertices,self.nvertices),dtype=np.float32)
        for i in range(self.nvertices):
            for j in range(self.nvertices):
                if (i != j): #and ( i != self.end_vertex ): # do not set values for non-existing edges
                    self.distance_matrix[i][j] = self.get_vertex_by_id(i).get_edge_towards_vertex(j).d
            
        
            
        
        
    def shrink_to_subset_of_vertices(self,list_of_vertices):
        ## list of vertices MUST include start and end vertex
        
        # remove all vertices not in list_of_vertices
        to_remove = set(self.vertices.keys())-set(list_of_vertices)
        for i in to_remove:
            del self.vertices[i]
                
        # adjust size, start, end
        npoints = len(list_of_vertices)
                
        self.nvertices = len(self.vertices)
        
     #   self.set_start_vertex(0)
     #   self.end_vertex = self.nvertices - 1
        
        
        # remove all edges towards edges that have been removed
        for i in self.vertices:
            to_remove = []
            for j in self.vertices[i].edges:
                if self.vertices[i].edges[j].dest.id not in list_of_vertices:
                    to_remove.append(j)
            for k in to_remove:
                del self.vertices[i].edges[k]
                
        
        #recomput distances
     #   self.distance_matrix = np.zeros((npoints,npoints),dtype=np.float32)
     #   for i in range(npoints):
     #       for j in range(npoints):
     #           if i!= j:
     #               self.distance_matrix[i][j] = self.get_vertex_by_id(i).get_edge_towards_vertex(j).d
                    
                
    
        
    def neighbors(self,v):  # returns the list of neighbors of v -> for compatibility with NN
        retval = []
        n = self.vertices[v]
        edges = n.get_edges()
        for i in edges:
            retval.append(i.dest.id)
        return retval
    
    def number_of_nodes(self): # duplicate for code compatibility
        return self.nvertices
    
    def get_number_of_vertices(self):
        return self.nvertices
    
    def set_start_vertex(self,idx):
        self.start_vertex = idx
        self.vertices[idx].set_visited()
        
    def get_vertices(self):
        return self.vertices
    
    def get_total_reward(self):
        if self.total_reward < 0:
            self.total_reward = 0
            for id in self.vertices:
                self.total_reward += self.vertices[id].get_value()
        return self.total_reward
    
    def get_reward(self,id):
        return self.vertices[id].get_value()
    
    def get_vertex_by_id(self,idx):
        return self.vertices[idx]
    
    def get_n_vertices(self):
        return self.nvertices
    
    def get_start_vertex(self):
        return self.start_vertex  # returns the id of the vertex
    
    def set_end_vertex(self,idx):
        self.end_vertex = idx
    
    def get_end_vertex(self):
        return self.end_vertex  # returns the id of the vertex
    
    def get_budget(self):
        return self.budget
    
    def set_budget(self,value):
        self.budget = value
        
    def number_of_unvisited_vertices(self):
        retval = 0
        for i in self.vertices:
            if not self.vertices[i].get_visited():
                retval += 1
        return retval
    
    def clear_visits(self):
        for i in self.vertices:
            self.vertices[i].clear_visited()
        self.vertices[self.start_vertex].set_visited()
    
    # # this should be better implemented
    # def shortest_path(self,start,goal):
    #     parent  = [-1]*self.nvertices
    #     g = [float('inf')]* self.nvertices
    #     g[start] = 0
    #     open = [start]
    #     while len(open)>0:
    #         min = g[open[0]]
    #         p = 0
    #         for i in open:
    #             if g[open[i]] < min:
    #                 p = i
    #                 min = g[open[i]]
    #         v = self.vertices[open[p]]
    #         open.pop(p)
    #         for e in v.edges:
    #             if g[e.]
                    
    
class SharedOrienteeringVertex:
    def __init__(self,v):
        self.vertex = v
        self.visited = False
        self.id = self.vertex.id
        self.value = self.vertex.value
        
    def set_visited(self):
        self.visited = True
            
    def clear_visited(self):
        self.visited = False
        
    def get_visited(self):
        return self.visited
    
    def get_edge_towards_vertex(self,id):
        return self.vertex.get_edge_towards_vertex(id)
    
    
class SharedOrienteeringGraph:
    def __init__(self,og):
        self.assign_graph(og)
        
    def get_reward(self,idx):
        return self.graph.get_reward(idx)
            
    def assign_graph(self,og):
        self.graph = og
        self.vertices = {}
        for key in og.vertices.keys():
            n = SharedOrienteeringVertex(og.vertices[key])
            self.vertices[key] = n
        
    def get_budget(self):
        return self.budget
    
    def set_budget(self,value):
        self.budget = value
        
    def number_of_nodes(self):
        return self.graph.number_of_nodes()
    
    def get_vertex_by_id(self,n):
        return self.graph.get_vertex_by_id(n)
    
    def get_vertices(self):
        return self.graph.get_vertices()
    
    def get_end_vertex(self):
        return self.graph.get_end_vertex()


if __name__ == "__main__":
    og = OrienteeringGraph('graph_test.mat')
