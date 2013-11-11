#!/usr/bin/env python
#
# Name:  Age Dating Spectra Fitting Program
#
# Author: Thuso S Simon
#
# Date: 11th of Nov, 2013
#TODO:  
#    
#
#    vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#    Copyright (C) 2011  Thuso S Simon
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    For the GNU General Public License, see <http://www.gnu.org/licenses/>.
#    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#History (version,date, change author)
#
#
#
"""Bean counting and distributed mcmc helpers"""
from mpi4py import MPI as mpi
import time as Time
import cPickle as pik
import csv
#import pylab as lab
import os
import numpy as nu


class Topologies(object):
    """Topologies( cpus='max'. top='cliques', k_max=16)
    Defines different topologies used in communication. Will probably affect
    performance if using a highly communicative topology.
    Topologies include :
    all, ring, cliques and square.
    all - every worker is allowed to communicate with each other, no buffer
    ring -  the workers are only in direct contact with 2 other workers
    cliques - has 1 worked connected to other head workers, which talks to all the other sub workers
    square - every worker is connect to 4 other workers
    cpus is number of cpus (max or number) to run chains on, k_max is max number of ssps to combine"""

    def Single(self):
        #Using mpi but want to run chains independantly
        #if mpi is avalible
        if not mpi is None:
            self.comm = mpi.COMM_SELF
            self.size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
            self.comm_world = mpi.COMM_SELF
            self.rank_world = self.comm_world.Get_rank()
            self.size_world = self.comm_world.Get_size()
        else:
            self.comm = None
            self.size = 1
            self.rank = 0
            self.comm_world = None
            self.rank_world = 0
            self.size_world = 1



    def All(self):
        #all workers talk to eachother
        self.comm = mpi.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.comm_world = mpi.COMM_WORLD
        self.rank_world = self.comm_world.Get_rank()
        self.size_world = self.comm_world.Get_size()

    def Ring(self):
        '''makes ring topology'''
        self.comm_world = mpi.COMM_WORLD
        self.rank_world = self.comm_world.Get_rank()
        self.size_world = self.comm_world.Get_size()
        r_index = range(2,2 * self.size_world+2,2)
        index = range(self.size_world)
        edges = []
        for i in index:
            if i - 1 < 0:
                edges.append(max(index))
            else:
                edges.append( i-1)
            if i + 1 > max(index):
                edges.append(min(index))
            else:
                edges.append( i + 1)
        self.comm = self.comm_world.Create_graph(r_index, edges, True)
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def Cliques(self, N=10):
        #N = 3
        self.comm_world = mpi.COMM_WORLD
        self.rank_world = self.comm_world.Get_rank()
        self.size_world = self.comm_world.Get_size()
        #setup comunication arrays to other workers + 1 from world
        head_nodes = nu.arange(N)
        workers = []
        for i in xrange(N):
            workers.append([i])
        j = 0
        for i in xrange(max(head_nodes) + 1, self.size_world):
            workers[j].append(i)
            j+=1
            if j == N:
                j=0
        #make index and edges
        index,edges = [],[]
        #workers
        j = 0
        #print 'world',self.size_world 
        for i in range(self.size_world - len(head_nodes)):
            if len(index) == 0:
                index.append(len(workers[j]))
            else:
                index.append(index[-1] + len(workers[j]))
            edges.append(workers[j])
            if (i + 1) % (len(workers[j]) - 1) == 0:
                j += 1
            if j >= len(workers):
                break
        #head nodes
        for i in head_nodes:
            temp = nu.unique(nu.hstack((workers[i],head_nodes)))
            edges.append(list(temp[temp != i]))
            index.append(index[-1] + len(edges[-1]))
        n_edge =[]
        for i in edges:
            for j in i:
                n_edge.append(j)
        self.comm = self.comm_world.Create_graph(index, n_edge, True)
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def Square(self):
        #Each worker communicates with max of 4 other workers
        Nrow = 3
        #make grid cartiesian grid
        self.comm_world = mpi.COMM_WORLD
        self.rank_world = self.comm_world.Get_rank()
        self.size_world = self.comm_world.Get_size()
        #make grid
        Ncoulms = self.size_world/Nrow
        if self.size_world % Nrow != 0:
            print 'Warrning: Not cylindrical, workers may not work correctly'
        tot = 0
        grid = []
        for i in xrange(Ncoulms):
            for j in range(Nrow):
                grid.append(nu.array([j,i]))
                tot += 1
        #fill in grid if points left over
        i = 0
        while tot < self.size_world:
            grid.append((j,i))
            i += 1
        grid = nu.array(grid)   
        edges,ind=[],[]
        #make comunication indicies
        for i in range(self.size_world):
            #find 4 closest workers
            min_dist = nu.zeros((4,2)) +9999999
            for k in range(min_dist.shape[0]):
                for j in range(self.size_world):
                    if (min_dist[k][0] > nu.sqrt((grid[i][0] - grid[j][0])**2 + (grid[i][1] - grid[j][1])**2) 
                        and i != j):
                        if not nu.any(min_dist[:,1] == j):
                            min_dist[k][0] = nu.sqrt((grid[i][0] - grid[j][0])**2 + (grid[i][1] - grid[j][1])**2)
                            min_dist[k][1] = nu.copy(j)
            #if on edged of grid, wrap around
            if nu.any(grid[i] == 0): #top or left side
                if grid[i][0] == 0: #top
                    #find one on bottom
                    index = min_dist[:,0].argmax()
                    Index = nu.nonzero(nu.logical_and(grid[:,0] == Nrow - 1,grid[:,1] == grid[i,1]))[0]
                    min_dist[index] = [0,Index[0].copy()]
                if grid[i][1] == 0: #left
                    index = min_dist[:,0].argmax()
                    Index = nu.nonzero(nu.logical_and(grid[:,0] == grid[i,0],grid[:,1] == Ncoulms-1))[0]
                    min_dist[index]= [0,Index[0].copy()]
            if nu.any(grid[i]  == Ncoulms - 1): #right side and maybe bottom
                if grid[i][1] == Ncoulms - 1: #right
                    index = min_dist[:,0].argmax()
                    Index = nu.nonzero(nu.logical_and(grid[:,1] == 0,grid[:,0] == grid[i,0]))[0]
                    min_dist[index] = [0,Index[0].copy()]
                if grid[i][0] == Nrow - 1: #bottom
                    index = min_dist[:,0].argmax()
                    Index = nu.nonzero(nu.logical_and(grid[:,1] == grid[i,1],grid[:,0] == 0))[0]
                    min_dist[index] = [0,Index[0].copy()]
            '''if grid[i][0]  == Nrow - 1: #def bottom
                index = min_dist[:,0].argmax()
                Index = nu.nonzero(nu.logical_and(grid[:,1] == grid[i,1],grid[:,0] == 0))[0]
                min_dist[index] = [0,Index[0].copy()]'''
            if nu.any(grid[i]  == Ncoulms): #extra grid on right side
                print 'bad'
            t =[]
            for k in range(min_dist.shape[0]):
                t.append(int(min_dist[k,1]))
            edges.append(t)
            if len(ind) == 0:
                ind.append(min_dist.shape[0])
            else:
                ind.append(ind[-1] + min_dist.shape[0])
        n_edge =[] 
        for i in edges:
            for j in i:
                n_edge.append(j)
        self.comm = self.comm_world.Create_graph(ind, n_edge, True)
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
    
    def get_neighbors(self,rank):
        '''for All, since doesn't use make cartisian grid'''
        return range(self.size)

    #====Update stuff====
    def thuso_min(self, x, y):
            if x[0] >y[0]:
                return y
            else:
                return x

    def init_sync_var(self):
        '''initates send and recive varibles'''
        #print self.send_to,self.reciv_from,self.swarmChi
        if self.rank_world == 0:
            #does stop signal to all
            self.iter_stop = nu.ones((1,self.comm_world.size),dtype=int)
            for i in xrange(1,self.comm_world.size):
                self._stop_buffer.append(self.comm_world.Send_init((
                            self.iter_stop[:,i],mpi.INT),dest=i,tag=1))
                #recive best chi,param and current iteration from others
            #make chibest,parambest and current temp arrays
            self._chibest = {}
            self._parambest = {}
            self._current = {}
            for i in xrange(1,self.size_world):
                self._chibest[str(i)] = nu.zeros((1,1)) + nu.inf
                self._parambest[str(i)] = nu.zeros_like(self.parambest) + nu.nan
                self._current[str(i)] = nu.zeros((1,1),dtype=int)
                #make buffer for recv and send best param 
                self._update_buffer.append(self.comm_world.Recv_init((
                            self._chibest[str(i)],mpi.DOUBLE),source=i,tag=2))
                self._update_buffer.append(self.comm_world.Recv_init((
                            self._parambest[str(i)],mpi.DOUBLE),source=i,tag=3))
                self._update_buffer.append(self.comm_world.Recv_init((
                            self._current[str(i)],mpi.INT),source=i,tag=4))
                #self._update_buffer = []      
        else:
            self._stop_buffer.append(self.comm_world.Recv_init((self.iter_stop,mpi.INT), source=0,tag=1))
          #send best chi to root and recive best chi from root
            self._update_buffer.append(self.comm_world.Send_init((self.chibest,mpi.DOUBLE),dest=0,tag=2))
            self._update_buffer.append(self.comm_world.Send_init((self.parambest,mpi.DOUBLE),dest=0,tag=3))
            self._update_buffer.append(self.comm_world.Send_init((self.current,mpi.INT),dest=0,tag=4))
        #send best param as part of swarm and recive swarm
        for i in xrange(len(self.send_to)):
                #send my stuff
            self.buffer.append(self.comm.Send_init((self.swarm[:,0],mpi.DOUBLE),dest=self.send_to[i],tag=5))
            self.buffer.append(self.comm.Send_init((self.swarmChi[:,0],mpi.DOUBLE),dest=self.send_to[i],tag=6))
        for i in xrange(len(self.reciv_from)):
            #recive other stuff
            self.buffer.append(self.comm.Recv_init((self.swarmChi[:,i+1],mpi.DOUBLE),source=self.reciv_from[i],tag=6))
            self.buffer.append(self.comm.Recv_init((self.swarm[:,i+1],mpi.DOUBLE),source=self.reciv_from[i],tag=5))
  
    def get_best(self, op=False):
       #updates chain info
       #checks to see if should stop
        mpi.Prequest.Startall(self._update_buffer)
        if self.rank_world == 0:
            mpi.Prequest.Waitany(self._update_buffer)
            if op:
                mpi.Prequest.Startall(self._stop_buffer)
                mpi.Prequest.Waitall(self._stop_buffer)
              #find best fit
            for i in self._current.keys():
                #print self._parambest[i],i
                if self._current[i] > 199:
                    self.global_iter += self._current[i]
                    self._current[i][0] = 0
                
                if self._chibest[i] < self.chibest:
                    self.chibest = self._chibest[i] + 0
                    self.parambest = self._parambest[i].copy()
                    num = (nu.isfinite(self.parambest).sum() - 6)/3
                    print '%i has best fit with a chi of %2.2f and %i' %(int(i),self.chibest,num)                    
                    sys.stdout.flush()
        else:
            mpi.Prequest.Testall(self._update_buffer)
            if self.comm_world.Iprobe(source=0, tag=1):
                Time.sleep(5)
                mpi.Prequest.Startall(self._stop_buffer)
                mpi.Prequest.Waitall(self._stop_buffer)
            if self.current > 199:
                self.current = nu.array([[0]])
                #print self.iter_stop

    def swarm_update(self,param, chi,bins):
        '''Updates positions of swarm using topology'''
        for kk in xrange(len(self.swarm[:,0])):
            if kk<bins*3+2+4:
                self.swarm[:,0][kk] = param[kk]
            else:
                self.swarm[:,0][kk] = nu.nan
        self.swarmChi[:,0] = chi
        mpi.Prequest.Startall(self.buffer)
        mpi.Prequest.Testany(self.buffer)
        
    def make_swarm(self):
        #who to send to and who to recieve from
        try:
            self.send_to = nu.array(self.comm.Get_neighbors(self.rank))
        except AttributeError:
            #if all no get_neighbors will be found
            self.send_to = nu.array(self.get_neighbors(self.rank))
        self.send_to = nu.unique(self.send_to[self.send_to != self.rank])
        self.reciv_from = []
        try:
            for i in xrange(self.size):
                if nu.any(nu.array(self.comm.Get_neighbors(i)) == self.rank) and i != self.rank:
                    self.reciv_from.append(i)
            self.reciv_from = nu.array(self.reciv_from)
        except AttributeError:
            self.reciv_from = self.send_to.copy()
        #makes large array for sending and reciving
        self.swarm = nu.zeros([len(self.reciv_from)+1, self._k_max * 3 + 2 + 4],order='Fortran') + nu.nan
        self.swarmChi = nu.zeros((1,len(self.reciv_from)+1),order='Fortran') + nu.inf

    def __init__(self, top = 'cliques', k_max=10):
        self._k_max = k_max
        #number of iterations done
        self.current = nu.array([[0]])
        self.global_iter = 0
        #number of workers to create
        #local_cpu = cpu_count()
        if not mpi is None:
            comm = mpi.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
        else:
            comm = None
            size = 1
            rank = 0
        #commuication buffers
        #[(param,source),(chi,source)]
        self.buffer = []
        self._stop_buffer = []
        self._update_buffer = []
        #simple manager just devides total processes up
        self.iter_stop = nu.array([[True]],dtype=int)
        self.chibest = nu.array([[nu.inf]],dtype=float)
        self.parambest = nu.ones(k_max * 3 + 2 + 4) + nu.nan
        #check if topology is in list
        if not top in ['all', 'ring', 'cliques', 'square','single']:
            raise ValueError('Topology is not in list.')
        if top.lower() == 'all':
            self.All()
        elif top.lower() == 'ring':
            self.Ring()
        elif top.lower() == 'cliques':
            self.Cliques()
        elif top.lower() == 'square':
            self.Square()
        elif top.lower() == 'single':
            self.Single()
        #print self.iter_stop
        try:
            self.make_swarm()
            self.init_sync_var()
        except:
            pass
