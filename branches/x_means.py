#!/usr/bin/env python
#
# Name:  X-means clusering
#
# Author: Thuso S Simon
#
# Date: 28 of April, 2011
#TODO: Add G-means and PD-means, weighted clustering
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
""" A program that does X-mean and G-mean and other type of clusering. Uses scipy clustering modual for clustering"""

from scipy.cluster import vq as sci
import numpy as nu

def xmeans(point):
    #does x_means clustering on a set of points ans splits up into x clusers
    #uses scipy's kmeans clusering to and the baysian information critera
    #to coose correct number of clusters
    cluser,asocate_point=sci.kmeans2(point,2)
    #plot lab.plot(point[asocate_point==0,0],point[asocate_point==0,1],'.',point[asocate_point==1,0],point[asocate_point==1,1],'.')
    
    #determine if should split
    while True: #make sure kmeans converges
        try:
            BIC2=Like(point[asocate_point==0],cluser[0])+Like(point[asocate_point==1],cluser[1])
            BIC1=Like(point,nu.mean(point,axis=0))
            break
        except ZeroDivisionError:
            pass
    if nu.sum(BIC2-BIC1)>nu.log(10): #accept split
        k=3
        pre_cluser,pre_asocate_point=sci.kmeans2(point,k-1)
        while nu.sum(BIC2-BIC1)>nu.log(5) or nu.sum(BIC2-BIC1)==0.0:
            cluser,asocate_point=sci.kmeans2(point,k)
            try:
                BIC2,BIC1=[0.0]*cluser.shape[1],[0.0]*cluser.shape[1]
            except IndexError: #if 1 dimensional
                BIC2,BIC1=0.0,0.0
            for i in range(k):
                try:
                    BIC2=BIC2+Like(point[asocate_point==i],cluser[i])
                except ZeroDivisionError:
                    continue
                if i<k-2:
                    try:
                        BIC1=BIC1+Like(point[pre_asocate_point==i],pre_cluser[i])
                    except ZeroDivisionError:
                        pre_cluser,pre_asocate_point=sci.kmeans2(point,k-1)
                        continue
            if nu.sum(BIC2-BIC1)>nu.log(5):
                pre_cluser,pre_asocate_point=cluser.copy(),asocate_point.copy()
                k+=1
               # print k
    else:
        return nu.mean(point,axis=0),nu.zeros(len(point))
        
    return pre_cluser,pre_asocate_point

def xmean_spit(point):
    #splits each cluster and tests to see if good split, Hierarical way
    
    #starts with 1 cluster and increases from there
    cluster_old=nu.mean(point,axis=0)
    associate_old =nu.zeros(len(point))
    #starts with 2 clusers and compares
    cluster_new,associate_new=sci.kmeans2(point,2)
    #calculate if should keep split
    BIC2=Like(point[asocate_point==0],cluster_new[0])+Like(point[asocate_point==1],cluster_new[1])
    BIC1=Like(point,cluster_old)

    #determine if should split
    while True: #make sure kmeans converges
        try:
            BIC2=Like(point[asocate_point==0],cluster_new[0])+Like(point[asocate_point==1],cluster_new[1])
            BIC1=Like(point,nu.mean(point,axis=0))
            break
        except ZeroDivisionError:
            pass
    if nu.sum(BIC2-BIC1)>nu.log(10): #accept split, split into sub clusers
        new_point={'1':point[asocate_point==0,:],'2':point[asocate_point==1,:]} 
        while nu.sum(BIC2-BIC1)>nu.log(5) or nu.sum(BIC2-BIC1)==0.0:
            cluser,asocate_point=sci.kmeans2(point,k)
            try:
                BIC2,BIC1=[0.0]*cluser.shape[1],[0.0]*cluser.shape[1]
            except IndexError: #if 1 dimensional
                BIC2,BIC1=0.0,0.0
            for i in range(k):
                try:
                    BIC2=BIC2+Like(point[asocate_point==i],cluser[i])
                except ZeroDivisionError:
                    continue
                if i<k-2:
                    try:
                        BIC1=BIC1+Like(point[pre_asocate_point==i],pre_cluser[i])
                    except ZeroDivisionError:
                        pre_cluser,pre_asocate_point=sci.kmeans2(point,k-1)
                        continue
            if nu.sum(BIC2-BIC1)>nu.log(5):
                pre_cluser,pre_asocate_point=cluser.copy(),asocate_point.copy()
                k+=1
               # print k
    else:
        return nu.mean(point,axis=0),nu.zeros(len(point))
        
    return pre_cluser,pre_asocate_point
  

def Like(x,mu):
    #determines the ln likelihood that the points are assocated with the clusers
    #assuming gaussan distribution
    #calculate maximul likelyhood sigma 
    var=(len(x)-1)/float(len(x))*sum((x-mu)**2)
    return (-len(x)/2.)*nu.log(2*nu.pi*var)-1/(2.*var)*sum((x-mu)**2)
    

    
def kdtree(point_list, depth=0):
#makes trees to partition points into cluster for speed up of x-means
    #from wikipeada 
    if not point_list:
        return
 
    # Select axis based on depth so that axis cycles through all valid values
    k = len(point_list[0]) # assumes all points have the same dimension
    axis = depth % k
 
    # Sort point list and choose median as pivot element
    point_list.sort(key=lambda point: point[axis])
    median = len(point_list) // 2 # choose median
 
    # Create node and construct subtrees
    node = Node()
    node.location = point_list[median]
    node.left_child = kdtree(point_list[:median], depth + 1)
    node.right_child = kdtree(point_list[median + 1:], depth + 1)
    return node


    
def toyI():
    import pylab as lab
    #1-d 3 clusers
    x=[]
    for i in range(50):
        x.append(nu.random.randn())
                 
    for i in range(50):
        x.append(nu.random.randn()+10)
    for i in range(50):
        x.append(nu.random.randn()-10)

    x=nu.array(x)
    #find true clusers with kmeans
    clus,assos=sci.kmeans2(x,3)
    #plot assosation
    fig=lab.figure()
    pl=fig.add_subplot(111)
    pl.hold(True)
    pl.plot(x[assos==0],nu.ones(len(x[assos==0])),'.',x[assos==1],nu.ones(len(x[assos==1])),'.',x[assos==2],nu.ones(len(x[assos==2])),'.',label='Data')
    #try xmeans 
    clus,assos=xmeans(x)
    pl.plot(clus,nu.ones(len(clus)),'k+',ms=20,mew=5,label='x-mean centroids')
    pl.legend()
    lab.show()
    
def toyII():
    import pylab as lab
    #2-d with 3 clusters
    x=nu.zeros([750,2])
    for i in range(250):
        x[i,:]=nu.random.randn(2)+[0,-nu.sqrt(75)/2.]
                 
    for i in range(250,500):
        x[i,:]=nu.random.randn(2)+[5,5]
    for i in range(500,750):
        x[i,:]=nu.random.randn(2)+[-5,5]

    clus,assos=sci.kmeans2(x,3)
    fig=lab.figure()
    pl=fig.add_subplot(111)
    pl.hold(True)
    pl.plot(x[assos==0,0],x[assos==0,1],'.',x[assos==1,0],x[assos==1,1],'.',x[assos==2,0],x[assos==2,1],'.')
    clus,assos=xmeans(x)
    for i in range(clus.shape[0]):
        pl.plot(clus[i,0],clus[i,1],'k+',ms=20,mew=5)
    lab.show()

def toyIII():
    import pylab as lab
    #2-d with 5 clusers at random places in 10X10 square and random var
    x=nu.zeros([1000,2])

    std=nu.random.rand(2)
    mu=nu.random.rand(2)*20-10   
    print 'mean is (%f,%f) and var is (%f,%f)' %(mu[0],mu[1],std[0],std[1])
    for i in range(1000):
        x[i,:]=nu.random.randn(2)*std+mu
        if i/200.==nu.round(i/200.) and i!=0:
            std=nu.random.rand(2)
            mu=nu.random.rand(2)*20-10   
            print 'mean is (%f,%f) and var is (%f,%f)' %(mu[0],mu[1],std[0],std[1])

                 

    clus,assos=sci.kmeans2(x,5)
    fig=lab.figure()
    pl=fig.add_subplot(111)
    pl.hold(True)
    for i in nu.unique(assos):
        pl.plot(x[assos==i,0],x[assos==i,1],'.')
    clus,assos=xmeans(x)
    for i in range(clus.shape[0]):
        pl.plot(clus[i,0],clus[i,1],'k+',ms=20,mew=5)
    lab.show()
