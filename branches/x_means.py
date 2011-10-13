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

def xmean(points,min_points=5):
    #does x_means clustering on a set of points ans splits up into x clusers
    #uses scipy's kmeans clusering to and the baysian information critera
    #to coose correct number of clusters
   
    max_cluster=int(round(sum(points.shape)/float(min_points)+1))
    #inital split
    #cluster,asocate_point=sci.kmeans2(points,2)
    
    if BIC1(points)<BIC2(points):
        return {'0':points}
    #check for min points
    means,asocate_point=sci.kmeans2(points,2)
    if sum(asocate_point)<=min_points or points.shape[0]-sum(asocate_point)<=min_points:
        return {'0':points}
     #start split algorithm
    n_cluster=1 #python numbering 0= 1 cluster
    out={}
    for i in range(2):# star splitting
        out[str(i)]=nu.copy(points[asocate_point==i,:])
    #associate=nu.copy(asocate_point)
    i=0
    while i<max_cluster:
        if BIC1(out[str(i)])>BIC2(out[str(i)]):#create new cluser
            means,asocate_point=sci.kmeans2(out[str(i)],2)
            if sum(asocate_point)<=min_points or out[str(i)].shape[0]-sum(asocate_point)<=min_points: #check for min points
                i+=1
                #check exit conditions
                if i==n_cluster+1:
                    break
            else:
                n_cluster+=1    
                try:
                    out[str(n_cluster)]= nu.copy(out[str(i)][asocate_point==0,:]) #create new clust before overwriting
                    out[str(i)]=nu.copy(out[str(i)][asocate_point==1,:])
                except IndexError:
                    print nu.unique(asocate_point)
        else:
            i+=1
            #check exit conditions
            if i==n_cluster+1:
                break
    return out
            
def BIC1(points):
    #finds BIC of cluster with 1 mean: from xmeans paper
    
    mean=nu.mean(points,0)
    n_points=nu.prod(points.shape)
    cov=nu.cov(points.T)
    mean_minus=points-mean
    try:
        n_dim=points.shape[1]
    except IndexError:
        n_dim=1
    try:
        cov_diag=nu.prod(cov.diagonal())
    except ValueError:
        cov_diag=cov
    invcov=cov**-1
    a=0 #a=sum(Transpose(pt[j]-mean).invcov.(pt[j]-mean))
    for i in xrange(points.shape[0]):
        a+=nu.sum(nu.dot(mean_minus[i],nu.dot(invcov,mean_minus[i].T)))
                        
    
    #return n_dim*nu.log(2*nu.pi)+2*nu.log(cov_diag)+2*a+n_dim*nu.log(n_points)
    return n_points*n_dim*nu.log(2*nu.pi)+n_points*nu.log(cov_diag)+2*a+0.5*n_dim*(n_dim+3.)*nu.log(n_points)
        
def BIC2(points):
    #find BIC with cluster with 2 means: from xmeans paper
    n_points_tot=float(nu.prod(points.shape))
    i=0
    while True: #if gives LinAlgError try again till sucess or 100 times
        try:
            mean,asocate_point=sci.kmeans2(points,2)
            break
        except: #nu.linalg.LinAlgError:
            i+=1
            if i==100:
                return nu.inf
            
    try:
        n_dim=points.shape[1]
    except IndexError:
        n_dim=1
    out=n_points_tot*n_dim*nu.log(2*nu.pi)+2*n_points_tot*nu.log(n_points_tot)+ (n_dim**2+3*n_dim+1)*nu.log(n_points_tot)
    for i in range(2):
        x=points[asocate_point==i,:]
        n_points=nu.prod(x.shape)
        mean_minus=x-mean[i]
        cov=nu.cov(x.T)
        invcov=cov**-1
        try:
            cov_diag=nu.prod(cov.diagonal())
        except ValueError:
            cov_diag=cov
        n_points=nu.prod(x.shape)
        a=0 #a=sum(Transpose(pt[j]-mean).invcov.(pt[j]-mean))
        for j in xrange(x.shape[0]):
            a+=nu.sum(nu.dot(mean_minus[j],nu.dot(invcov,mean_minus[j].T)))

        out+= n_points*nu.log(cov_diag)+2*a-2*n_points*nu.log(n_points)
    
    return out
    
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
        x.append(nu.random.randn()*.1)
                 
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
    clus=xmeans(x)
    for i in clus.keys():
        pl.plot(clus[i],nu.ones(len(clus[i])),'k+',ms=20,mew=5,label='x-mean centroids '+i)
    pl.legend()
    lab.show()
    
def toyII():
    import pylab as lab
    #2-d with 3 clusters
    x=nu.zeros([750,2])
    for i in range(250):
        x[i,:]=nu.random.randn(2)+[0,-nu.sqrt(75)/2.]
                 
    for i in range(250,750):
        x[i,:]=nu.random.randn(2)+[5,5]
    for i in range(500,750):
        x[i,:]=nu.random.randn(2)+[-5,5]

    clus,assos=sci.kmeans2(x,3)
    fig=lab.figure()
    pl=fig.add_subplot(111)
    pl.hold(True)
    pl.plot(x[assos==0,0],x[assos==0,1],'.',x[assos==1,0],x[assos==1,1],'.',x[assos==2,0],x[assos==2,1],'.')
    clus=xmeans(x)
    for i in clus.keys():
        pl.plot(clus[i][:,0],clus[i][:,1],'k+',ms=20,mew=5)
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
    clus=xmeans(x)
    for i in clus.keys():
        pl.plot(clus[i][:,0],clus[i][:,1],'k+',ms=20,mew=5)
    lab.show()
