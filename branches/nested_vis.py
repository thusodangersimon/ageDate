#!/usr/bin/env python
#
# Name:  Nested vis
#
# Author: Thuso S Simon
#
# Date: 10/3/11
# TODO: make multi-dimesional plots (muliple figures for each age,metal axis
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
""" makes visulaizations to diagnose nested sampling"""


from nested_samp import *
import pylab as lab
from matplotlib.patches import Ellipse

def elipse_vis(points,bins,N,elogf=1.1):
    #slowly steps through points showing how eliplse changed
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])
    #plot first N points
    fig=lab.figure()
    current=fig.add_subplot(111)
    #plot chi squared landscape using griddata
    metal,age=nu.linspace(metal_unq[0],metal_unq[-1],200),nu.linspace(age_unq[0],age_unq[-1],200)
    z=lab.griddata(points[:,0],points[:,1],points[:,-1],metal,age)
    current.pcolor(metal,age,nu.log10(z))
    #lab.colorbar()
    for i in xrange(N):
        current.plot(points[i,0],points[i,1],'yo',markersize=5)
    current.set_xlim((metal_unq.min(),metal_unq.max()))
    current.set_ylim((age_unq.min(),age_unq.max()))
    current.set_xlabel('Log Metalicity')
    current.set_ylabel('Age')
    #plot current elipse
    curren_ellipse=Ellipse(xy=[metal_unq[0]+metal_unq.ptp()/2.,
                               age_unq[0]+age_unq.ptp()/2.], 
                           width=metal_unq.ptp()*elogf,height=age_unq.ptp()*elogf)
    current.add_artist(curren_ellipse)
    curren_ellipse.set_alpha(.5)
    #curren_ellipse.set_facecolor(nu.random.rand(3))
    #current.lines.pop                         
    fig.canvas.show()
    for i in xrange(i,points[:,0].shape[0]): #add and remove ploints like in program
        #remove last point and add new one in red
        current.lines[i-N-1].set_visible(False)
        #make last new point blue
        current.lines[i-1].set_color('y')
        current.plot(points[i,0],points[i,1],'ro',markersize=5)
        #change ellipse size
        current.artists.pop(0)
        curren_ellipse=Ellipse(xy=nu.mean(points[i-N+1:i+1,:2],0),
                                   width=points[i-N+1:i+1,0].ptp()*elogf,height=points[i-N+1:i+1,1].ptp()*elogf)
        curren_ellipse.set_alpha(.5)
        current.add_artist(curren_ellipse)
        fig.canvas.show()

def draw_ellipse():
    s,t=nu.mat(nu.identity(2)),nu.mat(nu.identity(2))
    X,Y=nu.linspace(-2.6,-2.25,200),nu.linspace(7.8,8.3,200)
    x=nu.mat(nu.ones(2))
    #points=[]
    #for i in range(400):
    #    points.append(nu.random.multivariate_normal([0,0],nu.array([[2,.342],[1,-.3242]])))
    #points=nu.array(points)
    cov=nu.cov(points.T)
    s=nu.mat(cov)
    mean=points.mean(0)
    out=nu.zeros([200,200])
    for i in range(200):
        for j in range(200):
            x[0,:2]=nu.array([X[i],Y[j]])-mean
            if x*s**-1*x.T<=7:
                out[i,j]=1
    out=out.T
