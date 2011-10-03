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
    axis=current.plot(points[:N,0],points[:N,1],'o',markersize=5)
    current.set_xlim((metal_unq.min(),metal_unq.max()))
    current.set_ylim((age_unq.min(),age_unq.max()))
    #plot current elipse
    curren_ellipse=Ellipse(xy=rand(2)*10, width=rand(), height=rand())
'''    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(rand())
    e.set_facecolor(rand(3))
    current.lines.pop
'''
                           
    fig.canvas.show()
    for i in xrange(N,points[N:,:].shape[0]): #add and remove ploints like in program
        
