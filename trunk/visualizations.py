#!/usr/bin/env python
#
# Name:  
#
# Author: Thuso S Simon
#
# Date: Sep. 1, 2011
# TODO: 
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
"""
programs to visualize the Age date program
"""
import Age_date as ag ##temp import##
import numpy as nu
from scipy.signal import convolve
import pylab as lab
from multiprocessing import Pool,pool

def make_chi_grid(data,dust=None,points=500, Metal=None, Age=None):
    'makes a 3d pic of metal,age,chi with input spectra 2-D only'
    fun=ag.MC_func(data)
    ag.spect = fun.spect
    #create grid
    if nu.any(Metal) or nu.any(Age):
        metal,age = nu.meshgrid(Metal,Age)
    else:
        metal,age=nu.meshgrid(nu.linspace(fun._metal_unq.min(),
                                          fun._metal_unq.max(),points),
                              nu.linspace(fun._age_unq.min(),
                                          fun._age_unq.max(),points))

    param = nu.array(zip(metal.ravel(),age.ravel(),nu.ones_like(age.ravel())))
    #probably need to handel dust in a better way
    dust_param = nu.zeros([len(param),2])
    if nu.any(dust):
        dust_param[:,0] = dust[0]
        dust_param[:,1] = dust[1]
        
    
    #start making calculations for chi squared value
    po,out=Pool(),[]
    for i in xrange(len(param)):
        out.append(po.apply_async(func,args = (fun.data,param[i],dust_param[i],
                            fun._lib_vals ,fun._age_unq,fun._metal_unq)))
    b=nu.array(map(get, out))
    po.close()
    po.join()
    if nu.any(Metal) or nu.any(Age):
        return b[:,-1].reshape(points,points)
    else:
        return metal,age,b[:,-1].reshape(points,points)

def get(f):
    return f.get()

def func(data,param,param_dust,lib_vals,age_unq,metal_unq):

    bins = param.shape[0] / 3
    model = ag.get_model_fit_opt(param, lib_vals, age_unq,metal_unq, bins)
    model = ag.dust(nu.hstack((param, param_dust)), model)
    N,chi = ag.N_normalize(data, model, bins)
    return nu.hstack((param, param_dust,chi))


def histo_plot(x,y,z=None):
    '''plots 2-d histogram from frequency in poins (x,y)
    will plot x,y,z pcolor plot with 1 sigma contours if z is given'''

    assert len(x.shape) == 1 and len(y.shape) == 1
    if nu.any(z):
        assert z.shape[0] == x.shape[0]
        
    #create histogram
    Z,X,Y = nu.histogram2d(x,y,[200,200])
    Z = Z.T
    #get likelhood brute force if z not there
    if nu.any(z):
        Zz = lab.griddata(x,y,z,X,Y)
    else:
        Zz = make_chi_grid(data, points=100, metal=X, age=Y)
    #get hist plot ready from pylab docs
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    lab.figure(1, figsize=(8,8))

    axMain = lab.axes(rect_scatter)
    axHistx = lab.axes(rect_histx)
    axHisty = lab.axes(rect_histy)
    #plot 1 sigma contors
    axMain.contour(X[3:],Y[3:],blur_image(Z,1),
                nu.percentile(Z[Z.nonzero()],[16,84]),color=('red','red'))
    axMain.pcolor(X,Y,Zz,cmap='gray')
    axHistx.hist(x, bins=200)
    axHisty.hist(y, bins=200, orientation='horizontal')

    axMain.set_xlabel('Metalicity')
    axMain.set_ylabel('Age')

    axHistx.set_xlim( axMain.get_xlim() )
    axHisty.set_ylim( axMain.get_ylim() )
    lab.show()

#from scipy cookbook
def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = nu.mgrid[-size:size+1, -sizey:sizey+1]
    g = nu.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()
   
def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
    size n. The optional keyword argument ny allows for a different
    size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = convolve(im,g, mode='valid')
    return improc
