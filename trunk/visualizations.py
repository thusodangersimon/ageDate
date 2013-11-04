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



def bayes_hist_bins(t):
    """Bayesian Blocks Implementation

    By Jake Vanderplas.  License: BSD
    Based on algorithm outlined in http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    Parameters
    ----------
    t : ndarray, length N
        data to be histogrammed

    Returns
    -------
    bins : ndarray
        array containing the (N+1) bin edges

    Notes
    -----
    This is an incomplete implementation: it may fail for some
    datasets.  Alternate fitness functions and prior forms can
    be found in the paper listed above.
    """
    # copy and sort the array
    t = nu.sort(t)
    N = t.size

    # create length-(N + 1) array of cell edges
    edges = nu.concatenate([t[:1],
                            0.5 * (t[1:] + t[:-1]),
                            t[-1:]])
    block_length = t[-1] - edges

    # arrays needed for the iteration
    nn_vec = nu.ones(N)
    best = nu.zeros(N, dtype=float)
    last = nu.zeros(N, dtype=int)

    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for K in xrange(N):
        # Compute the width and count of the final bin for all possible
        # locations of the K^th changepoint
        width = block_length[:K + 1] - block_length[K + 1]
        count_vec = nu.cumsum(nn_vec[:K + 1][::-1])[::-1]

        # evaluate fitness function for these possibilities
        fit_vec = count_vec * (nu.log(count_vec) - nu.log(width))
        fit_vec -= 4  # 4 comes from the prior on the number of changepoints
        fit_vec[1:] += best[:K]

        # find the max of the fitness: this is the K^th changepoint
        i_max = nu.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]

    #-----------------------------------------------------------------
    # Recover changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points =  nu.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points]


def make_chi_grid(data,dust=None,losvd=None,points=500, Metal=None, Age=None):
    'makes a 3d pic of metal,age,chi with input spectra 2-D only'
    fun=ag.MC_func(data)
    fun.autosetup()
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
    dust_param = nu.zeros_like(param[:,:2])
    if nu.any(dust):
        dust_param[:,0] = dust[0]
        dust_param[:,1] = dust[1]
    losvd_param = nu.zeros((dust_param.shape[0],4))
    if nu.any(losvd):
        for i in range(len(losvd)):
            losvd_param[:,i] = losvd[i]
    #start making calculations for chi squared value
    po,out=Pool(),[]
    for i in xrange(len(param)):
        out.append(po.apply_async(func,args = (fun.data,param[i],dust_param[i],losvd_param[i],
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

def func(data,param,param_dust,losvd,lib_vals,age_unq,metal_unq):

    bins = param.shape[0] / 3
    model = ag.get_model_fit_opt(param, lib_vals, age_unq,metal_unq, bins)
    model = ag.dust(nu.hstack((param, param_dust)), model)
    model = ag.LOSVD(model,losvd,[data[:,0].min(), data[:,0].max()])
    model = ag.data_match(data, model, bins,True)
    param[slice(2,bins*3,3)],chi = ag.N_normalize(data, model, bins)
    return nu.hstack((param, param_dust,losvd,chi))


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
