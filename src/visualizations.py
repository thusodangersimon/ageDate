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
#import Age_date as ag ##temp import##
import numpy as nu
from scipy.signal import convolve
import pylab as lab
from multiprocessing import Pool,pool
"""
Code to plot a contour from an MCMC chain
Author: Michelle Knights (2013)
Modified: Jonathan Zwart (12 August 2013)
"""

import sys,os
import numpy
import numpy as np
import pylab
from scipy import interpolate
#from lumfunc import *
#import line_profiler #-should install
#from utils import *
#from settings import *

#-------------------------------------------------------------------------------


def findconfidence(H):
    """
    Finds the 95% and 68% confidence intervals, given a 2d histogram
    of the likelihood
    """


    H2 = H.ravel()
    H2 = numpy.sort(H2)
    
    #Cut out the very low end
    #H2 = H2[H2>100]

    #Loop through this flattened array until we find the value in the
    #bin which contains 95% of the points
    tot = sum(H2)
    tot95=0
    tot68=0

    #Changed this to 68% and 30% C.I
    for i in range(len(H2)):
        tot95 += H2[i]
        if tot95 >= 0.05*tot:
            N95 = H2[i]
            #print i
            break

    for i in range(len(H2)):
        tot68 += H2[i]
        if tot68>=0.32*tot:
            N68 = H2[i]
            break   
    return max(H2),N95,N68

#-------------------------------------------------------------------------------

def contour(chain,p,**kwargs):
    """
    Original alias for contourSingle
    """
    return contourSingle(chain,p,**kwargs)

#-------------------------------------------------------------------------------

def contourSingle(chain,p,**kwargs):
    """
    #Given a chain, labels and a list of which parameters to plot, plots the contours
    # Arguments:
    # chain=an array of the chain (not using weights, i.e. each row counts only once)
    # p= a list of integers: the two parameters you want to plot (refers to two columns in the chain)
    #kwargs:        labels= the labels of the parameters (list of strings)
    #               col=a tuple of the two colours for the contour plot
    #               line=boolean whether or not to just do a line contour plot
    #               outfile='outf.png'
    """

    # !!!! BEWARE THE BINSIZE --- PLOT IS A STRONG FUNCTION OF THIS
    binsize=50
    H,xedges,yedges=numpy.histogram2d(chain[:,p[0]],chain[:,p[1]],bins=(binsize,binsize))
    
    x=[]
    y=[]
    z=[]
    for i in range(len(xedges[:-1])):
        for j in range(len(yedges[:-1])):
            x.append(xedges[:-1][i])
            y.append(yedges[:-1][j])
            z.append(H[i, j])

    SMOOTH=False
    if SMOOTH:
        sz=50
        smth=80e6
        spl = interpolate.bisplrep(x, y, z,  s=smth)
        X = numpy.linspace(min(xedges[:-1]), max(xedges[:-1]), sz)
        Y = numpy.linspace(min(yedges[:-1]), max(yedges[:-1]), sz)
        Z = interpolate.bisplev(X, Y, spl)
    else:
        X=xedges[:-1]
        Y=yedges[:-1]
        Z=H

    #I think this is the weird thing I have to do to make the contours work properly
    X1=numpy.zeros([len(X), len(X)])
    Y1=numpy.zeros([len(X), len(X)])
    for i in range(len(X)):
        X1[ :, i]=X
        Y1[i, :]=Y
    X=X1
    Y=Y1
    
    N100,N95,N68 = findconfidence(Z)

    if 'col' in kwargs:
        col=kwargs['col']
    else:
        col =('#a3c0f6','#0057f6') #A pretty blue

    if 'labels' in kwargs:
        labels=kwargs['labels']
    else:
        labels = ['x', 'y']

    pylab.clf()

    if 'line' in kwargs and kwargs['line']==True:
        pylab.contour(X, Y,Z,levels=[N95,N68,N100],colors=col, linewidth=100)
    else:
        pylab.contourf(X, Y,Z,levels=[N95,N68,N100],colors=col)
    pylab.xlabel(labels[p[0]],fontsize=22)
    pylab.ylabel(labels[p[1]],fontsize=22)

    
    if 'outfile' in kwargs:
        outfile=kwargs['outfile']
        pylab.savefig(outfile)
        #pylab.close()
    else:
        pylab.show()

    return

#-------------------------------------------------------------------------------


def contourTri(chain,**kwargs):
    """
    #Given a chain, labels and a list of which parameters to plot, plots the contours
    # Arguments:
    # chain=an array of the chain (not using weights, i.e. each row counts only once)
    # p= a list of integers: the two parameters you want to plot (refers to two columns in the chain)
    #kwargs:        labels= the labels of the parameters (list of strings)
    #               col=a tuple of the two colours for the contour plot
    #               line=boolean whether or not to just do a line contour plot
    #               outfile='triangle.png'
    #               binsize=50
    #               reconstruct=boolean whether or not to plot reconstruction
    #               autoscale=boolean whether or not to autoscale axes
    #               ranges=dictionary of plot range lists, labelled by
    #                      parameter name, e.g. {'A':[0.0,1.0],etc.}
    #               title=outdir
    p is now ignored
    """

    # !!!! BEWARE THE BINSIZE --- PLOT IS A STRONG FUNCTION OF THIS
    if 'binsize' in kwargs:
        binsize=kwargs['binsize']
    else:
        binsize=50
    print 'Using binsize = %i' % binsize

    if 'labels' in kwargs:
        labels=kwargs['labels']
    else:
        labels = ['x', 'y']

    if 'ranges' in kwargs:
        ranges=kwargs['ranges']
    else:
        ranges=None

    if 'title' in kwargs:
        title=kwargs['title']
    else:
        title=''


    p = range(len(labels))
    pairs = trianglePairs(p)
    nparams = len(p)

    # Start setting up the plot
    ipanel=0; ax={}
    pylab.clf()
    for panel in pairs:
        ipanel+=1        
        H,xedges,yedges=numpy.histogram2d(chain[:,panel[0]],chain[:,panel[1]],bins=(binsize,binsize))

        x=[]
        y=[]
        z=[]
        for i in range(len(xedges[:-1])):
            for j in range(len(yedges[:-1])):
                x.append(xedges[:-1][i])
                y.append(yedges[:-1][j])
                z.append(H[i, j])

        SMOOTH=False
        if SMOOTH:
            sz=50
            smth=80e6
            spl = interpolate.bisplrep(x, y, z,  s=smth)
            X = numpy.linspace(min(xedges[:-1]), max(xedges[:-1]), sz)
            Y = numpy.linspace(min(yedges[:-1]), max(yedges[:-1]), sz)
            Z = interpolate.bisplev(X, Y, spl)
        else:
            X=xedges[:-1]
            Y=yedges[:-1]
            Z=H
    
        #I think this is the weird thing I have to do to make the contours work properly
        X1=numpy.zeros([len(X), len(X)])
        Y1=numpy.zeros([len(X), len(X)])
        for i in range(len(X)):
            X1[ :, i]=X
            Y1[i, :]=Y
        X=X1
        Y=Y1
    
        N100,N95,N68 = findconfidence(Z)

        if 'col' in kwargs:
            col=kwargs['col']
        else:
            col =('#a3c0f6','#0057f6') #A pretty blue

        # Now construct the subplot
        ax[ipanel]=pylab.subplot2grid((nparams,nparams),panel[::-1]) # Reverse quadrant

        if 'line' in kwargs and kwargs['line']==True:
            pylab.contour(X, Y,Z,levels=[N95,N68,N100],colors=col, linewidth=100)
        else:
            pylab.contourf(X, Y,Z,levels=[N95,N68,N100],colors=col)

        if 'truth' in kwargs:
            truth=kwargs['truth']
            pylab.plot(truth[labels[panel[0]]],truth[labels[panel[1]]],'g+',\
            markersize=20)

        # Set the axis labels only for left and bottom:
        #print ax[ipanel].get_xlabel(),ax[ipanel].get_ylabel()
        if panel[1] == (nparams-1):
            #ax[ipanel].set_xlabel(labels[panel[0]],fontsize=2)
            ax[ipanel].set_xlabel(labels[panel[0]],fontsize=8)
        else:
            ax[ipanel].set_xlabel('')
            ax[ipanel].get_xaxis().set_ticklabels([])
        if panel[0] == 0:
            #ax[ipanel].set_ylabel(labels[panel[1]],fontsize=2)
            ax[ipanel].set_ylabel(labels[panel[1]],fontsize=8)
        else:
            ax[ipanel].set_ylabel('')
            ax[ipanel].get_yaxis().set_ticklabels([])

        # Set plot limits
        autoscale=True
        if 'autoscale' in kwargs:
            autoscale=kwargs['autoscale']
        if not autoscale:
            xlo=ranges[labels[panel[0]]][0]
            xhi=ranges[labels[panel[0]]][1]
            ylo=ranges[labels[panel[1]]][0]
            yhi=ranges[labels[panel[1]]][1]
            pylab.xlim(xlo,xhi)
            pylab.ylim(ylo,yhi)

        # Some housekeeping
        #pylab.xticks(fontsize=8)
        #pylab.yticks(fontsize=8)
        pylab.xticks(fontsize=2)
        pylab.yticks(fontsize=2)


    # Set up the 1-D plots on the diagonal
    for iparam in range(nparams):
        #        b=numpy.histogram(R,bins=bins)
        J,edges=numpy.histogram(chain[:,iparam],density=True,bins=binsize)
        ax1d=pylab.subplot2grid((nparams,nparams),(iparam,iparam))
        pylab.plot(edges[:-1],J,color='k')
        #print iparam,nparams,labels[iparam]

        if 'truth' in kwargs:
            truth=kwargs['truth']
            pylab.axvline(truth[parameters[iparam]],color='g')

        if iparam == 0:
            ax1d.set_ylabel(labels[iparam],fontsize=8)
        if iparam == (nparams-1):
            ax1d.set_xlabel(labels[iparam],fontsize=8)

        # Set plot limits
        if not autoscale:
            xlo,xhi=ranges[parameters[iparam]]
            pylab.xlim(xlo,xhi)
        if iparam < (nparams-1):
            ax1d.get_xaxis().set_ticklabels([])
        ax1d.get_yaxis().set_ticklabels([])
        pylab.xticks(fontsize=2)
        pylab.yticks(fontsize=2)

    axinfo=pylab.subplot2grid((nparams,nparams),(0,nparams-3))
    axinfo.get_xaxis().set_visible(False)
    axinfo.get_yaxis().set_visible(False)
    pylab.axis('off')
    pylab.title(title)

    # Plot the truth - this needs to be generalized for non-lumfunc
    if 'truth' in kwargs:
        truth=kwargs['truth']
        note=['nparams %i\n truth:' % nparams]
        for k,v in truth.items():
            notelet='%s = %4.2f' % (k,v)
            note.append(notelet)
        pylab.text(0,0,'\n'.join(note))

        if 'reconstruct' in kwargs:
            reconstruct=kwargs['reconstruct']
            axrecon=pylab.subplot2grid((nparams,nparams),(0,nparams-2),\
                                       rowspan=2,colspan=2)
            axrecon.set_xscale('log')
            axrecon.set_yscale('log')
            pylab.xticks(fontsize=2)
            pylab.yticks(fontsize=2)
            median_bins=medianArray(reconstruct[0])
            dnds=calculateDnByDs(median_bins,reconstruct[1])
            dndsN=calculateDnByDs(median_bins,ksNoisy)
            print median_bins
            print dnds
            print truth.items()
            recon=numpy.zeros(numpy.shape(median_bins))
            post=numpy.zeros(numpy.shape(median_bins))
            print '# i Smedian ks dnds dndsS2.5 NRecon dndsRecon dndsS2.5Recon log10dnds log10dndsR diffR dndsN'
            if nparams == 4:
                (C,alpha,Smin,Smax)\
                  =(truth['C'],truth['alpha'],truth['Smin'],truth['Smax'])
                area=10.0 # Hack
                # Reconstruct powerLaw points given truth
                for i in range(len(median_bins)):
                    recon[i]=powerLawFuncS(median_bins[i],\
                                                   C,alpha,Smin,Smax,area)
                    post[i]=powerLawFuncS(median_bins[i],\
                                                  9.8,-0.63,0.04,14.1,area)
                #recon *= lumfunc.ksRaw
                #dndsR=lumfunc.calculateDnByDs(median_bins,recon)
                # **** XXXX Where does the 1000 come from!? :(( XXXX
                dndsR=recon*1000.0
                dndsP=post*1000.0
                # cols: i Smedian ks dnds dndsS2.5 NRecon dndsRecon
                # dndsS2.5Recon log10dnds log10dndsR diffR dndsN
                for i in range(len(median_bins)):
                    print '%i %f %i %i %i %i %i %i %f %f %i %i' % (i,median_bins[i],\
                                                  reconstruct[-1][i],dnds[i],\
                      dnds[i]*median_bins[i]**2.5,recon[i],dndsR[i],\
                      dndsR[i]*median_bins[i]**2.5,numpy.log10(dnds[i]),\
                        numpy.log10(dndsR[i]),int(dndsR[i]-dnds[i]),dndsN[i])

                      #print recon
            pylab.xlim(reconstruct[0][0],reconstruct[0][-1])
            #pylab.ylim(1.0e2,1.0e8)

            #pylab.plot(median_bins,dnds*numpy.power(median_bins,2.5)*lumfunc.sqDeg2sr,'+')
            power=2.5
            pylab.plot(median_bins,dnds*sqDeg2sr*numpy.power(median_bins,power),'+')
            pylab.plot(median_bins,dndsR*sqDeg2sr*numpy.power(median_bins,power),'-')
            pylab.plot(median_bins,dndsN*sqDeg2sr*numpy.power(median_bins,power),'+')
            pylab.plot(median_bins,dndsP*sqDeg2sr*numpy.power(median_bins,power),'-')
            #pylab.plot(dnds,dndsR*numpy.power(median_bins,1.0))
            #b=lumfunc.simtable(lumfunc.bins,a=-1.5,seed=1234,noise=10.0,dump=False)

    if 'outfile' in kwargs:
        outfile=kwargs['outfile']
        pylab.savefig(outfile,figsize=(8.27, 11.69),dpi=400)
        print 'Look in %s' % outfile
        #pylab.close()
    else:
        pylab.show()

    return

#-------------------------------------------------------------------------------


def trianglePairs(inlist):
    """
    """
    pairs=[]
    for i in inlist:
        for j in inlist:
            if j > i:
                pairs.append((i,j))

    return pairs
    


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

def make_sfh_plot(param, model=None):
        '''(dict(ndarray)-> Nonetype
        Make plot of sfh vs log age of all models in param 
        '''
        import pylab as lab
        if not model is None:
            x,y = [], []
            for i in param[model]:
                x,y = [] , []
                #make square bracket
                x.append(i[1]-i[0]/2.)
                x.append(i[1]-i[0]/2.)
                x.append(i[1]+i[0]/2.)
                x.append(i[1]+i[0]/2.)
                y.append(i[3]-50)
                y.append(i[3])
                y.append(i[3])
                y.append(i[3]-50)
                lab.plot(x,y,'b',label=model)
            lab.legend()
            lab.show()
        else:
            for i in param.keys():
                pass
        return x,y
            
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    """
    """

    infile=sys.argv[1];
    outfile=sys.argv[2];

    #parameters=lumfunc.parameters['C','alpha','Smin','Smax']

#Testing all functionality
#c=pylab.loadtxt('chain_2d_banana.txt')
#contour(c,[0,1], labels=['1', '2'], col=('#3bf940','#059a09'),line=True)

    trueS = float(sys.argv[3]);

    ratio_true=0.667;
    eccen_true = np.sqrt(1-ratio_true*ratio_true)
    flatt_true = 1-np.sqrt(1-eccen_true*eccen_true);

# Set up the parameters                                                     
    parameters=['S','l','m','maj','min','pa','rat','eccen','flatt']
#plotRanges={'S':[0.0,100.0],
#           'alpha':[-2.5,-0.1],
#           'Smin':[0.0,5.0],
#           'Smax':[0.0,50.0]}
    plotTruth={'S':trueS,
          'l':0.0,
          'm':0.0,
          'maj':60.0,
          'min':40.0,
	  'pa':90.0,
	  'rat':ratio_true,
	  'eccen':eccen_true,
          'flatt':flatt_true}

    print parameters, plotTruth

    contourTri(pylab.loadtxt(infile),line=True,outfile=outfile,col=('red','blue'),labels=parameters,truth=plotTruth)
    #contourTri(pylab.loadtxt('2-post_equal_weights_derived.dat'),line=True,outfile='tri_post_nov1.png',col=('red','blue'),labels=parameters,truth=plotTruth)

    # Run as e.g.
#contour_plot.contourTri(pylab.loadtxt('chains-4-all-10deg-130812/1-post_equal_weights.dat'),line=True,outfile='chains-4-all-10deg-130812/test.png',col=('red','blue'),labels=lumfunc.parameters,ranges=lumfunc.plotRanges,truth=lumfunc.plotTruth,reconstruct=(lumfunc.medianArray(lumfunc.bins),lumfunc.ksNoisy),autoscale=False,title='title')

    #contour_plot.contourTri(pylab.loadtxt('chains-3-fixSmin-10deg-130812/1-post_equal_weights.dat'),line=True,outfile='test.png',col=('red','blue'),labels=lumfunc.parameters)

    #import pymultinest
    #a=pymultinest.Analyzer(len(lumfunc.parameters),'chains-4-all-mm-10deg-130815/1-')
    #s=a.get_stats()
    #print s
    
    sys.exit(0)
