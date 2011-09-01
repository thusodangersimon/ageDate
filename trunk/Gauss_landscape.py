#!/usr/bin/env python
#
# Name:  Gauss_landscape
#
# Author: Thuso S Simon
#
# Date: 17 of April, 2011
#TODO: Make init have pror input, 
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
""" Creates a  N-1 dimensional gaussan landscape for use with a fitting program of user choise the -1 diemsion becames the liklihood. User can input: number of gaussian, possitions and FWHM and the program out puts a N dimesional file repersenting those values. Can also add noise to liklihood. The program will also output total area for prosterior calculations """

_module_name='gauss_param_space'

import numpy as nu
import pylab as lab
import mpl_toolkits.mplot3d.axes3d as lab3
from mpl_toolkits.mplot3d import Axes3D
import sympy as sy

class Param_landscape:
#creates the function for the land scape and makes it callable like a real problem
    def __init__(self,N=None,num_gaus=None):
        if N and N>1: #set number of dimensions
            self._N=N
        else: #if not set, choose 3 as default
            self._N=3
            
        if num_gaus:#sets number of peaks
            self._num_gaus=num_gaus
        else:#default is 1
            self._num_gaus=1

        #set boudaries where gaussian peaks are located
        self._bounds=nu.zeros([self._N-1,2])
        self._mu=nu.zeros([self._N-1,self._num_gaus]) #[dimensions,num of peaks]
        self._std=nu.zeros([self._N-1,self._num_gaus])
        self._amp=nu.array([None]*self._num_gaus)
        for i,j in enumerate(self._bounds[:,0]):
            self._bounds[i,:]=nu.array([-10,10])

        #set mu of each gausian in N space
            self._mu[i,:]=nu.diff(self._bounds[i,:])*nu.random.rand(self._num_gaus)+nu.mean(self._bounds[i,0])
#set sigma for each dimension and correlation
            self._std[i,:]=5*nu.random.rand(self._num_gaus)
        
        #make gaussian func
        

#callable fuction the returns likihood of space

    def likeihood_value(self,*args): #input and outputs a single value or vector
        #make gaussian func
#no coverance
        try: 
            #pass if want to make a [M,N] grid of points
            out=nu.ones([args[0].shape[0],args[0].shape[1],self._num_gaus])
            for i,ii in enumerate(args): 
                for j in range(self._num_gaus): #for each peak run a gauss calc
                    out[:,:,j]=out[:,:,j]*self.func(ii,self._mu[i,j],self._std[i,j],self._amp[j])
            return nu.sum(out,axis=2)
            
        except AttributeError: #if a single number make 1-d array
            temp_args=[]
            for i in args:
                if i.__class__==[].__class__:
                    temp_args.append(nu.array(i)) #if  a list
                else:
                    temp_args.append(nu.array([i])) #if int 
            args=temp_args
           
        except IndexError: #if float or shape=(N,)
            if len(args[0].shape)<1:
                temp_args=[]
                for i in args:
                    temp_args.append(nu.array([i]))
                args=temp_args

        out=nu.ones([args[0].shape[0],self._num_gaus])

        for i,ii in enumerate(args): #if a list turn into an numpy array
           for j in range(self._num_gaus): #for each peak run a gauss calc
                out[:,j]=out[:,j]*self.func(ii,self._mu[i,j],self._std[i,j],self._amp[j])
               
        return nu.sum(out,axis=1)

    def plot(self): #plots up to 3-d image of landscape
        if self._N>3:
            print 'to many dimesions'
            rase
        elif self._N==3: #plot in 3-d
            fig = lab.figure()
            ax = Axes3D(fig)
            #Z=self.likeihood_value(X)+self.likeihood_value(Y)
            X, Y = nu.meshgrid(nu.arange(self._bounds[0,0],self._bounds[0,1]+0.05 , 0.05),nu.arange(self._bounds[1,0],self._bounds[1,1]+0.05, 0.05) )
            Z=self.likeihood_value(X,Y)
            ax.plot_surface(X, Y, nu.log(Z))
            lab.show()
            
        
    def evidence(self,plot=None):
        #calculated the evidence or total area for landscape using erf's, uses 
        #symbolic package to compute integrals
        num_varable=self._mu.shape[0]
        #import varible
        from sympy.abc import x
        #check for amplitude
        if all(self._amp):
            amp=self._std*1.0
            for i in range(self._std.shape[1]):
                amp[:,i]=self._amp[i]+amp[:,1]*.0
        else: #use standar amp for normalization
            amp=self._std*1.0
            for i in range(self._std.shape[1]):
                amp[:,i]=1/(sy.sqrt(2*sy.pi)*amp[:,i])

        ans=sy.Mul(0)
        #eval integral, multiply same peaks add other peaks
        for i in range(self._num_gaus):
            multi_ans=sy.Mul(1.0)
            
            for j in range(num_varable):
                func=str(amp[j,i])+'*sy.exp(-(x-'+str(self._mu[j,i])+')**2/(2*'+str(self._std[j,i])+'**2))'
                multi_ans=multi_ans*sy.integrate(eval(func),(x,self._bounds[j,0],self._bounds[j,1]))
            ans=ans+multi_ans

        if plot: #if want plot of function
            from sympy.abc import x,y
            main_plot=sy.Mul(0)
            for i in range(self._num_gaus):
                multi_ans=sy.Mul(1.0)
                for j in range(2):
                    if j==1:
                        func=str(amp[j,i])+'*sy.exp(-(x-'+str(self._mu[j,i])+')**2/(2*'+str(self._std[j,i])+'**2))'
                    else: #put in y varible
                        func=str(amp[j,i])+'*sy.exp(-(y-'+str(self._mu[j,i])+')**2/(2*'+str(self._std[j,i])+'**2))'

                    multi_ans=multi_ans*sy.Mul(eval(func))
                main_plot=main_plot+multi_ans

            p=sy.Plot(sy.log(main_plot),[x,self._bounds[0,0],self._bounds[0,1]],[y,self._bounds[1,0],self._bounds[1,1]])
            p.wait_for_calculations()
            return float(ans.evalf())/(nu.prod(nu.diff( self._bounds,axis=1))),main_plot
        else:
            return float(ans.evalf())/nu.prod(nu.diff( self._bounds,axis=1))

    def func(self,x,mu=0,std=1,amp=None):
        #calculates gaussian function for landscape
        if amp:
            return amp*nu.exp(-(x-mu)**2/(2*std**2))
        else: #noramized to 1 in -inf,inf
            return 1/(nu.sqrt(2*nu.pi)*std)*nu.exp(-(x-mu)**2/(2*std**2))

################Toy Models##################################################
import nested_samp as nest
def toyI(N=300):
    toyI= Param_landscape()
    #set 2 d gausian on origen with prior beeing a 2*unit square
    toyI._mu[0,:]=[0]   
    toyI._mu[1,:]=[0]
    toyI._std[0,:]=[1]
    toyI._std[1,:]=toyI._std[0,:]
    toyI._bounds[0,:]=[-1.,1.]
    toyI._bounds[1,:]=[-1.,1.]
   #run nested sampling on it
    evidence,prior_vol,evid_error,old_points=nest.nested(toyI,N)
    lab.plot(prior_vol,nu.log(evidence))
    lab.title('Toy model 1, 2-D gausian at orgien')
    lab.xlabel('Prior Volume')
    lab.ylabel('ln evidence')
    #compare evidence to analytical one
    anyilitical=toyI.evidence()
    print 'Nested program calcuated evidence as %f, anaylitical is %f' %(sum(evidence),anyilitical/4.)
    lab.show()
   

def toyII(N=300):
    #3 gausians on a triangle with vertex at (0,-0.22),(-0.25,0.22),(0.25,0.22)
    #with std=1 and amplitudes of (5,3,9)
    toyII= Param_landscape(3,3)
    toyII._mu[0,:]=[0.0,-.25,.25]   
    toyII._mu[1,:]=[-.22,.22,.22]
    toyII._std[0,:]=[1.,1.,1.]
    toyII._std[1,:]=toyII._std[0,:]
    toyII._amp=[5.,3.,9.]
    toyII._bounds[0,:]=[-1.,1.]
    toyII._bounds[1,:]=[-1.,1.]

   #run nested sampling on it
    evidence,prior_vol,evid_error,old_points=nest.nested(toyII,N)
    lab.plot(prior_vol,nu.log(evidence))
    lab.title('Toy model 2, 3, 2-D gausian on Triangle')
    lab.xlabel('Prior Volume')
    lab.ylabel('ln evidence')
    #compare evidence to analytical one
    anyilitical=toyII.evidence()
    print 'Nested program calcuated evidence as %f, anaylitical is %f' %(sum(evidence),anyilitical/4.)
    lab.show()


def toyIII(C=2,N=300):
    from scipy.special import gamma
    #Skillin (2004) model where 1 gausian in C dimensions with a
    #flat prior on a unit circle
    #Z=(C/2)!*(2*sigma**2)**(C/2)
    #L=exp(-r**2/(2*sigma**2)) where r**2=sum(theda**2_i,i=0..C)
    toyIII=Param_landscape(C,1)
    #set std and amp, and change bounds
    toyIII._std=nu.zeros([toyIII._N-1,toyIII._num_gaus])+1.0
    toyIII._amp=nu.array([1.]*C)
    for i,j in enumerate(toyIII._bounds[:,0]):
        toyIII._bounds[i,:]=nu.array([-1,1])
    #set mu's following r<1 and on unit circle
    while sum (toyIII._mu**2)**.5>1.:
        toyIII._mu=nu.random.rand(len(toyIII._mu))*2-1

    Z=gamma(C/2.+1)*(2.*toyIII._std[0,0]**2)**(C/2.)
    evidence,prior_vol,evid_error,old_points=nest.nested(toyIII,N)
    print 'Nested program calcuated evidence as %f, anaylitical is %f' %(sum(evidence),Z)
