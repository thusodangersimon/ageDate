#! /usr/bin/python

import os
from numpy import *
import pyfits as fits
import pylab as pl
from multiprocessing import *
import time as Time

def Main(data = '1gaussian.clean.fits',iterations = 1000,skip = 10,blobs = 7):
#load params
    try:
        data1 = fits.getdata(data)
        fluxd=zeros([128,128],float)
        fluxd[:,:]=data1
    except ValueError:
        print 'cannot get file using internal blob'
        fluxd = lambda x,y: 150*exp(-x**2/25.-y**2/25.)
        x,y=meshgrid(linspace(-10,10,128),linspace(-10,10,128))
        fluxd= fluxd(x,y)
  
    chi2new=zeros([iterations,blobs],float)
    chi2old=inf

    range_x,range_y=fluxd.shape
    range_flux=10**3
#range_flux=fluxd.max()
    range_sigma=15.0

    x=zeros([iterations,blobs],float)
    y=zeros([iterations,blobs],float)
    flux=zeros([iterations,blobs],float)
    sigma=zeros([iterations,blobs],float)

#main MCMC program
    
    x_new =range_x *random.rand(blobs)
    stddev_x=random.rand(blobs)*20.
    running_x = copy(x_new)
    y_new= range_y*random.rand(blobs)
    stddev_y =random.rand(blobs)*2.0 
    running_y = copy(y_new)
    flux_new =range_flux*random.rand(blobs) + 50.0
    stddev_flux = random.rand(blobs)*20.
    running_flux = copy(flux_new)
    sigma_new =10*random.rand(blobs) + 5.0
    stddev_sigma = random.rand(blobs)*2.0
    running_sigma = copy(sigma_new)

    j,p=meshgrid(range(127),range(127))

    naccept,nreject=1,1
    chi2new=zeros([iterations,blobs],float)
    for i in range(iterations):
        #print naccept/(nreject+.0)
        modelimage=zeros(fluxd.shape)
        for m in range(blobs):
#   print 'iteration '+str(i)+' start'
            #changed boundary conditions
            flux_new[m] = float(running_flux[m]+stddev_flux[m]*random.randn())
            while check(flux_new[m],range_flux):
                flux_new[m] = float(running_flux[m]+stddev_flux[m]
                                    *random.randn())
            sigma_new[m]= float(running_sigma[m]+stddev_sigma[m]*random.randn())
            while check(flux_new[m],range_flux):
                sigma_new[m]= float(running_sigma[m]+stddev_sigma[m]*random.randn())

            x_new[m] = float(running_x[m]+stddev_x[m]*random.randn())
            if x_new[m] > range_x: 
                x_new[m] = x_new[m] - range_x
            if x_new[m] < 0.0: 
                x_new[m] = x_new[m] + range_x

            y_new[m] = float(running_y[m]+stddev_y[m]*random.randn())
            if y_new[m] > range_y: 
                y_new[m] = y_new[m] - range_y
            if y_new[m] < 0.0:  
                y_new[m] = y_new[m] + range_y

         
#   chi2new[i,m]=0.0 
            for k in range(blobs):
                modelimage[j,p] = modelimage[j,p]+exp(-((j-x_new[k])/sigma_new[k])**2 - ((p-y_new[k])/sigma_new[k])**2)*flux_new[k]
     
            chi2new[i,m] = sum((fluxd-modelimage)**2)

            if chi2new[i,m] < chi2old:  
                chi2old = copy(chi2new[i,m])
                print chi2old
                running_x[m] = copy(x_new[m])
                running_y[m]=  copy(y_new[m])
                running_flux[m]= copy(flux_new[m] )
                running_sigma[m]=copy(sigma_new[m])
                naccept+=1
            else:  
                delta_chi2 = chi2new[i,m]-chi2old
                if random.uniform(0,1) < exp(-(delta_chi2)/2.0): 
                    chi2old = copy(chi2new[i,m])
                    print chi2old
                    running_x[m] = copy(x_new[m])
                    running_y[m]=  copy(y_new[m])
                    running_flux[m]= copy(flux_new[m] )
                    running_sigma[m]=copy(sigma_new[m])
                    naccept+=1
                else:
                    chi2new[i,m]=copy(chi2old)
                    nreject+=1
   
            x[i,m] = copy (running_x[m])
            y[i,m] = copy(running_y[m])
            flux[i,m]= copy(running_flux[m])
            sigma[i,m]=copy(running_sigma[m])
        '''
        if naccept/(nreject+.0) < .34:
            stddev_x = stddev_x/2.05
            stddev_y = stddev_y/2.05
            stddev_flux = stddev_flux/2.05
            stddev_sigma= stddev_sigma/2.05
        if naccept/(nreject+.0) > .4:
            stddev_x = stddev_x*1.05
            stddev_y = stddev_y*1.05
            stddev_flux = stddev_flux*1.05
            stddev_sigma= stddev_sigma*1.05


        '''
        if i > skip:
            std_x =zeros([i+1]) #take the std. of only availabe value in iterations column 
            std_y =zeros([i+1])
            std_flux=zeros([i+1])
            std_sigma=zeros([i+1])
            for n in range(i):
                std_x[n] = x[n,m]
                std_y[n] = y[n,m]
                std_flux[n]=flux[n,m]
                std_sigma[n]=sigma[n,m]
        
            stddev_x[m] = std(std_x)
            stddev_y[m] = std(std_y)
            stddev_flux[m] = std(std_flux)
            stddev_sigma[m]=std(std_sigma) 
           

  
#  print 'iteration '+ str(i) +' finish'
#    hdu = fits.PrimaryHDU(modelimage)
 #   hdu.writeto('gau'+str(i)+'.fits')
        pl.imshow(modelimage)
        pl.title('sim with '+str(blobs)+' blobs')
        if i<10:
            filename = 'gau00'+str(i)+'.png'
        elif i<100 and i>=10:
            filename = 'gau0'+str(i)+'.png'
        else:
            filename = 'gau'+str(i)+'.png'
        pl.savefig(filename, dpi=100)
        pl.clf()
  #hdu = fits.PrimaryHDU(modelimage)
  #hdu.writeto('gau'+str(i)+'.fits')


    command = ('mencoder',
           'mf://*.png',
           '-mf',
           'type=png:w=800:h=600:fps=25',
           '-ovc',
           'lavc',
           '-lavcopts',
           'vcodec=mpeg4',
           '-oac',
           'copy',
           '-o',
           'fit_movie.avi')

    os.spawnvp(os.P_WAIT, 'mencoder', command)

    return x,y,flux,sigma,chi2new


def gau_fuc():
    #BI VARIATE gaussain function
    pass

def check(val,range_test):
    #checks to see if params are in range
    if val > range_test and val<0.0:  
        return True
    else:
        return False

