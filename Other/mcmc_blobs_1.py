#! /usr/bin/python

import os
from numpy import random 
from numpy import *
import pyfits as fits
import pylab as pl

def check(val,range_test):
    #checks to see if params are in range
    if val > range_test and val<0.0:  
        return True
    else:
        return False

iterations = 1000
skip = 10
blobs=10
try:
  data1 = fits.getdata('1gaussian.clean.fits') #observed data
  fluxd=zeros([128,128],float)
  fluxd[:,:]=data1
except ValueError:
  fluxd = lambda x,y: 150*exp(-x**2/25.-y**2/25.)
  x,y=meshgrid(linspace(-10,10,128),linspace(-10,10,128))
  fluxd= fluxd(x,y)

chi2new=zeros([iterations,blobs],float)
chi2old=10.0e30

range_x=128
range_y=128
range_flux=150.0
range_sigma=15.0

x=zeros([iterations,blobs],float)
y=zeros([iterations,blobs],float)
flux=zeros([iterations,blobs],float)
sigma=zeros([iterations,blobs],float)

flux_new=zeros([blobs],float)
x_new=zeros([blobs],float)
y_new=zeros([blobs],float)
sigma_new=zeros([blobs],float)

running_flux=zeros([blobs],float)
running_x=zeros([blobs],float)
running_y=zeros([blobs],float)
running_sigma=zeros([blobs],float)

stddev_x = zeros([blobs],float) 
stddev_y=  zeros([blobs],float)
stddev_flux=zeros([blobs],float)
stddev_sigma=zeros([blobs],float)

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
  print i
  j,p=meshgrid(range(127),range(127))
  for m in range(blobs):
#   print 'iteration '+str(i)+' start'
   
    flux_new[m] = float(running_flux[m]+2.38*
                        stddev_flux[m]*random.randn()/sqrt(dof))
    while check(flux_new[m],range_flux):
      flux_new[m] = float(running_flux[m]+2.38*
                          stddev_flux[m]*random.randn()/sqrt(dof))
            
    x_new[m] = float(running_x[m]+2.38*
                     stddev_x[m]*random.randn()/sqrt(dof))
    while check(flux_new[m],range_flux):
      x_new[m] = float(running_x[m]+2.38*
                       stddev_x[m]*random.randn()/sqrt(dof))
    y_new[m] = float(running_y[m]+2.38*
                     stddev_y[m]*random.randn()/sqrt(dof))
    while check(flux_new[m],range_flux):
      y_new[m] = float(running_y[m]+2.38*
                       stddev_y[m]*random.randn()/sqrt(dof))
    sigma_new[m]= float(running_sigma[m]+2.38*
                        stddev_sigma[m]*random.randn()/sqrt(dof))
    while check(flux_new[m],range_flux):
      sigma_new[m]= float(running_sigma[m]+2.38*
                          stddev_sigma[m]*random.randn()/sqrt(dof))

    modelimage=zeros([128,128],float)
    chi2new=zeros([iterations,blobs],float)
#   chi2new[i,m]=0.0

    
    for l in range(blobs):
        modelimage[j,p] = modelimage[j,p] + exp(-((j-x_new[l])/sigma_new[l])**2 - ((p-y_new[l])/sigma_new[l])**2) * flux_new[l]
     
        chi2new[i,m] = sum((fluxd-modelimage)**2)
        
    if chi2new[i,m] < chi2old:  
        chi2old = copy(chi2new[i,m])
        print chi2old
        running_x[m] = copy(x_new[m])
        running_y[m]=  copy(y_new[m])
        running_flux[m]= copy(flux_new[m] )
        running_sigma[m]=copy(sigma_new[m])
    else:  
        delta_chi2 = chi2new[i,m]-chi2old
        if random.uniform(0,1) < exp(-(delta_chi2)/2.0): 
            chi2old = chi2new[i,m]
            print chi2old
            running_x[m] = copy(x_new[m])
            running_y[m]=  copy(y_new[m])
            running_flux[m]= copy(flux_new[m] )
            running_sigma[m]=copy(sigma_new[m])
     
        else:
            chi2new[i,m]= copy(chi2old)
   
  x[i,m] =copy( running_x[m])
  y[i,m] = copy(running_y[m])
  flux[i,m]=copy(running_flux[m])
  sigma[i,m]=copy(running_sigma[m])
   
  if i > skip:
    std_x =zeros([i+1]) #take the std. of only availabe value in iterations column 
    std_y =zeros([i+1])
    std_flux=zeros([i+1])
    std_sigma=zeros([i+1])
    for n in range(i):
      std_x[n] = copy(x[n,m])
      std_y[n] = copy(y[n,m])
      std_flux[n]= copy(flux[n,m])
      std_sigma[n]=copy(sigma[n,m])
      
    stddev_x[m] = std(std_x)
    stddev_y[m] = std(std_y)
    stddev_flux[m] = std(std_flux)
    stddev_sigma[m]=std(std_sigma) 
  

  
#  print 'iteration '+ str(i) +' finish'
  pl.imshow(modelimage)
  pl.title('sim with '+str(blobs)+' blobs')
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


