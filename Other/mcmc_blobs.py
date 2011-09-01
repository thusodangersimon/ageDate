#! /usr/bin/python

import os
from numpy import random 
from numpy import *
import pyfits as fits
import pylab as pl


iterations = 1000
dof = 4
skip = 10
blobs=10
data1 = fits.getdata('1gaussian.clean.fits') #observed data
fluxd=zeros([128,128],float)
fluxd[:,:]=data1
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

for k in range(blobs):
  x_new[k] = 128*random.uniform(0,1)
  stddev_x[k]=random.uniform(0,1)*2.0
  running_x[k] = x_new[k]
  y_new[k]= 128*random.uniform(0,1)
  stddev_y[k] =random.uniform(0,1)*2.0 
  running_y[k] = y_new[k]
  flux_new[k]=100*random.uniform(0,1) + 50.0
  stddev_flux[k] = random.uniform(0,1)*2.0
  running_flux[k] = flux_new[k]
  sigma_new[k]=10*random.uniform(0,1) + 5.0
  stddev_sigma[k] = random.uniform(0,1)*2.0
  running_sigma[k]=sigma_new[k]

for i in range(iterations):
  print i
  for m in range(blobs):
#   print 'iteration '+str(i)+' start'
   
   flux_new[m] = float(running_flux[m]+2.38*stddev_flux[m]*random.random()/sqrt(dof))
   if flux_new[m] > range_flux:  
        flux_new[m] = flux_new[m] - range_flux
   if flux_new[m] < 0.0: 
        flux_new[m] = flux_new[m] + range_flux
   if flux_new[m] > range_flux:  
        flux_new[m] = flux_new[m] - range_flux
   if flux_new[m] < 0.0: 
        flux_new[m] = flux_new[m] + range_flux
   if flux_new[m] > range_flux:  
        flux_new[m] = flux_new[m] - range_flux
   if flux_new[m] < 0.0: 
        flux_new[m] = flux_new[m] + range_flux
   
   x_new[m] = float(running_x[m]+2.38*stddev_x[m]*random.random()/sqrt(dof))
   if x_new[m] > range_x: 
       x_new[m] = x_new[m] - range_x
   if x_new[m] < 0.0: 
       x_new[m] = x_new[m] + range_x
   if x_new[m] > range_x: 
       x_new[m] = x_new[m] - range_x
   if x_new[m] < 0.0: 
       x_new[m] = x_new[m] + range_x
   if x_new[m] > range_x: 
       x_new[m] = x_new[m] - range_x
   if x_new[m] < 0.0: 
       x_new[m] = x_new[m] + range_x   

   y_new[m] = float(running_y[m]+2.38*stddev_y[m]*random.random()/sqrt(dof))
   if y_new[m] > range_y: 
      y_new[m] = y_new[m] - range_y
   if y_new[m] < 0.0:  
      y_new[m] = y_new[m] + range_y
   if y_new[m] > range_y: 
      y_new[m] = y_new[m] - range_y
   if y_new[m] < 0.0:  
      y_new[m] = y_new[m] + range_y
   if y_new[m] > range_y: 
      y_new[m] = y_new[m] - range_y
   if y_new[m] < 0.0:  
      y_new[m] = y_new[m] + range_y
   
   sigma_new[m]= float(running_sigma[m]+2.38*stddev_sigma[m]*random.random()/sqrt(dof))
   if sigma_new[m] > range_sigma:
       sigma_new[m] = sigma_new[m] - range_sigma
   if sigma_new[m] < 0.0: 
       sigma_new[m] = sigma_new[m] + range_sigma
   if sigma_new[m] > range_sigma:
       sigma_new[m] = sigma_new[m] - range_sigma
   if sigma_new[m] < 0.0: 
       sigma_new[m] = sigma_new[m] + range_sigma
   if sigma_new[m] > range_sigma:
       sigma_new[m] = sigma_new[m] - range_sigma
   if sigma_new[m] < 0.0: 
       sigma_new[m] = sigma_new[m] + range_sigma

   modelimage=zeros([128,128],float)
   chi2new=zeros([iterations,blobs],float)
#   chi2new[i,m]=0.0

   j,p=meshgrid(range(127),range(127))
   for l in range(blobs):
     modelimage[j,p] = modelimage[j,p] + exp(-((j-x_new[l])/sigma_new[l])**2 - ((p-y_new[l])/sigma_new[l])**2) * flux_new[l]
     
   chi2new[i,m] = sum((fluxd-modelimage)**2)

   if chi2new[i,m] < chi2old:  
      chi2old = chi2new[i,m]
      running_x[m] = x_new[m]
      running_y[m]=  y_new[m]
      running_flux[m]=flux_new[m] 
      running_sigma[m]=sigma_new[m]
   else:  
      delta_chi2 = chi2new[i,m]-chi2old
      if random.uniform(0,1) < exp(-(delta_chi2)/2.0): 
         chi2old = chi2new[i,m]
         running_x[m] = x_new[m]
         running_y[m] = y_new[m]
         running_flux[m]=flux_new[m]
         running_sigma[m]=sigma_new[m] 
      
      else:
       chi2new[i,m]=chi2old
   
   x[i,m] = running_x[m]
   y[i,m] = running_y[m]
   flux[i,m]=running_flux[m]
   sigma[i,m]=running_sigma[m]
   
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
