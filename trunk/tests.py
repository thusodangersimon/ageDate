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
scap develoment programs
"""


#from Age_date import *
from mpi4py import MPI
import numpy as nu

def quick_cov_MCMC(x,y,params,func=[],constrants=[],sigma=0.8,itter=10**5,quiet=False):
    pypar=MPI.COMM_WORLD
    proc = pypar.size                                
    myid = pypar.Get_rank()
    param_info,sigma_tag,end_tag=0,1,2 #tags for sending and reciving
     #master process handels cov calculations and storring of out data
    out_param=[]
    out_chi=[]
    i,j,i_out=0,0,0
    sigma=nu.identity(len(params))*sigma
    #print myid
    if myid!=0:
        param=nu.array([params,params])#set input parameters [0]=old [1]=new
        if not func: #see if inputed a function
            raise
        #first fit
        y_fit=func(x,param[1])
        #start up chi
        chi=[nu.sum((y_fit-y)**2),nu.inf]
        if nu.isnan(chi[0]):
            chi[0]=nu.inf
        chibest=nu.inf
        parambest=nu.copy(param[0])
        out_param.append(nu.copy(param[0]))
        out_chi.append(nu.copy(chi[0]))

       #start mcmc
        keep_iter=True
        while keep_iter:
            i+=1
            i_out+=1
            #print i_out
            if i%1000==0 and not quiet:
                print 'current accptence rate %2.2f and chi2 is %2.2f' %(j/(i+1.0)*100.0,chi[1])
                #print "my id is %i" %myid
            # print param[1] 
             #select new param
            param[1]=nu.random.multivariate_normal(param[0],sigma)
            for ii in range(len(params)):
                i2=0
                while param[1][ii]<=constrants[ii][0] or param[1][ii]>=constrants[ii][1]: 
                    param[1][ii]=param[0][ii]+nu.random.randn()*sigma[ii,ii]
                    i2+=1
                    if i2>50:#sigma may be too big
                        sigma[ii,ii]=sigma[ii,ii]/1.05
            #sample new distribution
            y_fit=func(x,param[1])
            chi[1]=nu.sum((y_fit-y)**2)
            if nu.isnan(chi[1]):
                chi[1]=nu.inf
        
            #decide to accept or not
            a=nu.exp((chi[0]-chi[1])/2.0)
            #metropolis hastings
            if a>=1: #acepted
                chi[0]=chi[1]+0.0
                param[0]=param[1]+0.0
                out_param.append(nu.copy(param[0]))
                out_chi.append(nu.copy(chi[0]))
                j+=1
                if chi[0]<chibest:
                    chibest=chi[0]+0.0
                    parambest=param[0]+0.0
                    #if not quiet:
                    print 'best fit value for %3.2f,%3.2f with chi2=%4.2f' %(parambest[0],parambest[1],chibest)
                    #print i

            else:
                if a>nu.random.rand():#false accept
                    chi[0]=chi[1]+0.0
                    param[0]=param[1]+0.0
                    j+=1
                    out_param.append(nu.copy(param[0]))
                    out_chi.append(nu.copy(chi[0]))

                else:
                    out_param.append(nu.copy(param[0]))
                    out_chi.append(nu.copy(chi[0]))
            
            if i_out==50:
                pypar.isend([out_param,out_chi,j,i],dest=0,tag=param_info)
                #print 'here 1'
                sigma=pypar.recv(source=0, tag=sigma_tag)
                keep_iter=pypar.recv(source=0, tag=end_tag)
                #print sigma
                i_out=0
                out_param,out_chi=[],[]
        pypar.Barrier()
    else:
        while True:
            status=MPI.Status()
            temp=pypar.recv(source=MPI.ANY_SOURCE,tag=param_info,status=status)
            temp_param,temp_chi,temp_j,temp_i=temp
            for ii in temp_param:
                out_param.append(nu.copy(ii))
            for ii in temp_chi:
                out_chi.append(nu.copy(ii))
            j+=temp_j
            i+=temp_i
            #print float(j/(i+j))
       #change sigma with acceptance rate
            if float(j/(i+j))>.24 and any(sigma.diagonal()<3): #too many aceptnce increase sigma
                sigma=sigma*5.0
            elif float(j/(i+j))<.34 and any(sigma.diagonal()>10**-5): #not enough
                sigma=sigma/5.0
        #change sigma with cov matrix  
            if i>1000 and i%500==0:
                sigma=nu.cov(nu.array(out_param)[i-1000:i,:].T)
            pypar.isend(sigma,status.Get_source(),tag=sigma_tag)
            #print 'on %i out of %i' %(i,itter)
            if i<itter:
                pypar.isend(True,status.Get_source(),tag=end_tag)
            else:
                keep_iter=False
                for i in range(1,proc):
                    pypar.send(sigma,i,tag=sigma_tag)
                    pypar.send(keep_iter,i,tag=end_tag)
                out_param,out_chi=nu.array(out_param),nu.array(out_chi)
                chibest=out_chi.min()
                parambest=out_param[chibest==out_chi,:][0]
                pypar.Barrier()
                print 'ending'
                #pypar.finalize()
                return out_chi,out_param,parambest,chibest


if __name__=='__main__':	
 #import thuso_quick_fits as T
 #import asciidata
    import numpy as np
    import pylab as pl
    import time
 #Enter the time and mag (note it should be in one combined file with 2 columns)
 #targets=asciidata.open('combinednoave.dat')
    temp=nu.loadtxt('combinednoave.dat')
    x=temp[:,0]
    y=temp[:,1]

 #x,y=nu.array(x),nu.array(y) 
    func=lambda x,p:p[0]*nu.sin(2*nu.pi*x/0.065714+p[1])+p[2]*nu.sin(2*nu.pi*x/p[3]+p[4])	#The vector p gives the fit parameters - change this to any form that you need.
    param=[1.,0.,1.,0.035,0.]	#Original guess - can be way off - must equal the number of unknowns in the above line

 #Now set the limits (this example is for three parameters, lower and upper limits [0,inf]
    const=nu.zeros([5,2])
    const[2,0]=-10
    const[:,1]=const[:,1]+4*nu.pi	#Set the upper limit as infinity and the lower as 0
 #Run the program
    t=time.time()
    Chi,Param,outparam,outchi=quick_cov_MCMC(x,y,param,func,const,itter=3*10**7,sigma=0.02)#,quiet=True)	#When 'quiet' is false, it displays all the guesses
    print time.time()-t
    Chi,Param=np.array(Chi),np.array(Param)
 #####Plot to check fit:
    print 'your best fit parameters are: ',outparam
    print 'your best fit chi squared value is: ', outchi
 #Plot the seperate nights data below one another:
    xplot=[x[0]-np.floor(x[0])];ysineplot=[func(x[0],outparam)];yplot=[y[0]]#;prewhitened=[y[0]-func(x[0],outparam)]	#Plotting vectors
    move = 0	#How much the plot must be moved down (0 for first night, 'move' for next ... )
    for i in range(1,len(x)):	#Split up the plots
        if int(x[i])-int(x[i-1])<1:
            xplot.append(x[i]-np.floor(x[i]))	#Modded so that the x-axis starts at 0
            yplot.append(y[i]+move)
            ysineplot.append(func(x[i],outparam)+move)
    #prewhitened.append(y[i]-func(x[i],outparam))
        else:
            pl.scatter(xplot,yplot,s=3)
            pl.plot(xplot,ysineplot)
            move = move + 5	#Move future plots down by an amount 'move'
            xplot=[x[i]-np.floor(x[i])];ysineplot=[func(x[i],outparam)+move];yplot=[y[i]+move]	
    #Restart the lists
    #prewhitened.append(y[i]-func(x[i],outparam))
 #Comment out if you don't want the prewhitened light curve printed:   
 #for q in range(0,len(x)):
 #  print str(x[q])+' '+str(prewhitened[q])
 #Plot
    pl.scatter(xplot,yplot,s=5)
    pl.plot(xplot,ysineplot)
    yl,yu = pl.ylim()
    pl.ylim(yu,yl)
    pl.show()


