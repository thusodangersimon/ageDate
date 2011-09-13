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
from Age_MCMC import * ##temp import##

def make_chi_grid(data,points=500):
    #makes a 3d pic of metal,age,chi with input spectra 2-D only
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0])
    #create grid
    metal,age=nu.meshgrid(nu.linspace(nu.log10(lib_vals[0][:,0].min()),
                                      nu.log10(lib_vals[0][:,0].max()),points),
                          nu.linspace(lib_vals[0][:,1].min(),
                                      lib_vals[0][:,1].max(),points))
    #age,N=nu.meshgrid(nu.linspace(lib_vals[0][:,1].min(),
    #                                  lib_vals[0][:,1].max(),points),
    #              nu.linspace(0,700,points))   
    N=nu.array([[1,1],[1,1]])
    out=age*0
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])
    #start making calculations for chi squared value
    po,work=Pool(),[]
    for i in range(points):
         #print '%i out of %i' %(i+1,points)
         #sys.stdout.flush()
         work.append(po.apply_async(poly_grid,(metal,age,N,lib_vals,age_unq,metal_unq,1,points,data,out,i,)))
    po.close()
    po.join()
    for i in work:
        out=out+i.get()
    return metal,age,N,out

def poly_grid(metal,age,N,lib_vals,age_unq,metal_unq,bins,points,data,out,i):
    for j in range(points):     
        model=get_model_fit([metal[i,j],age[i,j],1],lib_vals,age_unq,metal_unq,bins)
        model=data_match(model,data)
        out[i,j]=sum((data[:,1] - model)**2)

    print '%i out of %i' %(i+1,points)
    sys.stdout.flush()
    return out

def make_marg_chi_grid(data,points,bins,axis=0):
    #makes a marginalized chi grid for the axis specified just for bins=2 right now
    lib_vals=get_fitting_info(lib_path)
    metal_unq=nu.unique(lib_vals[0][:,0])
    age_unq=nu.unique(lib_vals[0][:,1])
    bin=nu.linspace(age_unq.min(),age_unq.max(),bins+1)
    
    poss_age={} #all possible vals for age and metal with point fineness
    for i in range(bins):
        poss_age[str(i)]=nu.linspace(bin[i],bin[i+1],num=points)
    poss_metal=nu.logspace(nu.log10(metal_unq[0]),nu.log10(metal_unq.max()),num=points)

    #multiprocess splitting of work
    splits=nu.int32(nu.linspace(0,points,num=cpu_count()+1))
    work=[]
    itter=Value('i', 0)
    q=Queue()
    for i in range(cpu_count()):
        work.append(Process(target=make_grid_multi,args=(data,poss_metal,
                                                         poss_age['0'],
                                                         poss_age['1'],
                                                         range(splits[i],splits[i+1]),itter,q)))
        work[-1].start()
    count=1
    temp=[]
    while count<=cpu_count():
        print '%i out of %i iterations done' %(itter.value,points**4)
        if q.qsize()>0:
            temp.append(q.get())
            count+=1
        else:
            Time.sleep(3)
    #collect data
    for i,j in enumerate(temp):
        if i==0:
            out,index=j
        else:
            out=j[0]+out
    return poss_metal,poss_age['0'],out

def make_grid_multi(data,metal,age,other_age,iss,itter,q):
    #makes chi grid for multiprocess
    out=nu.zeros([len(metal),len(other_age)])
    lib_vals=get_fitting_info(lib_path)
    metal_unq=nu.unique(lib_vals[0][:,0])
    age_unq=nu.unique(lib_vals[0][:,1])
    for i in iss:#x axis plot age
        for j in xrange(len(metal)):#plot metal
            for k in xrange(len(other_age)): #marginal age
                for l in xrange(len(metal)): #marginal metal
                    param=nu.array([metal[j],age[i],metal[k],other_age[l],1,1])
                    model=get_model_fit(param,lib_vals,age_unq,metal_unq,2)
                    model=data_match(model,data)
                    out[i,j]=out[i,j]+sum((
                            data[:,1]-normalize(data,model)*model))**2
                    itter.value=itter.value+1

    q.put((out,iss))

def plot_model(param,bins):
    #takes parameters and returns spectra associated with it
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0])
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])
    for i in range(0,bins,3):
        param[i]=nu.log10(param[i])
    return get_model_fit(param,lib_vals,age_unq,metal_unq,bins)
