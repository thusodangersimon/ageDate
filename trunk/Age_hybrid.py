#!/usr/bin/env python
#
# Name:  Hybrid nnls and MC
#
# Author: Thuso S Simon
#
# Date: Oct. 20 2011
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
''' Fits spectra with a non-negitive least squares fit and finds uncertaties 
in a multitude of ways, using grid,MCMC and population fitting'''

from Age_date import *

def nn_ls_fit(data,max_bins=16,min_norm=10**-4,spect=spect):
    #uses non-negitive least squares to fit data
    #spect is libaray array
    #match wavelength of spectra to data change in to appropeate format
    model={}
    for i in xrange(spect[0,:].shape[0]):
        if i==0:
            model['wave']=nu.copy(spect[:,i])
        else:
            model[str(i-1)]=nu.copy(spect[:,i])

    model=data_match_new(data,model,spect[0,:].shape[0]-1)
    index=nu.int64(model.keys())
    
    #nnls fit
    N,chi=nnls(nu.array(model.values()).T,data[:,1])
    N=N[index.argsort()]
    
    #check if above max number of binns
    if len(N[N>min_norm])>max_bins:
        #remove the lowest normilization
        print 'removing bins is not ready yet'
        raise
    current=info[N>min_norm]
    metal,age=[],[]
    for i in current:
        metal.append(float(i[4:10]))
        age.append(float(i[11:-5]))
    metal,age=nu.array(metal),nu.array(age)
    #check if any left
    if len(current)<2:
        return float(current[4:10]),float(current[11:-5]),N[N>min_norm]

    return metal[nu.argsort(age)],age[nu.argsort(age)],N[N>min_norm][nu.argsort(age)]

def info_convert(info_txt):
    #takes info array from Age_date.create_data and turns in floats for plotting
    metal,age=[],[]
    for i in info_txt:
        metal.append(float(i[4:10]))
        age.append(float(i[11:-5]))
    metal,age=nu.array(metal),nu.array(age)
    return metal[nu.argsort(age)],age[nu.argsort(age)]


def grid_fit(data,grid_points=500,spect=spect):
    #does nnls to fit data, then uses a adaptive grid to find uncertanty on params
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])

    #nnls fits
    best_metal,best_age,best_N=nn_ls_fit(data)
    #make iterations ready
    #match lib with data
    data_match_all(data)
    #out lists
    out={}
    #inital best fit
    bins=len(best_age)
    param=make_correct_params(best_metal,best_age,best_N)
    #make error testing lists
    grid=gen_lists(param,age_unq,metal_unq,bins,grid_points)
    norm_range=nu.array([0.,param.max()])
    for i in xrange(len(param)):
        pool=Pool()
        out[str(i)]=[]
        if any(i==nu.arange(0,bins*3,3)): #metal
            work=[]
            w=work.append
            for j in grid[str(i)]:
                w(pool.apply_async( multi_unitfor_grid,(data,param,lib_vals,metal_unq,age_unq,norm_range,bins,j,i)))
            pool.close()
            pool.join()
            for j in work:
                out[str(i)].append(j.get())
                
        elif any(i==nu.arange(1,bins*3,3)): #age
            work=[]
            w=work.append
            for j in grid[str(i)]:
                w(pool.apply_async( multi_unitfor_grid,(data,param,lib_vals,metal_unq,age_unq,norm_range,bins,j,i)))
            pool.close()
            pool.join()
            for j in work:
                out[str(i)].append(j.get())

        elif any(i==nu.arange(2,bins*3,3)): #norm
            work=[]
            w=work.append
            for j in grid[str(i)]:
                w(pool.apply_async( multi_unitfor_grid,(data,param,lib_vals,metal_unq,age_unq,norm_range,bins,j,i)))
            pool.close()
            pool.join()
            for j in work:
                out[str(i)].append(j.get())

    #get ready for output
    for i in out.keys():
        out[i]=nu.array(out[i])
        if any(int(i)==nu.arange(0,bins*3,3)): #take log of metals
            out[i][:,0]=nu.log10(out[i][:,0])

        out[i]=out[i][out[i][:,0].argsort(),:]
    return out,param

def gen_lists(param,age_unq,metal_unq,bins,points):
    #generates lin spaceds lists near values in param with num_points=points
    out={}
    for i in xrange(len(param)):
        if any(i==nu.arange(0,bins*3,3)): #metal
            #check to see of edge of param space
            if param[i]==metal_unq[0]:
                out[str(i)]=nu.linspace(metal_unq[0],metal_unq[1],points)
            elif param[i]==metal_unq[-1]:
                out[str(i)]=nu.linspace(metal_unq[-2],metal_unq[-1],points)
            else:
                out[str(i)]=nu.linspace(param[i]-nu.mean(nu.diff(metal_unq)),param[i]+nu.mean(nu.diff(metal_unq)),points)

        if any(i==nu.arange(1,bins*3,3)): #age
            if param[i]==age_unq[0]:
                out[str(i)]=nu.linspace(age_unq[0],age_unq[2],points)
            elif param[i]==age_unq[-1]:
                out[str(i)]=nu.linspace(age_unq[-3],age_unq[-1],points)
            else:
                out[str(i)]=nu.linspace(param[i]-nu.mean(nu.diff(age_unq))*2,param[i]+nu.mean(nu.diff(age_unq))*2,points)
 
        if any(i==nu.arange(2,bins*3,3)): #norm
            #make sure above zero
            norm=5.
            while True:
                if param[i]-norm<0:
                    norm=norm/1.05
                else:
                    break

            out[str(i)]=nu.linspace(param[i]-norm,param[i]+norm,points)

    return out

def make_correct_params(metal,age,norm):
    #turns seprate lists of age,metal,norm in to the correct format to be used for fitting
    out=nu.zeros(len(age)*3)
    index=0
    for i in range(0,len(age)*3,3):
        out[i:i+3]=[nu.log10(metal[index]),age[index],norm[index]]
        index+=1

    return out

def gen_new_param_uniform(index,metal_unq,age_unq,norm_range):
    #generates uniform parameters for testing   
    points=nu.zeros(index.shape)
    for k in index:
        if any(nu.array(range(0,index.shape[0],3))==k):#metalicity
            points[k]=(nu.random.random()*metal_unq.ptp()+metal_unq[0])
        else:#age and normilization
            if any(nu.array(range(1,index.shape[0],3))==k): #age
                points[k]=nu.random.rand()*age_unq.ptp()+age_unq[0]
            else: #norm stuff
                    points[k]=nu.random.random()*norm_range.ptp()+norm_range.min()
 
    return points

def multi_unitfor_grid(data,param,lib_vals,metal_unq,age_unq,norm_range,bins,j,i):
    #for multiprocessing
    #time=[]
    out=nu.array([j,0.])
    nu.random.seed(current_process().pid*nu.random.randint(1,999999999))
    index=xrange(2,bins*3,3)
    for k in xrange(200):
        #gen new vectors
        #t=Time.time()
        new_param=gen_new_param_uniform(nu.arange(len(param)),metal_unq,age_unq,norm_range)
        new_param[i]=nu.copy(j) #make sure correct place
        #calc chi
        model=get_model_fit_opt(new_param,lib_vals,age_unq,metal_unq,bins)
        #fastest way
        model['wave']= model['wave']*.0
        for ii in model.keys():
            if ii!='wave':
                model['wave']+=model[ii]*new_param[index[int(ii)]]
        
        out[1]+=nu.sum((data[:,1]-model['wave'])**2)
        #time.append(Time.time()-t)
    #print nu.mean(time)
    return out
