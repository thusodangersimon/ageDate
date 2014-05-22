#!/usr/bin/env python
#
# Name:  Database utilities
#
# Author: Thuso S Simon
#
# Date: 11th of Nov, 2013
#TODO:  
#    
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
""" Utilites used for makeing and accessing databases"""

import numpy as nu
import tables as tab
import os
from interp_utils import n_dim_interp
from sklearn.neighbors import NearestNeighbors as NN
import sqlite3
import io

'''Pytable utils for searching and adding spectra with params to a .h5 file'''

class CV_lib(tab.IsDescription):
    '''Storage param and spectra for white dwarf atomspheres'''
    Temp = tab.Float64Col()
    logg = tab.Float64Col()
    H = tab.Float64Col()
    Li = tab.Float64Col()
    Be = tab.Float64Col()
    B = tab.Float64Col()
    C = tab.Float64Col()
    N= tab.Float64Col()
    O= tab.Float64Col()
    F= tab.Float64Col()
    Ne= tab.Float64Col()
    Na= tab.Float64Col()
    Mg= tab.Float64Col()
    Al= tab.Float64Col()
    Si= tab.Float64Col()
    P= tab.Float64Col()
    S= tab.Float64Col()
    Cl= tab.Float64Col()
    Ar= tab.Float64Col()
    K= tab.Float64Col()
    Ca= tab.Float64Col()
    Sc= tab.Float64Col()
    Ti= tab.Float64Col()
    V= tab.Float64Col()
    Cr= tab.Float64Col()
    Mn= tab.Float64Col()
    Fe= tab.Float64Col()
    Co= tab.Float64Col()
    Ni= tab.Float64Col()
    Cu= tab.Float64Col()
    Ni= tab.Float64Col()
    Zn= tab.Float64Col()
    He= tab.Float64Col()
    spec = tab.FloatCol(shape=(5001,2))
    tried = tab.BoolCol()



def creat_lib(param,spec,abn_list,lib_path):
    '''(ndarray, list or ndarray,spectra,col_label,match labes, lib_path or tables object)-> None
    Creates or appends abn_list param to a new or open tables file
    abn_list is names of parameters but MUST be in order of param where temp and logg are first then elements
    '''
    #check if files exsists and is open
    if type(lib_path) is str:
        if not os.path.exists(lib_path):
            #create new database, and all beggining information
            lib = tab.open_file(lib_path, mode = "w")
            filters = tab.Filters(1,'lzo')
            #group = lib.create_group('/','param','CV params with spectra')
            lib.create_table(lib.root, 'param',CV_lib,
                             expectedrows=9000,byteorder='little',filters=filters)
            #set dflt values
            attr = lib.root.param.set_attr
            g_attr = lib.root.param.attrs
            for i in range(len(lib.root.param.coltypes)):
                if g_attr['FIELD_%i_NAME'%i] == 'spec':
                    continue
                if g_attr['FIELD_%i_NAME'%i]  == 'tried':
                    attr('FIELD_%i_FILL'%i,True)
                    continue
                attr('FIELD_%i_FILL'%i ,-9999999.0)
            
            #make temp node
            lib.flush()
            lib.close()
            lib = tab.open_file(lib_path, mode = 'a')
        else:
            #open in append mode
            lib = tab.open_file(lib_path,mode='a')
    else:
        #send a open table and will add to it
        assert type(lib_path) is tab.file.File
        lib = lib_path

    #add params and spectra to hdf5 lib
    row = lib.root.param.row
    assert len(param) == len(abn_list)+2
    for i in xrange(len(param)):
        if i == 0:
            row['Temp'] = param[i]
        elif i == 1:
            row['logg'] = param[i]
        else:
            #elements
            row[abn_list[i-2]] = param[i]
    row['spec'] = spec
    row.append()
    lib.flush()
    if type(lib_path) is str:
        return lib

def put_in_lib(tab, param, abn_list, spec, lock):
    '''Puts parametrs into hdf5 lib. Checks if spectrum is there if not puts
    holder in place.
    wait's till lock is true before writes table'''
    lock.acquire()
    row = tab.row
    for i in xrange(len(param)):
        if i == 0:
            row['Temp'] = param[i]
        elif i == 1:
            row['logg'] = param[i]
        else:
            #elements
            row[abn_list[i-2]] = param[i]
    row['spec'] = spec
    row.append()
    tab.flush()
    lock.release()

def skiki_NN(hdf5,col,param):
    '''Looks for with skit learn function the len(col)*2 nearest neighbors in an hdf5 database'''
    d = np.empty((rows*batches,))
    for i in range(batches):
        nbrs = NN(n_neighbors=len(col)*2, algorithm='ball_tree').fit(h5f.root.carray[i*rows:(i+1)*rows])
        distances, indices = nbrs.kneighbors(vec)  # put in dict?
        
def linear_NN(hdf5,col,param):
    '''same as kikik_NN but linear search'''
    d = np.empty((rows*batches,))
    for i in range(batches):
        d[i*rows:(i+1)*rows] = ((h5f.root.carray[i*rows:(i+1)*rows]-vec)**2).sum(axis=1)
 
def get_param_from_hdf5(hdf5,param,cols,all_param):
    '''searches hdf5 lib for rows of intrest. uses a binary like search'''
    go_on = False
    query = hdf5.read_where('(%s == %f) & (%s == %f)'%(cols[0],param[0],cols[1],param[1]))
    if len(query) > 0:
        #found something check other params
        for i in query:
            for j,col in enumerate(cols):
                if not i[col] == param[j]:
                    go_on = True
                    break
            else:
                #match!
                return i['spec']
    #not match try interp
    if len(query) == 0 or go_on:
        #found nothing get nearest neightbors
        Nei_clas = NN(len(param)*2).fit(all_param)
        index = nu.ravel(Nei_clas.kneighbors(param)[1])
        #get spec
        spec = []
        for i in index:
            spec.append(hdf5.cols.spec[i][:,1])
        spec = nu.asarray(spec)
        #get spec and interp
        print 'interpolating'
        return nu.vstack((hdf5.cols.spec[i][:,0],
                           n_dim_interp(all_param[index],param,spec))).T
    return []
    
    
def pik_hdf5(pik_path, out_hdf5_path):
    '''Turns a pickle file into an hdf5 database'''
    
    
if __name__ == '__main__':
    '''makes database from likelihood.CV_Fit'''
    from mpi4py import MPI as mpi
    from likelihood_class import CV_Fit
    comm = mpi.COMM_WORLD
    rank = comm.rank
    size = comm.size
    #load spectra maker
    
    bins = fun.models.keys()[0]
    #make persistant comunicators for exit status
    if rank == 0: pass
        #create data base
        #lib = tab.open_file('CV_lib.h5', 'w')
        #table = lib.create_table(lib.root, 'Param',CV_lib,"Param and spec")
    #create enough points to take a computer month to calculate
    #time per iteration
    t = 160
    t_month = 31*24*3600.
    itter = round(t_month/t)
   
    #itter = 50
    #params to use
    abn = fun._abn_lst
    out = []
    #draw points from uniform distribution
    grid = nu.random.rand(itter,len(abn)+2)
    #scale grid
    #Temp
    grid[:,0] =grid[:,0] * 20000 + 20000
    #logg
    grid[:,1] = grid[:,1] * 4 + 4
    #metals
    grid[:,2:] = grid[:,2:] *2 -1
    #round to the tenth
    grid = nu.around(grid,1)
    #look for duplicates
    grid = nu.unique(grid.view(nu.dtype((nu.void, grid.dtype.itemsize*grid.shape[1])))).view(grid.dtype).reshape(-1, grid.shape[1])
    #make spectra over cluster
    out = list(fut.map(convert,grid,**{'bins':'1','return_spect':True}))
    if rank == 0:
        pik.dump((out,grid),open('pre_hdf5.pik','w'),2)


####sql databases
def numpy_sql(path):
    '''Allows for storage of numpy arrays in sql databases. Can be used
    for opening and creating db'''
     #Converts np.array to TEXT when inserting
    sqlite3.register_adapter(nu.ndarray, adapt_array)

    # Converts TEXT to np.array when selecting
    sqlite3.register_converter("array", convert_array)
    con = sqlite3.connect(path)
    return con



def adapt_array(arr):
    out = io.BytesIO()
    nu.save(out, arr)
    out.seek(0)
    # http://stackoverflow.com/a/3425465/190597 (R. Hill)
    return buffer(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return nu.load(out)
