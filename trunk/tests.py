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
import tables as tab
import numpy as nu

#create an hdf5 database
#['Temp','g','H','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Ni','Zn','point']
#class defining col
class CV_lib(tab.IsDescription):
    Temp = tab.Float32Col(pos=1)
    g= tab.Float32Col(pos=2)
    H= tab.Float32Col(pos=3)
    Li= tab.Float32Col(pos=4)
    Be= tab.Float32Col(pos=5)
    B= tab.Float32Col(pos=6)
    C= tab.Float32Col(pos=7)
    N= tab.Float32Col(pos=8)
    O= tab.Float32Col(pos=9)
    F= tab.Float32Col(pos=10)
    Ne= tab.Float32Col(pos=11)
    Na= tab.Float32Col(pos=12)
    Mg= tab.Float32Col(pos=13)
    Al= tab.Float32Col(pos=14)
    Si= tab.Float32Col(pos=15)
    P= tab.Float32Col(pos=16)
    S= tab.Float32Col(pos=17)
    Cl= tab.Float32Col(pos=18)
    Ar= tab.Float32Col(pos=19)
    K= tab.Float32Col(pos=20)
    Ca= tab.Float32Col(pos=21)
    Sc= tab.Float32Col(pos=22)
    Ti= tab.Float32Col(pos=23)
    V= tab.Float32Col(pos=24)
    Cr= tab.Float32Col(pos=25)
    Mn= tab.Float32Col(pos=26)
    Fe= tab.Float32Col(pos=27)
    Co= tab.Float32Col(pos=28)
    Ni= tab.Float32Col(pos=29)
    Cu= tab.Float32Col(pos=30)
    Ni= tab.Float32Col(pos=31)
    Zn= tab.Float32Col(pos=32)
    spec = tab.FloatCol(pos=33,shape=(5001,2))
    


if __name__ == __'main__':
    #create data base
    lib = tab.open_file('CV_lib.h5', 'w')
    table = lib.create_table(lib.root, 'Param',CV_lib,"Param and spec")
    #create enough points to take a computer month to calculate
    #time per iteration
    t = 220.
    t_month = 31*24*3600.
    itter = round(t_month/t)
    #params to use
    abn = ['H','He','C','N','O','Si','P','S']
    nbins = round(nu.exp((len(abn)+2)/nu.log(itter)))
    out = []
    lin = nu.linspace
    for T in lin(20000,40000,nbins):
        for g in lin(4,8,nbins):
            for H in lin(-1,1,nbins):
                for He in lin(-1,1,nbins):
                    for C in lin(-1,1,nbins):
                        for N in lin(-1,1,nbins):
                            for O in lin(-1,1,nbins):
                                for Si in lin(-1,1,nbins):
                                    for P in lin(-1,1,nbins):
                                        for S in lin(-1,1,nbins):
                                            out.append([T,g,H,He,C,N,O,Si,P,S])
