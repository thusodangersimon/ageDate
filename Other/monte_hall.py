#!/usr/bin/env python
#
# Name:  Monte Hall problem
#
# Author: Thuso S Simon
#
# Date: 25 April 2012
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
"""does monte carlo against 2 situations (always change, always stay) to see if probablilities are correct"""

import numpy as nu
import pylab as lab

class host(object):
    def __init__(self):
        doors=nu.zeros(3)
        i=nu.random.randint(3)
        doors[i]+=1
        self._true_door=doors
        self._out_door=nu.array(['closed']*3)

    def cont_choose(self,choise):
        'index of door to be choose'
        if nu.sum(self._out_door=='closed')==3: #first choise
            if choise>3 or choise<1: #out of range
                print 'Must choose door 1,2, or 3'
                return False
            #open door that is not equl to choise and not correct
            true_ans=self._true_door.nonzero()[0]+1
            if true_ans==choise: #can open other 2 doors
                i=nu.arange(1,4)[nu.arange(1,4)!=choise][nu.random.randint(2)]-1
                self._out_door[i]='nothing'
                i=nu.nonzero(nu.logical_and(nu.arange(1,4)!=i+1,nu.arange(1,4)!=choise))[0]
                print self._out_door
                print 'type %i to stay or change to %i' %(choise,i+1)
                self._left_choise=nu.array([choise,i+1])
                return self._left_choise
            else: #only 1 door to open
                i=nu.nonzero(nu.logical_and(nu.arange(1,4)!=true_ans,nu.arange(1,4)!=choise))[0]
                self._out_door[i]='nothing'
                i=nu.nonzero(nu.logical_and(nu.arange(1,4)!=i+1,nu.arange(1,4)!=choise))[0]
                print self._out_door
                print 'type %i to stay or change to %i' %(choise,i+1)
                self._left_choise=nu.array([choise,i+1])
                return self._left_choise
        else: #second choise for true answer
            if not nu.any(choise==self._left_choise):
                print 'Wrong choise, choose either %i or %i' %(self._left_choise[0],
                                                               self._left_choise[1])
                return False
            true_ans=self._true_door.nonzero()[0]+1
            if choise==true_ans:
                print 'Congradulations you chose correctly'
                return True
            else:
                print 'Sorry you did not choose correctly'
                return False

class contestant(object):
    def __init__(self,mode):
        'has to modes, change always (ca) or stay'
        if mode=='ca' or mode=='change always':
            self.mode=change
        else:
            self.mode=stay

        self._first_choice=nu.random.randint(3)+1


    def ca(self,avalible_values):
        return avalible_values[avalible_values!=self._first_choice]

    def stay(self,avalible_values):
        return self._first_choice


if __name__=='__main__':

    prob_stay,prob_change=[],[]
    mode='stay'
    for i in range(10**2):
        h=host()
        c=contestant(mode)
        

    mode='ca'
    for i in range(10**2):
        h=host()
        c=contestant(mode)
