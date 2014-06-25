#!/usr/bin/env python
#
# Name:  boundary value
#
# Author: Thuso S Simon
#
# Date: May 9th 2012
#TODO: 
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
''' tells from a set of points if another point lies with in them'''

import numpy as nu

def find_boundary(points):
    ''' takes in a list of points and returns the vertex of each side of the 
    polygon that it makes
    '''
    #find first point on boundary
    points = points[nu.argsort(points[:,0])]
    index = []
    #make sure starting point is a vertex of hull
    for i in [points[:,0].min(), points[:,0].max()]:
        for j in [points[:,1].min(), points[:,1].max()]:
            #if the points exsist in points start there
            if nu.any(nu.all(points == [i,j],axis=1)):
                index, = nu.nonzero(nu.all(points == [i,j],axis=1))
                new_index = [index[0]]
                #put vertex point at start of array
                points = points[nu.hstack((index[0],
                                           range(index),
                                           range(index+1,len(points))))]
                break
        if index:
            break
    vertex = points[0] + 0
    hull = [vertex + 0]
    rightmost = points[1] + 0 
    #find  border
    while  True:
        for i in points:
            #sees if point is on the right side of righmost
            if ((nu.all(i == vertex) and 
                nu.all(i == rightmost)) and 
                is_right(vertex, rightmost, i)) <= 0:
                #if on line make sure further away than other point
                if ((is_right(vertex, rightmost, i)) == 0 and
                ((i- vertex)**2).sum() > ((rightmost- vertex)**2).sum()):
                    rightmost = i + 0
                elif is_right(vertex, rightmost, i) < 0:
                    rightmost = i + 0
                
        vertex = rightmost + 0
        hull.append(vertex +0)
        if nu.all(hull[0] == vertex):
            break
    if len(hull) < 3:
        raise ValueError('Convex hull failed, try again')
    return nu.array(hull)
    
def is_right(P0, P1, P2):
    #tests if a point is Left|On|Right of an infinite line.
    #Input:  three points P0, P1, and P2 
    #Return: >0 for P2 left of the line through P0 and P1
    #=0 for P2 on the line
    #<0 for P2 right of the line
    #See: the January 2001 Algorithm on Area of Triangles
    return (P1[0] - P0[0])*(P2[1] - P0[1]) - (P2[0] - P0[0])*(P1[1] - P0[1]);
    
def point_in_polygon(point,poly):
    #tells if point is in polygon
    n = len(poly)
    inside = False
    x,y = point
    p1x,p1y = poly[0]
    for i in xrange(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = nu.copy([p2x,p2y])

    return inside

def pinp_wbounds(point,poly):
    #tells if point is in polygon and hadels if its on a boundary
    
    p1x,p1y = poly[0]
    #bound = 'on Boundary'
    n = len(poly)
    x,y = point
    count = False
    #//check all rays
    for i in range(n):
        if nu.all(point == [p1x,p1y]):
            #if on vertex
            #return bound
            return True
        p2x,p2y = poly[i % n]

        #//ray is outside of our interests
        if y < min(p1y, p2y) or y > max(p1y, p2y):
            #//next ray left point
            p1x = p2x
            p1y = p2y
            continue
        #//ray is crossing over by the algorithm (common part of)
        if y > min(p1y, p2y) and y < max(p1y, p2y):
            #//x is before of ray
            if x <= max(p1x, p2x):
                #//overlies on a horizontal ray
                if p1y == p2y and x >= min(p1x, p2x):
                    #return bound
                    return True
                #//ray is vertical
                if p1x == p2x:
                    #//overlies on a ray
                    if p1x == x:
                        #return bound
                        return True
                    #//before ray
                    else:
                        count = not count

                #//cross point on the left side
                else:
                    #//cross point of x
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x

                    #//overlies on a ray
                    if abs(x - xinters) < 10**-6: #abritray small number
                        #return bound
                        return True
                    #//before ray
                    if x < xinters:
                        count = not count
        #//special case when ray is crossing through the vertex
        else:
        
            #//p crossing over p2
            if y == p2y and x <= p2x:
                #//next vertex
                p3x,p3y = poly[(i+1) % n]
                
                #//p.y lies between p1.y & p3.y
                if y >= min(p1y, p3y) and y <= max(p1y, p3y):
                    count = not count
        #//next ray left point
        p1x,p1y = p2x,p2y
    #%//EVEN
    return count

def test(points):
    #tests if all possible points in a square are in points and plots
    import pylab as lab
    #makes points for square
    x_unq = nu.unique(points[:,0])
    y_unq = nu.unique(points[:,1])
    #make accept and reject lists
    accept, reject =[], []
    hull = find_boundary(points)
    #sort which points are in boundary
    for i in x_unq:
        for j in y_unq:
            if pinp_wbounds(nu.array([i,j]), hull):
                accept.append(nu.array([i,j]))
            else:
                reject.append(nu.array([i,j]))

    #plot them
    accept = nu.array(accept)
    fig = lab.figure()
    plt = fig.add_subplot(111)
    plt.plot(accept[:,0],accept[:,1],'.',markersize=10,label='in bounds')
    if reject:
        reject = nu.array(reject)
        plt.plot(reject[:,0],reject[:,1],'+' ,
                 markersize=10,label='out of bounds')
    plt.legend()
    lab.show()
