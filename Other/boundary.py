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
import matplotlib.animation as animation
import pylab as lab



def find_boundary(points):
    ''' takes in a list of points and returns the vertex of each side of the 
    polygon that it makes
    Jarvis Convex Hull algorithm.
    points is a list of CGAL.Point_2 points
    '''
    #find first point on boundary
    unq_x = nu.unique(points[:,0])
    unq_y = nu.unique(points[:,1])
    hull = []
    for i in [unq_x.min(),unq_x.max()]:
        for j in unq_y:
            if nu.any(nu.logical_and(points[:,0] == i, 
                                     points[:,1] == j)):
                hull.append([i,j])
                break
        if hull:
            break
    r, u = r0, r0*nu.inf
    #put all other points in array
    remainingPoints = []
    for i in points:
        if not nu.all(i == hull):
            remainingPoints.append(i)
    remainingPoints = nu.array(remainingPoints)
    #find  border
    while  nu.all(remainingPoints):
        u = remainingPoints[nu.random.randint(len(remainingPoints))]
        for i in points:
            if nu.all(i != u) and (not is_clockwise(r, u, i) or 
                           is_collinear(r, u, i)):
                u = i
        r = u
        #points.remove(r)
        hull.append(r)
        remainingPoints = remainingPoints[nu.all(remainingPoints == 
                                                 r, axis=1) == False]
    return nu.array(hull)
 
def is_clockwise(first, second=None, third=None):
    '''takes 3 points or a list of points
    if their are 3 points it will see if 
    third point is clockwise from last 2.
    If their is just a list of points will 
    test to see if increasing clockwise'''
    if not nu.all(second):
        vertex = first
    else:
        vertex = nu.vstack((first, second, third))
    if vertex.shape[0] < 3:
      raise TypeError('Did no input in the right shape')
    #ported from c code. Writen by G. Adam Stanislav.
    count = 0
    for i in range(vertex.shape[0]) :
       j = (i + 1) % vertex.shape[0]
       k = (i + 2) % vertex.shape[0]
       z  = (vertex[j,0] - vertex[i,0]) * (vertex[k,1] - vertex[j,1])
       z -= (vertex[j,1] - vertex[i,1]) * (vertex[k,0] - vertex[j,0])
       if (z < 0):
           count -= 1
       elif(z > 0):
           count += 1
    if count > 0:
        return True
    else:
        return False

def is_collinear(first, second, third):
    '''tells if points lie on a line'''
    test_points = nu.vstack((first, second, third))
    m_b = nu.polyfit(test_points[:-1,0], test_points[:-1,1],1)
    if nu.polyval(m_b, third[0]) == third[1]:
        return True
    else:
        return False
      
    #y-axis vertex finding
def in_boundary(vertex, point):
    '''tells if a point is in a polygon takes input from find_boundary'''
    assert len(point) == vertex.shape[1]
    #take dot product to see if right (positive) or left (negitive)
    #of lines
    for i in xrange(vertex.shape[0]):
        #for last point in array wrap arround to firts point
        if vertex.shape[0] - 1 == i:
            bound_vec = vertex[0] - vertex[i]
        else:
            bound_vec = vertex[i + 1] - vertex[i]
        point_vec = point - vertex[i]
        if nu.dot(point_vec, bound_vec) < 0:
            return False

    return True


def disp(points=None):
    #shows the javis program in action
    import Age_date as ag
    points=ag.get_fitting_info('/home/thuso/Phd/Spectra_lib')[0]
 
    fig = lab.figure()
    plt = fig.add_subplot(111)
    plt.plot(points[:,0], points[:,1],'b.')#,animated=True)
    fig.canvas.draw()
    unq_x = nu.unique(points[:,0])
    unq_y = nu.unique(points[:,1])
    hull = []
    for i in [unq_x.min(),unq_x.max()]:
        for j in unq_y:
            if nu.any(nu.logical_and(points[:,0] == i, 
                                     points[:,1] == j)):
                hull.append([i,j])
                break
        if hull:
            break
    r0 = nu.array(hull[0])
    r, u = r0, r0*nu.inf
    #put all other points in array
    remainingPoints = []
    for i in points:
        if not nu.all(i == hull):
            remainingPoints.append(i)
    remainingPoints = nu.array(remainingPoints)
    #find  border
    kk = 0
    while  nu.all(remainingPoints): #and kk < 1:
        u = remainingPoints[nu.random.randint(len(remainingPoints))]
        plt.plot([r0[0],u[0]], [r0[1],u[1]],'g')
        fig.canvas.draw()
        for i in points:
            plt.plot([u[0],i[0]], [u[1],i[1]],'r')
            fig.canvas.draw()
            if nu.all(i != u) and (not is_clockwise(r, u, i) or 
                           is_collinear(r, u, i)):
                #change color of line
                del plt.lines[-1]
                plt.lines[-1].set_markerfacecolor('b')
                fig.canvas.draw()
                u = i
                r = u
                hull.append(r)
                remainingPoints = remainingPoints[nu.all(remainingPoints == 
                                                         r, axis=1) == False]
            else:
                del plt.lines[-1]
                fig.canvas.draw()
        del plt.lines[-1]
        fig.canvas.draw()
        kk += 1
    #return nu.array(hull)
 
def convex_hull(points):
    """Computes the convex hull of a set of 2D points.
 
    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """
 
    # Sort the points lexicographically (numpy array
    #are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = points[nu.argsort(points[:,0])]
  
    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
 
    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
 
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
 
    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    return lower[:-1] + upper[:-1]
 
 
# Example: convex hull of a 10-by-10 grid.
#assert convex_hull([(i/10, i%10) for i in range(100)]) == [(0, 0), (9, 0), (9, 9), (0, 9)]



if __name__== '__main__':
    
   #lab.ion()
    import Age_date as ag
    points=ag.get_fitting_info('/home/thuso/Phd/Spectra_lib')[0]
 
    fig = lab.figure()
    plt = fig.add_subplot(111)
    plt.plot(points[:,0], points[:,1],'b.',animated=True)
    #fig.canvas.draw()
    unq_x = nu.unique(points[:,0])
    unq_y = nu.unique(points[:,1])
    hull = []
    for i in [unq_x.min(),unq_x.max()]:
        for j in unq_y:
            if nu.any(nu.logical_and(points[:,0] == i, 
                                     points[:,1] == j)):
                hull.append([i,j])
                break
        if hull:
            break
    r0 = nu.array(hull[0])
    r, u = r0, r0*nu.inf
    #put all other points in array
    remainingPoints = []
    for i in points:
        if not nu.all(i == hull):
            remainingPoints.append(i)
    remainingPoints = nu.array(remainingPoints)
    #find  border
    kk = 0
    #while  nu.all(remainingPoints) and kk < 1:
    u = remainingPoints[nu.random.randint(len(remainingPoints))]
    plt.plot([r0[0],u[0]], [r0[1],u[1]],'g')
    
    def ah(i):
        global u,r
        #for i in points[:10]:
        plt.plot([u[0],i[0]], [u[1],i[1]],'r')
        #fig.canvas.draw()
        if nu.all(i != u) and (not is_clockwise(r, u, i) or 
                               is_collinear(r, u, i)):
                #change color of line
            del plt.lines[-1]
            plt.lines[-1].set_markerfacecolor('b')
            #fig.canvas.draw()
            u = i
            r = u
            hull.append(r)
            remainingPoints = remainingPoints[nu.all(remainingPoints == 
                                                         r, axis=1) == False]
        else:
            del plt.lines[-1]
            #fig.canvas.draw()
        try:
            del plt.lines[-1]
        except IndexError:
            pass
        #fig.canvas.draw()
        #kk += 1

    ani = animation.FuncAnimation(fig, ah, points, interval=25, blit=True)
