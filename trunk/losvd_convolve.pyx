#cython: profile=False
#cython: boundscheck=False
#cython: wraparound=False
# filename: losvd_convolve.pyx

import numpy as nu
cimport numpy as nu
from time import time
cimport cython

#gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/share/apps/include/python2.7 -o downgrade_res.so downgrade_res.c

def convolve( nu.ndarray[nu.float64_t, ndim=1] x,nu.ndarray[nu.float64_t, ndim=1] y, nu.ndarray[nu.float64_t, ndim=1] losvd_param):
   '''convolve(array, kernel)
	 does convoluton useing input kernel '''
   #alocate parameters
   cdef nu.ndarray[ nu.float64_t, ndim=1] u, g, ys, k
   cdef int i, i1, i2, m1, m2, Len_kernel, Len_data
   cdef float diff_wave
   #start covloution
   ys = nu.zeros_like(y)
   diff_wave = nu.mean(nu.diff(x))
   Len_data = len(x)
   for  i in xrange(Len_data):   
      kernel = gauss(diff_wave, x[i], losvd_param[0], losvd_param[2], losvd_param[3])
      Len = len(kernel)  
      m2 = i + (Len - 1)/2 + 1
      m1 = i - (Len - 1)/2 
      if m1 < 0:
         m1 = 0
         i1 = (Len - 1)/2 - i
         i2 = Len 
         k = kernel[i1:i2] / kernel[i1:i2].sum()
      elif m2 > Len_data - 1:
         m2 = Len_data - 1
         i1 = 0
         i2 = kernel.shape[0] - ((Len - 1)/2 - m2 + i) - 1
         k = kernel[i1:i2] / kernel[i1:i2].sum()
      else:
         i1, i2 = 0, kernel.shape[0] - 1
         k = kernel
      #u, g = nu.zeros(m2 - m1), nu.zeros(m2 - m1)
      u = x[m1:m2]
      g = y[m1:m2] * k
      ys[i] = trapz(g, u, diff_wave)
      #xs[i] = x[i]
   return ys



cdef  inline trapz(nu.ndarray y, nu.ndarray x, float d):
    #taken from numpy removed diff for speed
    #y = asanyarray(y)
    #x = asanyarray(x)
    #d = diff(x)
    # reshape to correct shape
    return (d * (y[1:] + y[:-1]) / 2.0).sum()

cdef inline gauss(float diff_wave, float wave_current, float sigma, float h3, float h4):
   '''inline gauss(nu.ndarray diff_wave, float wave_current float sigma, float h3, float h4)
	Returns value of gaussian-hermite function normalized to area = 1'''
   cdef float vel_scale, c, logl_simga, temp_out
   cdef int N
   cdef nu.ndarray[ nu.float64_t, ndim=1] x, y, out
   c = 299792.458
   vel_scale = diff_wave / wave_current * c
   logl_sigma = nu.log(1. + sigma/c) / vel_scale * c
   N = nu.ceil( 5.*logl_sigma)
   x = nu.arange(2*N+1,dtype=float) - N
   y = x / logl_sigma
   out = nu.exp(-y**2/2.) / logl_sigma #/ nu.sqrt(2.*nu.pi)
   out *= ( 1.+ h3 * 2**.5 / (6.**.5) * (2 * y**3 - 3 * y) + h4 / (24**.5) * (4 * y**4 - 12 * y**2 + 3))
   temp_out = nu.sum(out)
   if not temp_out == 0:
      out /= temp_out

   return out

cdef inline locate(nu.ndarray xx, float x):
     #does binary search
    cdef int j, n, jl, ju, jm
    jl = 0
    ju, n = len(xx) - 1,  len(xx) - 1 
    while ju-jl > 1:
        jm = (ju + jl) / 2
        if xx[n] > xx[0] == x > xx[jm]:
            jl = jm +0
        else:
            ju = jm + 0
    return jl
#C  (C) Copr. 1986-92 Numerical Recipes Software '5s!61"25:5<.

if __name__ == '__main__':
   import cPickle as pik
   import downgrade_res as res
   import pylab as lab
   from time import time
   data=lab.np.loadtxt('ssp_0.0200_10.021189.spec')
   t = time() 
   d = res.downgrade_resolution(data,3.)
   print time() - t

