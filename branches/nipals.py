import numpy as np,numpy.linalg

def doit(data, maxpc=None, 
			 conv = 1e-8, max_it = 100000):
	"""Perform Principal Component Analysis using the NIPALS algorithm.
	"""
	X = np.matrix(data)  # ndat, npix
	npix = X.shape[1]
	eigenv = np.zeros((maxpc, npix))
	import time 
	for i in range(maxpc):
		it = 0
		t = X[0, :] #1,npix
		# initialize difference
		diff = conv + 1

		while diff > conv:
			#t1=time.time()
			# increase iteration counter
			it += 1
			# Project X onto t to find corresponding loading p
			# and normalize loading vector p to length 1
			p = (X * t.T) / (t * t.T) # ndat,1
			p /= np.sqrt(p.T * p)

			# project X onto p to find corresponding score vector t_new
			t_new = p.T * X 
			# difference between new and old score vector
			tdiff = t_new - t
			diff = (tdiff * tdiff.T)
			t = t_new
			if it > max_it:
				msg = ('PC#%d: no convergence after'
					   ' %d iterations.'% (i, max_it))
				raise Exception(msg)
			#t2=time.time()
			#print t2-t1

		# store ith eigenvector in result matrix
		eigenv[i, :] = t
		# remove the estimated principal component from X
		D = np.matrix(np.outer(p, t), copy=False)
		X -= D
		#D = (D * p)
		#d[i] = (D*D).sum()/(tlen-1)
		#exp_var += d[i]/var
		#if des_var and (exp_var >= self.desired_variance):
		#	self.output_dim = i + 1
		#	break
	return eigenv
	#self.d = d[:self.output_dim]
	#self.v = eigenv[:self.output_dim, :].T
	#self.explained_variance = exp_var