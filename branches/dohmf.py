import numpy as np
import scipy
import numpy.random as nprand
import scipy.linalg
import scipy.stats
import ctypes
import scipy
import multiprocessing as mp


class config:
	nthreads = mp.cpu_count()
	chunksize = 10


def preprocess_data_pca(dat, edat):
	ndat, npix = dat.shape
	meandat = np.mean(dat, axis=0)
	meandat = meandat / ((meandat ** 2).sum()) ** .5
	proj = np.dot(dat, meandat)	 # projection
	shrink = edat.mean(axis=0)[None, :]
	return shrink, meandat


def get_firstvec(dat, edat, pc=10):
	shrink, meandat = preprocess_data_pca(dat, edat)
	import nipals
	eigs = nipals.doit(dat / shrink, pc - 1)
#	eigs = scipy.linalg.svd(eigs)[2][:(pc-1)] # orthonorm
	eigs = eigs * shrink  # scaling fixed
	eigs = np.vstack((meandat[None, :], eigs))	# add mean spec to the eigens
	eigs = scipy.linalg.svd(eigs)[2][:(pc)]	 # orthonorm
	return eigs


def get_data(ndat=1000, npix=100, ncens=5, xmax=10, err0=0.1, outlfrac=0, outlerrmult=10):
	xgrid = np.linspace(-xmax, xmax, npix)
	# cens = nprand.uniform(-xmax,xmin,size=ncens)
	cens = np.linspace(-xmax, xmax, ncens)

	cenids = nprand.randint(ncens, size=ndat)
	sig = 1
	arr = []
	earr = []

	for i in range(ndat):
		curerr0 = nprand.exponential(err0)
		errs0 = curerr0 + np.zeros(npix)
		# errs0 = np.random.uniform(0,err0,size=npix)
		errs = np.random.normal(0, errs0, size=npix)
		xind = np.random.uniform(size=npix) < outlfrac
		errs[xind] = errs[xind] * outlerrmult
		y = scipy.stats.norm.pdf(xgrid, cens[cenids[i]], sig) + errs
		arr.append(y)
		earr.append(errs0)
	return np.array(arr), np.array(earr)


def get_pca(arr):
	arrm = np.matrix(arr)
	eigvals, eigvecs = scipy.linalg.eigh(arrm.T * arrm)
	return eigvals, eigvecs


def project_only(dat, edat, vecs, getChisq=False):
	ncomp = vecs.shape[1]
	ndat = len(dat)
	npix = len(dat[0])
	Gs = np.matrix(vecs) 
		# no need to make it fancy, we aren't going to change it
	As = shared_zeros_matrix(ndat, ncomp)
	data_struct.dat = dat
	data_struct.edat = edat
	data_struct.Gs = Gs
	data_struct.As = As

	if config.nthreads>1:
		pool = mp.Pool(config.nthreads)
		deltas1 = pool.map(doAstep, range(ndat), chunksize=config.chunksize)
		pool.close()
		pool.join()
	else:
		deltas1 = map(doAstep, range(ndat))

	if getChisq:
		As = data_struct.As
		chisqs=((np.array(dat-np.array((As * vecs.T)))/edat)**2).sum(axis=1)
		ret = As, chisqs
	else:
		ret = As 
	return ret


def shared_zeros_matrix(n1, n2):
	if config.nthreads==1:
		return np.matrix(np.zeros((n1,n2)), copy=False)
	shared_array_base = mp.Array(ctypes.c_double, n1 * n2)
	shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
	shared_array = shared_array.reshape(n1, n2)
	shared_array = np.matrix(shared_array, copy=False)
	return shared_array


def copy_as_shared(mat):
	n1, n2 = mat.shape
	shmat = shared_zeros_matrix(n1, n2)
	shmat[:, :] = mat[:, :]
	return shmat


class data_struct:
	dat = None
	edat = None
	As = None
	Gs = None
	GsOld = None
	ncomp = None
	npix = None
	eps = None


def doAstep(i):
	# we use the fact that dat,edat aren't changed on the way,
	# so they shouldn't be copied to a different thread
	dat, edat = data_struct.dat, data_struct.edat
	Gs, As = data_struct.Gs, data_struct.As
	Fi = np.matrix(dat[i] / edat[i] ** 2, copy=False) * Gs
	
	# Covi = np.matrix(np.diag(1. / edat[i] ** 2), copy=False)
	# Gi = Gs.T * Covi * Gs
	# del Covi
	Gi = Gs.T * np.matrix((1./edat[i]**2)[:,None] * np.asarray(Gs),copy=False)
	
	
	Ai = scipy.linalg.solve(Gi, Fi.T, sym_pos=True)
	newAi = Ai.flatten()
	oldAi = As[i, :]
	delta = scipy.nanmax(np.abs((newAi - oldAi) / (np.abs(oldAi).max() + 1e-100)))
	As[i, :] = newAi
	return delta

def doGstep(j):

	dat, edat = data_struct.dat, data_struct.edat
	Gs, As = data_struct.Gs, data_struct.As
	
	# Covj = np.matrix(np.diag(1. / edat[:, j] ** 2), copy=False)
	# Aj = As.T * Covj * As
	# del Covj
	# the rewrite uses the fact that 
	# diagonal matrix times matrix can be rewritten as 
	# np.matrix(xs[:,None]*np.asarray(Gs)) ==
	# np.matrix(np.diag(xs))*Gs
	Aj = As.T * np.matrix((1. / edat[:, j] ** 2)[:,None] * np.asarray(As), copy=False)
	
	Fj = As.T * np.matrix((dat[:,j] / (edat[:,j]) ** 2), copy=False).T
	Gj = scipy.linalg.solve(Aj, Fj, sym_pos=True)
	newGj = Gj.flatten()
	oldGj = Gs[j, :]
	delta = scipy.nanmax(np.abs((newGj - oldGj) / (np.abs(oldGj).max() + 1e-100)))
	Gs[j, :] = newGj
	return delta


def doGstepSmooth(j):
	dat, edat = data_struct.dat, data_struct.edat
	Gsold, Gs, As = data_struct.Gsold, data_struct.Gs, data_struct.As
	npix = data_struct.npix
	ncomp = data_struct.ncomp
	eps = data_struct.eps


	if j > 0 and j < (npix - 1):
		mult = 2 
	else:
		mult = 1
	# Covj = np.matrix(np.diag(1. / edat[:, j] ** 2))
	# Aj = As.T * Covj * As + mult * eps * np.identity(ncomp)
	# del Covj
	# rewrite of less performant code 
	Aj = As.T * np.matrix((1. / edat[:, j] ** 2)[:,None] * np.asarray(As), copy=False)
	Aj[np.arange(ncomp),np.arange(ncomp)] = np.asarray(
			Aj[np.arange(ncomp),np.arange(ncomp)]
			+ (mult * eps) * np.ones(ncomp)
			).flatten()
	
	if j > 0 and j < (npix - 1):
		Fj = As.T * np.matrix(dat[:,j] / (edat[:,j]) ** 2, copy=False).T + \
			eps * (Gsold[j - 1, :] + Gsold[j + 1, :]).T
	elif j == 0:
		Fj = As.T * np.matrix(dat[:,j] / (edat[:,j]) ** 2, copy=False).T + \
			eps * Gsold[1, :].T
	elif j == npix - 1:
		Fj = As.T * np.matrix(dat[:,j] / (edat[:,j]) ** 2, copy=False).T + \
			eps * Gsold[npix - 2, :].T

	Gj = scipy.linalg.solve(Aj, Fj, sym_pos=True)
	newGj = Gj.flatten()
	oldGj = Gsold[j, :]
	delta = scipy.nanmax(np.abs((newGj - oldGj) / (np.abs(oldGj).max() + 1e-100)))
	Gs[j, :] = newGj
	return delta


def get_hmf(dat, edat, vecs, nit=5, convergence=0.01):
	"""
	dat should have the shape Nobs,Npix
	edat the same thing
	vecs should have the shape (npix, ncomp)
	returns the eigen vectors and the projections vector
	"""
	ncomp = vecs.shape[1]
	ndat = len(dat)
	npix = len(dat[0])
	# a step
#	As = np.matrix(np.zeros((ndat,ncomp)))
#	Gs = np.matrix(vecs)
	As = shared_zeros_matrix(ndat, ncomp)
	Gs = copy_as_shared(vecs)
	data_struct.dat = dat
	data_struct.edat = edat
	data_struct.Gs = Gs
	data_struct.As = As


	for i in range(nit):
		deltas1 = mapper(doAstep, range(ndat))	
		deltas2 = mapper(doGstep, range(npix))

		curconv = scipy.nanmax([scipy.nanmax(deltas1), scipy.nanmax(deltas2)])
		print curconv
		if curconv < convergence:
			break

	# orthogonalize
	Gs, As = orthogonalize(Gs, As)
	return Gs, As


def orthogonalize(G, A):
	# do the orthogonalization and reordering
	eigv, neweigvec = scipy.linalg.eigh(G.T * G)
	sortind = np.argsort(-eigv)
	neweigvec = neweigvec[:, sortind]
		# reorder in descending eigv
	newGs = (G * np.matrix(neweigvec))	# reproject
	newAs = (A * np.matrix(neweigvec))
	return newGs, newAs

def mapper(func, dataset):
	if config.nthreads>1:
		pool = mp.Pool(config.nthreads)
		result = pool.map(func, dataset, chunksize=config.chunksize)
		pool.close()
		pool.join()
	else:
		result = map(func, dataset)
	return result

def get_hmf_smooth(dat, edat, vecs, nit=5, eps=0.01, convergence=0.01):
	"""
	dat should have the shape Nobs,Npix
	edat the same thing
	vecs should have the shape (npix, ncomp)
	returns the eigen vectors and the projections vector
	"""
	ncomp = vecs.shape[1]
	ndat = len(dat)
	npix = len(dat[0])

	As = shared_zeros_matrix(ndat, ncomp)
	Gs = copy_as_shared(vecs)
	Gsold = shared_zeros_matrix(Gs.shape[0], Gs.shape[1])
	
	#arrays used for processing
	data_struct.Gs = Gs
	data_struct.Gsold = Gsold
	data_struct.As = As

	# input data
	data_struct.dat = dat
	data_struct.edat = edat

	# parameters
	data_struct.eps = eps
	data_struct.ncomp = ncomp
	data_struct.npix = npix

	for i in range(nit):
		# a step
		deltas1 = mapper(doAstep, range(ndat))	

		data_struct.Gsold, data_struct.Gs = data_struct.Gs, data_struct.Gsold
		# swapping variables, because we going to update Gs while still using
		# original Gs from A step

		# g step

		deltas2 = mapper(doGstepSmooth, range(npix))	

		curconv = scipy.nanmax([scipy.nanmax(deltas1), scipy.nanmax(deltas2)])
		print curconv
		if curconv < convergence:
			break

	Gs, As = orthogonalize(Gs, As)
	return Gs, As


def rescaler(Gs, As):
	eigvals, eigvecs = scipy.linalg.eigh(As.T * As)
	return Gs * np.matrix(eigvecs).T, As * np.matrix(eigvecs).T


def full_loop():
	arr, earr = get_data(npix=101)
	eigva, eigve = get_pca(arr)
	neweigve, As = get_hmf(arr, earr, eigve[:, -5:])
