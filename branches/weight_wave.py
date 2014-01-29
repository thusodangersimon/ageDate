from scipy.linalg import svd
import numpy as nu
import pylab as lab
import multiprocessing as multi
import likelihood_class as lik
import dohmf as hmf
from sklearn.decomposition import PCA
stats_dist = lik.stats_dist

def weight_wave(data, approx_age, age_sigma,approx_metal=None,metal_sigma=None,meth='pca'):
    '''Trys to deteremine which wavelength are important for best fit of data.
uses aprrox age or metals to decide what areas are important'''

    #needs age or metal
    assert not ((approx_age is None) and (approx_metal is None)), 'Need to define age or metals'
    #import lik object
    fun = lik.VESPA_fit(data,spec_lib='cb07',use_dust=False,use_losvd=False)
    fun.SSP.is_matched=True

    #make age and metal grid
    age,Z =  nu.sort(fun._age_unq),nu.sort(fun._metal_unq)
    Age,met = nu.meshgrid(age,Z)
    Prior = stats_dist.norm.pdf(Age,approx_age,age_sigma)
    if approx_metal is not None and metal_sigma is not None:
        Prior *= stats_dist.norm.pdf(met, approx_metal,metal_sigma)
    prior = Prior.ravel()/nu.sum(Prior)
    met_age_list = nu.asarray(zip(Age.ravel(),met.ravel()))
    spec_lib = nu.zeros((len(met_age_list),data.shape[0]))
    #fit for normalizaton
    for i,j in enumerate(met_age_list):
        spec_lib[i] = fun.lik({'1':{'gal':nu.asarray([nu.hstack((10**-5,j[:],0.))])}},'1',True)[1] 
        spec_lib[i] *= lik.ag.normalize(fun.data,spec_lib[i])
        spec_lib[i] = (spec_lib[i] - fun.data[:,1])**2
    #remove and row with nans or inf
    index= nu.where(nu.isfinite(spec_lib.sum(1)))[0]
    spec_lib = spec_lib[index,:]
    prior = prior[index]
    met_age_list = met_age_list[index]
    if meth.lower() == 'hmf':
        #do hmf
        eigs = hmf.get_firstvec(spec_lib,nu.ones_like(spec_lib),10).T
        loadings,scores = hmf.get_hmf_smooth(spec_lib,nu.ones_like(spec_lib),eigs,nit=130)
        loadings = nu.asarray(loadings).T
        scores = nu.asarray(scores)
        index = 10
    else:
        #do pca
        pca = PCA(10)
        #tell how ages and metalicity go together
        scores = pca.fit_transform(spec_lib)
        var = pca.explained_variance_ratio_
        #show how wavelengths correlate
        loadings = pca.components_.T
        #project loadings on scores
        #get componets with >80% of varance or atleast 2 componetns
        #index = nu.searchsorted(var.cumsum(),0.8)
    #if index < 1:
    #    index = 2
    #index = 2
    #loadings = loadings[:,:index]
    #scores = scores[:,:index]
    proj = nu.zeros_like(spec_lib)
    #theda_scores = nu.asarray(map(nu.math.atan,scores[:,1]/scores[:,0]))
    #theda_loadings = nu.asarray(map(nu.math.atan,loadings[:,1]/loadings[:,0]))
    #normalize so projections are between 2 and 0
    for i in xrange(scores.shape[0]):
        rep_scores = nu.ones_like(loadings) * scores[i]/nu.sqrt(nu.sum(scores[i]**2))
        proj[i] = nu.sum(rep_scores*loadings,1)
        
    #make negitve values less than 1 positive values >1 with max at 2 min at 0
    #multiply by prior
    for i in range(len(prior)):
        proj[i] *= prior[i]
    proj[proj < 0] = 0
    proj = proj.sum(0)
    #make values between -1&1
    proj *= (2 / proj.ptp())
    #shift so min =0
    #proj += nu.abs(proj.min()+proj.min()*0.001)
    return proj
   

def chi_plot(data, weighted_data, grid=20,proc=multi.cpu_count()):
    '''makes chi plot of before and after. Uses multiprocessing'''
    #initalize grid
    fun = lik.VESPA_fit(data,weights=weighted_data,spec_lib='cb07',use_dust=False,use_losvd=False)
    #make age and metal grid
    age,Z =  fun._age_unq,nu.linspace(fun._metal_unq.min(),fun._metal_unq.max(),grid)
    met,Age = nu.meshgrid(age,Z)
    
    met_age_list = nu.asarray(zip(Age.ravel(),met.ravel(),range(Age.size)))
    chi,wchi = nu.zeros_like(Age),nu.zeros_like(Age)
    #multiprocess
    pool = []
    #make queues to recieve and put data
    qin,qout = multi.Manager().Queue(),multi.Manager().Queue()
    #start up process
    for i in range(proc):
        pool.append(multi.Process(target=lik_calc,args=(data,None,qin,qout)))
        pool[-1].start()
    #send data
    map(qin.put,met_age_list)
    #recive and process
    for i in met_age_list:
        try:
            tchi,index = qout.get(timeout=1)
        except:
            continue
        
        chi[nu.unravel_index(int(index),chi.shape)] = nu.copy(tchi)
    #weighted data
    pool = []
    qin,qout = multi.Manager().Queue(),multi.Manager().Queue()
    for i in range(proc):
        pool.append(multi.Process(target=lik_calc,args=(data,weighted_data,qin,qout)))
        pool[-1].start()
    #send data
    map(qin.put,met_age_list)
    #recive and process
    for i in met_age_list:
        try:
            tchi,index = qout.get(timeout=1)
        except:
            continue
        
        wchi[nu.unravel_index(int(index),chi.shape)] = nu.copy(tchi)
    return chi,wchi,Age,met
       
def lik_calc(data,weight,qin,qout):
    '''data,age,Z'''
    fun = lik.VESPA_fit(data,weights=weight,spec_lib='cb07',use_dust=False,use_losvd=False)
    while qin.qsize() == 0:
        pass
    while qin.qsize()>0:
        Z,age,i = qin.get(timeout=1)
        temp_spec = fun.lik({'1':{'gal':nu.asarray([nu.hstack((10**-5,age,Z,1.))])}},'1',True)[1]
        temp_spec *= lik.ag.normalize(fun.data,temp_spec)
        chi = nu.sum((fun.data[:,1] - temp_spec)**2)
        qout.put((chi,i))
        
def post_conturo(x,y,wchi,n_contour=10):
    '''Makes chi into a contour plot with intervals c1 and c2'''
    #get evidence
    evi = -nu.inf
    for i in chi.ravel()[nu.isfinite(chi.ravel())]:
        evi = nu.logaddexp(evi,-i)
    con = nu.sort(nu.concatenate(([.68,95.],nu.arange(1,n_contour)/float(n_contour))))
    fig = lab.figure()
    plt = fig.add_subplot(111)
    plt.contour(x,y,-chi,evi+nu.log(con),cmap='prism',linewidth=3)
    #set binary
    
    
if __name__ == '__main__':
    from time import time
    #make some test data and see if recoverable
    fun = lik.VESPA_fit(nu.ones((500,2)),spec_lib='cb07',use_dust=False,use_losvd=False)
    age = 10.
    fun.SSP.is_matched = True
    a = fun.SSP.get_sed(10**(age - 9),10**(-1.55))
    data=nu.vstack((a,a)).T
    data[:,0]=fun.SSP.sed_ls
    data=data[::-1]

    #make real fake data in a wavelength range
    index = nu.searchsorted(data[:,0],[3000,8000])
    data = data[index[0]:index[1]]
    #data[:,1] *= nu.random.rand(data.shape[0])*data[:,1].min()
    chi, wchi, Z, Age = chi_plot(data,weight_wave(data,10.,.1,meth='pca'),30)
    fig = lab.figure()
    plt1 = fig.add_subplot(121)
    plt2 = fig.add_subplot(122)
    plt1.pcolor(Z,Age,nu.log10(chi),vmin=0,vmax=nu.nanmax(nu.log10(chi)))
    #lab.colorbar()
    plt2.pcolor(Z,Age,nu.log10(wchi),vmin=0,vmax=nu.nanmax(nu.log10(chi)))
    #lab.colorbar()
    plt1.set_title('Orignal')
    plt2.set_title('Weighted')
    lab.show()
