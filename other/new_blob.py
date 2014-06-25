#! /usr/bin/python2.7

import numpy as nu
import pylab as pl
import pyfits as fits


def MCMC(data,itter,blobs,plot=True):
    #mcmc for fitting blobs to 2-d image
    

    #set up parameters
    param_list=['x','y','sigma_x','sigma_y','flux']
    #Naccept={'x': 1.,'y': 1.,'sigma_x': 1.,'sigma_y': 1.,'flux': 1.0}
    #Nreject={'x': 1.,'y':1.,'sigma_x':1.,'sigma_y':1.,'flux':1.}
    Naccept,Nreject=1.,1.
    active_param={'x':nu.zeros(blobs),'y':nu.zeros(blobs)
                  ,'sigma_x':nu.zeros(blobs),'sigma_y':nu.zeros(blobs)
                  ,'flux':nu.zeros(blobs)}
    out_param={'x':[],'y':[],'sigma_x':[],'sigma_y':[] ,'flux':[]}
    chi=[]
    limits={'x':[0,data.shape[0]-1],'y':[0,data.shape[1]-1],
            'sigma_x':[0,data.shape[1]-1],'sigma_y':[0,data.shape[1]-1],
            'flux':[0,data.max()]}
    step={'x':5.0,'y':5.0,'sigma_x':5.0,'sigma_y':5.0,'flux':12.0}
    #randomly put params
    for j in param_list:
        active_param[j]=nu.random.rand(len(active_param[j]))*limits[j][1]
        out_param[j].append(active_param[j])
    #calculate 1st chi val
    model=nu.zeros(data.shape)
    x,y=nu.float64(nu.meshgrid(range(data.shape[1]),range(data.shape[1])))
    for i in xrange(blobs):
        model=model+gauss_2d(x,y,active_param['x'][i],active_param['y'][i],
                             active_param['flux'][i],active_param['sigma_x'][i],
                             active_param['sigma_y'][i])
    chi.append(nu.sum((data-model)**2))

    #star MCMC
    for ii in xrange(itter):
        for i in xrange(blobs):
            #move blob
            for j in param_list:
                #print ii,i,j
                active_param[j][i] = change_param(out_param[j][-1][i],
                                                  limits[j],step[j])

            #check chi and MH condition
            model=nu.zeros(data.shape)
            for k in xrange(blobs):
                model=model+gauss_2d(x,y,active_param['x'][k],
                                     active_param['y'][k],
                                     active_param['flux'][k],active_param['sigma_x'][k],
                                     active_param['sigma_y'][k])
            chi.append(nu.sum((data-model)**2))
            #mh critera
            if nu.random.rand()<nu.min([1.0,nu.exp((chi[-2]-chi[-1])/2.0)]):
                Naccept+=1
                if min(chi)==chi[-1]:
                    print 'Best chi2 is %f' %chi[-1]
                #accept and false accept
                for j in param_list:
                    out_param[j].append(nu.copy(
                        active_param[j]))
            else:
                Nreject+=1
                chi[-1]=nu.copy(chi[-2])
                for j in param_list:
                    out_param[j].append(nu.copy(
                        out_param[j][-1]))
     #aceptance rate stuff
            if Naccept/Nreject<.4 and step[j]<50:
                for j in param_list:
                    step[j]=step[j]*1.05
            elif Naccept/Nreject>.5 and step[j]>1:
                for j in param_list:
                    step[j]=step[j]/1.05



        if plot:
            pl.imshow(model,vmin=0,vmax=300)
            pl.title('sim with chi of'+str(chi[-1])+' blobs')
            if ii<10:
                filename = 'gau000'+str(ii)+'.png'
            elif ii<100 and ii>=10:
                filename = 'gau00'+str(ii)+'.png'
            elif ii<1000 and ii>=100:
                filename = 'gau0'+str(ii)+'.png'
            else:
                filename = 'gau'+str(ii)+'.png'
            pl.savefig(filename, dpi=50)
            pl.clf()

    #save as numpy arrays
    chi=nu.array(chi)
    for i in out_param.keys():
        out_param[i]=nu.array(out_param[i])
        
    return out_param,chi,step


def change_param(param,limits,sigma):
    #moves param with normal dist and checks limits
    try:
        new=param+nu.random.randn()*sigma
        while new<limits[0] or new>limits[1]:
            new=param+nu.random.randn()*sigma
        return new
    except ValueError:
        print param,limits,sigma
        raise
    except TypeError:
        print param,limits,sigma
        raise
 
def gauss_2d(x,y,mu_x,mu_y,A,sig_x,sig_y):
    return A*nu.exp(-(x-mu_x)**2/(2.0*sig_x**2)-(y-mu_y)**2/(2.0*sig_y**2))

def plot_param(active_param,chi):
    #plot best fit blobs on a 128x128 grid takes only numpy arrays
    x,y=nu.float64(nu.meshgrid(range(128),
                               range(128)))
    model=nu.zeros([128,128])
    #find best chi value
    index=nu.nonzero(min(chi)==chi)[0]
    
    for k in xrange(active_param[active_param.keys()[0]].shape[1]):
        model=model+gauss_2d(x,y,active_param['x'][index[0]][k],
                             active_param['y'][index[0]][k],
                             active_param['flux'][index[0]][k],
                             active_param['sigma_x'][index[0]][k],
                             active_param['sigma_y'][index[0]][k])
    #plot
    pl.figure()
    pl.imshow(model,vmin=0,vmax=300)
    pl.title('best fit image, chi='+str(min(chi)))
    pl.show()
    return model
'''
def nest_elips(like_obj,N=500,elogf=1.06):
    #Does Ellipsoidal Nested Sampling as discribed in Feroz and Hobson (2008)
    #the elipse is s'pose to map the iso-likelhoood contours of the likelihood

    points=nu.zeros([N,like_obj._bounds.shape[0]+1])
    for i in range(1,1+points[:,1:].shape[1]): #assing points in bounds
        points[:,i]=nu.random.rand(len(points[:,i]))*nu.diff(like_obj._bounds[i-1])+nu.mean(like_obj._bounds[i-1,0])
        
    #calculate likeihood for each point and sort
    points[:,0]=like_obj.likeihood_value(points[:,1], points[:,2]) #2-d
    index=nu.argsort(points[:,0])
    points=points[index,:]
    #store discared points for later inference
    old_points=[] #[prior weight,points]
    old_points_txt='likelihood[-1]*weight[-1]'
    for i in range(len(points[0,1:])):
        old_points_txt=old_points_txt+',points[0,1:]['+str(i)+']'
    #initalize prior volume and evidence calculation
    prior_vol,likelihood,weight=[1],[],[]
    #evid=[0]
    i,n_fail=1,0
    #start nested sampling
    while (points[-1,0]-points[0,0])*prior_vol[-1]>10**-4: ####put in stop condition later
        #step 2 and 3
        likelihood.append(points[0,0])
        weight.append((nu.exp(-(i-1.0)/(N+.0))-nu.exp(-(i+1.0)/(N+.0)))/2.0)
        #evid.append(points[0,0]*(nu.exp(-(i-1.0)/(N+.0))-nu.exp(-(i+1.0)/(N+.0)))/2.0)
        prior_vol.append(nu.exp(-i/(N+.0)))
        #recode old point for later inference
        old_points.append([eval(old_points_txt)])
        #step 4 get new liklihood value
           #transform axis using Shaw's method
        
        #find new sampling by L_new>L_old ###slow
        temp_points=nu.zeros([1,points[:,1:].shape[1]+1])
        temp_points[0,1:]=elogf*(coord_trans(points[:,1:])*unit_cir_dis(points[:,1:].shape[1])).ravel()+nu.mean(points[1:,1:],axis=0) #new point
        temp_points[0,0]=like_obj.likeihood_value(temp_points[0,1],temp_points[0,2])
        while points[0,0]>=temp_points[0,0]:
            temp_points[0,1:]=elogf*(coord_trans(points[:,1:])*unit_cir_dis(points[:,1:].shape[1])).ravel()+nu.mean(points[1:,1:],axis=0) #new point
            temp_points[0,0]=like_obj.likeihood_value(temp_points[0,1],temp_points[0,2])
            n_fail+=1
        #print abs(i-n_fail)/float(i)
        #insert new point, sort
        points[0,:]=temp_points+0.0
        index=nu.argsort(points[:,0])
        points=points[index,:]
        i+=1

    #change old_point_txt to match new stuff
    old_points_txt='likelihood[-1]*weight[-1]'
    for j in range(len(points[0,1:])):
        old_points_txt=old_points_txt+',points[j-i,1:]['+str(j)+']'
 
#calculate evidence for remaing points
    for j in range(i+1,i+len(points[:,0])):
        #evid.append(points[j-i,0]*(nu.exp(-(j-1.0)/(N+.0))-nu.exp(-(j+1.0)/(N+.0)))/2.0)
        likelihood.append(points[j-i-1,0])
        weight.append((nu.exp(-(j-1.0)/(N+.0))-nu.exp(-(j+1.0)/(N+.0)))/2.0)
        prior_vol.append(nu.exp(-(j)/(N+.0)))
        old_points.append([eval(old_points_txt)])
    #turn list into for manipulation
    likelihood,weight,prior_vol,old_points=nu.array(likelihood),nu.array(weight),nu.array(prior_vol),nu.array(old_points)[:,0]
    #calculate the uncertany in evidence
    #calculate entropy
    H=nu.nansum(likelihood*weight/sum(likelihood*weight)*nu.log(likelihood/sum(likelihood*weight)))
    #print H
    #calculate uncertanty in evidemce
    evid=likelihood*weight
    evid_error=nu.sqrt(H/float(N))

    return evid,prior_vol[1:],evid_error,old_points

class Param_landscape:
#manages likelihood of input function
    def __init__(self,num_gaus=None,data=None):
        if num_gaus and not data: #set number of 2-d blobs or input data
            self._data=make_data()
        else:
            self._data=data
            
        #set boudaries where gaussian peaks are located
        self._bounds=[128,128]

    def likeihood_value(self,*args): #input and outputs a single value or vector
        #make gaussian func
#no coverance
        try: 
            #pass if want to make a [M,N] grid of points
            out=nu.ones([args[0].shape[0],args[0].shape[1],self._num_gaus])
            for i,ii in enumerate(args): 
                for j in range(self._num_gaus): #for each peak run a gauss calc
                    out[:,:,j]=out[:,:,j]*self.func(ii,self._mu[i,j],self._std[i,j],self._amp[j])
            return nu.sum(out,axis=2)
'''
def make_data(blobs=2):
	#makes data for quick testing
    x,y=nu.meshgrid(nu.linspace(-10,10,128),nu.linspace(-10,10,128))
    return gauss_2d( x, y, 0.0,0.0,150,5,5)+gauss_2d( x, y, -4,-5.0,100,3,9)

def make_movie():
    import os
    command = ('mencoder',
           'mf://*.png',
           '-mf',
           'type=png:w=800:h=600:fps=25',
           '-ovc',
           'lavc',
           '-lavcopts',
           'vcodec=mpeg4',
           '-oac',
           'copy',
           '-o',
           'fit_movie.avi')

    os.spawnvp(os.P_WAIT, 'mencoder', command)
