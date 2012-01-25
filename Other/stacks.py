


import numpy as nu
import os
from scipy.optimize import leastsq

def load_stack(infile):
	#loads files from Rita's vespa output
	temp=open(infile)
	#get number of bins needed to take
	N_bin_txt=temp.readline(99)
	Nbin=''
	for i in N_bin_txt:
		if i=='#':
			break
		Nbin+=i
	Nbin=float(Nbin)
	#grab redshift line
	redshift_temp=temp.readline(99)
	redshift=''
	for i in redshift_temp:
		if i=='#':
			break
		redshift+=i
	redshift=float(redshift)
	#look for start of information about star formation history
	for i in temp:
		if i.find('#Bin')>0:
			out=[]
			for j,i in enumerate(temp): #found data not grab it
				if j>=Nbin:
					break
				out.append(nu.array(i.split(),dtype=nu.float64))
			break

	return nu.array(out),redshift

def translate_bin(data):
	#takes output from Vespa and turns into useable format
	#data should be [Bin,Width,SFF,Error_SFF,Metallicity,Error_met]
	#in Fig 1 Trojero 2009 decided width means 2 below and 1 up
	bin_table=nu.array([
			[0, .002],
			[1,.074],
			[2,.177],
			[3,.275],
			[4,.425],
			[5,.658],
			[6,1.02],
			[7,1.57],
			[8,2.44],
			[9,3.78],
			[10,5.84],
			[11,7.44],
			[12,8.24],
			[13,9.04],
			[14,10.28],
			[15,11.52],
			[16,13.8]])
	age_uncert=[]
	#calculate age from table using [self,down,up,down,up,etc...]
	for i in data:
		bin_index=[int(i[0])+1]
		for j in range(1,int(i[1])):
			bin_index.append(bin_index[-1]+0)
			if j%2==0: #if even go up
				bin_index[-1]+=j
			else: #odd go down 
				bin_index[-1]-=j

		i[0]= 10**nu.log10(bin_table[bin_index,1]).mean()#take log mean, its what she does in paper
		#i[0]= bin_table[bin_index,1].mean()
		age_uncert.append(i[3]*bin_table[bin_index,1].ptp()/2.) #propigation of uncert
				
	#calc SFH weighted age of stack 
	age= nu.sum(data[:,0]*data[:,2])/nu.sum(data[:,2])
	#calc age bin uncertaty
	age_unc_out=nu.sum(nu.array(age_uncert))**.5
	return age,age_unc_out

def calc_hubble(indir):
	#takes all stacks in file and calculates hubble constant for each redshift
	if indir[-1]!='/':
		indir+='/'
	files=os.listdir(indir)
	temp=[]
	for i in files:
		if i[-4:]!='.txt': #make sure txt file
			print 'wrong input file detected and skipped'
			continue
		
		temp_out=load_stack(indir+i)
		temp.append(nu.copy([translate_bin(temp_out[0])[0],temp_out[1],
				     translate_bin(temp_out[0])[1]]))

	age_red=nu.array(temp)
	age_red=age_red[nu.argsort(age_red[:,1]),:]
	redshifts=nu.unique(age_red[:,1])
	age_uncert=redshifts*0.
	age=redshifts*0.
	for j,i in enumerate(redshifts):
		index=nu.nonzero(age_red[:,1]==i)[0]
		age[j]=nu.mean(age_red[index,0])
		age_uncert[j]=nu.sum(age_red[index,2]**2)**.5 #uncert propigation
	return redshifts,age,age_uncert,-1/(1+redshifts[:-1])*nu.diff(age)/nu.diff(redshifts)

def fit(x,y,y_uncert):
	func=lambda a,x: a[0]*x**a[1]
	def f(a,x,y=y,yerr=y_uncert):
		return (y - func(a,x))/yerr
	return leastsq(f, [-1000.,2.1823],args=(x))
    

if __name__=='__main__':
	from matplotlib import rc #latex modual
	import pylab as lab
	from thuso_quick_fits import quick_MCMC
	#power law fit
	func=lambda x,a: a[0]*x**a[1]+a[2]
	H_fit=lambda z,p2: -(p2[0]*p2[1])**-1*(z**(p2[1]-1)+z**(p2[1]))**-1 #calculate hubble
	redshifts,age,age_err,H=calc_hubble('/home/thuso/Phd/stellar_models/stacks_SFHs')
	print 'fitting with MCMC, please be patent'
	chi,param,p2,chi_best=quick_MCMC(redshifts,age,age_err,[10.,5.,1],func,itter=5*10**6,quiet=True)
	#calculate uncertantiy in parameters
	param,uncer=nu.array(param),[]
	for i in range(len(p2)):
		uncer.append([0,0])
		quantile,y=nu.histogram(param[1000:,i],nu.unique(param[1000:,i]))
		quantile=nu.cumsum(quantile)/(nu.sum(quantile)+.0)
		uncer[-1]=[y[nu.abs(quantile-.16)==nu.abs(quantile-.16).min()],
			   y[nu.abs(quantile-.84)==nu.abs(quantile-.84).min()]]
	rc('text', usetex=True)
	#p2,success = fit(redshifts,age,age_err)
	hub_con=978. #conversion from gyr to km/s/mpc not correct but gives right values
	#plotting
	fig=lab.figure()
	plot=fig.add_subplot(121)
	plot.plot(redshifts,func(redshifts,p2))
	plot.errorbar(redshifts,age,age_err,marker='.',linestyle='.')
	plot.set_xlabel('Redshift')
	plot.set_ylabel('Age (Gyr)')
	plot.set_title(r'Z vs Age $Age=%2.1f^{%2.1f}_{%2.1f} *Z^{%2.4f^{%2.1f}_{%2.1f}}$' 
		       %(p2[0],uncer[0][1]-p2[0],-uncer[0][0]+p2[0],
			 p2[1],uncer[0][1]-p2[0],-uncer[0][0]+p2[0]))

	plot1=fig.add_subplot(122)
	plot1.plot(redshifts[:-1],H_fit(redshifts[:-1],p2)*hub_con,redshifts[:-1],H*hub_con,'.')
	plot1.set_ylim((min(H*hub_con),max(H*hub_con)))
	plot1.set_xlabel('Redshift')
	plot1.set_ylabel(r'H(Z) [$\frac{km}{s*Mpc}$]')
	plot1.set_title(r'Hubble vs Redshift $H_{0}=%2.2f$' %(H_fit(0.001,p2)*hub_con))
	lab.show()
	
