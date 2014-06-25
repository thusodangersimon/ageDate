'''EzGal 
==========
A python module for calculating observables (magnitudes, masses, mass-to-light ratios, etc...) from standard SPS models as a function of filter, formation redshift, and redshift. A web interface to EzGal is available for calculating models and getting results immediately.

This is a fork of the EZGAL for running with Age_Date. See http://www.baryons.org/ for more info about EZGAL.

'''



import src
import convert #todo
__author__ = 'Conor Mancone, Anthony Gonzalez'
__email__ = 'cmancone@gmail.com'
__ver__ = '2.0'

ezgal = src.ezgal.ezgal
model = ezgal
astro_filter = src.astro_filter.astro_filter
ezgal_light = src.ezgal_light.ezgal_light
wrapper = src.wrapper.wrapper
weight = src.weight.weight
utils = src.utils
sfhs = src.sfhs

__all__ = ["model", "utils", "wrapper", "sfhs", "weight"]



def interpolate( values, xs, models=None, key=None, return_wrapper=False ):
	""" models = ezgal.interpolate( values, xs, models, return_wrapper=False )
	
	or
	
	models = ezgal.interpolate( values, models, key=meta_key, return_wrapper=False )
	
	Interpolate between EzGal models and return new models.
	`models` is a list of EzGal model objects or filenames of EzGal compatible files.
	`xs` is the values of the models to be interpolated between and `values` is a list
	of values for the new models to be interpolated at.
	
	Alternatively you can ignore xs and specify the name of a meta key
	to use to build the interpolation grid.
	
	Returns a list of EzGal model objects or a single EzGal model if a scalar is passed
	for `values`.  Alternatively, set return_wrapper=True and it will return an ezgal wrapper
	object containing the fitted models objects.
	
	All model SEDs must have the same age/wavelength grid. """

	# what calling sequence was used?
	if models is None and key is not None:
		return wrapper( xs ).interpolate( key, values, return_wrapper=return_wrapper )

	# make sure we have everything we need...
	if len( models ) != len( xs ): raise ValueErrors( 'xs list has a different length than models list!' )

	# return interpolated models
	return wrapper( models, extra_data=xs, extra_name='interp' ).interpolate( 'interp', values, return_wrapper=return_wrapper )


def add_meta_data_batch(infiles, outfile):
    '''ezgal.add_meta_data( infile, outfile, model, metallicity, imf, sfh)

    If a model does not come with meta data, this program will add it in.
    model is refence name for model eg. bc03, miles ...
    metallicity is metallicity of model spectra
    imf is the inital mass funtion.
    SFH tells if model is an ssp or other type

    Returns Nonetype'''
    from glob import glob
    #get file list
    if not outfile.endswith('/'):
        outfile += '/'
    if not infiles.endswith('/'):
        infiles += '/'
        
    files = glob(infiles + '*.model')
    for i in files:
        temp_models = model(i)
        if temp_models.has_meta_data:
            continue
        #get meta data from name
        meta_data = {}
        #remove path and other stuff
        temp_name = i[i.rfind('/')+1:i.rfind('.')]
        try:
            meta_data['model'],meta_data['sfh'],junk,meta_data['met'],meta_data['imf'] = temp_name.split('_')
        except ValueError:
            #basti has 1 more alpha enhansment param wich i will put in
            meta_data['model'],meta_data['sfh'],junk,meta_data['alpha'],junk,meta_data['met'],meta_data['imf'] = temp_name.split('_')
        #save model in outfiles
        temp_models.set_meta_data(meta_data)
        temp_models.save_model(outfile + temp_name + '.model')
        print "setting %s.model with meta data:"%temp_name,meta_data
