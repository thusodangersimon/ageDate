import ezgal as gal
import spectra_lib_utils as sp_util
from  database_utils import numpy_sql
from multiprocessing import Pool
import numpy as nu
import os

'''creates CSP for LRGs makes 2 different models, Sigle burst at age t
or exponental delciling sfh with tau as scale'''


def make_exp(tau, lib_path='/home/thuso/Phd/stellar_models/ezgal', imf='chab',
             model='cb07'):
    '''Uses ezgal to make exponentaly decreasing sfh with scale tau (GYR) for
     all metalicites of input spec librabry'''
    # Load ssp models
    SSP = sp_util.load_ezgal_lib(model, lib_path, imf)
    # make sure is matched for interpolatioin
    SSP.is_matched = True
    #make exp
    for ssp in SSP:
        yield ssp.make_exponential(tau)

    
def make_burst(tau, lib_path='/home/thuso/Phd/stellar_models/ezgal', imf='chab',
             model='cb07'):
    '''Uses ezgal to make sigle burst of sf with length tau (GYR) for
     all metalicites of input spec librabry'''
    SSP = sp_util.load_ezgal_lib(model, lib_path, imf)
    # make sure is matched for interpolatioin
    SSP.is_matched = True
    #make exp
    for ssp in SSP:
        yield ssp.make_burst(tau)


def make_csp_lib(csp_type, csp_num=10, save_path='.'):
    '''Makes a csp library. does n steps of tau in GYR to 0.05-11 GYR.
    With multicore processing
    '''
    # get burst function
    assert csp_type.lower() in ['burst', 'exp'], "csp_type must be 'burst' or 'exp'"
    if csp_type == 'burst':
        model = make_burst
    else:
        model = make_exp
    #pool = Pool()
    tau = nu.linspace(0.1, 11, csp_num)
    models = map(model, tau)
    # save spectra to database
    save_name = os.path.join(save_path, '%s_dtau_%d.db'%(csp_type,csp_num))
    conn = numpy_sql(save_name)
    c = conn.cursor()
    #create table
    c.execute('''CREATE TABLE %s (imf text, model text, tau real, age real,
    metalicity real ,
    spec array)'''%csp_type)

    for csp in models:
        for meta_gal in csp:
            for age in nu.log10(1+meta_gal.ages):
                if age == 0:
                    age = nu.log10(1*10**5)
                if nu.isclose(10**(age-9), 20):
                    data = nu.vstack((meta_gal.ls, meta_gal.get_sed(20))).T
                else:
                    data = nu.vstack((meta_gal.ls,
                                      meta_gal.get_sed(10**(age-9)))).T
                # reverse
                data = data[-1:0:-1]
                if 'length' in meta_gal.meta_data:
                    length = float(meta_gal.meta_data['length'])
                elif 'tau' in meta_gal.meta_data:
                    length = float(meta_gal.meta_data['tau'])
                
                insert  = (meta_gal.meta_data['imf'],meta_gal.meta_data['model']
                           , length, age,
                           nu.log10(float(meta_gal.meta_data['met'])), data,)
                c.execute('INSERT INTO %s VALUES (?,?,?,?,?,?)'%csp_type, insert)
            # save
            conn.commit()
    conn.close()


def index_DB(db):
    '''Index database made like above to improve query speed.'''
    # get table name
    table_name = db.execute('select * from sqlite_master').fetchall()[0][1]
    db.execute('CREATE UNIQUE INDEX i ON %s (imf, model, tau, age, metalicity)'%table_name)
    db.commit()

    
if __name__ == '__main__':
    
    #make_csp_lib('burst')
    
    make_csp_lib('exp')
