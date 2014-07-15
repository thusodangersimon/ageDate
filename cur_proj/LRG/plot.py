import numpy as nu
import pylab as lab
import acor
import triangle as tri
from os import path as Path
import os
import pandas as pd

def load_param(path):
    '''loads in parameters'''
    all_data = recover_save_param(path)
    #retive only params
    params = {}
    for i in all_data:
        if 'param' in all_data[i]:
            params[i] = all_data[i]['param']
    return params


def load_debug():
    pass

def recover_save_param(path):
    ''' '''
    if not Path.exists(Path.join(path,'save_files')):
        raise OSError('Path does not exist.')
    out_all = {}
    walker = os.walk(Path.join(path,'save_files'))
    for dirpath, dirnames, filenames in walker:
        for model in  dirnames:
            out_all[model] = {}
        for files in filenames:
            if Path.splitext(files)[1] == '.csv':
                temp_model = Path.split(dirpath)[1]
                file = Path.splitext(files)[0]
                try:
                    out_all[temp_model][file] = nu.loadtxt(Path.join(dirpath,
                                                                     files))
                except ValueError:
                    # Load with pandas
                    out_all[temp_model][file] = pd.DataFrame.from_csv(Path.join(
                                                              dirpath, files),
                                                              sep=' ',
                                                              index_col=None)

    return out_all
                
            
