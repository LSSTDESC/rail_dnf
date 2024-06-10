"""
Implementation of the DNF algorithm
See https://academic.oup.com/mnras/article/459/3/3078/2595234
for more details
"""

import math
import numpy as np
from sklearn import neighbors
from rail.estimation.estimator import CatEstimator, CatInformer

def _computemagdata(data, column_names, err_names):
    """
    make a dataset consisting of N-1 colors and errors in quadrature.******
    """
    numcols = len(column_names)
    numerrcols = len(err_names)
    if numcols != numerrcols:  # pragma: no cover
        raise ValueError("number of magnitude and error columns must be the same!")
    
    magdata = np.array(data[column_names[0]])
    errdata = np.array(data[err_names[0]])
    
    # Iteramos desde el segundo elemento
    for i in range(1, numcols):
        tmpmag = np.array(data[column_names[i]])
        tmperr = np.array(data[err_names[i]])
        
        magdata = np.vstack((magdata, tmpmag))
        errdata = np.vstack((errdata, tmperr))
        
    return magdata.T, errdata.T
    
class DNFInformer(CatInformer):
    """Descripcion of the funcion ***
    """
    name = 'DNFInformer'
    config_options = CatInformer.config_options.copy()
    config_options.update(bands=SHARED_PARAMS,          
                          err_bands=SHARED_PARAMS,    
                          redshift_col=SHARED_PARAMS,
                          mag_limits=SHARED_PARAMS,
                          nondetect_val=SHARED_PARAMS,
                          nondetect_replace=Param(bool, False, msg="set to True to replace non-detects,"
                                                  " False to ignore in distance calculation"))

    def __init__(self, args, comm=None):
        """ Constructor"""  
        CatInformer.__init__(self, args, comm=comm)
        
    def run(self):
        if self.config.hdf5_groupname:
            training_data = self.get_data('input')[self.config.hdf5_groupname]
        else:
            training_data = self.get_data('input')
        specz = np.array(training_data[self.config['redshift_col']])
        
        # replace nondetects
        for col, err in zip(self.config.bands, self.config.err_bands):
            if np.isnan(self.config.nondetect_val):  
                mask = np.isnan(training_data[col])
            else:
                mask = np.isclose(training_data[col], self.config.nondetect_val)
            if self.config.nondetect_replace:
                training_data[col][mask] = self.config.mag_limits[col]
                training_data[err][mask] = 1.0  # Discutir este valor
            else:
                training_data[col][mask] = np.nan
                training_data[err][mask] = np.nan
                
        mag_data, mag_err = _computemagdata(training_data,
                                              self.config.bands,
                                              self.config.err_bands)           
        
        self.model = dict(train_mag=mag_data, train_err=mag_err, truez=specz,
                          nondet_choice=self.config.nondetect_replace)
        self.add_data('model', self.model)        
   

class DNFEstimator(CatEstimator):
    """ Aqui habra que escribir una descripcion de la funcion"""
    
    name = 'DNFEstimator'
    config_options = CatEstimator.config_options.copy()
    config_options.update(zmin=SHARED_PARAMS,
                          zmax=SHARED_PARAMS,
                          nzbins=SHARED_PARAMS,
                          bands=SHARED_PARAMS,
                          err_bands=SHARED_PARAMS,
                          nondetect_val=SHARED_PARAMS,
                          mag_limits=SHARED_PARAMS,
                          redshift_col=SHARED_PARAMS,
                          # seed=Param(int, 66, msg="random seed used in selection mode"),
                          # ppf_value=Param(float, 0.68, msg="PPF value used in Mahalanobis distance"),
                          selection_mode=Param(int, 1, msg="select which mode to choose the redshift estimate:"
                                               "0: ENF, 1: ,ANF, 2: DNF"), # Habra que detallar mas que significa cada caso
                          min_n=Param(int, 25, msg="minimum number of training galaxies to use"),
                          min_thresh=Param(float, 0.0001, msg="minimum threshold cutoff"),
                          min_dist=Param(float, 0.0001, msg="minimum Mahalanobis distance"),
                          bad_redshift_val=Param(float, 99., msg="redshift to assign bad redshifts"),
                          bad_redshift_err=Param(float, 10., msg="Gauss error width to assign to bad redshifts")

                          
    def __init__(self, args, comm=None):
        """ Constructor:
        Do Estimator specific initialization 
        ¿necesito algun parametro mas?"""
        self.truezs = None
        self.model = None    # Asegurar este parametro y su necesidad
        self.zgrid = None
        CatEstimator.__init__(self, args, comm=comm)
        usecols = self.config.bands.copy()
        usecols.append(self.config.redshift_col)
        self.usecols = usecols

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        if self.model is None:  # pragma: no cover
            return
        self.train_mag = self.model['train_mag']
        self.train_err = self.model['train_err']
        self.truez = self.model['truez']
        self.nondet_choice = self.model['nondet_choice']
                          


def greetings() -> str:
    """A friendly greeting for a future friend.

    Returns
    -------
    str
        A typical greeting from a software engineer.
    """
    return "Hello from LINCC-Frameworks!"


def meaning() -> int:
    """The meaning of life, the universe, and everything.

    Returns
    -------
    int
        The meaning of life.
    """
    return 42
    
    

