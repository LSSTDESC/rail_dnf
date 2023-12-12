"""
Implementation of the DNF algorithm
See https://academic.oup.com/mnras/article/459/3/3078/2595234
for more details
"""

import math
import numpy as np
from sklearn import neighbors
from rail.estimation.estimator import CatEstimator, CatInformer





class DNFInformer(CatInformer):
    """Descripcion of the funcion ***
    """
    name = 'DNFInformer'
    config_options = CatInformer.config_options.copy()  # Segun vea lo que necesito cambiar lo que esta comentado
    config_options.update(bands=SHARED_PARAMS,                  #filter_name
                          err_bands=SHARED_PARAMS,              #filter_name_err 
#                          redshift_col=SHARED_PARAMS,
                          mag_limits=SHARED_PARAMS,
                          nondetect_val=SHARED_PARAMS,
                          nondetect_replace=Param(bool, False, msg="set to True to replace non-detects,"
                                                  " False to ignore in distance calculation"))

    def __init__(self, args, comm=None):
        """ Constructor
        Do CatInforme specific initialization"""  # no tengo claro que hace estopero parece que es necesario para inicializar
        CatInformer.__init__(self, args, comm=comm)
        
    def run(self):
        training_data = self.get_data('input')
        
        # replace nondetects
        for col, err in zip(self.config.bands, self.config.err_bands):
            if np.isnan(self.config.nondetect_val):  
                mask = np.isnan(training_data[col])
            else:
                mask = np.isclose(training_data[col], self.config.nondetect_val)
            if self.config.nondetect_replace:
                training_data[col][mask] = self.config.mag_limits[col] # Discutir este valor
                training_data[err][mask] = 1.0  # Discutir este valor
            else:
                training_data[col][mask] = np.nan
                training_data[err][mask] = np.nan
           
        
   

#class DNFEstimator(CatEstimator):





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
    
    

