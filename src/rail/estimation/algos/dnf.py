"""An example module containing simplistic functions."""


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
    
    
class DNFInformer(CatInformer):
    """Descripcion of the funcion ***
    """
    name = 'DNFInformer'
#    config_options = CatInformer.config_options.copy()
#    config_options.update(bands=SHARED_PARAMS,
#                          err_bands=SHARED_PARAMS,
#                          redshift_col=SHARED_PARAMS,
#                          mag_limits=SHARED_PARAMS,
#                          nondetect_val=SHARED_PARAMS,
#                          nondetect_replace=Param(bool, False, msg="set to True to replace non-detects,"
#                                                  " False to ignore in distance calculation"))


#class DNFEstimator(CatEstimator):
