import numpy as np
import pytest
# from rail.core.stage import RailStage
from rail.utils.testing_utils import one_algo
from rail.estimation.algos import dnf


def test_dnf():
    train_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'hdf5_groupname': 'photometry',
                         'model': 'model.tmp'}
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'hdf5_groupname': 'photometry',
                         'min_n': 15, 'bad_redshift_val': 99.,
                         'bad_redshift_err': 10.,
                         'model': 'model.tmp',
                         'nondetect_replace': True}

    train_algo = dnf.DNFInformer
    pz_algo = dnf.DNFEstimator
    results, rerun_results, _ = one_algo("DNF", train_algo, pz_algo, train_config_dict, estim_config_dict)
    #assert np.isclose(results.ancil['mode'], rerun_results.ancil['mode']).all()
    #assert np.isclose(results.ancil['mean'], rerun_results.ancil['mean']).all()
