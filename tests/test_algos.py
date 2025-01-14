import pytest
# from rail.core.stage import RailStage
from rail.utils.testing_utils import one_algo
from rail.estimation.algos import dnf


def test_dnf_ANF():
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
    results, rerun_results, _ = one_algo("DNF_ANF", train_algo, pz_algo, train_config_dict, estim_config_dict)


def test_dnf_ENF():
    train_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'hdf5_groupname': 'photometry',
                         'model': 'model.tmp'}
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'hdf5_groupname': 'photometry',
                         'min_n': 15, 'bad_redshift_val': 99.,
                         'bad_redshift_err': 10.,
                         'model': 'model.tmp',
                         'nondetect_replace': True,
                         'selection_mode': 0}

    train_algo = dnf.DNFInformer
    pz_algo = dnf.DNFEstimator
    results, rerun_results, _ = one_algo("DNF_ENF", train_algo, pz_algo, train_config_dict, estim_config_dict)


def test_dnf_DNF():
    train_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'hdf5_groupname': 'photometry',
                         'model': 'model.tmp'}
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'hdf5_groupname': 'photometry',
                         'min_n': 15, 'bad_redshift_val': 99.,
                         'bad_redshift_err': 10.,
                         'model': 'model.tmp',
                         'nondetect_replace': True,
                         'selection_mode': 2}

    train_algo = dnf.DNFInformer
    pz_algo = dnf.DNFEstimator
    results, rerun_results, _ = one_algo("DNF_DNF", train_algo, pz_algo, train_config_dict, estim_config_dict)


def test_dnf_badval():
    train_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'hdf5_groupname': 'photometry',
                         'model': 'model.tmp'}
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'hdf5_groupname': 'photometry',
                         'min_n': 15, 'bad_redshift_val': 99.,
                         'bad_redshift_err': 10.,
                         'model': 'model.tmp',
                         'nondetect_replace': True,
                         'selection_mode': "BADVAL"}

    train_algo = dnf.DNFInformer
    pz_algo = dnf.DNFEstimator
    with pytest.raises(TypeError):
        results, rerun_results, _ = one_algo("DNF_DNF", train_algo, pz_algo, train_config_dict, estim_config_dict)


def test_dnf_badint():
    train_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'hdf5_groupname': 'photometry',
                         'model': 'model.tmp'}
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'hdf5_groupname': 'photometry',
                         'min_n': 15, 'bad_redshift_val': 99.,
                         'bad_redshift_err': 10.,
                         'model': 'model.tmp',
                         'nondetect_replace': True,
                         'selection_mode': 99}

    train_algo = dnf.DNFInformer
    pz_algo = dnf.DNFEstimator
    with pytest.raises(ValueError):
        results, rerun_results, _ = one_algo("DNF_DNF", train_algo, pz_algo, train_config_dict, estim_config_dict)
