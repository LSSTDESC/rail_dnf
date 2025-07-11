"""
Implementation of the DNF algorithm

DNF (Directional Neighbourhood Fitting) is a nearest-neighbor approach for photometric redshift estimation developed at the CIEMAT (Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas) at Madrid. DNF computes the photo-z hyperplane that best fits the directional neighbourhood of a photometric galaxy in the training sample.

See https://academic.oup.com/mnras/article/459/3/3078/2595234
for more details.

"""

# import math
import numpy as np
import qp
from sklearn import neighbors
# from scipy.stats import chi2
from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatEstimator, CatInformer
from rail.core.common_params import SHARED_PARAMS


def _computemagdata(data, column_names, err_names):
    """
    Constructs a dataset containing N-1 magnitudes (or fluxes) and their corresponding
    errors combined in quadrature.

    Returns:
    - magdata: numpy array (N_samples, N_features) containing magnitude data.
    - errdata: numpy array (N_samples, N_features) containing error data.
    """
    numcols = len(column_names)
    numerrcols = len(err_names)
    if numcols != numerrcols:  # pragma: no cover
        raise ValueError("number of magnitude and error columns must be the same!")

    magdata = np.array(data[column_names[0]])
    errdata = np.array(data[err_names[0]])

    # Iterate through the remaining columns
    for i in range(1, numcols):
        tmpmag = np.array(data[column_names[i]])
        tmperr = np.array(data[err_names[i]])

        magdata = np.vstack((magdata, tmpmag))
        errdata = np.vstack((errdata, tmperr))

    return magdata.T, errdata.T


class DNFInformer(CatInformer):
    """
    A class for photometric redshift estimation.

    This class extends `CatInformer` and processes photometric data to train
    for estimating redshifts. It handles missing data by replacing
    non-detections with predefined magnitude limits and assigns errors accordingly.

    """
    name = 'DNFInformer'
    config_options = CatInformer.config_options.copy()
    config_options.update(bands=SHARED_PARAMS,
                          err_bands=SHARED_PARAMS,
                          redshift_col=SHARED_PARAMS,
                          mag_limits=SHARED_PARAMS,
                          nondetect_val=SHARED_PARAMS,
                          hdf5_groupname=SHARED_PARAMS)

    def __init__(self, args, **kwargs):
        """ Constructor
        Do CatInformer specific initialization, then check on bands """
        super().__init__(args, **kwargs)

    def run(self):
        if self.config.hdf5_groupname:
            training_data = self.get_data('input')[self.config.hdf5_groupname]
        else:
            training_data = self.get_data('input')  # pragma: no cover
        specz = np.array(training_data[self.config['redshift_col']])

        # replace nondetects
        for col, err in zip(self.config.bands, self.config.err_bands):
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                mask = np.isnan(training_data[col])
            else:
                mask = np.isclose(training_data[col], self.config.nondetect_val)

            training_data[col][mask] = self.config.mag_limits[col]
            training_data[err][mask] = 1.0  # could also put 0.757 for 1 sigma, but slightly inflated seems good

        mag_data, mag_err = _computemagdata(training_data,
                                            self.config.bands,
                                            self.config.err_bands)

        # Training euclidean metric
        clf = neighbors.KNeighborsRegressor()
        clf.fit(mag_data, specz)

        # Training variables
        Tnorm = np.linalg.norm(mag_data, axis=1)

        self.model = dict(train_mag=mag_data, train_err=mag_err, truez=specz, clf=clf, train_norm=Tnorm)
        self.add_data('model', self.model)


class DNFEstimator(CatEstimator):
    """
    A class for estimating photometric redshifts using the DNF method.

    This class extends `CatEstimator` and predicts redshifts based on photometric.
    It supports multiple selection  modes for redshift estimation, processes missing data, and generates probability
    density functions (PDFs) for photometric redshifts.

    Metrics (selection_mode):
    - ENF (1): Euclidean neighbourhood. It's a common distance metric used in kNN (k-Nearest Neighbors) for photometric redshift prediction.
    - ANF (2): uses normalized inner product for more accurate photo-z predictions. It is particularly recommended when working with datasets containing more than four filters.
    - DNF (3): combines Euclidean and angular metrics, improving accuracy, especially for larger neighborhoods, and maintaining proportionality in observable content.
    """

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
                          selection_mode=Param(int, 1, msg="select which mode to choose the redshift estimate:"
                                               "0: ENF, 1: ANF, 2: DNF")
                          )

    def __init__(self, args, **kwargs):
        """ Constructor:
        Do Estimator specific initialization
        """
        self.truezs = None
        self.model = None
        self.zgrid = None
        self.metric = "ANF"
        super().__init__(args, **kwargs)
        usecols = self.config.bands.copy()
        usecols.append(self.config.redshift_col)
        self.usecols = usecols
        # set up selection mode metric choice
        if self.config.selection_mode == 0:
            self.metric = "ENF"
            print("using metric ENF")
        elif self.config.selection_mode == 1:
            self.metric = "ANF"
            print("using metric ANF")
        elif self.config.selection_mode == 2:
            self.metric = "DNF"
            print("using metric DNF")
        else:
            raise ValueError("invalid value for config parameter selection_mode! Valid values are 0, 1, and 2")

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        if self.model is None:  # pragma: no cover
            return
        self.train_mag = self.model['train_mag']
        self.train_err = self.model['train_err']
        self.truez = self.model['truez']
        self.clf = self.model['clf']
        self.Tnorm = self.model['train_norm']

    def _process_chunk(self, start, end, data, first):

        print(f"Process {self.rank} estimating PZ PDF for rows {start:,} - {end:,}")
        # replace nondetects
        for col, err in zip(self.config.bands, self.config.err_bands):
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                mask = np.isnan(data[col])
            else:
                mask = np.isclose(data[col], self.config.nondetect_val)

            data[col][mask] = self.config.mag_limits[col]
            data[err][mask] = 1.0  # could also put 0.757 for 1 sigma, but slightly inflated seems good

        test_mag, test_mag_err = _computemagdata(data,
                                                 self.config.bands,
                                                 self.config.err_bands)
        num_gals = test_mag.shape[0]
        ncols = test_mag.shape[1]

        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)

        photoz, photozerr, photozerr_param, photozerr_fit, z1, nneighbors, de1, d1, id1, C, pdfs = \
            dnf_photometric_redshift(
                self.train_mag,
                self.train_err,
                self.truez,
                self.clf,
                self.Tnorm,
                test_mag,
                test_mag_err,
                self.zgrid,
                metric=self.metric,
                fit=True,
                pdf=True,
                Nneighbors=80,
                presel=4000
            )

        ancil_dictionary = dict()
        qp_dnf = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=pdfs))

        ancil_dictionary.update(DNF_Z=photoz, photozerr=photozerr,
                                photozerr_param=photozerr_param, photozerr_fit=photozerr_fit, DNF_ZN=z1,
                                nneighbors=nneighbors, de1=de1, d1=d1, id1=id1)  # , C=C, Vpdf=Vpdf
        qp_dnf.set_ancil(ancil_dictionary)

        self._do_chunk_output(qp_dnf, start, end, first, data=data)


def dnf_photometric_redshift(T, Terr, z, clf, Tnorm, V, Verr, zgrid, metric='ANF', fit=True, pdf=True, Nneighbors=80, presel=500):
    """
    Compute the photometric redshifts for the validation or science sample.

    Returns
    -------
        - photoz: Estimated photometric redshift.
        - photozerr: Error on the photometric redshift.
        - photozerr_param: Redshift error due to parameters.
        - photozerr_fit:Redshift error due to fit.
        - z1: Closest redshift estimate.
        - nneighbors: Number of neighbors considered.
        - de1: Distances Euclidea to the closest neighbor.
        - d1: Distances to the closest neighbor.
        - id1: Index of the closest neighbor.
        - C: Additional computed parameters.
        - zpdf: Matrix containing the redshifts of neighboring galaxies.
        - wpdf: Matrix of weights corresponding to the neighboring redshifts.
        - Vpdf: Probability Density Functions (PDFs) for the photometric redshifts of the validation set.
    """

    C = 0  # coefficients by default

    # Step 0: Manage NaNs
    V, Verr = manage_nan(V, Verr)

    # Step 1: Preselection
    NEIGHBORS, Ts, Tsnorm, de1, Nvalid = preselection(V, Verr, Nneighbors, presel, T, clf, Tnorm, z)

    # Step 2: Metric computation
    NEIGHBORS, z1, d1, id1 = metric_computation(V, NEIGHBORS, Ts, Tsnorm, metric, Nneighbors)

    # Step 3: Compute mean redshift removing outliers
    photoz, photozerr, photozerr_param, photozerr_fit, nneighbors, Vpdf, NEIGHBORS = compute_photoz_mean_routliers(NEIGHBORS, Verr, pdf, Nvalid, zgrid)

    # Step 4: Optional fitting of redshifts
    if fit:
        photoz, photozerr, photozerr_param, photozerr_fit, nneighbors, C, Vpdf = compute_photoz_fit(
            NEIGHBORS,
            V,
            Verr,
            T,
            z,
            fit,
            photoz,
            photozerr,
            photozerr_param,
            photozerr_fit,
            pdf,
            zgrid
        )

    # Return
    return (
        photoz,
        photozerr,
        photozerr_param,
        photozerr_fit,
        z1,
        nneighbors,
        de1,
        d1,
        id1,
        C,
        Vpdf,
    )


def validate_columns(V, T):
    """
    Validates that the columns of T and V have the same names.

    Parameters
    ----------
    T : np.ndarray
      Training data.
    V : np.ndarray
      Validation data.

    Raises
    ------
    ValueError
      If the column names of T and V do not match.
    """
    if not np.array_equal(T.dtype.names, V.dtype.names):  # pragma: no cover
        raise ValueError("The columns of T and V do not match. Please ensure that both T and V have the same features.")


def manage_nan(V, Verr):
    '''
    Change NaNs by 0 in V and Verr to use only proper measurements
    '''
    V[np.isnan(V)] = 0.0
    V[np.isnan(Verr)] = 0.0
    Verr[np.isnan(V)] = 0.0
    Verr[np.isnan(Verr)] = 0.0
    return V, Verr


def preselection(V, Verr, Nneighbors, presel, T, clf, Tnorm, z):
    """
    Perform the preselection process for photometric redshift estimation.
    """
    # Ensure V is set
    if V is None:  # pragma: no cover
        raise ValueError("Validation data 'V' is not set. Ensure it is initialized.")

    # Size training
    Ntrain = T.shape[0]
    Nneighbors_presel = presel if Ntrain > presel else Ntrain

    # Validation variables
    validate_columns(V, T)

    Nvalid = V.shape[0]
    nneighbors = np.full(Nvalid, Nneighbors, dtype='int')

    # Compute distances and indices for preselected neighbors
    Ndistances, Nindices = clf.kneighbors(V, n_neighbors=Nneighbors_presel)

    # Handle cases where distance is zero
    mask = Ndistances[:, 0] == 0
    Ndistances[mask] = np.roll(Ndistances[mask], -1, axis=1)
    Nindices[mask] = np.roll(Nindices[mask], -1, axis=1)
    Ndistances[mask, -1] = Ndistances[mask, 0]
    Nindices[mask, -1] = Nindices[mask, 0]

    # Store distances and indices
    de1 = Ndistances[:, 0]

    # Initialize NEIGHBORS array to store indices, distances, and redshifts
    NEIGHBORS = np.zeros(
        (Nvalid, Nneighbors_presel),
        dtype=[('index', 'i4'), ('distance', 'f8'), ('z', 'f8')]
    )
    NEIGHBORS['index'] = Nindices
    NEIGHBORS['distance'] = Ndistances
    NEIGHBORS['z'] = z[Nindices]

    # Extract training preselection data
    Ts = T[Nindices]
    Tsnorm = Tnorm[Nindices]

    return NEIGHBORS, Ts, Tsnorm, de1, Nvalid


def metric_computation(V, NEIGHBORS, Ts, Tsnorm, metric, Nneighbors):
    """
    Compute distances based on the selected metric, sort neighbors, and store the closest neighbors.
    """
    # Compute distances based on the chosen metric
    if metric == 'ENF':
        NEIGHBORS['distance'] = compute_euclidean_distance(V, Ts)
    elif metric == 'ANF':
        NEIGHBORS['distance'] = compute_angular_distance(V, Ts, Tsnorm)
    elif metric == 'DNF':
        NEIGHBORS['distance'] = compute_directional_distance(V, Ts, Tsnorm)

    # Get the indices of the k nearest neighbors
    partial_indices = np.argpartition(NEIGHBORS['distance'], Nneighbors, axis=1)[:, :Nneighbors]

    # Use those indexes to select neighbors
    row_indices = np.arange(NEIGHBORS.shape[0])[:, None]
    top_k_neighbors = NEIGHBORS[row_indices, partial_indices]

    # Sort only those k neighbors (so they are sorted by distance)
    sorted_order = np.argsort(top_k_neighbors['distance'], axis=1)
    top_k_neighbors = np.take_along_axis(top_k_neighbors, sorted_order, axis=1)

    # top_k_neighbors is equivalent to NEIGHBORS[:,:Nneighbors] sorted
    NEIGHBORS = top_k_neighbors

    # Store nearest redshift, distance and index
    z1 = NEIGHBORS[:, 0]['z']
    id1 = NEIGHBORS[:, 0]['index']
    d1 = NEIGHBORS[:, 0]['distance']

    return NEIGHBORS, z1, d1, id1


def compute_euclidean_distance(V, Ts):
    """
    Compute distances based on Euclidean metric.
    """

    D = V[:, np.newaxis, :] - Ts
    Dsquare = D[:] * D[:]
    D2 = np.sum(Dsquare[:], axis=2)
    d = np.sqrt(D2)
    return d


def compute_angular_distance(V, Ts, Tsnorm):
    """
    Compute distances based on angular (ANF) metric.
    """

    Vnorm = np.linalg.norm(V, axis=1)  # [:,np.newaxis])
    # Vnorm2 = np.sum(V**2, axis=1) #[:,np.newaxis])
    pescalar = np.sum(V[:, np.newaxis, :] * Ts, axis=2)  # np.inner(self.V[:], Ts[:,:])
    normalization = Vnorm[:, np.newaxis] * Tsnorm
    NIP = pescalar / normalization
    alpha = np.sqrt(1.0 - NIP**2)
    return alpha


def compute_directional_distance(V, Ts, Tsnorm):
    """
    Compute distances based on directional (DNF) metric.
    """

    d1 = compute_euclidean_distance(V, Ts)
    d2 = compute_angular_distance(V, Ts, Tsnorm)
    d = d1 * d2
    return d


def compute_photoz_mean_routliers(NEIGHBORS, Verr, pdf, Nvalid, zgrid):
    """
    Compute the mean photometric redshift removing outliers
    """

    # Extract distances and redshifts from neighbors
    distances = NEIGHBORS['distance']
    zmatrix = NEIGHBORS['z']
    indices = NEIGHBORS['index']

    # --- Outlier Detection and Weighting ---
    # Calculate mean distance for each sample
    median_absolute_deviation = distances.mean(axis=1)

    # Define the threshold for outlier detection
    threshold = median_absolute_deviation  # Adjust multiplier if needed (e.g., *2)

    # Create a mask for non-outlier distances
    outliers_weights = distances < threshold[:, None]

    # Update the number of valid neighbors per sample
    nneighbors = np.sum(outliers_weights, axis=1)
    cutNneighbors = np.max(nneighbors)  # Maximum number of valid neighbors

    # --- Distance Weighting ---
    # Compute inverse distances for weighting
    inverse_distances = 1.0 / distances

    # Apply the outlier weigh to inverse distances and distances
    inverse_distances = inverse_distances * outliers_weights
    distances = distances * outliers_weights

    # Normalize weights by the sum of inverse distances per sample
    row_sum = inverse_distances.sum(axis=1, keepdims=True)
    wmatrix = inverse_distances / row_sum
    wmatrix = np.nan_to_num(wmatrix)  # Handle potential NaN values from division by zero

    # --- Photometric Redshift Computation ---
    # Compute the weighted mean redshift for each sample
    photozmean = np.sum(zmatrix * wmatrix, axis=1)
    photoz = photozmean

    # --- Redshift Error Computation ---
    # Compute the standard deviation of redshifts (fit error)
    zt = photozmean[:, np.newaxis]  # column vector
    zerrmatrix = (zmatrix - zt) ** 2

    # Compute the error based in the fit and the parameters
    Verrnorm = np.linalg.norm(Verr, axis=1)
    photozerr_fit = np.sqrt(np.sum(zerrmatrix * wmatrix, axis=1))
    photozerr_param = np.std(NEIGHBORS['z'], axis=1)

    # Combine errors to calculate the total redshift error
    photozerr = np.sqrt(photozerr_param**2 + photozerr_fit**2)

    # --- PDF Computation Setup ---
    # Select the top Nneighbors redshifts and weights for PDF computation
    zpdf = zmatrix[:, :cutNneighbors]
    wpdf = wmatrix[:, :cutNneighbors]

    # Update NEIGHBORS array to include only the top Nneighbors
    NEIGHBORS = NEIGHBORS[:, :cutNneighbors]

    if pdf:
        Vpdf = compute_pdfs(zpdf, wpdf, pdf, Nvalid, zgrid)
    else:  # pragma: no cover
        Vpdf = None

    return photoz, photozerr, photozerr_param, photozerr_fit, nneighbors, Vpdf, NEIGHBORS


def compute_photoz_fit(NEIGHBORS, V, Verr, T, z, fit, photoz, photozerr, photozerr_param, photozerr_fit, pdf, zgrid):
    """
    Compute the photometric redshift fit by iteratively removing outliers.
    """
    # Initialize output parameters
    Nvalid = V.shape[0]
    nfilters = V.shape[1]
    Ntrain = T.shape[0]

    fitIterations = 4
    badtag = 99

    rss = badtag * np.zeros(Nvalid, dtype='double')
    photoz = photoz
    photozerr = photozerr
    photozerr_param = photozerr_param
    photozerr_fit = photozerr_fit
    nneighbors = np.zeros(Nvalid, dtype='double')
    C = np.zeros((Nvalid, nfilters + 1), dtype='double')

    # Increase dimensionality of validation and training data for offsets in fit
    Te = np.hstack([T, np.ones((Ntrain, 1), dtype='double')])
    Ve = np.hstack([V, np.ones((Nvalid, 1), dtype='double')])

    # Loop over all validation samples
    for i in range(0, Nvalid):
        NEIGHBORSs = NEIGHBORS[i]  # Get neighbors for the current sample
        nneighbors[i] = len(NEIGHBORSs)
        # if nneighbors[i] < nfilters:
        #       continue

        # Perform iterative fitting
        for h in range(0, fitIterations):
            # Build the design matrix (A) and target vector (B) for the neighbors
            A = Te[NEIGHBORSs['index']]
            B = z[NEIGHBORSs['index']]

            # Solve the least squares problem
            X = np.linalg.lstsq(A, B, rcond=-1)
            residuals = B - np.dot(A, X[0])  # Compute residuals

            # Identify outliers using a 3-sigma threshold
            abs_residuals = np.abs(residuals)
            sigma3 = 3.0 * np.mean(abs_residuals)
            selection = (abs_residuals < sigma3)

            # Update the number of selected neighbors
            nsel = np.sum(selection)

            # If enough neighbors remain, update NEIGHBORSs; otherwise, stop iteration
            if nsel < nfilters:  # pragma: no cover
                break
            NEIGHBORSs = NEIGHBORSs[selection]
            nneighbors[i] = nsel

            # Save the solution vector
        C[i] = X[0]

        # Compute the photometric redshift fit for the current sample
        photoz[i] = np.inner(X[0], Ve[i])
        rss[i] = np.sum(X[1]**2)

        # Calculate error metrics
        photozerr_param = np.sqrt(np.sum((C[:, :-1] * Verr) ** 2, axis=1))
        photozerr_fit = np.sqrt(rss / (nneighbors - nfilters))
        photozerr_neig = np.std(NEIGHBORS['z'], axis=1)
        photozerr = np.sqrt(photozerr_param**2 + photozerr_fit**2 + photozerr_neig**2)

    Vpdf = None
    if pdf:
        Vpdf = compute_pdfs_fit(photoz, photozerr, zgrid)

    return photoz, photozerr, photozerr_param, photozerr_fit, nneighbors, C, Vpdf


def compute_pdfs(zpdf, wpdf, pdf, Nvalid, zgrid):
    """
    Compute the PDFs from neighbor redshifts and weights

    Parameters:
    - zpdf: (Nvalid, Nneighbors) array with redshift values of neighbors.
    - wpdf: (Nvalid, Nneighbors) array with corresponding weights.
    - pdf: bool, if True, compute PDFs.
    - Nvalid: int, number of galaxies.
    - zgrid: (Nz,) array, redshift grid.

    Returns:
    - Vpdf: (Nvalid, Nz) array with probability distributions.
    """
    if not pdf:
        return np.zeros((Nvalid, len(zgrid)))

    Nz = len(zgrid)
    Vpdf = np.zeros((Nvalid, Nz), dtype='double')

    # Select only the first 5 neighbors
    zpdf_top5 = zpdf[:, :5]
    wpdf_top5 = wpdf[:, :5]

    # Expand dimensions to facilitate comparison with the grid
    zpdf_exp = zpdf_top5[:, :, np.newaxis]  # (Nvalid, Nneighbors, 1)
    zgrid_exp = zgrid[np.newaxis, np.newaxis, :]  # (1, 1, Nz)

    # Create a weight matrix based on proximity to grid points
    weights = np.exp(-((zpdf_exp - zgrid_exp) ** 2) / (2 * (0.05 ** 2)))  # Gaussian with sigma=0.05
    weights *= wpdf_top5[:, :, np.newaxis]  # Apply the original weights

    # Sum the weights for each grid point
    Vpdf = weights.sum(axis=1)

    # Normalize each row to sum to 1 (correct PDFs)
    Vpdf /= Vpdf.sum(axis=1, keepdims=True) + 1e-12  # Avoid division by zero

    return Vpdf


def compute_pdfs_fit(photoz, photozerr, zgrid):
    """
    Computed the gaussian PDFs for the objects

    Parameters:
    - photoz : z mean values
    - photozzerr : zerr values
    - zgrid: grid

    Return:
    ---------
    pdfs : np.ndarray

    """
    z = np.asarray(photoz)[:, None]
    zerr = np.asarray(photozerr)[:, None]

    # Direct formula of the Gaussian
    norm_factor = 1 / (np.sqrt(2 * np.pi) * zerr)
    exponent = -0.5 * ((zgrid - z) / zerr) ** 2
    pdfs = norm_factor * np.exp(exponent)

    # Normalization using numerical integration
    pdfs /= np.trapz(pdfs, zgrid, axis=1)[:, None]

    return pdfs
