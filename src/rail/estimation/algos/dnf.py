"""
Implementation of the DNF algorithm
See https://academic.oup.com/mnras/article/459/3/3078/2595234
for more details
"""

import math
import numpy as np
import qp
from sklearn import neighbors
from scipy.stats import chi2
from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatEstimator, CatInformer
from rail.core.common_params import SHARED_PARAMS
from sklearn import neighbors

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
                          nondetect_val=SHARED_PARAMS)

    def __init__(self, args, comm=None):
        """ Constructor
        Do CatInformer specific initialization, then check on bands """ 
        CatInformer.__init__(self, args, comm=comm)
        
    def run(self):
        if self.config.hdf5_groupname:
            training_data = self.get_data('input')[self.config.hdf5_groupname]
        else:
            training_data = self.get_data('input')
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

        self.model = dict(train_mag=mag_data, train_err=mag_err, truez=specz, clf=clf, train_norm = Tnorm
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
                          # radius=SHARED_PARAMS,
                          # num_neighbors=SHARED_PARAMS,
                          # seed=Param(int, 66, msg="random seed used in selection mode"),
                          # ppf_value=Param(float, 0.68, msg="PPF value used in Mahalanobis distance"),
                          selection_mode=Param(int, 1, msg="select which mode to choose the redshift estimate:"
                                               "0: ENF, 1: ANF, 2: DNF"), # Habra que detallar mas que significa cada caso
                          min_n=Param(int, 25, msg="minimum number of training galaxies to use"), # ¿Necesitamos un minimo de galaxas?
                          # min_thresh=Param(float, 0.0001, msg="minimum threshold cutoff"),  # Esto no tengo claro para que es
                          # min_dist=Param(float, 0.0001, msg="minimum Mahalanobis distance"), # Esto supongo que lo podemos eliminar
                          bad_redshift_val=Param(float, 99., msg="redshift to assign bad redshifts"),
                          bad_redshift_err=Param(float, 10., msg="Gauss error width to assign to bad redshifts") # Habria que cambiar el msg ¿?
                          )

                          
    def __init__(self, args, comm=None):
        """ Constructor:
        Do Estimator specific initialization 
        """
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
        self.clf = self.model['clf']
        self.Tnorm = self.model['train_norm']        
        
    def _process_chunk(self, start, end, data, first): # hay que dejar este nombre porque es el que ponen todos
        
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
                metric='ANF', 
                fit=True, 
                pdf=True, 
                Nneighbors=80, 
                presel=500
            )

        ancil_dictionary = dict()
        qp_dnf = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=pdfs))


        ancil_dictionary.update(DNF_Z=photoz,photozerr=photozerr,
                           photozerr_param=photozerr_param,photozerr_fit=photozerr_fit,DNF_ZN=z1,
                           nneighbors=nneighbors,de1=de1,d1=d1,id1=id1) #, C=C, Vpdf=Vpdf
        qp_dnf.set_ancil(ancil_dictionary)
        
        self._do_chunk_output(qp_dnf, start, end, first)
     

def dnf(T,z,V,Verr,zbins,pdf=True,bound=False,radius=2.0,Nneighbors=80,magflux='mag',metric='DNF',coeff=True):
    """
    def dnf(T,z,V,Verr,zbins,pdf=True,bound=True,radius=2.0,Nneighbors=80,magflux='flux',metric='DNF')
    
    Computes the photo-z by Directional Neighborhood Fit (Copyright (C) 2015, Juan de Vicente)
  
    Input parameters:
      T: 2-dimensional array with magnitudes of the training sample
      z: 1-dimensional array with the spectroscopic redshift of the training sample
      V: 2-dimensional array with magnitudes of the photometric sample
      Verr: 2-dimensional array with magnitudes errors of the photometric sample
      zbins: 1-dimensional numpy array with redshift bins for photo-z PDFs
      pdf: True for pdf computation
      bound: True to ensure photometric redshifts remain inside the training redshift range.
      radius: Euclidean-radius for euclidean neighbors preselection to speed up and avoid outliers.
      Galaxies without neighbors inside this radius are tagged with photoz_err=99.0 and should be removed from statistical analysis.
      Nneighbors: Number of neighbors to construct the photo-z hyperplane predictor (number of neighbors for the fit)
      magflux: 'mag' | 'flux'
      metric: 'ENF' | 'ANF' | 'DNF' (euclidean|angular|directional)
      coeff: True for returning the fitting coeffients 
    Return:
      photoz: 1-dimesional dnf-photoz array for the photometric sample
      photozerr: 1-dimensional photoz error estimation array. Takes the value 99.0 for galaxies with unreliable photo-z
      photozerr_param: photo-z uncertainty due to photometry uncertainty
      photozerr_fit: photo-z uncertainty coming from fit residual
      Vpdf: 2-dimensional photo-z PDFs array when pdf==1, 0 when pdf==0
      z1: 1-dimesional photo-z array to be used for histogramming N(z). When computing n(z) per bin, use dnf-photoz for galaxy classification in bins and z1 for n(z) histogramming.
      nneighbors: 1-dimensional array with the number of neighbors used in the photo-z estimation for each galaxy
      de1: 1-dimensional array with the Euclidean magnitude distance to the nearest neighbor for each galaxy
      d1: 1-dimensional array with the metric-distance to the nearest neighbor for each galaxy
      id1: 1-dimensional array with the position of the nearest-neighbor for each galaxy (with metric-distance)
      C: C=fit-coeficients when coeff==True, otherwise C=0
    """      
    
    nfilters=T.shape[1]
    Nvalid=V.shape[0]
    Ntrain=T.shape[0]
    
    if Ntrain>4000: #2000
        Nneighbors_presel=4000 #2000
    else:
        Nneighbors_presel=Ntrain
        
    #neighbor preselection within radius mag-euclidean metric
    clf=neighbors.KNeighborsRegressor(n_neighbors=Nneighbors_presel)
    clf.fit(T,z)  #multimagnitude-redshift association from the training sample
    #photoz=clf.predict(V)
    Vdistances,Vneighbors= clf.kneighbors(V,n_neighbors=Nneighbors_presel)  #neighbors computation
    de1=Vdistances[:,0]  #euclidean nearest-neighbor euclidean distance
    d1=Vdistances[:,0]  #nearest-neighbor metric distance (to be properly overwritten latter)
    Vclosest=Vneighbors[:,0]
    id1=Vclosest
    

    #In case of giving fluxes compute closest distance in magnitude
    if magflux=='flux':
        for i in range(Nvalid):
            magV=-2.5*np.log10(V[i])
            magT=-2.5*np.log10(T[Vclosest[i]])
            diff=magV-magT
            dmag=np.sqrt(np.inner(diff,diff))
            de1[i]=dmag
        
   
    #output declaration
    photoz=np.zeros(Nvalid,dtype='double')
    z1=np.zeros(Nvalid,dtype='double')
    photozerr=np.zeros(Nvalid,dtype='double')
    photozerr_param=np.zeros(Nvalid,dtype='double')
    photozerr_fit=np.zeros(Nvalid,dtype='double')
    nneighbors=np.zeros(Nvalid,dtype='double')
      
    #auxiliary variable declaration
    pescalar=np.zeros(Ntrain,dtype='double')
    D2=np.zeros(Ntrain,dtype='double')
    Tnorm=np.zeros(Ntrain,dtype='double')
    Tnorm2=np.zeros(Ntrain,dtype='double')    
    #max and min training photo-zs
    maxz=np.max(z)
    minz=np.min(z)

    if coeff==True:
         C=np.zeros((Nvalid,nfilters+1),dtype='double')
    else:
         C=0

    ########pdf bins##########
    nbins=len(zbins)-1 
    bincenter=(np.double(zbins[1:])+np.double(zbins[:-1]))/2.0
    if pdf==True:
     Vpdf=np.zeros((Nvalid,nbins),dtype='double')
    else:
        Vpdf=0

    #training flux/mag norm pre-calculation
    for t,i in zip(T,range(Ntrain)):
     Tnorm[i]=np.linalg.norm(t)
     Tnorm2[i]=np.inner(t,t)

    #for offset of the fit
    Te=np.ones((Ntrain,nfilters+1),dtype='double')  
    Te[:,:-1]=T
    Ve=np.ones((Nvalid,nfilters+1),dtype='double')  
    Ve[:,:-1]=V

    #to computed neighbor pre-selection within mag radius in case of fluxes  
    ratiomax=np.power(10.0,radius/2.5)
    
    #photo-z computation
    for i in range(0,Nvalid):
        #neighbors pre-selection within mag radius
        if magflux=='mag':
              selection=Vdistances[i]<radius
        elif magflux=='flux':
              selection=np.ones(Nneighbors_presel,dtype='bool') 
              for j in range(0,nfilters):
                  ratio1=V[i][j]/T[Vneighbors[i],j]
                  ratio2=T[Vneighbors[i],j]/V[i][j]
                  selectionaux=np.logical_and(ratio1<ratiomax,ratio2<ratiomax) 
                  selection=np.logical_and(selection,selectionaux)

        Vneighbo=Vneighbors[i][selection]
        Vdistanc=Vdistances[i][selection]
        
        Eneighbors=Vneighbo.size #euclidean neighbors within mag radius
        if Eneighbors==0:  #probably bad photo-zs
            nneighbors[i]=0
            photozerr[i]=99.0
            photozerr_param[i]=99.0
            photozerr_fit[i]=99.0
            photoz[i]=z[Vclosest[i]]
            continue

        #declaration of auxiliary array to store neighbors features during photo-z computation 
        NEIGHBORS=np.zeros(Eneighbors,dtype=[('pos','i4'),('distance','f8'),('z_true','f8')])
        #copy of euclidean preselection previously computed  
        NEIGHBORS['z_true']=z[Vneighbo] #photo-z
        NEIGHBORS[:]['pos']=Vneighbo 
        Ts=T[Vneighbo] #flux/mag  
        
        if metric=='ENF':
             D=V[i]-Ts
             Dsquare=D*D
             D2=np.sum(Dsquare,axis=1)
             NEIGHBORS['distance']=D2
        elif metric=='ANF':
             Tsnorm=Tnorm[Vneighbo] 
             Vnorm=np.linalg.norm(V[i])
             pescalar=np.inner(V[i],Ts)
             normalization=Vnorm*Tsnorm
             NIP=pescalar/normalization
             alpha2=1-NIP*NIP
             NEIGHBORS['distance']=alpha2
        elif metric=='DNF':  #ENF*ANF
             D=V[i]-Ts
             Dsquare=D*D
             D2=np.sum(Dsquare,axis=1)
             

             Tsnorm=Tnorm[Vneighbo] 
             Vnorm=np.linalg.norm(V[i])
             pescalar=np.inner(V[i],Ts)
             normalization=Vnorm*Tsnorm
             NIP=pescalar/normalization
             alpha2=1-NIP*NIP

             D2norm=D2/(Vnorm*Vnorm) #normalized distance to do it more interpretable
             NEIGHBORS['distance']=alpha2*D2norm 
     
        NEIGHBORSsort=np.sort(NEIGHBORS,order='distance')
        z1[i]=NEIGHBORSsort[0]['z_true']
        d1[i]=NEIGHBORSsort[0]['distance']
        id1[i]=NEIGHBORSsort[0]['pos']

        #if the galaxy is found in the training sample
        if NEIGHBORSsort[0]['distance']==0.0:
            #photoz[i]=NEIGHBORSsort[0]['z_true']
            if Eneighbors>1:
                z1[i]=NEIGHBORSsort[1]['z_true']
            #if nneighbors[i]==0:
            #   photozerr[i]=0.001
            #    photozerr_param[i]=0.001
            #    photozerr_fit[i]=0.0
            #else:
            #    photozerr_fit[i]=NEIGHBORSsort['z_true'].std()
            #    photozerr_param[i]=0.0
            #    photozerr[i]=photozerr_param[i]

            if pdf==True:
                zdist=photoz[i] #-0.01 #-residuals  #for p
                hist=np.double(np.histogram(zdist,zbins)[0])
                Vpdf[i]=hist/np.sum(hist)
            #continue
        
        #limiting the number of neighbors to Nneighbors parameter
        if Eneighbors>Nneighbors:
                NEIGHBORSsort=NEIGHBORSsort[0:Nneighbors]  #from 1 in case to exclude the own galaxy
                neigh=Nneighbors
        else:
                neigh=Eneighbors
        
        nneighbors[i]=neigh
        
            
        #nearest neighbor photo-z computation when few neighbors are found (fitting is not a good option)
        if neigh<10: #<30 
            photoz[i]=np.inner(NEIGHBORSsort['z_true'],1.0/NEIGHBORSsort['distance'])/np.sum(1.0/NEIGHBORSsort['distance']) #weighted by distance
            if neigh==1:
                photozerr_param[i]=0.1
                photozerr[i]=0.1
            else:
                #photozerr[i]=np.std(NEIGHBORSsort['z_true'])
                photozerr_fit[i]=np.sqrt(np.inner((NEIGHBORSsort['z_true']-photoz[i])**2,1.0/NEIGHBORSsort['distance'])/np.sum(1.0/NEIGHBORSsort['distance']))  #weighted by distance
                photozerr[i]=photozerr_fit[i]
            if pdf==True:
                        if photozerr[i]==0:
                            s=1
                        else:
                            s=photozerr[i]
                        zdist=np.random.normal(photoz[i],s,neigh)
                        #zdist=NEIGHBORSsort['z_true']
                        hist=np.double(np.histogram(zdist,zbins)[0])
                        sumhist=np.sum(hist)
                        if sumhist==0.0:
                            Vpdf[i][:]=0.0
                        else:
                            Vpdf[i]=hist/sumhist 
            continue

    
        #Fitting when large number of neighbors exists. Removing outliers by several iterations  
        fititerations=4
        for h in range(0,fititerations):
            A=Te[NEIGHBORSsort['pos']]  
            B=z[NEIGHBORSsort['pos']]
            X=np.linalg.lstsq(A,B)
            residuals=B-np.dot(A,X[0])
    
        
            if h==0:  #PDFs computation
                photoz[i]=np.inner(X[0],Ve[i])
            
                if pdf==True:
                    zdist=photoz[i]+residuals
                    #zdist=NEIGHBORSsort['z_true']
                    hist=np.double(np.histogram(zdist,zbins)[0])
                    sumhist=np.sum(hist)
                    if sumhist==0.0:
                        Vpdf[i][:]=0.0
                    else:
                        Vpdf[i]=hist/sumhist 
                    
        
            #outlayers are removed after each iteration 
            absresiduals=np.abs(residuals)      
            sigma3=3.0*np.mean(absresiduals)
            selection=(absresiduals<sigma3)
            #NEIGHBORSsort=NEIGHBORSsort[selection]
            
            
            nsel=np.sum(selection)
            nneighbors[i]=nsel
            if nsel>10:
                NEIGHBORSsort=NEIGHBORSsort[selection]
            else:
                break
            
        C[i]=X[0]
        photoz[i]=np.inner(X[0],Ve[i])
        neig=NEIGHBORSsort.shape[0]
        nneighbors[i]=neig
        if X[1].size!=0:               
            photozerr_param[i]=np.sqrt(np.inner(np.abs(X[0][:-1])*Verr[i],np.abs(X[0][:-1])*Verr[i]))
            photozerr_fit[i]=np.sqrt(X[1]/neig)
        else:
            photoz[i]=np.inner(NEIGHBORSsort['z_true'],1.0/NEIGHBORSsort['distance'])/np.sum(1.0/NEIGHBORSsort['distance']) #weighted by distance
            if neigh==1:
                photozerr_param[i]=0.1
            else:
                #photozerr[i]=np.std(NEIGHBORSsort['z_true'])
                photozerr_fit[i]=np.sqrt(np.inner((NEIGHBORSsort['z_true']-photoz[i])**2,1.0/NEIGHBORSsort['distance'])/np.sum(1.0/NEIGHBORSsort['distance']))  #weighted by distance
                #photozerr[i]=photozerr_fit[i]

                #photozerr_fit[i]=0.01

        #photoz bound
        if bound==True:
            if photoz[i]< minz or photoz[i]>maxz:
               photozerr_fit[i]+=np.abs(photoz[i]-NEIGHBORSsort[0]['z_true'])
               photoz[i]=NEIGHBORSsort[0]['z_true']
      
        
                    

        percent=np.double(100*i)/Nvalid
        
        if i % 1000 ==1:
            print('progress: ',percent,'%')

    photozerr=np.sqrt(photozerr_param**2+photozerr_fit**2)
    
    return photoz,photozerr,photozerr_param,photozerr_fit,Vpdf,z1,nneighbors,de1,d1,id1,C 

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
    
    

