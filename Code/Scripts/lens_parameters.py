import numpy as np
from scipy.stats import norm, truncnorm
import paltas.Sampling.distributions as dist
import pickle
import random

# load in a PSF kernel
from astropy.io import fits
from lenstronomy.Util import kernel_util

psf_fits_file = '/Users/Logan/AppData/Local/Programs/Python/Python312/Lib/site-packages/paltas/datasets/hst_psf/STDPBF_WFC3UV_F814W.fits'

def perturberparameters(sample_num,module_path):
    # load in focus diverse PSF maps
    with fits.open(psf_fits_file) as hdu:
        psf_kernels = hdu[0].data
    psf_kernels = psf_kernels.reshape(-1,101,101)
    psf_kernels[psf_kernels<0] = 0

    # normalize psf_kernels to sum to 1
    psf_sums = np.sum(psf_kernels,axis=(1,2))
    psf_sums = psf_sums.reshape(-1,1,1)
    normalized_psfs = psf_kernels/psf_sums

    # pick random weights to create PSF
    weights = np.random.uniform(size=np.shape(normalized_psfs)[0])
    weights /= np.sum(weights)
    weighted_sum = np.sum(weights.reshape(len(weights),1,1) * normalized_psfs,axis=0)
    np.save(module_path+'/Files/weighted_sum.npy', weighted_sum)

    #Defining parameters
    full_dict = []
    for i in range(sample_num):
        cross_object_1 = dist.DuplicateScatter(dist=norm(loc=0,scale=0.07).rvs,scatter=0.005)()
        cross_object_2 = dist.DuplicateScatter(dist=norm(loc=0,scale=0.07).rvs,scatter=0.005)()
        cross_object_3 = dist.DuplicateXY(x_dist=norm(loc=0.0,scale=0.1).rvs,
                    y_dist=norm(loc=0.0,scale=0.1).rvs)()
        cross_object_4 = dist.RedshiftsPointSource(z_lens_min=0,z_lens_mean=0.5,z_lens_std=0.2,
    		z_source_min=0,z_source_mean=2,z_source_std=0.4)()

        param_dict = {
    	'z_lens': cross_object_4[0],
    	'gamma_md': truncnorm(-(2.05/.1),np.inf,loc=2.05,scale=0.1).rvs(),
    	'theta_E_md': truncnorm(-(.7/.08),np.inf,loc=0.7,scale=0.08).rvs(),
    	'e1_md': norm(loc=0,scale=0.2).rvs(),
    	'e2_md': norm(loc=0,scale=0.2).rvs(),
    	'center_x_md': cross_object_1[0],
    	'center_y_md': cross_object_2[0],
    	'gamma1_md': norm(loc=0,scale=0.12).rvs(),
    	'gamma2_md': norm(loc=0,scale=0.12).rvs(),
    	'p_center_x': truncnorm(-1.2,1.25,loc=0,scale=1).rvs(),
    	'p_center_y': truncnorm(-1.2,1.25,loc=0,scale=1).rvs(),
    	'z_source': cross_object_4[2],
    	'mag_app_source': truncnorm(-3./2.,3./2.,loc=23.5,scale=7./3.).rvs(),
    	'R_sersic_source': truncnorm(-(5./8.),np.inf,loc=0.5,scale=0.5).rvs(),
    	'n_sersic_source': truncnorm(-1.25,np.inf,loc=3.,scale=1.).rvs(),
    	'e1_source': truncnorm(-2.5,2.5,loc=0,scale=0.18).rvs(),
    	'e2_source': truncnorm(-2.5,2.5,loc=0,scale=0.18).rvs(),
    	'center_x_source': cross_object_3[0],
    	'center_y_source': cross_object_3[1],
        'z_lens_light': cross_object_4[1],
    	'mag_app_light': truncnorm(-3./2.,3./2.,loc=20,scale=2.).rvs(),
    	'R_sersic_light': truncnorm(-(1./.8),np.inf,loc=1.0,scale=0.8).rvs(),
    	'n_sersic_light': truncnorm(-1.25,np.inf,loc=3.,scale=2.).rvs(),
    	'e1_light': truncnorm(-2.5,2.5,loc=0,scale=0.18).rvs(),
    	'e2_light': truncnorm(-2.5,2.5,loc=0,scale=0.18).rvs(),
    	'z_point_source': cross_object_4[3],
    	'x_point_source': cross_object_3[2],
    	'y_point_source': cross_object_3[3],
    	'mag_app_point_source': truncnorm(-3./2.,3./2.,loc=22.,scale=2.).rvs(),
    	#'mag_pert': dist.MultipleValues(dist=truncnorm(-1/0.3,np.inf,1,0.3).rvs,num=10)(),
        } 
        
        full_dict.append(list(param_dict.values()))
    return full_dict