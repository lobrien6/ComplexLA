# Includes a PEMD deflector with external shear, and Sersic sources. 

import numpy as np
from scipy.stats import norm
from paltas.MainDeflector.simple_deflectors import PEMDShear
from paltas.Sources.sersic import SingleSersicSource
from paltas.PointSource.single_point_source import SinglePointSource
from paltas.Substructure.subhalos_dg19 import SubhalosDG19
from lenstronomy.Util import kernel_util

file = '../../Data-Tables/substructure_parameters_catalog.csv'

# Define the numerics kwargs.
kwargs_numerics = {'supersampling_factor':1}

# This is always the number of pixels for the CCD; size of your cutout 
numpix = 80

# Define arguments that will be used multiple times
output_ab_zeropoint = 25.1152
catalog = True

# other flags (all are booleans) you can use: subtract_lens, subtract_source ,
# doubles_quads_only, no_singles, compute_caustic_area

# this can be None, single int, or a list/np.array/pd.Series of elements
index = None

def draw_psf_kernel():
    weighted_sum = np.load('../Files/weighted_sum.npy')
    return kernel_util.degrade_kernel(weighted_sum,4)

# you're appending root path to each file path you provide!
# so they should be in the same directory
config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'file': file,
		'parameters':{
			'z_lens': 'z_lens',
			'gamma': 'gamma_md',
			'theta_E': 'theta_E_md',
			'e1': 'e1_md', 
			'e2': 'e2_md',
			'center_x': 'center_x_md', 
			'center_y':'center_y_md',
			'gamma1': 'gamma1_md', 
			'gamma2': 'gamma2_md', 
            'M200': 10**13,
			'ra_0':0.0, 'dec_0':0.0
		}
	},
	'lens_light':{
		'class': SingleSersicSource,
		'file': file,
		'parameters':{
			'z_source': 'z_lens_light',
			'mag_app': 'mag_app_light', # LENS APPARENT MAG
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic': 'R_sersic_light',
			'n_sersic': 'n_sersic_light',
			'e1': 'e1_light', 
			'e2': 'e2_light', 
			'center_x': 'center_x_md',
			'center_y': 'center_y_md'
			}
	},
	'source':{
		'class': SingleSersicSource,
		'file': file,
		'parameters':{
			'z_source': 'z_source',
			'mag_app': 'mag_app_source', # SOURCE APPARENT MAG
			'output_ab_zeropoint':output_ab_zeropoint,
            'R_sersic': 'R_sersic_source',
			'n_sersic': 'n_sersic_source',
			'e1': 'e1_source', 
			'e2': 'e2_source', 
			'center_x': 'center_x_source',
			'center_y': 'center_y_source'
		}
	},
    'point_source':{
		'class': SinglePointSource,
		'file': file,
		'parameters':{
			'z_source': 'z_source',
            'z_point_source': 'z_point_source',
			'x_point_source': 'x_point_source',
			'y_point_source': 'y_point_source',
			'mag_app': 'mag_app_point_source', # POINT SOURCE APPARENT MAG
			'output_ab_zeropoint':output_ab_zeropoint,
			'compute_time_delays': False
		}
	},
    'subhalo':{
        'class': SubhalosDG19,
        'file' : file,
        'parameters':{
            'sigma_sub': 'sigma_sub',
            'shmf_plaw_index': 1.9,
            'm_pivot': 1e10,'m_min': 1e7,'m_max': 1e10,
            'c_0': 17,
            'conc_zeta': -0.25,
            'conc_beta': 0.7,
            'conc_m_ref': 1e8,
            'dex_scatter': 0.13,
            'k1':0.0, 'k2':0.0
        }
    },
	'cosmology':{
		'file': None,
		'parameters':{
			'cosmology_name': 'planck18'
		}
	},
    'psf':{
		'file': None,
		'parameters':{
			'psf_type':'PIXEL',
			'kernel_point_source':draw_psf_kernel,
			'point_source_supersampling_factor':1
		}
	},
	'detector':{
		'file': None,
		'parameters':{
			'pixel_scale':0.04,'ccd_gain':1.5,'read_noise':3.0,
			'magnitude_zero_point':output_ab_zeropoint,
			'exposure_time':1400,'sky_brightness':21.9,
			'num_exposures':1,'background_noise':None
		}
	},
    'drizzle':{
        'file': None,
		'parameters':{
        		'supersample_pixel_scale':0.040,'output_pixel_scale':0.040,
        		'wcs_distortion':None,
        		'offset_pattern':[(0,0),(0.5,0.5)],
        		'psf_supersample_factor':1
        }
    }
}