import os
import numpy as np
import paltas.Sampling.distributions as dist
from paltas.MainDeflector.perturbers import MainDeflectorWithPerturber
from paltas.Sources.sersic import SingleSersicSource
from paltas.Sources.sersic import SersicPerturber
from paltas.PointSource.single_point_source import SinglePointSource
import pickle
from paltas.Utils.cosmology_utils import get_cosmology, apparent_to_absolute

output_ab_zeropoint = 25.139883
catalog = False
path = os.path.abspath(os.path.join('..'))+'/'

kwargs_numerics = {'supersampling_factor':1}

numpix = 80

compute_caustic_area = True

# load PSF map from lenstronomy fitting results
f = open(path+'Files/PSJ1606-2333_results.txt','rb')
multi_band_list,_,_,_ = pickle.load(f)
psf_map = multi_band_list[0][1]['kernel_point_source']

cosmology_params = { 'cosmology_name': 'planck18' }
cosmo = get_cosmology(cosmology_params)

config_dict = {
	'main_deflector':{
		'class': MainDeflectorWithPerturber,
		'parameters':{
			'z_lens':0.500,
			'gamma': 1.9156268331,
			'theta_E': 0.6997810337,
			'e1': -0.2601939426,
			'e2': -0.1282661622,
			'center_x': -0.0020882773,
			'center_y': 0.0492052357,
			'gamma1': 0.0165289103,
			'gamma2': 0.0855005994,
			'ra_0': 0.0000000000,
			'dec_0': 0.0000000000,
            'p_theta_E': 0.07080846591210048,
            'p_center_x': -0.28556169454510827,
            'p_center_y': -1.1317849659620933
		}
	},
	'source':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':2.000,
			'mag_app': 21.0791555025,
			'output_ab_zeropoint': output_ab_zeropoint,
			'R_sersic': 0.6611145558,
			'n_sersic': 3.9720031048,
			'e1,e2': dist.EllipticitiesTranslation(q_dist=1,phi_dist=0),
			'center_x': 0.0332554221,
			'center_y': 0.0272838146
		}
	},
	'lens_light':{
		'class': SersicPerturber,
		'parameters':{
			'z_source':0.500,
			'mag_app': 20.6540853862,
			'output_ab_zeropoint': output_ab_zeropoint,
			'R_sersic': 1.5221246244,
			'n_sersic': 5.9960772470,
			'e1': -0.2179993341,
			'e2': -0.1395888312,
			'center_x': 0.0346518914,
			'center_y': 0.0257911257,
            'p_z_source':0.500,
            'p_amp': 29.9385619019969,
            'p_output_ab_zeropoint': output_ab_zeropoint,
            'p_R_sersic': 0.07592075244357199,
            'p_n_sersic': 1.8502723946132715,
            'p_center_x': -0.28556169454510827,
            'p_center_y': -1.1317849659620933
		}
	},
	'point_source':{
		'class': SinglePointSource,
		'parameters':{
			'z_point_source':2.000,
			'mag_app': 20.0043560243,
			'output_ab_zeropoint': output_ab_zeropoint,
			'x_point_source': 0.0332601073,
			'y_point_source': 0.0272852269,
			'compute_time_delays':False,
			'mag_pert':[1.094,0.721,0.957,1.228]
		}
	},
	'lens_equation_solver':{
		'parameters':{
			'search_window': 3.200,
			'min_distance': 0.040
		}
	},
	'cosmology':{
		'parameters':cosmology_params
	},
	'psf':{
		'parameters':{
			'psf_type':'PIXEL',
			'kernel_point_source':psf_map,
			'point_source_supersampling_factor':1
		}
	},
	'pixel_grid':{
		'parameters':{
			'ra_at_xy_0': 1.6000000000,
			'dec_at_xy_0': 1.5600000000,
			'transform_pix2angle':np.array([[-0.04000000,0.00000000],
				[0.00000000,-0.04000000]])
		}
	},
	'detector':{
		'parameters':{
			'pixel_scale':0.04,
			'ccd_gain':1.5,
			'read_noise':3.0,
			'magnitude_zero_point':output_ab_zeropoint,
			'exposure_time':1366.566,
			'sky_brightness':21.936,
			'num_exposures':1,
			'background_noise':None
		}
	}
}