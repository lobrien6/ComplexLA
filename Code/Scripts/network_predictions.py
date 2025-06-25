import os
import sys
from deep_lens_modeling import network_predictions
import time

#define a function for a general case of n situations
def Predictions(config_file, image_folder, sample_num=1):
    """ Generates network predictions from images
    
    Parameters
    ----------
    config_file : str
        File location for the config file being used.
    image_path : str
        Image folder location where the images will be stored.
    sample_num : int
        Number of mock sample lenses. Default 1

    Returns
    -------
    y_test_wop : float
        Truth values of lens parameters.
    y_pred_wop, y_pred_wp, y_pred_wpl : float
        Prediction values for the wop, wp, and wpl cases.
    std_pred_wop, std_pred_wp, std_pred_wpl : float
        Standard deviation of the prediction values for the wop, wp and wpl cases.
    prec_pred_wop, prec_pred_wp, prec_pred_wpl : array
        Predicted precision matrix for the wop, wp and wpl cases.

    """

    #Generating the images
    os.system(f'python /Users/Logan/AppData/Local/Programs/Python/Python312/Lib/site-packages/paltas/paltas/generate.py {config_file} {image_folder} --n {sample_num-1} --tf_record')

    # Compute predictions for test sets
    path_to_weights = '../files/xresnet34_068--14.58.h5'
    path_to_norms = '../files/norms.csv'

    learning_params = ['main_deflector_parameters_theta_E','main_deflector_parameters_gamma1',
                       'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
                       'main_deflector_parameters_e1','main_deflector_parameters_e2',
                       'main_deflector_parameters_center_x','main_deflector_parameters_center_y',
                       'source_parameters_center_x','source_parameters_center_y']

    model_predictions = network_predictions.NetworkPredictions(path_to_weights,path_to_norms,
                                                               learning_params,loss_type='diag',model_type='xresnet34',norm_type='lognorm')

    # generate network predictions
    y_test, y_pred, std_pred, prec_pred = model_predictions.gen_network_predictions(image_folder)

    return y_test, y_pred, std_pred, prec_pred