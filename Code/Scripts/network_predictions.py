import os
from deep_lens_modeling import network_predictions

def Predictions(sample_num, path, config_path, image_path):
    """ Generates network predictions from images for the wop, wp, and wpl cases
    
    Parameters
    ----------
    sample_num : int
        Number of mock sample lenses.
    path : str
        Main notebook path.
    config_path : str
        File path of the config files for the wop, wp, and wpl cases.
    image_path : str
        File path of the image files for the wop, wp, and wpl cases.

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

    #Assigning paths to variables
    config_file_wop = path+config_path+'/perturber_sample_wop_config.py'
    image_folder_wop = path+image_path+'/perturber_sample_wop_image'

    config_file_wp = path+config_path+'/perturber_sample_wp_config.py'
    image_folder_wp = path+image_path+'/perturber_sample_wp_image'

    config_file_wpl = path+config_path+'/perturber_sample_wpl_config.py'
    image_folder_wpl = path+image_path+'/perturber_sample_wpl_image'

    #Generating the images
    os.system(f'python /Users/Logan/AppData/Local/Programs/Python/Python312/Lib/site-packages/paltas/paltas/generate.py {config_file_wop} {image_folder_wop} --n {sample_num-1} --tf_record')
    os.system(f'python /Users/Logan/AppData/Local/Programs/Python/Python312/Lib/site-packages/paltas/paltas/generate.py {config_file_wp} {image_folder_wp} --n {sample_num-1} --tf_record')
    os.system(f'python /Users/Logan/AppData/Local/Programs/Python/Python312/Lib/site-packages/paltas/paltas/generate.py {config_file_wpl} {image_folder_wpl} --n {sample_num-1} --tf_record')

    # Compute predictions for test sets
    path_to_weights = path+'/files/xresnet34_068--14.58.h5'
    path_to_norms = path+'/files/norms.csv'

    learning_params = ['main_deflector_parameters_theta_E','main_deflector_parameters_gamma1',
                       'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
                       'main_deflector_parameters_e1','main_deflector_parameters_e2',
                       'main_deflector_parameters_center_x','main_deflector_parameters_center_y',
                       'source_parameters_center_x','source_parameters_center_y']
    learning_params_names = [r'$\theta_\mathrm{E}$',r'$\gamma_1$',r'$\gamma_2$',r'$\gamma_\mathrm{lens}$',r'$e_1$',
                             r'$e_2$',r'$x_{lens}$',r'$y_{lens}$',r'$x_{src}$',r'$y_{src}$']

    model_predictions = network_predictions.NetworkPredictions(path_to_weights,path_to_norms,
                                                               learning_params,loss_type='diag',model_type='xresnet34',norm_type='lognorm')

    # generate network predictions
    y_test_wop, y_pred_wop, std_pred_wop, prec_pred_wop = model_predictions.gen_network_predictions(test_folder=image_folder_wop)
    y_test_wp, y_pred_wp, std_pred_wp, prec_pred_wp = model_predictions.gen_network_predictions(test_folder=image_folder_wp)
    y_test_wpl, y_pred_wpl, std_pred_wpl, prec_pred_wpl = model_predictions.gen_network_predictions(test_folder=image_folder_wpl)

    return y_test_wop, y_pred_wop, std_pred_wop, prec_pred_wop, y_pred_wp, std_pred_wp, prec_pred_wp, y_pred_wpl, std_pred_wpl, prec_pred_wpl