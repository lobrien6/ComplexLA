#from paltas.Configs.config_handler import ConfigHandler
from paltas.Configs.config_handler import ConfigHandler

def PaltasModel(config_file):
    """ Calls the ConfigHandler script to generate the im and metadata variables from the config file

    Parameters
    ----------
    config_file : str
        Config file containing lens parameters.

    Returns
    -------
    im : np.array
        Numpy array of the generated image.
    metadata : dict
        Dictionary of corresponding sampled values.

    """
    perturber_sample = ConfigHandler(config_file)
    im,metadata = perturber_sample.draw_image(new_sample=False)

    return im,metadata

"""
def PaltasModelWoS(path):
     Calls the ConfigHandler script to generate the im and metadata variables from the WoS config file

    Parameters
    ----------
    path : str
        File path for the main directory.
    sample_num : int
        Number of mock sample lenses.

    Returns
    -------
    im : np.array
        Numpy array of the generated image.
    metadata : dict
        Dictionary of corresponding sampled values.

    
    test_sample_i = ConfigHandler('../Configs/substructure_WoS_config.py')
    n = 0
    im_list = []
    metadata_list = []
    #while n<sample_num:
    for i in range(0,100):
        im,metadata = test_sample_i.draw_image(new_sample=True)
        im_list.append(im)
        metadata_list.append(metadata)
        #n = n+1

    return im_list,metadata_list

def PaltasModelWS(path,sample_num):
     Calls the ConfigHandler script to generate the im and metadata variables from the WS config file

    Parameters
    ----------
    path : str
        File path for the main directory.
    sample_num : int
        Number of mock sample lenses.

    Returns
    -------
    im : np.array
        Numpy array of the generated image.
    metadata : dict
        Dictionary of corresponding sampled values.

    
    test_sample_i = ConfigHandler(path+'Configs/substructure_WS_config.py')
    n = 0
    im_list = []
    metadata_list = []
    #while n<sample_num:
    for i in range(0,sample_num-1):
        im,metadata = test_sample_i.draw_image(new_sample=True)
        im_list.append(im)
        metadata_list.append(metadata)
        #n = n+1

    return im_list,metadata_list
"""