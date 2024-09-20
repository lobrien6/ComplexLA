from paltas.Configs.config_handler import ConfigHandler
import matplotlib.pyplot as plt

def PaltasModelWoP(path):
    perturber_sample = ConfigHandler(path+'Configs/perturber_sample_wop_config.py')
    im,metadata = perturber_sample.draw_image(new_sample=False)

    return im,metadata

def PaltasModelWP(path):
    test_sample_i = ConfigHandler(path+'Configs/perturber_sample_wp_config.py')
    im,metadata = test_sample_i.draw_image(new_sample=False)

    return im,metadata

def PaltasModelWPL(path):
    test_sample_i = ConfigHandler(path+'Configs/perturber_sample_wpl_config.py')
    im,metadata = test_sample_i.draw_image(new_sample=False)

    return im,metadata