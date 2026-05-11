"""
This module contains functions to call paltas scirpts that will generate test sets of simulated lenses.
Images of the lenses are stored in user defined locations.
"""
from astropy.visualization import AsinhStretch, ImageNormalize
import matplotlib.pyplot as plt
import numpy as np
import os

class TestSetGeneration():
    """
    Class for generation samples of lenses and displaying the images.

    Args
    ----
    sample_num : int
        Number of desired simulated lenses
    sets : array
        Array of test set types to be generated
    """

    def __init__(self,sample_num,sets):
        self.sample_num = sample_num
        self.sets = sets

    def generate_images(self):
        for set in self.sets:
            config_file = '../complexLA/Configs/'+set+'_config.py'
            image_path = '../images_for_network/'+set

            import subprocess
            command = f'python /Users/Logan/AppData/Local/Programs/Python/Python312/Lib/site-packages/paltas/paltas/generate.py {config_file} {image_path} --n {self.sample_num-1} --tf_record'
            print(f"Running: {command}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            print("Exit code:", result.returncode)

            #os.system(f'python /Users/Logan/AppData/Local/Programs/Python/Python312/Lib/site-packages/paltas/paltas/generate.py {config_file} {image_path} --n {self.sample_num} --tf_record')
   
    def display_sample(self,save_fig=False):

        #Initializing array that will store the images from the image path
        for set in self.sets:
            im = []
            image_path = '../images_for_network/'+set

            #Populating image array
            for file in os.listdir(image_path):
                if file.endswith('.npy'):
                    im.append(file)

            #Defining details of the grid
            im = np.asarray(im)
            fig,axs = plt.subplots(10,10,figsize=(10,10))
            n_cols = 10
            norm = ImageNormalize(np.load(image_path+'/'+im[1]),stretch=AsinhStretch())

            for i in range(0,self.sample_num-1):
                axs[i//n_cols,i%n_cols].imshow(np.load(image_path+'/'+im[i]), norm=norm)
                axs[i//n_cols,i%n_cols].set_xticks([])
                axs[i//n_cols,i%n_cols].set_yticks([])
            
            print(set)
            plt.show()
            if save_fig==True:
                fig.savefig('../docs/figures/'+set+'_sample.png',bbox_inches='tight')