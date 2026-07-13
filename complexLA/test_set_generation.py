"""
This module contains functions to call paltas scirpts that will generate test sets of simulated lenses.
Images of the lenses are stored in user defined locations.
"""
from astropy.visualization import AsinhStretch, ImageNormalize
import matplotlib.colors as mcolors
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
        """
        Generates a test set of a user defined size of simulated lens systems.
        """
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
        """
        Displays the test set images in a grid.

        Parameters
        ----------
        save_fig : bool
            Determines if the test set images should be saved to the local device
        """

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

    def residuals(self,sample_num,set,save_fig=False):
        """
        Displays the residuals between the complex mass model test sets and the simple main deflector test set.

        Parameters
        ----------
        sample_num : int
            Number of simulated lenses in the test set
        set : str
            Test set to be compared with the simple main deflector test set
        save_fig : bool
            Determines if the test set images should be saved to the local device
        """
        #Initializing array that will store the images from the image path
        im = []
        im_md = []
        image_path = '../images_for_network/'+set
        image_path_md = '../images_for_network/main_deflector'

        #Populating image array
        for file in os.listdir(image_path):
            if file.endswith('.npy'):
                im.append(file)

        for file in os.listdir(image_path_md):
            if file.endswith('.npy'):
                im_md.append(file)

        #Setting the norm and color scale
        resid_norm =mcolors.TwoSlopeNorm(vmin=-0.025,vcenter=0,vmax=0.025)

        #Defining details of the grid
        fig,axs = plt.subplots(10,10,figsize=(10,10))
        n_cols = 10

        for i in range(0,sample_num-1):
            imris = axs[i//n_cols,i%n_cols].imshow(np.load(image_path+im[i])-np.load(image_path_md+im_md[i]), norm=resid_norm,cmap='bwr')
            axs[i//n_cols,i%n_cols].set_xticks([])
            axs[i//n_cols,i%n_cols].set_yticks([])

        fig.colorbar(imris, ax=axs)
        plt.show()
        if save_fig==True:
            fig.savefig('../docs/figures/resid_'+set+'.png',bbox_inches='tight')
