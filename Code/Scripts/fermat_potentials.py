# Imports
import matplotlib.colors as colors
import matplotlib.pyplot as plt

import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.util as util
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Analysis.td_cosmography import TDCosmography
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

# Declaring arrays
kwargs_model = dict(
    lens_model_list=['EPL', 'SHEAR'],
    point_source_model_list=['SOURCE_POSITION'],
    cosmo=FlatLambdaCDM(H0=70.0, Om0=0.3)
    )
lens_model = LensModel(['EPL', 'SHEAR'])
td_cosmo = TDCosmography(0.5, 2., kwargs_model, cosmo_fiducial=FlatLambdaCDM(H0=70.0, Om0=0.3))

def kwargs_lens_from_y_pred(y_pred):
     return [{'theta_E':y_pred[0],'e1':y_pred[4],'e2':y_pred[5],'gamma':y_pred[3],
                      'center_x':y_pred[6],'center_y':y_pred[7]},
                     {'gamma1':y_pred[1],'gamma2':y_pred[2]}]

def lens_plot_from_y_pred(y_pred):
    kwargs_lens = kwargs_lens_from_y_pred(y_pred)
    kwargs_ps = [{'ra_source':y_pred[8],'dec_source':y_pred[9]}]
    fig,ax = plt.subplots(1,1)
    lens_plot.lens_model_plot(ax, lensModel=lens_model, kwargs_lens=kwargs_lens,
                            sourcePos_x=y_pred[8],
                            sourcePos_y=y_pred[9],
                            point_source=True, with_caustics=True, fast_caustic=False,
                            coord_inverse=True,numpix=80,deltapix=0.04)
    plt.show()

def fermat_surface_plot_from_y_pred(y_pred,numPix=140,deltaPix=0.02):
    """Trying to heatmap fermat potential instead of contouring it
    """
    kwargs_lens = kwargs_lens_from_y_pred(y_pred)
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix)
    data = ImageData(**kwargs_data)
    x_grid, y_grid = data.pixel_coordinates
    # ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = lensModelExt.critical_curve_caustics(
    #    kwargs_lens, compute_window=_frame_size, grid_scale=deltaPix/2.)
    x_grid1d = util.image2array(x_grid)
    y_grid1d = util.image2array(y_grid)
    fermat_surface = lens_model.fermat_potential(
        x_grid1d, y_grid1d, kwargs_lens, y_pred[8], y_pred[9]
    )
    fermat_surface = util.array2image(fermat_surface)
    fig,axs = plt.subplots(figsize=(14,14))
    im = axs.matshow(fermat_surface)
    fig.colorbar(im,ax=axs)
    axs.set_yticks([])
    axs.set_xticks([])
    axs.set_title('Fermat Potential')
    # GET POINT SOURCE IMAGES ON THERE
    solver = LensEquationSolver(lens_model)
    theta_x, theta_y = solver.image_position_from_source(
        y_pred[8],
        y_pred[9],
        kwargs_lens,
        search_window=np.max(data.width),
        min_distance=data.pixel_width,
    )
    fps = fermat_potential_at_image_positions(y_pred,theta_x,theta_y)
    #fig2,axs2 = plt.subplots(2,2,figsize=(10,10))
    labels = ['A','B','C','D']
    for i in range(len(theta_x)):
        x_ = theta_x[i]/deltaPix + numPix/2
        y_ = theta_y[i]/deltaPix + numPix/2
        axs.scatter(x_,y_,marker='*',color='red')
        axs.text(x_,y_,labels[i])
        """
        im2 = axs2[i//2,i%2].matshow(fermat_surface,vmin=fps[i]-0.02,vmax=fps[i]+0.02)
        axs2[i//2,i%2].scatter(x_,y_,marker='*',color='red')
        axs2[i//2,i%2].text(x_,y_,labels[i])
        x_min = int(x_)-20
        x_max = int(x_)+20
        y_min = int(y_)-20
        y_max = int(y_)+20
        axs2[i//2,i%2].set_xlim([x_min,x_max])
        axs2[i//2,i%2].set_ylim([y_min,y_max])
        axs2[i//2,i%2].set_yticks([])
        axs2[i//2,i%2].set_xticks([])
        fig2.colorbar(im2,ax=axs2[i//2,i%2])
        """
def compare_fermat_at_im_positions(y_truth,y_pred,numPix=140,deltaPix=0.02):
    """Trying to heatmap fermat potential instead of contouring it
    """
    # kwarg_lens for truth values
    kwargs_lens_truth = kwargs_lens_from_y_pred(y_truth)
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix)
    data = ImageData(**kwargs_data)
    x_grid, y_grid = data.pixel_coordinates
    # ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = lensModelExt.critical_curve_caustics(
    #    kwargs_lens, compute_window=_frame_size, grid_scale=deltaPix/2.)
    x_grid1d = util.image2array(x_grid)
    y_grid1d = util.image2array(y_grid)
    fermat_surface_truth = lens_model.fermat_potential(
        x_grid1d, y_grid1d, kwargs_lens_truth, y_truth[8], y_truth[9]
    )
    fermat_surface_truth = util.array2image(fermat_surface_truth)
    # start with im positions and raytrace back to a source position?...
    # or use the source position provided by the lens model? b/c the images won’t trace back to the same source pos...
    
    # TRUTH PS POSITIONS
    solver = LensEquationSolver(lens_model)
    theta_x, theta_y = solver.image_position_from_source(
        y_truth[8],
        y_truth[9],
        kwargs_lens_truth,
        search_window=np.max(data.width),
        min_distance=data.pixel_width,
    )
    fps = fermat_potential_at_image_positions(y_truth,theta_x,theta_y)

    # kwarg_lens for predictions
    kwargs_lens_pred = kwargs_lens_from_y_pred(y_pred)
    fermat_surface_pred = lens_model.fermat_potential(
        x_grid1d, y_grid1d, kwargs_lens_pred, y_pred[8], y_pred[9]
    )
    fermat_surface_pred = util.array2image(fermat_surface_pred)

    # Prediction PS positions
    theta_x_pred, theta_y_pred = solver.image_position_from_source(
        y_pred[8],
        y_pred[9],
        kwargs_lens_pred,
        search_window=np.max(data.width),
        min_distance=data.pixel_width,
    )
    
    fps_pred = fermat_potential_at_image_positions(y_pred,theta_x,theta_y)
    fig2,axs2 = plt.subplots(4,3,figsize=(10,10))
    labels = ['A','B','C','D']
    for i in range(len(theta_x)):
        x_ = theta_x[i]/deltaPix + numPix/2
        y_ = theta_y[i]/deltaPix + numPix/2
        x_pred_ = theta_x_pred[i]/deltaPix + numPix/2
        y_pred_ = theta_y_pred[i]/deltaPix + numPix/2
        im2 = axs2[i,0].matshow(fermat_surface_truth,vmin=fps[i]-0.02,vmax=fps[i]+0.02)
        axs2[i,0].scatter(x_,y_,marker='*',color='red')
        axs2[i,0].text(x_,y_,labels[i])
        x_min = int(x_)-20
        x_max = int(x_)+20
        y_min = int(y_)-20
        y_max = int(y_)+20
        axs2[i,0].set_xlim([x_min,x_max])
        axs2[i,0].set_ylim([y_min,y_max])
        axs2[i,0].set_yticks([])
        axs2[i,0].set_xticks([])
        fig2.colorbar(im2,ax=axs2[i,0])
        im3 = axs2[i,1].matshow(fermat_surface_pred,vmin=fps_pred[i]-0.02,vmax=fps_pred[i]+0.02)
        axs2[i,1].scatter(x_,y_,marker='*',color='red')
        axs2[i,1].scatter(x_pred_,y_pred_,marker='x',color='black')
        axs2[i,1].text(x_,y_,labels[i])
        axs2[i,1].set_xlim([x_min,x_max])
        axs2[i,1].set_ylim([y_min,y_max])
        axs2[i,1].set_yticks([])
        axs2[i,1].set_xticks([])
        fig2.colorbar(im3,ax=axs2[i,1])
        resid_fermat = fermat_surface_truth-fermat_surface_pred
        resid_norm = colors.TwoSlopeNorm(vmin=-0.05,vcenter=0,vmax=0.05)
        im4 = axs2[i,2].matshow(resid_fermat,norm=resid_norm,cmap='bwr')
        fig2.colorbar(im4,ax=axs2[i,2])

def image_positions_from_y_pred(y_pred):
    #theta_x = []
    #theta_y = []
    #for arr in y_pred:
    solver = LensEquationSolver(lens_model)
    kwargs_lens = kwargs_lens_from_y_pred(y_pred)
    theta_x, theta_y = solver.image_position_from_source(y_pred[8], y_pred[9], kwargs_lens)
    #theta_x.append(theta_x_val)
    #theta_y.append(theta_y_val)
    return theta_x,theta_y

def fermat_potential_at_image_positions(y_pred,x_im,y_im):
    kwargs_lens = kwargs_lens_from_y_pred(y_pred)
    return lens_model.fermat_potential(x_im,y_im,kwargs_lens,x_source=y_pred[8],y_source=y_pred[9])