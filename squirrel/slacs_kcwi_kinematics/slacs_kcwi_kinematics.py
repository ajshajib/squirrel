'''
##########################################################
##########################################################

01/04/2023 - Shawn Knabel (shawnknabel@gmail.com) - Chih-Fan Chen

This python script creates a class called slacs_kcwi_kinematics.
Its purpose is to take a mosaic'ed datacube of a SLACS lens galaxy and create kinematic maps.
Several pieces must be done beforehand and input.

##########################################################
##########################################################
'''

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pathlib # to create directory

from ppxf.ppxf import ppxf
from pathlib import Path
from scipy import ndimage
from urllib import request
from scipy import ndimage
from time import perf_counter as clock
from scipy import interpolate
from astropy.visualization import simple_norm
import astropy.units as u
from vorbin.voronoi_2d_binning import voronoi_2d_binning
import pyregion

# my functions
import sys
sys.path.append("/home/shawnknabel/Documents/slacs_kinematics/my_python_packages/ppxf_kcwi_util_022423")

import ppxf.ppxf_util as ppxf_util
from os import path
ppxf_dir = path.dirname(path.realpath(ppxf_util.__file__))
import ppxf.sps_util as sps_util

c = 299792.458 # km/s

##############
# Utility functions

def register_sauron_colormap():
    """
    Regitsr the 'sauron' and 'sauron_r' colormaps in Matplotlib

    """
    cdict = {'red':[(0.000,   0.01,   0.01),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.4,    0.4),
                 (0.414,   0.5,    0.5),
                 (0.463,   0.3,    0.3),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.7,    0.7),
                 (0.590,   1.0,    1.0),
                 (0.668,   1.0,    1.0),
                 (0.834,   1.0,    1.0),
                 (1.000,   0.9,    0.9)],
        'green':[(0.000,   0.01,   0.01),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.85,   0.85),
                 (0.414,   1.0,    1.0),
                 (0.463,   1.0,    1.0),
                 (0.502,   0.9,    0.9),
                 (0.541,   1.0,    1.0),
                 (0.590,   1.0,    1.0),
                 (0.668,   0.85,   0.85),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.9,    0.9)],
         'blue':[(0.000,   0.01,   0.01),
                 (0.170,   1.0,    1.0),
                 (0.336,   1.0,    1.0),
                 (0.414,   1.0,    1.0),
                 (0.463,   0.7,    0.7),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.0,    0.0),
                 (0.590,   0.0,    0.0),
                 (0.668,   0.0,    0.0),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.9,    0.9)]
         }

    rdict = {'red':[(0.000,   0.9,    0.9),
                 (0.170,   1.0,    1.0),
                 (0.336,   1.0,    1.0),
                 (0.414,   1.0,    1.0),
                 (0.463,   0.7,    0.7),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.3,    0.3),
                 (0.590,   0.5,    0.5),
                 (0.668,   0.4,    0.4),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.01,   0.01)],
        'green':[(0.000,   0.9,    0.9),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.85,   0.85),
                 (0.414,   1.0,    1.0),
                 (0.463,   1.0,    1.0),
                 (0.502,   0.9,    0.9),
                 (0.541,   1.0,    1.0),
                 (0.590,   1.0,    1.0),
                 (0.668,   0.85,   0.85),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.01,   0.01)],
         'blue':[(0.000,   0.9,    0.9),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.0,    0.0),
                 (0.414,   0.0,    0.0),
                 (0.463,   0.0,    0.0),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.7,    0.7),
                 (0.590,   1.0,    1.0),
                 (0.668,   1.0,    1.0),
                 (0.834,   1.0,    1.0),
                 (1.000,   0.01,   0.01)]
         }

    sauron = colors.LinearSegmentedColormap('sauron', cdict)
    sauron_r = colors.LinearSegmentedColormap('sauron_r', rdict)
    plt.register_cmap(cmap=sauron)
    plt.register_cmap(cmap=sauron_r)

def de_log_rebin(delog_axi, value, lin_axi):
    '''
    :param delog_axi: input the value by np.exp(logLam1)
    :param value: flux at the location of np.exp(logLam1) array
    :param lin_axi: linear space in wavelength that we want to intepolate
    :return: flux at the location of linear space in wavelength
    '''
    inte_sky = interpolate.interp1d(delog_axi, value, bounds_error=False)
    sky_lin = inte_sky(lin_axi)
    return sky_lin

def getMaskInFitsFromDS9reg(input,shape,hdu):
    '''
    Returns 2D pixel mask from .reg file created in DS9.
    '''
    r = pyregion.open(input)
    mask = r.get_mask(shape=shape,hdu=hdu)
    return mask

def poisson_noise(T_exp, gal_lin, std_bk_noise, per_second=False):
    '''
    This means that the pixel uncertainty of pixel i (sigma_i) is obtained
    from the science image intensity pixel i (d_i) by:
    sigma_i^2 = scale * (d_i)^power + const
    The first term represents noise from the astrophysical source, and the
    second term is background noise (including read noise etc.).
    When power=1 and scale=1 with d_i in counts, the astrophysical source noise
    (=1*d_i^1=d_i) is Poisson. Suyu 2012 and Suyu et al. 2013a have somels

    description of this.

    To construct the weight map using the esource_noise_model:
    -- set power=1
    -- obtain const by estimating the variance of the background (i.e., const = sigma_bkgd^2 from an empty part of of the science image).
    -- the scale is 1 if d_i is in counts, but otherwise it needs to account for exposure time if d_i is in counts per second.

    Since the unit of the KCWI data is flux/AA (see fits header),
    I need to compute scale with appropriate multiplications/divisions of
    the exposure time (T_exp).  In this case, scale should be 1/texp so that
    the units are in counts/sec for sigma_i^2 (since d_i needs to be
    multiplied by texp to get to counts for Poisson noise estimation,
    but then divided by texp^2 to get to counts/sec).

    :param T_exp: the total exposure time of the dataset
    :param gal_lin: input data
    :param bk_noise: standard deviation of the background noise
    :param per_second: set True if it is in the unit of counts/second
    :return: poisson noise
    '''

    const = std_bk_noise**2
    if per_second:
        scale= 1/T_exp
        sigma2 = scale * (gal_lin) + const
    else:
        scale = 1.
        sigma2 = scale * (gal_lin) + const

    if (sigma2<0).any():
        sigma2[sigma2 < 0] = const

    if np.isnan(sigma2).any():
        sigma2[np.isnan(sigma2)] = const

    poisson_noise = np.sqrt(sigma2)

    return poisson_noise

def find_nearest(array, value):
    '''
    :param array: wavelength array
    :param value: wavelength that we want to get the index
    :return: the index of the wavelength
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

##############
# start
register_sauron_colormap()


class slacs_kcwi_kinematics:
    
    '''
    slacs_kcwi_kinematics Purpose:
    -------------------
    For a SLACS galaxy with reduced and mosaic'ed IFU datacube, take some inputs to create a stellar kinematic map.
    
    Calling Sequence:
    -------------------
    
    .. code-block:: python
        from slacs_kcwi_kinematics import slacs_kcwi_kinematics

        # data directory
        data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'
        # Set 'obj_name', 'z', 'T_exp'
        obj_name = 'SDSSJ0029-0055'
        obj_abbr = obj_name[4:9] # e.g. J0029
        zlens = 0.227 # lens redshift
        T_exp = 1800*5 # exposure time in seconds... this is where I made the disastrous mistake
        lens_center_x,lens_center_y = 61, 129
        # other necessary directories ... Be very careful! This is how we will make sure we are using the correct files moving forward.
        mos_dir = f'{data_dir}mosaics/{obj_name}/' # files should be loaded from here but not saved
        kin_dir = f'{data_dir}kinematics/{obj_name}/'
        #------------------------------------------------------------------------------
        # Kinematics systematics initial choices
        # aperture
        aperture = 'R2'
        # wavelength range
        wave_min = 3400
        wave_max = 4300 # CF set to 428
        # degree of the additive Legendre polynomial in ppxf
        degree = 4 # 90/25 = 3.6 round up
        #------------------------------------------------------------------------------
        # Information specific to KCWI and templates
        kcwi_scale = 0.1457
        ## R=3600. spectral resolution is ~ 1.42A
        FWHM = 1.42 #1.42
        ## initial estimate of the noise
        noise = 0.014
        # velocity scale ratio
        velscale_ratio = 2
        #------------------------------------------------------------------------------
        # variable settings in ppxf and utility functions
        # cut the datacube at lens center, radius given here
        radius_in_pixels = 21
        # target SN for voronoi binning
        #vorbin_SN_targets = np.array([10, 15, 20])
        SN = 15
        #KCWI mosaic datacube
        mos_name = f'KCWI_{obj_abbr}_icubes_mosaic_0.1457'
        kcwi_datacube = f'{mos_dir}{mos_name}.fits'
        #spectrum from the lens center # using R=2
        central_spectrum_file = f'{mos_dir}{obj_abbr}_central_spectrum_{aperture}.fits' 
        background_spectrum_file = f'{mos_dir}{obj_abbr}_spectrum_background_source.fits'
        background_source_mask_file = f'{mos_dir}{obj_abbr}_background_source_mask.reg'
        sps_name = 'emiles'

        j0029_kinematics = slacs_kcwi_kinematics(
                                                 mos_dir=mos_dir,
                                                 kin_dir=kin_dir,
                                                 obj_name=obj_name,
                                                 kcwi_datacube_file=kcwi_datacube,
                                                 central_spectrum_file=central_spectrum_file,
                                                 background_spectrum_file=None,
                                                 background_source_mask_file=background_source_mask_file,
                                                 zlens=zlens,
                                                 exp_time=T_exp,
                                                 lens_center_x=lens_center_x,
                                                 lens_center_y=lens_center_y,
                                                 aperture=aperture,
                                                 wave_min=wave_min,
                                                 wave_max=wave_max,
                                                 degree=degree,
                                                 sps_name=sps_name,
                                                 pixel_scale=kcwi_scale,
                                                 FWHM=FWHM,
                                                 noise=noise,
                                                 velscale_ratio=velscale_ratio,
                                                 radius_in_pixels=radius_in_pixels,
                                                 SN=SN,
                                                 plot=True,
                                                 quiet=False
                                                    )

        # convenience function will do all the below automatically
        #j0029_kinematics.run_slacs_kcwi_kinematics()
        # Visualize the summed datacube
        j0029_kinematics.datacube_visualization()
        # rebin the central spectrum in log wavelengths and prepare for fitting
        j0029_kinematics.log_rebin_central_spectrum()
        # same with background spectrum
        j0029_kinematics.log_rebin_background_spectrum()
        # prepare the templates from the sps model
        j0029_kinematics.get_templates()
        # set up the wavelengths that will be fit, masks a couple gas lines
        j0029_kinematics.set_up_mask()
        # fit the central spectrum to create the global_template
        j0029_kinematics.ppxf_central_spectrum()
        # crop the datacube to a smaller size
        j0029_kinematics.crop_datacube()
        # create a S/N map to get the Voronoi binning going
        j0029_kinematics.create_SN_map()
        # select the spaxels S/N > 1 that will be binned
        j0029_kinematics.select_region()
        # bin the selected spaxels to the target S/N
        j0029_kinematics.voronoi_binning()
        # fit each bin spectrum with global_template
        j0029_kinematics.ppxf_bin_spectra(plot_bin_fits=plot_bin_fits)
        # create the 2d kinematic maps from ppxf fits
        j0029_kinematics.make_kinematic_maps()
        # plot those maps
        j0029_kinematics.plot_kinematic_maps()
        
        ###
        Nothing will be saved. Individual outputs can be saved as arrays, or the whole class instance can be saved to, e.g. a pkl file with python's pickle or dill module.
        ###
        import dill as pickle
        # to save as a pickle
        with open('test_pickle.pkl', 'wb') as file:
            pickle.dump(j0029_kinematics, file)
        # to reload the pickle
        with open('test_pickle.pkl', 'rb') as file:
            tommy_pickles = pickle.load(file)

    Input Parameters
    ------------------
    
    mos_dir: str
        Path to directory containing mosaic datacubes and extracted central and background source spectra.
        
    kin_dir: str
        Path to directory where outputs will be saved.
        
    obj_name: str
        Object identifier for filenames, plotting, etc.
        
    kcwi_datacube_file: str
        Path to .fits file containing the mosaic'ed KCWI datacube.
    
    central_spectrum_file: str
        Path to .fits file containing the 1D extracted spectrum from the center of the galaxy (e.g. with the program QFitsView)
        
    background_spectrum_file: str
        Path to .fits file containing the 1D extracted spectrum from the background source (e.g. with the program QFitsView)
        
    zlens: float
        Redshift of the foreground deflector lens galaxy
        
    exp_time: float or int
        Total exposure time in seconds of the mosaic'ed datacube (seconds)
        
    lens_center_x, lens_center_y: int
        Row index (x-coordinate) and column index (y-coord) of "roughly" central pixel of foreground lens galaxy, e.g. from DS9 or QFitsView, used for centering
        
    aperture: int
        Size of the aperture in pixels used to extract the central spectrum of the foreground lens galaxy "central_spectrum_file" from e.g. QFitsView (pixels)
        
    wave_min, wave_max: float or int
        Wavelength in Angstroms of the minimum and maximum of the desired fitting range. Actual fitting will ignore any part of the spectrum not inside this range (Angstroms)
        
    degree: int
        Degree of the additive polynomial used to help fit the continuum during kinematic fitting. *Very* rough rule of thumb is (wave_max - wave_min) / 250 Angstroms, rounded up. It will overfit and bias the kinematics if degree is too high. It will be unable to correct wiggles in the continuum if too low.
        
    sps_name: str
        Name of the simple stellar population used with ppxf, e.g. 'emiles'. Other templates can be used, but this is the easiest. I will likely add functionality to do what we originally did with Xshooter (more flexibility, extra steps). The sps model files will be in the ppxf module directory (should be at least). Maybe check the latest version of ppxf from github.
        
    pixel_scale: float
        Pixel scale of the datacube. KCWI is 0.1457 arcseconds per pixel (after we have made the pixels square) (arcseconds / pixel)
        
    FWHM: float
        Estimate of instrument spectral FWHM in Angstroms. KCWI is 1.42 (Angstroms)
        
    noise: float
        Rough initial estimate of the noise (I might have a bug here... Maybe I need to redo it the way CF did)
        
    velscale_ratio: int
        The ratio of desired resolution of the template spectra with relation to the datacube spectra. We tend to use 2, which means the template spectra are sampled at twice the resolution of the data
        
    radius_in_pixels: int
        Radius in pixels taken for cropping the datacube to a smaller square. We tend to use 21 pixels, which is just over 3 arcseconds. This is about the range at which we are able to get S/N per pixel > 1, generally (pixels)
        
    bin_target_SN: float
        Target signal-to-noise ratio for Voronoi binning. The spaxels (spatial pixels, used interchangeably with pixels here) will be binned so that they are close to this target S/N before fitting kinematics to each bin. We tend to use 15
        
    pixel_min_SN: float
        Minimum SN for a pixel to be included in fitting, during self.select_region, tend to use 1
        
    plot: boolean
        Plot the steps throughout, True or False
        
    quiet: boolean
        Suppress some of the wordier outputs, True or False
    
    Output Parameters
    -----------------

    Stored as attributes of the ``slacs_kcwi_kinematics`` class:
    
    .rest_wave_range: tuple (2,)
        Deredshifted wavelength range of datacube in restframe of foreground deflector lens
    
    .rest_FWHM: float
        Adjusted datacube resolution in Angstroms to restframe of foreground deflector lens
        
    .central_spectrum: array (N,)
        Deredshifted (to foreground deflector restframe), log-rebinned 1D spectrum of foreground deflector galaxy from self.central_spectrum_file, of size N (size of the datacube in spectral elements)
    
    .rest_wave_log: array (N,)
        Log of log-rebinned wavelengths in restframe of datacube spectra (log Angstroms), of size N (size of the datacube in spectral elements)
        
    .rest_wave: array (N,)
        Log-re-binned wavelengths in restframe of datacube spectra (Angstroms), of size N (size of the datacube in spectral elements)
    
    .central_velscale: float
        Resolution in km/s of the datacube
        
    .background_spectrum: array (N,)
        Deredshifted (to foreground deflector restframe), log-rebinned 1D spectrum of background source galaxy from self.background_spectrum_file, of size N (size of the datacube in spectral elements)
    
    .wave_range_templates: array (size slightly larger than N)
        Restframe wavelength range of stellar templates to be used for fitting, should be slightly larger than the range of the galaxy we want to fit, but can be much larger (just takes more time)
        
    .templates: array (number of templates, templates_wave.size) # could be opposite :)
        Array containing the stellar templates from sps models, used for fitting the global template spectrum (i.e. the central_spectrum), size will be the number of templates along one axis and the range of wavelengths of templates along the other (self.templates_wave), which will be the range of templates multiplied by the input velscale (diff(wave_range_templates)*velscale, for the total number of sampled wavelengths of the templates)
        
    .templates_wave: array(diff(wave_range_templates)*velscale,)
        Array of template wavelengths, sampled at velscale times the resolution of the datacube spectra.
        
    .mask: array (unsure of size)
        Array of indices for the wavelengths of spectra that will be included in the fit, masking so that it is between wave_min and wave_max, keeping good pixels, and excluding MgII lines at ~2956-3001, input for ppxf
        
    .central_spectrum_ppxf: instance of ppxf
        Instance of ppxf with information about the central spectrum fit, can be used to recover, e.g., the weights of the stellar templates in the fit
        
    .nTemplates: int
        Number of stellar template spectra used to fit the central foreground galaxy spectrum (global_template)
        
    .global_template: array (N,)
        Weighted sum of stellar template spectra that make up the model of the central galaxy spectrum (central_spectrum); does not include the polynomial, ackground_source components, or kinematics. Will be used as the single template to fit the individual spatially-binned spectra to make the kinematic map. Essentially, this keeps the weights of the stellar templates uniform throughout the bins, which saves time and avoids over-fitting for stellar population deviations. Each bin could also be fit individually with new weights for the templates, but this project does not require that (and the data isn't good enough to determine different stellar populations).
        
    .global_template_wave: array (same as templates_wave)
        Probably redundant, just the wavelengths of the sum of templates, should be the same as the variable templates_wave
        
    .cropped_datacube: array(N, 2*radius_in_pixels+1, 2*radius_in_pixels+1)
        Datacube cropped spatially to square dimensions determined by radius_in_pixels. Wavelengths and fluxes are unaffected.
        
    .SN_per_AA: array_like(cropped_datacube)...(2*radius_in_pixels+1, 2*radius_in_pixels+1)
        Signal-to-noise map of cropped datacube spaxels, for Voronoi binning
    
    .voronoi_binning_input: array (Npix = num pixels S/N > 1, 4)
        Array containing x-coord, y-coord, signal-to-noise, and dummy noise variable for pixels in region where SN_per_AA > 1. np.vstack((xx_1D, yy_1D, SN_1D, np.ones(SN_1D.shape[0]))). Input for voronoi_2d_binning in method self.voronoi_binning
        
    .voronoi_binning_output: array (Npix = num pixels S/N > 1, 3)
        Array of outputs from Voronoi binning, which contains the x-coordinate, y-coordinate, and assigned Voronoi bin number for each of the spaxels that are in the binning (S/N > 1). This allows one to connect any measured bin values, e.g. velocity dispersion, to the individual spaxels that belong to the bin where it was measured.
        
    .voronoi_binning_data: array (nbins, N)
        Array containing the summed spectra (size N) of all of the spaxels that make up each Voronoi bin. This is the "stacked" bin spectrum that is fitted with the global_template in order to measure the kinematics of the bin.
        
    .nbins: int
        Number of spatial Voronoi bins
        
    .bin_kinematics: array (nbins, 5)
        Array of velocity dispersion VD, velocity V, error dVD, error dV, chi2 for each of the spatial Voronoi bins. Errors are determined in a bit of a funky way and will be replaced by sampling at some point. For now, errors are the "formal errors" from ppxf multiplied by the sqrt(chi2) for the fit.
        
    .VD_2d, .V_2d, .dVD_2d, .dV_2d: array_like(cropped_datacube)
        Arrays of same shape as cropped_datacube (2*radius_in_pixels+1, square), with kinematic information assigned to each of the pixels individually (through voronoi_binning_output). The 2D kinematic map that is ready to be plotted. Spaxels that are not fit are Nan.
    '''

    
    def __init__(self,
                 mos_dir,
                 kin_dir,
                 obj_name,
                 kcwi_datacube_file,
                 central_spectrum_file,
                 background_spectrum_file,
                 background_source_mask_file,
                 zlens,
                 exp_time,
                 lens_center_x,
                 lens_center_y,
                 aperture,
                 wave_min,
                 wave_max,
                 degree,
                 sps_name,
                 pixel_scale,
                 FWHM,
                 noise,
                 velscale_ratio,
                 radius_in_pixels,
                 bin_target_SN,
                 pixel_min_SN,
                 plot,
                 quiet):
        
        self.mos_dir = mos_dir
        self.kin_dir = kin_dir
        self.obj_name = obj_name
        self.kcwi_datacube_file = kcwi_datacube_file
        self.central_spectrum_file = central_spectrum_file
        self.background_spectrum_file = background_spectrum_file
        self.background_source_mask_file = background_source_mask_file
        self.zlens = zlens
        self.exp_time = exp_time
        self.lens_center_x = lens_center_x
        self.lens_center_y = lens_center_y
        self.aperture = aperture
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.degree = degree
        self.sps_name = sps_name
        self.pixel_scale = pixel_scale
        self.FWHM = FWHM
        self.noise = noise
        self.velscale_ratio = velscale_ratio
        self.radius_in_pixels = radius_in_pixels
        self.bin_target_SN = bin_target_SN
        self.pixel_min_SN = pixel_min_SN
        self.plot = plot
        self.quiet = plot  
        
########################################################################################################

    def run_slacs_kcwi_kinematics(self, fit_poisson_noise=False, plot_bin_fits=False):
        '''
        Convenience function runs all the steps in a row for ease.
        plot_bin_fits- if True, will plot each ppxf fit for the bins, default is False (to save time)
        '''
        print(f'pPXF will now consume your soul and use it to measure the kinematics of {self.obj_name}.')
        
        # Visualize the summed datacube
        self.datacube_visualization()
        
        # rebin the central spectrum in log wavelengths and prepare for fitting
        self.log_rebin_central_spectrum()
        
        # same with background spectrum
        self.log_rebin_background_spectrum()
        
        # prepare the templates from the sps model
        self.get_templates()
        
        # set up the wavelengths that will be fit, masks a couple gas lines
        self.set_up_mask()
        
        # fit the central spectrum to create the global_template
        self.ppxf_central_spectrum()
        
        # crop the datacube to a smaller size
        self.crop_datacube()
        
        # create a S/N map to get the Voronoi binning going
        self.create_SN_map()
        
        # select the spaxels S/N > 1 that will be binned
        self.select_region()
        
        # bin the selected spaxels to the target S/N
        self.voronoi_binning()
        
        # fit each bin spectrum with global_template
        self.ppxf_bin_spectra(fit_poisson_noise, plot_bin_fits)
        
        # create the 2d kinematic maps from ppxf fits
        self.make_kinematic_maps()
        
        # plot those maps
        self.plot_kinematic_maps()
        
        print("Job's finished!")
        
        
#########################################
        
    def datacube_visualization(self):
        '''
        Function shows the mosaic'ed datacube summed over the wavelength axis. This is mostly just to get oriented to where things are in the image.
        '''
        
        # open the fits file and get the data
        data_hdu = fits.open(self.kcwi_datacube_file)
        datacube=data_hdu[0].data
        data_hdu.close()
        
        # norm for plotting
        norm = simple_norm(np.nansum(datacube, axis=0), 'sqrt')
        
        # plot
        plt.imshow(np.nansum(datacube, axis=0), origin="lower", norm=norm)
        plt.title('KCWI data')
        plt.colorbar(label='flux')
        plt.pause(1)
    
    
#########################################

    def log_rebin_central_spectrum(self):
        '''
        Function to deredshift and rebin the central foreground galaxy spectrum to log space and prepare the restframe wavelengths for proper fitting with the stellar template spectra later.
        '''
        
        # open the fits file and get the data
        hdu = fits.open(self.central_spectrum_file)
        
        # galaxy spectrum with linear wavelength spacing
        gal_lin = hdu[0].data
        h1 = hdu[0].header
        
        # wavelength range from fits header
        lamRange1 = h1['CRVAL1'] + np.array([0., h1['CDELT1']*(h1['NAXIS1'] - 1)])
        
        # Compute approximate restframe wavelength range
        self.rest_wave_range = lamRange1/(1+self.zlens)
        
        # Adjust resolution in Angstrom
        self.rest_FWHM = self.FWHM/(1+self.zlens)  
        
        # rebin to log wavelengths and calculate velocity scale (resolution)
        self.central_spectrum, self.rest_wave_log, self.central_velscale = \
                                            ppxf_util.log_rebin(self.rest_wave_range, gal_lin)
        
        # keep the rebinned wavelengths in Angstroms
        self.rest_wave = np.exp(self.rest_wave_log)
        
        
#########################################
    
    def log_rebin_background_spectrum(self):
        '''
        Function to deredshift (to the foreground deflector galaxy redshift zlens) and rebin the background source galaxy spectrum to log space and prepare the restframe wavelengths for proper fitting with the "sky" keyword in ppxf.
        '''
        
        if self.background_spectrum_file is not None:
            
            # open the fits file and get the data
            hdu = fits.open(self.background_spectrum_file)
            background_source_lin = hdu[0].data 
            
            # rebin to log wavelengths
            background_source, _, _ = ppxf_util.log_rebin(self.rest_wave_range, background_source_lin)
            
            # Normalize spectrum to avoid numerical issues
            self.background_spectrum = background_source/np.median(background_source)  
            
        else:
            # return an array of zeros
            self.background_spectrum = np.zeros_like(self.central_spectrum)
            
      
#########################################
            
    def get_templates(self):
        '''
        Function prepares the stellar template spectra from the sps model identified by sps_name.
        '''
        
        # take wavelength range of templates to be slightly larger than that of the galaxy restframe
        self.wave_range_templates = self.rest_wave_range[0]/1.2, self.rest_wave_range[1]*1.2
        
        # bring in the templates from ppxf/sps_models/
        basename = f"spectra_{self.sps_name}_9.0.npz"
        filename = path.join(ppxf_dir, 'sps_models', basename)
        
        # template library will be sampled at data resolution times the velscale_ratio in the given wavelength range
        sps = sps_util.sps_lib(filename, 
                               self.central_velscale/self.velscale_ratio, # resolution
                               self.rest_FWHM, # data FWHM in restframe
                               wave_range=self.wave_range_templates) # range for templates
        templates= sps.templates
        
        # keep templates and wavelength range of templates
        self.templates = templates.reshape(templates.shape[0], -1) 
        self.templates_wave = sps.lam_temp


#########################################
        
    def set_up_mask(self):
        '''
        Function prepares a mask of wavelengths for ppxf that determines the correct wavelength range and masks some gas lines.
        '''
        # take the pixels in the range
        # after de-redshift, the initial redshift is zero.
        goodPixels = ppxf_util.determine_goodpixels(self.rest_wave_log, self.wave_range_templates, 0)
        # find the indices of the restframe wavelengths that are closest to the min and max we want
        ind_min = find_nearest(self.rest_wave, self.wave_min)
        ind_max = find_nearest(self.rest_wave, self.wave_max)
        mask=goodPixels[goodPixels<ind_max]
        mask = mask[mask>ind_min]
        # mask gas lines
        boolen = ~((2956 < mask) & (mask < 2983))  # mask the Mg II
        mask = mask[boolen]
        boolen = ~((2983 < mask) & (mask < 3001))  # mask the Mg II
        # return the mask
        self.mask = mask[boolen]
    

#########################################
    
    def ppxf_central_spectrum(self):
        '''
        Function fits the central_spectrum with the stellar template spectra, a polynomial of specified degree, and the background source as the "sky" component.
        '''
        
        # some setup, starting guesses
        vel = c*np.log(1 + 0)   # eq.(8) of Cappellari (2017)
        start = [vel, 250.]  # (km/s), starting guess for [V, sigma]
        #bounds = [[-500, 500],[50, 450]] # not necessary
        t = clock()
        
        # create a noise array # Assume constant noise per AA
        noise_array = np.full_like(self.central_spectrum, self.noise) 
        
        # fit with ppxf
        pp = ppxf(self.templates, # templates for fitting
                  self.central_spectrum,  # spectrum to be fit
                  noise_array,
                  self.central_velscale, # resolution
                  start, # starting guess
                  plot=False, # no need to plot here, will plot after
                  moments=2, # VD and V, no others
                  goodpixels=self.mask, # mask we made
                  degree=self.degree, # degree of polynomial we specified
                  velscale_ratio=self.velscale_ratio, # resolution of templates wrt. data
                  sky=self.background_spectrum, # background source spectrum
                  lam=self.rest_wave, # wavelengths for fitting
                  lam_temp=self.templates_wave, # wavelenghts of templates
                 )

        #plot the fit
        # model
        model = pp.bestfit
        # background source
        background = self.background_spectrum * pp.weights[-1]
        # data
        data = pp.galaxy
        
        # linearize the wavelengths for plotting
        log_axis = self.rest_wave
        lin_axis = np.linspace(self.rest_wave_range[0], self.rest_wave_range[1], data.size)
        
        # rebin in linear space
        back_lin = de_log_rebin(log_axis, background, lin_axis) # background source fit
        model_lin = de_log_rebin(log_axis, model, lin_axis) # ppxf model
        data_lin = de_log_rebin(log_axis, data, lin_axis) # data
        noise_lin = data_lin - model_lin # noise
        
        # find the indices of the restframe wavelengths that are closest to the min and max we want for plot limits
        plot_ind_min = find_nearest(lin_axis, self.wave_min)
        plot_ind_max = find_nearest(lin_axis, self.wave_max)
        
        # make the figure
        plt.figure(figsize=(8,6))
        
        # plot the spectra, noise, etc
        plt.plot(lin_axis, data_lin, 'k-', label='data')
        plt.plot(lin_axis, model_lin, 'r-', label='model (lens+background)')
        plt.plot(lin_axis, data_lin - back_lin, 'm-',
                 label='remove background source from data', alpha=0.5) # if there is a background source to remove
        plt.plot(lin_axis, back_lin + np.full_like(back_lin, 0.9e-5), 'c-',label='background source', alpha=0.7)
        plt.plot(lin_axis, noise_lin + np.full_like(back_lin, 0.9e-5), 'g-',
                 label='noise (data - best model)', alpha=0.7)
        
        # set labels, axes, etc
        plt.legend(loc='best')
        plt.ylim(np.nanmin(noise_lin[plot_ind_min:plot_ind_max])/1.1, np.nanmax(data_lin[plot_ind_min:plot_ind_max])*1.1)
        plt.xlim(self.wave_min, self.wave_max)
        plt.xlabel('wavelength (A)')
        plt.ylabel('relative flux')
        plt.title(f'Velocity dispersion - {int(pp.sol[1])} km/s')
        plt.show()
        plt.pause(1)
        
        # show results of fit
        print("Formal errors:")
        print("     dV    dsigma   dh3      dh4")
        print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))
        print('Elapsed time in pPXF: %.2f s' % (clock() - t))
        
        # take the fit as attributes for future
        self.central_spectrum_ppxf = pp
        # number of templates
        self.nTemplates = pp.templates.shape[1]
        # global_template is what we use to fit the bins
        self.global_template = pp.templates @ pp.weights[:self.nTemplates]
        self.global_template_wave = pp.lam_temp
        
        
#########################################
        
    def crop_datacube(self):
        '''
        Function crops the datacube to a size determined by radius_in_pixels.
        '''
        
        # resulting datacube will have square spatial dim 2*r+1 pixels
        r = self.radius_in_pixels
        
        # open the datacube fits file and retrieve the data
        data_hdu = fits.open(self.kcwi_datacube_file)
        datacube=data_hdu[0].data
        data_hdu.close()
        
        # if there is a background source mask from DS9, background_source_mask_file should be the path to that file
        if self.background_source_mask_file is not None:
            
            # load the mask file
            self.background_source_mask = ~getMaskInFitsFromDS9reg(self.background_source_mask_file, datacube.shape[1:], data_hdu[0])*1
            
            # norm for plotting
            norm = simple_norm(np.nansum(datacube, axis=0), 'sqrt')
            # plot and check the mask
            plt.imshow(np.nansum(datacube, axis=0)*self.background_source_mask, origin="lower", norm=norm)
            plt.title('Masked KCWI data')
            plt.colorbar(label='flux')
            plt.pause(1)
            
        else:
            # if there is no background source, it will not mask anything
            self.background_source_mask = np.ones(datacube.shape[1:])
        
        # perform the crop
        # crop the mask
        self.background_source_mask = self.background_source_mask[
                                                                 self.lens_center_y - r-1:self.lens_center_y + r, 
                                                                 self.lens_center_x - r -1:self.lens_center_x + r]
        # crop the datacube and apply the mask
        self.cropped_datacube = datacube[:, 
                                         self.lens_center_y - r-1:self.lens_center_y + r, 
                                         self.lens_center_x - r -1:self.lens_center_x + r] \
                                                            * self.background_source_mask
        
        # norm for plotting
        norm = simple_norm(np.nansum(self.cropped_datacube, axis=0), 'sqrt')
        # plot to check the crop is successful
        plt.imshow(np.nansum(self.cropped_datacube, axis=0)*self.background_source_mask, origin="lower", norm=norm)
        plt.title('Cropped KCWI data')
        plt.colorbar(label='flux')
        plt.pause(1)

        
#########################################
        
    def create_SN_map(self):
        '''
        Function creates the S/N map for use in Voronoi binning.
        '''
        # estimate the noise from a blank section of sky
        noise_from_blank = self.cropped_datacube[self.wave_min:self.wave_max, 4-3:4+2,4-3:4+2]
        
        # blank space may be chopped by mask, take the opposite corner
        if noise_from_blank.std() == 0:
            noise_from_blank = self.cropped_datacube[self.wave_min:self.wave_max, -4-3:-4+2,-4-3:-4+2]
            
        # take the std of the blank patch of sky
        std = np.std(noise_from_blank)
        
        # sample the normal distribution of the noise
        s = np.random.normal(0, std, self.cropped_datacube.flatten().shape[0])
        
        # create a noise cube from it
        noise_cube = s.reshape(self.cropped_datacube.shape)

        ## use the noise spectrum and datacube
        # produced in the previous steps to estimate the S/N per AA. Since KCWI
        #  is  0.5AA/pixel, I convert the value to S/N per AA. Note that I use only the
        # region of CaH&K to estimate the S/N ratio (i.e. 3900AA - 4000AA).
        lin_axis = np.linspace(self.rest_wave_range[0], self.rest_wave_range[1], self.cropped_datacube.shape[0])
        # find indices for SN
        ind_min_SN = find_nearest(lin_axis, 3900)
        ind_max_SN = find_nearest(lin_axis, 4000)
        
        # show wavelengths used to calculate SN, just to be sure
        plt.plot(self.rest_wave, self.central_spectrum, c='r')
        plt.axvspan(3900, 4000, facecolor='black', alpha=0.5)
        plt.title('S/N wavelengths')
        plt.pause(1)

        # first, I need to estimate the flux/AA
        flux_per_half_AA = np.nanmedian(self.cropped_datacube[ind_min_SN:ind_max_SN, :, :],
                                      axis=0)

        #  convert from signal/0.5 A to signal/A
        flux_per_AA = 2 * flux_per_half_AA

        # show flux/AA
        plt.clf()
        p = plt.imshow(flux_per_AA, origin="lower")
        plt.title('flux per AA')
        plt.colorbar(p)
        plt.show()

        # then, I estimate the noise/AA.
        sigma_per_half_pixel = np.std(noise_cube[ind_min_SN:ind_max_SN,:,:], axis=0)
        sigma = np.sqrt(2) * sigma_per_half_pixel

        # then, estimate the poisson noise
        sigma_poisson = poisson_noise(self.exp_time, flux_per_AA, sigma, per_second=True)
        plt.imshow(sigma_poisson,origin="lower")
        plt.title('poisson noise')
        plt.colorbar()
        plt.show()
        
        # save the SN_per_AA to self
        self.SN_per_AA = flux_per_AA / sigma_poisson
        plt.imshow(self.SN_per_AA, origin="lower")
        plt.title('S/N ratio')
        plt.colorbar()
        plt.show()

        
#########################################

    def select_region(self):
        '''
        Function to select the pixels that will be used for Voronoi binning by minimum pixel SN.
        '''
        
        # center the SN map on highest value
        SN_y_center, SN_x_center = np.unravel_index(self.SN_per_AA.argmax(), self.SN_per_AA.shape)
        max_radius = 50 # in case the datacube has bright pixels unrelated to the deflector galaxy
        
        # make grid the same size as cropped datacube
        xx = np.arange(self.radius_in_pixels * 2 + 1)
        yy = np.arange(self.radius_in_pixels * 2 + 1)
        xx, yy = np.meshgrid(xx, yy)
        
        # calculate distance from center
        dist = np.sqrt((xx - SN_x_center) ** 2 + (yy - SN_y_center) ** 2)
        
        # create mask of SN pixels > than the minimum we set
        SN_mask = (self.SN_per_AA > self.pixel_min_SN) & (dist < max_radius)
        
        # mask the x-y grid and SN map
        xx_1D = xx[SN_mask]
        yy_1D = yy[SN_mask]
        SN_1D = self.SN_per_AA[SN_mask]
        
        # stack the masked x-y grid and SN for input to Voronoi binning
        # the fourth term here is just an array of 1s because vorbin takes the signal and noise separately
        self.voronoi_binning_input = np.vstack((xx_1D, yy_1D, SN_1D, np.ones(SN_1D.shape[0])))
        
        # plot the region that will be binned
        plt.imshow(SN_mask, origin="lower", cmap='gray')
        plt.imshow(self.SN_per_AA, origin="lower", alpha=0.9)  #
        plt.title('region selected for voronoi binning (S/N > %s)' % self.pixel_min_SN)
        plt.axis('off')
        plt.colorbar()
        plt.show()
        
        
#########################################        
    
    def voronoi_binning(self):
        '''
        Function takes the pixels and S/N selectd from the "select_region" function and bins using Voronoi binning technique.
        '''
        
        # take the voronoi binning input as x-y grid, S/N (and noise =1 )
        x, y, signal, noise = self.voronoi_binning_input
        
        # voronoi binning assigns bin numbers 
        binNum, xNode, yNode, bin_cen_x, bin_cen_y, sn, nPixels, scale = voronoi_2d_binning(
                x, y, signal, noise, self.bin_target_SN, plot=1, quiet=1)
        
        # plot the binning
        plt.tight_layout()
        plt.pause(1)
        plt.clf()
        
        # save luminosity-weighted center of each bin
        self.bin_centers = np.column_stack((bin_cen_x, bin_cen_y))
        
        # stack x-y pixel grid with the assigned bin number
        self.voronoi_binning_output = np.column_stack([x, y,binNum])
        
        # for each bin, sum the spectra of all pixels belonging to that bin
        self.voronoi_binning_data = \
                    np.zeros(
                                (
                                    int(np.max(self.voronoi_binning_output.T[2]))+1,
                                    self.cropped_datacube.shape[0]
                                )
                            )
        
        #  "check" allows plotting the pixels with their bin number
        check = np.zeros(self.cropped_datacube[0, :, :].shape)
        
        # loop through all the pixels and sum spectra in bins
        for i in range(self.voronoi_binning_output.shape[0]):
            
            # take x, y, and binNum for pixel
            wx = int(self.voronoi_binning_output[i][0])
            wy = int(self.voronoi_binning_output[i][1])
            num = int(self.voronoi_binning_output[i][2])
            
            # add this pixel spectrum to the appropriate bin spectrum
            self.voronoi_binning_data[num]=self.voronoi_binning_data[num]+self.cropped_datacube[:,wy,wx]
            
            # update the "check" array for plotting
            check[wy, wx] = num+1
        
        # save number of bins
        self.nbins = self.voronoi_binning_data.shape[0]
        print("Number of bins =", self.nbins)
        
        # plot to check the binning with luminosity weighted centers
        p=plt.imshow(check, origin="lower", cmap='sauron')
        plt.scatter(self.bin_centers[:,0], self.bin_centers[:,1], c='k', marker='.', s=2)
        plt.colorbar(p)
        plt.pause(1)
        
        
#########################################

    def ppxf_bin_spectra(self, fit_poisson_noise=False, plot_bin_fits=False):
        '''
        Function to loop through bin spectra constructed with "voronoi_binning" function. Fit each spectrum with ppxf using "global_template" constructed during the "ppxf_central_spectrum" function.
        
        Optional keywords:
        
        fit_poisson_noise - not working in this code yet, but if True a better noise contribution after assuming a uniform noise across all wavelengths
        
        plot_bin_fits - if True, each bin fit will be plotted with bin number and velocity dispersion measurement. Takes quite a bit more time, but is recommended.
        '''
        
        # bin_kinematics will be an array of five entries (mean velocity, velocity dispersion, errors on both, and chi2 value)
        self.bin_kinematics = np.zeros(shape=(0,5))
        
        # loop through nbins
        for i in range(self.nbins):
            
            # take the bin spectrum (data)
            bin_spectrum = self.voronoi_binning_data[i]
            
            # rebin the spectrum with restframe wavelengths in log space, "galaxy" is now the data to be fit
            galaxy, log_wavelengths, velscale = ppxf_util.log_rebin(self.rest_wave_range, bin_spectrum)
            
            # take the wavelengths of the data
            wavelengths = np.exp(log_wavelengths)
            
            # cut the data, background source, and wavelengths to the wave_min and wave_max we specified
            galaxy = galaxy[wavelengths>self.wave_min]
            background_source = self.background_spectrum.copy()
            background_source = background_source[wavelengths>self.wave_min]
            wavelengths = wavelengths[wavelengths>self.wave_min]
            galaxy = galaxy[wavelengths<self.wave_max]
            background_source = background_source[wavelengths<self.wave_max]
            wavelengths = wavelengths[wavelengths<self.wave_max]
            
            # take the log of the now-cut wavelength array
            log_wavelengths = np.log(wavelengths)
            
            # take the wavelength range of the global template
            lam_range_global_temp = np.array([self.global_template_wave.min(), self.global_template_wave.max()])
            
            # keep only the good pixels, after de-redshift, the initial redshift is zero.
            goodPixels = ppxf_util.determine_goodpixels(log_wavelengths, lam_range_global_temp, 0)
            
            # find the indices of wave_min and wave_max in wavelengths array
            ind_min = find_nearest(wavelengths, self.wave_min)
            ind_max = find_nearest(wavelengths, self.wave_max)
            
            # mask the appropriate wavelengths and gas emission lines
            mask=goodPixels[goodPixels<ind_max]
            mask = mask[mask>ind_min]
            boolen = ~((2956 < mask) & (mask < 2983))  # mask the Mg II
            mask = mask[boolen]
            boolen = ~((2983 < mask) & (mask < 3001))  # mask the Mg II
            mask = mask[boolen]
            
            # setup with initial guesses
            vel = c*np.log(1 + 0)   # eq.(8) of Cappellari (2017)
            start = [vel, 250.]  # (km/s), starting guess for [V, sigma]
            #bounds = [[-500, 500],[50, 450]] # not necessary
            t = clock()
            
            # set the initial noise as uniform across wavelengths from input noise estimate
            noise = np.full_like(galaxy, self.noise)
            
            # fit the bin spectrum with the global template using ppxf
            pp = ppxf(self.global_template, # global template created in "ppxf_central_spectrum" function
                      galaxy, # bin spectrum (data)
                      noise, # noise estimate
                      velscale, # velocity scale of data (resolution)
                      start, # initial guess
                      sky=background_source, # background source spectrum 
                      plot=False, # we plot later 
                      quiet=self.quiet, # suppress the outputs
                        moments=2, # fit only mean velocity and velocity dispersion
                      goodpixels=mask, # mask of wavelengths 
                        degree=self.degree, # degree of additive polynomial for fitting
                        velscale_ratio=self.velscale_ratio, # resolution ratio of data vs template
                        lam=wavelengths, # wavelengths in restframe of data
                        lam_temp=self.global_template_wave, # wavelenghts in restframe of the global template
                        )
            
            # Do another fit using the noise from the previous fit to make a better estimate of the Poisson noise
            # I have a bug here
            if fit_poisson_noise==True:
                
                # take the outputs of the previous fit
                data = pp.galaxy
                model = pp.bestfit
                log_axis = wavelengths
                
                # make linear axis of wavelengths and rebin data and model to linear axis
                lin_axis = np.linspace(self.wave_min, self.wave_max, data.size)
                data_lin = de_log_rebin(log_axis, data, lin_axis)
                model_lin = de_log_rebin(log_axis, model, lin_axis)
                # get the noise as the residual between data and model
                noise_lin = data_lin - model_lin
                
                # estimate poisson noise
                noise_poisson = poisson_noise(self.exp_time, model_lin,
                                      np.std(noise_lin[self.wave_min:self.wave_max]),
                                      per_second=True)
                
                # fit again with the better noise estimate
                pp = ppxf(self.global_template, 
                          galaxy, 
                          noise_poisson, 
                          velscale, 
                          start, 
                          sky=background_source, 
                          plot=False,#plot_bin_fits, 
                          quiet=self.quiet,
                            moments=2, 
                          goodpixels=mask,
                            degree=self.degree,
                            velscale_ratio=self.velscale_ratio,
                            lam=wavelengths,
                            lam_temp=self.global_template_wave,
                            )
            
            # for viewing each bin spectrum fit
            if plot_bin_fits==True:
                
                # take the background source, data, model
                background = background_source * pp.weights[-1]
                data = pp.galaxy
                model = pp.bestfit
                log_axis = wavelengths
                
                # rebin on linear axis
                lin_axis = np.linspace(self.wave_min, self.wave_max, data.size)
                back_lin = de_log_rebin(log_axis, background, lin_axis)
                model_lin = de_log_rebin(log_axis, model, lin_axis)
                data_lin = de_log_rebin(log_axis, data, lin_axis)
                # take noise as residual
                noise_lin = data_lin - model_lin
                
                # find the indices of the restframe wavelengths that are closest to the min and max we want for plot limits
                plot_ind_min = find_nearest(lin_axis, self.wave_min)
                plot_ind_max = find_nearest(lin_axis, self.wave_max)
                
                # make the figure
                plt.figure(figsize=(8,6))
                
                # plot data, model, background-subtracted data, and residual
                plt.plot(lin_axis, data_lin, 'k-', label='data')
                plt.plot(lin_axis, model_lin, 'r-', label='model ('
                                                             'lens+background)')
                plt.plot(lin_axis, data_lin - back_lin, 'm-',
                         label='remove background source from data', alpha=0.5)
                plt.plot(lin_axis, back_lin + np.full_like(back_lin, 0.9e-5), 'c-',label='background source', alpha=0.7)
                plt.plot(lin_axis, noise_lin + np.full_like(back_lin, 0.9e-5), 'g-',
                         label='noise (data - best model)', alpha=0.7)
                
                # set up axis, etc.
                plt.legend(loc='best')
                plt.ylim(np.nanmin(noise_lin[plot_ind_min:plot_ind_max])/1.1, np.nanmax(data_lin[plot_ind_min:plot_ind_max])*1.1)
                plt.xlim(self.wave_min, self.wave_max)
                plt.xlabel('wavelength (A)')
                plt.ylabel('relative flux')
                plt.title(f'Bin {i} - Velocity dispersion - {int(pp.sol[1])} km/s')
                plt.show()
                plt.pause(1)
                
            # save the kinematics and initial error estimates, as well as chi2 (V, VD, dV, dVD, chi2)
            # pp.error is formal error from ppxf; if fit is reliable, can be corrected with chi2 as shown below
            # ideally, we want to be able to instead sample the velocity space for more reliable error estimate
            self.bin_kinematics = np.vstack(
                                            (
                                                self.bin_kinematics, 
                                                 np.hstack( 
                                                             (
                                                                 pp.sol[:2],
                                                                 (pp.error*np.sqrt(pp.chi2))[:2],
                                                                 pp.chi2) 
                                                              )
                                             )
                                            )

            
#########################################

    def make_kinematic_maps(self):
        '''
        Function to make 2D maps of kinematic components and errors of all bins measured in "ppxf_bin_spectra" function.
        '''
        
        # make arrays of kinematic components and error of size number of pixels
        VD_array    =np.zeros(self.voronoi_binning_output.shape[0])
        dVD_array   =np.zeros(self.voronoi_binning_output.shape[0])
        V_array     =np.zeros(self.voronoi_binning_output.shape[0])
        dV_array    =np.zeros(self.voronoi_binning_output.shape[0])

        # loop through each pixel and assign the kinematics values from the bin measurements
        for i in range(self.voronoi_binning_output.shape[0]):
            
            # num is bin number
            num=int(self.voronoi_binning_output.T[2][i])
            # take bin kinematics
            vd = self.bin_kinematics[num][1]
            dvd = self.bin_kinematics[num][3]
            v = self.bin_kinematics[num][0]
            dv = self.bin_kinematics[num][2]
            # update the array with the pixel's assigned kinematics
            VD_array[i]=vd
            dVD_array[i]=dvd
            V_array[i]=v
            dV_array[i]=dv
        
        # stack the pixel kinematics with the pixel bin information
        self.pixel_details=np.vstack((self.voronoi_binning_output.T, VD_array, dVD_array, V_array, dV_array))
        
        # dimension of the square cropped datacube
        dim = self.radius_in_pixels*2+1
        
        # create the 2D kinematic maps by looping through each pixel and taking teh values from "self.pixel_details", pixels with no kinematics are 'nan'
        # velocity dispersion
        self.VD_2d=np.zeros((dim, dim))
        self.VD_2d[:]=np.nan
        for i in range(self.pixel_details.shape[1]):
            self.VD_2d[int(self.pixel_details[1][i])][int(self.pixel_details[0][i])]=self.pixel_details[3][i]
            
        # error in velocity dispersion
        self.dVD_2d=np.zeros((dim, dim))
        self.dVD_2d[:]=np.nan
        for i in range(self.pixel_details.shape[1]):
            self.dVD_2d[int(self.pixel_details[1][i])][int(self.pixel_details[0][i])]=self.pixel_details[4][i]
        
        # velocity
        self.V_2d=np.zeros((dim, dim))
        self.V_2d[:]=np.nan
        for i in range(self.pixel_details.shape[1]):
            self.V_2d[int(self.pixel_details[1][i])][int(self.pixel_details[0][i])]=self.pixel_details[5][i]
        
        # error in velocity
        self.dV_2d=np.zeros((dim, dim))
        self.dV_2d[:]=np.nan
        for i in range(self.pixel_details.shape[1]):
            self.dV_2d[int(self.pixel_details[1][i])][int(self.pixel_details[0][i])]=self.pixel_details[6][i]

            
#########################################

    def plot_kinematic_maps(self):
        '''
        Function to plot the kinematic maps made in "make_kinematic_maps"
        '''

        # velocity dispersion
        plt.figure()
        plt.imshow(self.VD_2d,origin='lower',cmap='sauron')
        cbar1 = plt.colorbar()
        cbar1.set_label(r'$\sigma$ [km/s]')
        #plt.savefig(target_dir + obj_name + '_VD.png')
        plt.pause(1)
        plt.clf()
        
        # error in velocity dispersion
        plt.figure()
        plt.imshow(self.dVD_2d, origin='lower', cmap='sauron',vmin=0, vmax=40)
        cbar2 = plt.colorbar()
        cbar2.set_label(r'd$\sigma$ [km/s]')
        #plt.savefig(target_dir + obj_name + '_dVD.png')
        plt.pause(1)
        plt.clf()
        
        # mean velocity
        # subtract the "bulk" velocity, small offset in galaxy velocity from redshift error
        bulk = np.nanmedian(self.V_2d)
        plt.figure()
        plt.imshow(self.V_2d-bulk,origin='lower',cmap='sauron',vmin=-100, vmax=100)
        cbar3 = plt.colorbar()
        cbar3.set_label(r'Vel [km/s]')
        plt.title("Velocity map")
        #plt.savefig(target_dir + obj_name + '_V.png')
        plt.pause(1)
        plt.clf()
        
        # error in velocity
        plt.figure()
        plt.imshow(self.dV_2d,origin='lower',cmap='sauron')
        cbar4 = plt.colorbar()
        cbar4.set_label(r'dVel [km/s]')
        plt.title("error on velocity")
        plt.pause(1)
        plt.clf()
        