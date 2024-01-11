# __author__: Shawn Knabel
# 2024/01/11
# test script for slacs_kcwi_kinematics.py

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pathlib # to create directory
import dill as pickle
from datetime import datetime
import os

from ppxf.ppxf import ppxf
from pathlib import Path
from scipy import ndimage
from urllib import request
from scipy import ndimage
from time import perf_counter as clock
from scipy import interpolate
from astropy.visualization import simple_norm
from astropy.modeling.models import Sersic2D
import astropy.units as u
from vorbin.voronoi_2d_binning import voronoi_2d_binning

# import the module
from slacs_kcwi_kinematics import slacs_kcwi_kinematics

# Universal parameters
# todays date
date = datetime.now().strftime("%Y_%m_%d")
# data directory
data_dir = f'{os.getcwd()}/data/' #'/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'

#------------------------------------------------------------------------------
# Kinematics systematics initial choices
# aperture
aperture = 'R2'
# wavelength range
wave_min = 3400
wave_max = 4300 ########### NOTE J0330 will be different
# degree of the additive Legendre polynomial in ppxf
degree = 4 # 900/250 = 3.6 round up

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
bin_target_SN = 15.
# minimum SN of pixels to be included in voronoi bining
pixel_min_SN = 1.
# stellar population
sps_name = 'emiles'

#------------------------------------------------------------------------------
# Object specific
# Set 'obj_name', 'z', 'T_exp'
obj_name = 'SDSSJ0029-0055'
obj_abbr = obj_name[4:9] # e.g. J0029
zlens = 0.227 # lens redshift
exp_time = 1800*5 # exposure time in seconds... 
lens_center_x,lens_center_y = 61, 129

# other necessary directories ... Be very careful! This is how we will make sure we are using 
mos_name = f'KCWI_{obj_abbr}_icubes_mosaic_0.1457'
kcwi_datacube = f'{data_dir}{mos_name}.fits'
#spectrum from the lens center # using R=2
central_spectrum_file = f'{data_dir}{obj_abbr}_central_spectrum_{aperture}.fits' 
background_spectrum_file = None #f'{data_dir}{obj_abbr}_spectrum_background_source.fits'
background_source_mask_file = None #f'{mos_dir}{obj_abbr}_background_source_mask.reg'

#------------------------------------------------------------------------------
# initialize the class
j0029_kinematics = slacs_kcwi_kinematics(
                                             mos_dir=data_dir,
                                             kin_dir=data_dir,
                                             obj_name=obj_name,
                                             kcwi_datacube_file=kcwi_datacube,
                                             central_spectrum_file=central_spectrum_file,
                                             background_spectrum_file=background_spectrum_file,
                                             background_source_mask_file=background_source_mask_file,
                                             zlens=zlens,
                                             exp_time=exp_time,
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
                                             bin_target_SN=bin_target_SN,
                                             pixel_min_SN=pixel_min_SN,
                                             plot=True,
                                             quiet=False
)

# run the kinematics extraction
j0029_kinematics.run_slacs_kcwi_kinematics(fit_poisson_noise=False, plot_bin_fits=False)

print('#########################################################')
print("Job's finished!")