# -*- coding: utf-8 -*-
"""
Environment
===========

Initialize a BrainJ environment.

Note
----
To initialize the main functions in a BrainJ script use:
>>> from BrainJ.Environment import *
"""
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__webpage__   = 'http://cellularimaging.org'
__download__  = 'http://www.github.com/lahmmond/BrainJ-Python'

###############################################################################
### Python
###############################################################################

import sys   
import os    

import tifffile
#import pims
import time

import numpy as np                
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from skimage import exposure, segmentation
from skimage.io import imread, imsave, imshow, util
from skimage.util import img_as_ubyte
from scipy import ndimage as ndi #Distance transformation

from IPython.display import clear_output, display



###############################################################################
### BrainJ
###############################################################################

import BrainJ.Main.Main as main
import BrainJ.Main.Timer as timer

#Utilities
#import BrainJ.Utilities.gui as gui

#CNN
#import BrainJ.CNN.unet as unet
#import BrainJ.CNN.multiclass_unet as unetmc

#image analysis
import BrainJ.ImageAnalysis.ImageAnalysis as imgan




###############################################################################
### All
###############################################################################

__all__ = ['sys', 'os', 'tifffile', 'time', 'np',
           'plt', 'figure', 'exposure', 
           'segmentation', 'imread', 'imsave', 'imshow',  'util', 'img_as_ubyte',
           'ndi', 'clear_output',
           'display', 'main', 'timer', 'imgan'];
