
# **Quick Script for Segmenting**

__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__webpage__   = 'http://cellularimaging.org'
__download__  = 'http://www.github.com/lahmmond/BrainJ-Python'

##%%

# Add the BrainJ install directory to the path
import sys
import os
import numpy as np

import sys
import warnings

import BrainJ.Main.Main as main
from BrainJ.Main.Timer import time, Timer


import segmentation_models as sm
from keras.models import load_model
from patchify import patchify, unpatchify

import cv2
from tifffile import imread, imwrite

from math import trunc

import skimage.io
from skimage.transform import rescale, downscale_local_mean
from skimage import color, data, filters, measure, morphology, segmentation, util, exposure, restoration


import cupy as cp

import matplotlib.pyplot as plt

from stardist.models import StarDist2D
import tensorflow as tf
from csbdeep.utils import download_and_extract_zip_file, plot_some, axes_dict, plot_history, Path, download_and_extract_zip_file, normalize
from csbdeep.data import RawData, create_patches
from csbdeep.io import load_training_data, save_tiff_imagej_compatible
from csbdeep.models import Config, CARE

#check gpu
if tf.test.gpu_device_name() == '':
    print('You do not have GPU access.')

else:
    print('You have GPU access')
    # !nvidia-smi

# %% Provide Brain dir or dirs: - convert to loop


brain_dirs = ["D:\Project_Data\BrainJ Datasets\BrainJ validation\Automated/"]# ,
# "D:/BrainJ Datasets/Brain2/",
# "D:/BrainJ Datasets/etc/"]



#image_dir = r'D:\Project_Data\BrainJ Datasets\BrainJ validation\Automated_DAPI\Images'+'/' #DAPI
#image_dir = r'D:\Project_Data\BrainJ Datasets\BrainJ validation\Automated\Raw_cfos'+'/' #cfos
image_dir = r'D:\Project_Data\BrainJ Datasets\BrainJ validation\Automated\tdtomato\Raw_tdtomato'+'/'

#1cfos, 2 tdtomato, 4 dapi. unless dapi alone
channel = 2

if channel == 1:
    rest_model_path = "D:/Dropbox/Github/BrainJ-Python/BrainJ/Models/2023_07_cfos_2D_1_6um_widefield_256x256_resnet34_V3_1.h5"
    rest_type = ("tf", "resnet34")
    max_int = 65535
    seg_model = ('StarDist2D', '2D_versatile_fluo')
    seg_thresh= 0.05
    normalize_range= (0, 100)
    scale= 1  # Additional scaling applied to data - scaleing up improves stardist on low res data
    preprocess= (0, 0, 0)  # Tophat, gaussian, intensity subtraction (background)
    prob_thresh= 0.65  # default 0.5, for dapi 0.05
    cell_size= (10, 180)  # min, max 30, 170 good for 1.6um sampling, 15 good for nuclei or smaller cells
    intensity_filter= 0  # Filter out detected cells below this intensity from results
    save_val_data= True  # Saves all validation data

if channel == 2:
    rest_model_path = "D:/Dropbox/Github/BrainJ-Python/BrainJ/Models/2023_02_Trap2_1_6um_widefield_resnet34_V2_2023_03_backbone_50epochs.hdf5"
    rest_type = ("tf", "resnet34")
    max_int = 20000
    seg_model = ('StarDist2D', '2D_versatile_fluo')
    seg_thresh = 0.05
    normalize_range = (0, 100)
    scale = 1  # Additional scaling applied to data - scaleing up improves stardist on low res data
    preprocess = (0, 0, 0)  # Tophat, gaussian, intensity subtraction (background)
    prob_thresh = 0.65  # default 0.5, for dapi 0.05
    cell_size = (10, 180)  # min, max 30, 170 good for 1.6um sampling, 15 good for nuclei or smaller cells
    intensity_filter = 0  # Filter out detected cells below this intensity from results
    save_val_data = True  # Saves all validation data

if channel == 4:
    rest_model_path = "D:/Dropbox/Github/BrainJ-Python/BrainJ/Models/DAPI_Enhanced_Care_2D_V2"
    rest_type = ("care", 0)
    seg_model = ('StarDist2D', '2D_versatile_fluo')
    seg_thresh = 0.05
    normalize_range = (20,99.8) #(0, 100)#
    scale = 1  # Additional scaling applied to data - scaleing up improves stardist on low res data
    preprocess = (0, 0, 0)  # Tophat, gaussian, intensity subtraction (background)
    prob_thresh = 0.05  # default 0.5, for dapi 0.05
    cell_size = (15,180)  # min, max 30, 170 good for 1.6um sampling, 15 good for nuclei or smaller cells
    intensity_filter = 0  # Filter out detected cells below this intensity from results
    save_val_data = True  # Saves all validation data

# Load in experiment parameters and analysis settings
settings, locations = main.initialize_brainJ(brain_dirs[0])

# Modify specific parameters and settings:
settings.save_intermediate_data = True
locations.annotations_table = "C:/Users/Luke_H/Desktop/BrainJ Atlas/ABA_CCF_25_2017/Atlas_Regions.csv"
settings.tiles_for_prediction = (2 ,2)
settings.c1_cell_analysis = True



print(f"Restoring and segmenting channel {channel}...")
# import series as a npy array
files = [f for f in os.listdir(image_dir) if f.endswith('.tif') or f.endswith('.tiff')]
# sort files to ensure they're in the correct order
files.sort()
# load all images into a list of numpy arrays
image = [imread(image_dir + f) for f in files]
slice_filenames = []
for f in files:
    slice_filenames.append(f)

# stack all images into a single numpy array
image = np.stack(image)
# remove zeroes to avoid care issues
image[image == 0] = settings.tissue_background

tiles_for_prediction = settings.tiles_for_prediction
validation_scale = settings.validation_scale
nms_threshold = settings.stard_nms_thresh
if save_val_data == True:
    print("Validation images will be saved.")

if len(image.shape) == 3:
    shapeY = image.shape[1]
    shapeX = image.shape[2]
else:
    shapeY = image.shape[0]
    shapeX = image.shape[1]
    image = np.expand_dims(image, axis=0)

# load restoration model
print(f"Restoration model = {rest_model_path}")
if rest_model_path != None:
    if rest_type[0] == 'care':
        if os.path.isdir(rest_model_path) is False:
            raise RuntimeError(rest_model_path, "not found, check settings and model directory")
        rest_model = CARE(config=None, name=rest_model_path)
        print(f"Section image shape: {image.shape}")
        print(f"Scale used: {scale}")
        # apply restoration model to channel
        print(f"Restoring image for channel {channel}")

        restored = np.empty((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint16)

        start_time = time.time()
        for slice_idx in range(image.shape[0]):
            loop_start_time = time.time()

            print(f"\rRestoring slice {slice_idx + 1} of {image.shape[0]}")  # , end="\r", flush=True)

            slice_img = image[slice_idx]
            with main.HiddenPrints():
                # rescale if necessary
                # tiles_for_prediction = tuple(x * trunc(scale) for x in tiles_for_prediction)

                # restore image
                restoredslice = rest_model.predict(slice_img, axes='YX', n_tiles=tiles_for_prediction)

                # convert to 16bit
                restoredslice = restoredslice.astype(np.uint16)
                # remove low intensities that are artifacts
                # as restored images have varying backgrounds due to high variability in samples. Detect background with median, then add the cutoff
                # cutoff = np.median(restored) + rest_type[1]
                # restored[restored < cutoff] = 0
                background = restoration.rolling_ball(restoredslice, radius=5)
                restoredslice = restoredslice - background
                restored[slice_idx] = restoredslice

                loop_end_time = time.time()
                loop_duration = loop_end_time - loop_start_time
                total_elapsed_time = loop_end_time - start_time
                avg_time_per_loop = total_elapsed_time / (slice_idx + 1)
                estimated_total_time = avg_time_per_loop * image.shape[0]

                print(f"{loop_duration:.2f} seconds. Estimated total time: {estimated_total_time:.2f} minutes")

        print("Complete.\n")

    if rest_type[0] == 'tf':
        print(f"Image shape: {image.shape}")
        print(f"Scale used: {scale}")
        print("Restoring image using loaded model...")

        if os.path.isfile(rest_model_path) is False:
            raise RuntimeError(rest_model_path, "not found, check settings and model directory")

        model = load_model(rest_model_path, compile=False)
        patch_size = 256
        BACKBONE = rest_type[1]
        threshold = 0.05
        preprocess_input = sm.get_preprocessing(BACKBONE)

        restored = np.empty((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)

        start_time = time.time()
        for slice_idx in range(image.shape[0]):
            loop_start_time = time.time()
            # print(f"\rSegmenting slice {slice_idx+1} of {image.shape[0]}", end="\r", flush=True)
            print(f"\rSegmenting slice {slice_idx + 1} of {image.shape[0]}")
            # Process one slice at a time.
            slice_img = image[slice_idx]
            # pad as required to dvide by patch size
            dim_x = slice_img.shape[1]
            dim_y = slice_img.shape[0]

            upsize_x = (np.ceil(dim_x / patch_size) * patch_size)  # Nearest size divisible by our patch size
            pad_x = int(upsize_x - dim_x)

            upsize_y = (np.ceil(dim_y / patch_size) * patch_size)  # Nearest size divisible by our patch size
            pad_y = int(upsize_y - dim_y)

            padded_image = np.pad(slice_img, ((0, pad_y), (0, pad_x)), constant_values=0)

            # convert image to float32
            #if channel == 1:
            #    padded_image = padded_image.astype(np.float32)  # temporary until retrained model for 0-1norm
           #else:
            padded_image = (padded_image.astype('float32')) / max_int

            # patchify
            patches = patchify(padded_image, (patch_size, patch_size),
                               step=(patch_size, patch_size))  # Step=256 for 256 patches means no overlap

            # print(image.shape, padded_image.shape, patches.shape)

            patched_prediction = []
            for i in range(patches.shape[0]):
                # for i in range(1):
                patch = patches[i, :, :, :]
                # only one channel so need to convert to rgb
                patch = np.stack((patch,) * 3, axis=-1)
                patch = np.squeeze(patch)
                # print(patch.shape)
                # run stack through prediction, update later to concantenate all images, and run as one stack
                # single_patch_img = np.stack((single_patch_img,)*1, axis=0)
                patch = preprocess_input(patch)
                with main.HiddenPrints():
                    pred = model.predict(patch)
                pred = np.where(pred > threshold, 255, 0)
                # pred = np.argmax(pred, axis=3)
                # pred = pred[0, :,:]

                patched_prediction.append(pred)
            # turn list into np array
            patched_prediction = np.array(patched_prediction)
            predicted_patches_reshaped = np.reshape(patched_prediction, (patches.shape[0], patches.shape[1], 256, 256))
            reconstructed_image = unpatchify(predicted_patches_reshaped, padded_image.shape)
            restoredslice = reconstructed_image[0:dim_y, 0:dim_x]
            #restoredslice = restoredslice.astype(np.uint8)
            restored[slice_idx] = restoredslice

            loop_end_time = time.time()
            loop_duration = loop_end_time - loop_start_time
            total_elapsed_time = loop_end_time - start_time
            avg_time_per_loop = total_elapsed_time / (slice_idx + 1)
            estimated_total_time = avg_time_per_loop * image.shape[0]

            # print(f"Segmenting slice {slice_idx+1} of {image.shape[0]} took {loop_duration:.2f} seconds. Estimated total time: {estimated_total_time/60:.2f} minutes")
            print(f"{loop_duration:.2f} seconds. Estimated total time: {estimated_total_time / 60:.2f} minutes")
        print("\nComplete\n")

else:
    print("No restoration selected. Using raw data for detection.")
    restored = image

# with main.HiddenPrints():
# preprocess if necessary

# label on restored image
print(f"Detecting cells for channel {channel}")

if seg_model[0] == 'StarDist2D':
    model = StarDist2D.from_pretrained(seg_model[1])

    labels = np.empty((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint32)

    if scale > 1:
        tiles_for_prediction = tuple(x * trunc(scale) for x in tiles_for_prediction)

    start_time = time.time()
    for slice_idx in range(restored.shape[0]):
        loop_start_time = time.time()

        print(f"\rDetecting cells in slice {slice_idx + 1} of {restored.shape[0]}")
        # print(f"\rDetecting cells in slice {slice_idx+1} of {restored.shape[0]}", end="\r", flush=True)
        # Process one slice at a time.
        slice_restored = restored[slice_idx]

        if scale > 1:
            slice_restored = rescale(slice_restored, scale, anti_aliasing=False)

        with main.HiddenPrints():

            slice_labels, _ = model.predict_instances(normalize(slice_restored, normalize_range[0], normalize_range[1]),
                                                      axes='YX',
                                                      prob_thresh=prob_thresh,  # default 0.5 #for dapi 0.05
                                                      nms_thresh=nms_threshold,  # default 0.4 #for dapi 0.3
                                                      n_tiles=tiles_for_prediction,
                                                      show_tile_progress=False,
                                                      verbose=False)
        # Rescale back for saving
        if scale > 1:
            slice_restored = rescale(slice_restored, 1 / scale, anti_aliasing=False)

            slice_labels = cv2.resize(slice_labels, (shapeX, shapeY), interpolation=cv2.INTER_NEAREST)

            # labels = rescale(labels, 1/scale, anti_aliasing=True)
            print(f"restored shape is now: {restored.shape} Label shape is now: {labels.shape}")

        labels[slice_idx] = slice_labels

        loop_end_time = time.time()
        loop_duration = loop_end_time - loop_start_time
        total_elapsed_time = loop_end_time - start_time
        avg_time_per_loop = total_elapsed_time / (slice_idx + 1)
        estimated_total_time = avg_time_per_loop * restored.shape[0]

        print(f"{loop_duration:.2f} seconds. Estimated total time: {estimated_total_time / 60:.2f} minutes")
        # print(f"Detecting cells in slice {slice_idx+1} of {restored.shape[0]} took {loop_duration:.2f} seconds. Estimated total time: {estimated_total_time/60:.2f} minutes")
    print("\nComplete.\n")

masks = labels > 1
# FILTER BASED ON SIZE HERE


print(f"Masks size = {(round(sys.getsizeof(masks) / (1024 ** 3), 3))} GB of type 8bit")
print(f"Labels size = {(round(sys.getsizeof(labels) / (1024 ** 3), 3))} GB of type {labels.dtype}")

#make output dir in image_dir
if not os.path.exists(image_dir + "/" + str(channel)):
    os.makedirs(image_dir + "/" + str(channel))

if not os.path.exists(image_dir + "/" + str(channel)+ "/restored"):
    os.makedirs(image_dir + "/" + str(channel)+ "/restored")

np.save(image_dir + "/" + str(channel) + '/_labels.npy', labels)
np.save(image_dir + "/" + str(channel) + '/_masks.npy', masks)

# save validation data:
if save_val_data == True:
    print(f"restored type is {restored.dtype} \n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if settings.validation_format == "tif":
            # imwrite(restore_val_dir+str(channel)+"/"+str(section)+".tif", restored.astype('uint16'), imagej=True)
            # imwrite(restore_val_dir+str(channel)+"/"+str(section)+".tif", labels.astype('uint16'), imagej=True)
            if rest_type[0] != 'tf':
                # restored = exposure.rescale_intensity(restored, in_range=(np.min(restored), np.max(restored)), out_range='uint16')
                restored.astype(np.uint16)

                for i, slice_2d in enumerate(restored):
                    #filename = f"section{i:04}.tif"
                    filename = slice_filenames[i]
                    filepath =image_dir + "/" + str(channel) + "/restored/" + filename
                    skimage.io.imsave(filepath, slice_2d.astype(np.uint16), plugin='tifffile', photometric='minisblack')
            if rest_type[0] == 'tf':
                restored = exposure.rescale_intensity(restored, in_range=(np.min(channel), np.max(restored)),
                                                      out_range='uint8')
                restored.astype(np.uint8)
                for i, slice_2d in enumerate(restored):
                    #filename = f"section{i:04}.tif"
                    filename = slice_filenames[i]
                    filepath = image_dir + "/" + str(channel) + "/restored/" + filename
                    skimage.io.imsave(filepath, slice_2d.astype(np.uint8), plugin='tifffile', photometric='minisblack')

            for i, slice_2d in enumerate(labels):
                #filename = f"section{i:04}.tif"
                filename = slice_filenames[i]
                filepath = image_dir + "/" + str(channel) + "/" + filename
                skimage.io.imsave(filepath, slice_2d.astype(np.float32), plugin='tifffile', photometric='minisblack')
        # if validation_format == "tif" and scale > 1:
        # imwrite(restore_val_dir+str(channel)+"/"+str(section)+".tif", rescale(restored, 1/scale, anti_aliasing=False).astype('uint16'), imagej=True)
        # imwrite(cell_val_dir+str(channel)+"/"+str(section)+".tif", rescale(labels, 1/scale, anti_aliasing=False).astype('uint16'), imagej=True)


print("")
