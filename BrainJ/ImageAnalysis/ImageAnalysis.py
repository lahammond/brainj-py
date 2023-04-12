# -*- coding: utf-8 -*-
"""
Image Analysis
==========


"""
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__webpage__   = 'http://cellularimaging.org'
__download__  = 'http://www.github.com/lahmmond/BrainJ-Python'


import os
import numpy as np
import pandas as pd
import ast


import BrainJ.Main.Main as main
import subprocess
from subprocess import Popen, PIPE

from BrainJ.Main.Timer import time, Timer

from csbdeep import data

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


##############################################################################
# Load in Locations and Settings
#class Load_Settings_and_Locations():
#  def __init__(self,Locations, Settings):
#      Locations = Locations
#      Settings = Settings


##############################################################################
# Cell Segmentation Functions
##############################################################################

def cell_detection(settings, locations):
    #count sections and confirm equal in all folders
    #modified to return a large df with a column "channel" that notes each channel a cell is located in.
      
    section_count = check_registered_sections(locations.registered_dir)

    rawcells = process_sections(section_count, settings, locations)
    #rawcells1, rawcells2, rawcells3, rawcells4 = process_sections(section_count, settings, locations)
    
    if settings.save_intermediate_data == True:
        rawcells.to_csv(locations.raw_measurements_dir + 'raw_cells.csv',index=False) #, compression='gzip')
                
    return rawcells

def check_registered_sections(registered_dir):
    global reg_channels
    reg_channels = main.count_dirs(registered_dir)

    if reg_channels >= 1:
        section_count = [file_i for file_i in os.listdir(registered_dir+"1/") if file_i.endswith('.tif')]
        print(len(section_count), "registered sections to be processed.")
        
    if reg_channels >= 2:
        if main.count_files(registered_dir+"2/") != len(section_count): print("Sections missing from", registered_dir,"2/")
            
    if reg_channels >= 3:
        if main.count_files(registered_dir+"3/") != len(section_count): print("Sections missing from", registered_dir,"3/")
            
    if reg_channels >= 4:
        if main.count_files(registered_dir+"4/") != len(section_count): print("Sections missing from", registered_dir,"4/")
    
    return section_count

def process_sections(section_count, settings, locations):
    #find a way to make these as needed elegantly
    #currently need all dataframes but should only require dfs being used
    rawcells = pd.DataFrame()

    for section in range(len(section_count)):
    #for section in range(1):

      print('Processing section', section+1, "of" ,len(section_count))

      # Run a function that processes the sections:
      #approx 1min/section for 1.6µm resolution brain sections
      rawcells = analyze_section(section, section_count, settings, locations, rawcells)
      print("Time elased = ", round(Timer.timers['section_processing']/60), "minutes. Estimated total time = ", round(Timer.timers['section_processing']/(section+1)*len(section_count)/60), "minutes.")

    return rawcells

@Timer(name= "section_processing", text="Section processing time: {:.1f} seconds.")
def analyze_section(section, section_count, settings, locations, rawcells):    
    #****had to make raw and restore images global for pipeline to work - didn't have this issue earlier?
    #find better way to to do this. shouldn't be an issue as they are called and used in the function??
    
    #tissue_background required to fix high contrast around edges of tissue during restoration - fills zero values with this value

    #Load in raw channel files
    global c1_raw
    if settings.c1_cell_analysis == True:
        c1_raw = imread(locations.registered_dir+"1/"+section_count[section])
        c1_raw[c1_raw == 0] = settings.tissue_background
    
    if settings.c2_cell_analysis == True:
        global c2_raw
        c2_raw = imread(locations.registered_dir+"2/"+section_count[section])
        c2_raw[c2_raw == 0] = settings.tissue_background
    if settings.c3_cell_analysis == True:
        global c3_raw
        c3_raw = imread(locations.registered_dir+"3/"+section_count[section])  
        c3_raw[c3_raw == 0] = settings.tissue_background
    if settings.c4_cell_analysis == True:
        global c4_raw
        c4_raw = imread(locations.registered_dir+"4/"+section_count[section])
        c4_raw[c4_raw == 0] = settings.tissue_background


    #Also - find sensible way of dealing with DAPI channel if necessary?

    # Find a cleaner way of operating optionally over multiple channels
    # Also how to pass prb and nms tresholds for each analysis

    #All channels provide own prob_thresholds in parameters
    # later on if using a different approach for DAPI e.g. UNet - then need to rewrite to allow different settings for DAPI

    if settings.c1_cell_analysis == True:
        print("Restoring and segmenting channel 1...")
        global c1_restore
        global c1_mask
        global c1_labels
        c1_restore, c1_labels, c1_mask = restore_and_segment(1, section, settings.c1_rest_model_path, settings.c1_rest_type, 
                                                             settings.c1_seg_model,
                                                             c1_raw, settings.c1_scale, settings.c1_preprocess, 
                                                             settings.c1_prob_thresh, settings.c1_normalize,
                                                             settings.c1_save_val_data,
                                                             settings, locations)
    if settings.c2_cell_analysis == True:
        print("Restoring and segmenting channel 2...")
        global c2_restore
        global c2_mask
        global c2_labels
        c2_restore, c2_labels, c2_mask = restore_and_segment(2, section, settings.c2_rest_model_path, settings.c2_rest_type, 
                                                             settings.c2_seg_model,
                                                             c2_raw, settings.c2_scale, settings.c2_preprocess, 
                                                             settings.c2_prob_thresh, settings.c2_normalize,
                                                             settings.c2_save_val_data,
                                                             settings, locations)
    if settings.c3_cell_analysis == True:
        print("Restoring and segmenting channel 3...")
        global c3_restore
        global c3_mask
        global c3_labels
        c3_restore, c3_labels, c3_mask = restore_and_segment(3, section, settings.c3_rest_model_path, settings.c3_rest_type, 
                                                             settings.c3_seg_model,
                                                             c3_raw, settings.c3_scale, settings.c3_preprocess, 
                                                             settings.c3_prob_thresh, settings.c3_normalize,
                                                             settings.c3_save_val_data,
                                                             settings, locations)
    if settings.c4_cell_analysis == True:
        print("Restoring and segmenting channel 4...")
        global c4_restore
        global c4_mask
        global c4_labels
        c4_restore, c4_labels, c4_mask = restore_and_segment(4, section, settings.c4_rest_model_path, settings.c4_rest_type, 
                                                             settings.c4_seg_model,
                                                             c4_raw, settings.c4_scale, settings.c4_preprocess, 
                                                             settings.c4_prob_thresh, settings.c4_normalize,
                                                             settings.c4_save_val_data,
                                                             settings, locations)

        #create slice table and validation image - after all images have been restored and segmented!
    if settings.c1_cell_analysis == True:
        print("Measuring channel 1...")
        if settings.c1_measure == True:
            cells1_slice_table, cells1_colored_by_area = measure_and_create_validation_image(1, section, c1_raw, 
                                                                                             c1_restore, c1_labels, 
                                                                                             settings.c1_scale,
                                                                                             settings.c1_save_val_data,
                                                                                             settings.c1_cell_size,
                                                                                             settings, locations)
            rawcells = pd.concat((rawcells, cells1_slice_table))
        else:
            cells1_slice_table, cells1_colored_by_area = measure_and_create_validation_image_centroid_only(1, section, c1_raw, 
                                                                                                           c1_restore, 
                                                                                                           c1_labels, 
                                                                                                           settings.c1_scale,
                                                                                                           settings.c1_save_val_data,
                                                                                                           settings.c1_cell_size,
                                                                                                           settings, locations)
            rawcells = pd.concat((rawcells, cells1_slice_table))
    if settings.c2_cell_analysis == True:
        print("Measuring channel 2...")
        if settings.c2_measure == True:
            cells2_slice_table, cells2_colored_by_area = measure_and_create_validation_image(2, section, c2_raw, 
                                                                                             c2_restore, c2_labels, 
                                                                                             settings.c2_scale,
                                                                                             settings.c2_save_val_data,
                                                                                             settings.c2_cell_size,
                                                                                             settings, locations)
            rawcells = pd.concat((rawcells, cells2_slice_table))
        else:
            cells2_slice_table, cells2_colored_by_area = measure_and_create_validation_image_centroid_only(2, section, c2_raw, 
                                                                                                           c2_restore, 
                                                                                                           c2_labels, 
                                                                                                           settings.c2_scale, 
                                                                                                           settings.c2_save_val_data,
                                                                                                           settings.c2_cell_size,
                                                                                                           settings, locations)
            rawcells = pd.concat((rawcells, cells2_slice_table))
    if settings.c3_cell_analysis == True:
        print("Measuring channel 3...")
        if settings.c3_measure == True:
            cells3_slice_table, cells3_colored_by_area = measure_and_create_validation_image(3, section, c3_raw, 
                                                                                             c3_restore, c3_labels, 
                                                                                             settings.c3_scale,
                                                                                             settings.c3_save_val_data,
                                                                                             settings.c3_cell_size,
                                                                                             settings, locations)
            rawcells = pd.concat((rawcells, cells3_slice_table))
        else:
            cells3_slice_table, cells3_colored_by_area = measure_and_create_validation_image_centroid_only(3, section, c3_raw, 
                                                                                                           c3_restore, 
                                                                                                           c3_labels, 
                                                                                                           settings.c3_scale, 
                                                                                                           settings.c3_save_val_data,
                                                                                                           settings.c3_cell_size,
                                                                                                           settings, locations)
            rawcells = pd.concat((rawcells, cells3_slice_table))
    if settings.c4_cell_analysis == True:
        print("Measuring channel 4...")
        if settings.c4_measure == True:
            cells4_slice_table, cells4_colored_by_area = measure_and_create_validation_image(4, section, c4_raw, 
                                                                                             c4_restore, c4_labels, 
                                                                                             settings.c4_scale,
                                                                                             settings.c4_save_val_data,
                                                                                             settings.c4_cell_size,
                                                                                             settings, locations)
            rawcells = pd.concat((rawcells, cells4_slice_table))
        else:
            cells4_slice_table, cells4_colored_by_area = measure_and_create_validation_image_centroid_only(4, section, c4_raw, 
                                                                                                           c4_restore, 
                                                                                                           c4_labels, 
                                                                                                           settings.c4_scale,
                                                                                                           settings.c4_save_val_data,
                                                                                                           settings.c4_cell_size,
                                                                                                           settings, locations)
            rawcells = pd.concat((rawcells, cells4_slice_table))
    
    #round to 2 decimal places - could do this somewhere in the function?
    rawcells = rawcells.round(2)
        
    return rawcells
   
def restore_and_segment(channel, section, rest_model_path, rest_type, seg_model, image, scale, preprocess, prob_thresh, normalize_range, saveval, settings, locations):
    #preprocess = list(map(int, (ast.literal_eval(preprocess))))
    #tiles_for_prediction = list(map(int, (ast.literal_eval(settings['tiles_for_prediction']))))
    tiles_for_prediction = settings.tiles_for_prediction
    #validation_scale = list(map(int, (ast.literal_eval(settings['validation_scale']))))
    validation_scale = settings.validation_scale
    #rest_type = ast.literal_eval(rest_type)
    #seg_model = ast.literal_eval(seg_model)
    #scale = int(scale)
    #prob_thresh = float(prob_thresh)
    #nms_threshold = float(settings['stard_nms_thresh'])
    nms_threshold = settings.stard_nms_thresh    
    #saveval = int(saveval)
    if saveval == True:
        print("Validation images will be saved.")

    #load restoration model
    print("Restoration model = ", rest_model_path)
    if rest_model_path != None:
        if rest_type[0] == 'care':
            rest_model = CARE(config=None, name=rest_model_path)
            print("Section image shape: ", image.shape)
            shape0 = image.shape[0]
            shape1 = image.shape[1]
            print("Scale used: ", scale)
            #apply restoration model to channel
            print("Restoring image for channel ", channel)

            with main.HiddenPrints():
                
                #rescale if necessary
                #tiles_for_prediction = tuple(x * trunc(scale) for x in tiles_for_prediction)

                #restore image
                restored = rest_model.predict(image, axes='YX', n_tiles=tiles_for_prediction)

                #convert to 16bit
               
                restored = restored.astype(np.uint16)
                #remove low intensities that are artifacts 
                #as restored images have varying backgrounds due to high variability in samples. Detect background with median, then add the cutoff
                #cutoff = np.median(restored) + rest_type[1]
                #restored[restored < cutoff] = 0
                background = restoration.rolling_ball(restored, radius=5)
                restored = restored - background
                
        if rest_type[0] == 'tf':
            print("Section image shape: ", image.shape)
            shape0 = image.shape[0]
            shape1 = image.shape[1]
            print("Scale used: ", scale)
            print("Restoring image using loaded model...")
            
            model = load_model(rest_model_path, compile=False)
            patch_size = 256
            BACKBONE = rest_type[1]
            preprocess_input = sm.get_preprocessing(BACKBONE)
            
            #pad as required to dvide by patch size
            dim_x = image.shape[1]
            dim_y = image.shape[0]
            
            upsize_x = (np.ceil(dim_x/patch_size)*patch_size) #Nearest size divisible by our patch size
            pad_x = int(upsize_x - dim_x)

            upsize_y = (np.ceil(dim_y/patch_size)*patch_size) #Nearest size divisible by our patch size
            pad_y = int(upsize_y - dim_y)
            
            padded_image = np.pad(image, ((0,pad_y),(0,pad_x)), constant_values=0)

            #convert image to float32  
            padded_image = (padded_image.astype('float32')) / 65535.

            #patchify
            patches = patchify(padded_image, (patch_size, patch_size), step=patch_size)  #Step=256 for 256 patches means no overlap

            print(image.shape, padded_image.shape, patches.shape)   
            
            patched_prediction = []
            for i in range(patches.shape[0]):
            #for i in range(1):          
                    patch = patches[i,:,:,:]
                    #only one channel so need to convert to rgb
                    patch = np.stack((patch,)*3, axis=-1)
                    #print(patch.shape)
                    # run stack through prediction, update later to concantenate all images, and run as one stack
                    #single_patch_img = np.stack((single_patch_img,)*1, axis=0)
                    with main.HiddenPrints():
                        pred = model.predict(patch)
                    #pred = np.argmax(pred, axis=3)
                    #pred = pred[0, :,:]

                    patched_prediction.append(pred)
            #turn list into np array
            patched_prediction = np.array(patched_prediction)
            predicted_patches_reshaped = np.reshape(patched_prediction, (patches.shape[0], patches.shape[1], 256,256) )
            reconstructed_image = unpatchify(predicted_patches_reshaped, padded_image.shape)
            restored = reconstructed_image[0:dim_y, 0:dim_x]
            #print(restored.dtype)
            #restored = exposure.rescale_intensity(restored, in_range=(np.min(restored), np.max(restored)), out_range='float32')
            #restored = exposure.rescale_intensity(restored, in_range=(np.min(restored), np.max(restored*2)), out_range='uint16')
            #restored = restored.astype(np.uint16)
            
    else:
        print("No restoration selected. Using raw data for detection.")
        restored = image
    
    
    #with main.HiddenPrints():
        #preprocess if necessary

    if preprocess[0] > 0:
        print("Using tophat filter: ", preprocess[0])
        filterSize =morphology.disk(preprocess[0])
        # Applying the Top-Hat operation
        restored = morphology.white_tophat(restored, filterSize)
        
    #remove  background
    if preprocess[2] > 0:
        restored = np.clip(restored.astype(np.int32) - preprocess[2], 0,65535).astype(np.uint16)
    
    #apply gaussian
    if preprocess[1] > 0:
        restored = filters.gaussian(restored, sigma=preprocess[1])

 

    #label on restored image
    print("Detecting cells for channel ", channel)
        
    if scale > 1:
        restored = rescale(restored, scale, anti_aliasing=False)
        tiles_for_prediction = tuple(x * trunc(scale) for x in tiles_for_prediction)
    
    if seg_model[0] == 'StarDist2D':
        model = StarDist2D.from_pretrained(seg_model[1])


        with main.HiddenPrints():
        
            labels, _ = model.predict_instances(normalize(restored, normalize_range[0],normalize_range[1]),
                                          axes='YX', 
                                          prob_thresh=prob_thresh, #default 0.5 #for dapi 0.05
                                          nms_thresh=nms_threshold,   #default 0.4 #for dapi 0.3
                                          n_tiles=tiles_for_prediction,
                                          show_tile_progress=False,
                                          verbose=False)
    # Rescale back for saving
    if scale >1:
        restored = rescale(restored, 1/scale, anti_aliasing=False)
            

        labels = cv2.resize(labels, (shape1,shape0), interpolation=cv2.INTER_NEAREST)
        #labels = rescale(labels, 1/scale, anti_aliasing=True)
        print("restored shape is now: ", restored.shape, " Label shape is now: ", labels.shape)
    
    
    masks = labels>1
    
    
    # FILTER BASED ON SIZE HERE
    
    
    
    #save validation data:
    if saveval == True:
        print("restored type is" + str(restored.dtype))
        if restored.dtype != "uint16":
            restored = exposure.rescale_intensity(restored, in_range=(np.min(restored), np.max(restored)), out_range='uint16')

            restored.astype(np.uint16)

        
        print("labels type is" + str(labels.dtype))
    
        if settings.validation_format == "tif":
            #imwrite(restore_val_dir+str(channel)+"/"+str(section)+".tif", restored.astype('uint16'), imagej=True)
            #imwrite(restore_val_dir+str(channel)+"/"+str(section)+".tif", labels.astype('uint16'), imagej=True)

            skimage.io.imsave(locations.restore_val_dir+str(channel)+"/"+str(section)+".tif", restored.astype(np.uint16), plugin='tifffile', photometric='minisblack')
            skimage.io.imsave(locations.cell_val_dir+str(channel)+"/"+str(section)+".tif", labels.astype(np.uint16), plugin='tifffile', photometric='minisblack')
        #if validation_format == "tif" and scale > 1:
            #imwrite(restore_val_dir+str(channel)+"/"+str(section)+".tif", rescale(restored, 1/scale, anti_aliasing=False).astype('uint16'), imagej=True)
            #imwrite(cell_val_dir+str(channel)+"/"+str(section)+".tif", rescale(labels, 1/scale, anti_aliasing=False).astype('uint16'), imagej=True)
            
        if settings.validation_format != "tif":
            
            #scale down by validation scale and save as jpeg
            #validation_scale = tuple(x * scale for x in validation_scale)
            
            #might not work - with larger files - may need to use skimage.io as above
            cv2.imwrite(locations.restore_val_dir+str(channel)+"/"+str(section)+".jpg", rescale(restored, 1/(validation_scale[0]), anti_aliasing=False).astype('uint16'), [cv2.IMWRITE_JPEG_QUALITY, settings.validation_jpeg_comp]) 
            #save filtered labels colored by area
            cv2.imwrite(locations.cell_val_dir+str(channel)+"/"+str(section)+".jpg", rescale(labels, 1/(validation_scale[0]), anti_aliasing=False).astype('uint32'), [cv2.IMWRITE_JPEG_QUALITY, settings.validation_jpeg_comp]) 
    print("")    
    return restored, labels, masks




##############################################################################
# Measurement Functions
##############################################################################

def measure_int_and_add_column(raw_image, labels, main_table, column_name):
    #Measure channel
    #print("Image shape: ", raw_image.shape, "Labels shape:", labels.shape)
    temp_table = pd.DataFrame(
        measure.regionprops_table(
            labels,
            intensity_image=raw_image,
            properties=['label', 'mean_intensity'], #area is volume for 3D images
        )
    )

    #rename mean intensity
    temp_table.rename(columns={'mean_intensity':column_name}, inplace=True)
    temp_col = temp_table[column_name]

    #combine columns with main table
    main_table = main_table.join(temp_col)
    
    return main_table


        
def measure_and_create_validation_image(channel, section, raw_image, restored_image, labels, scale, saveval, cell_size, settings, locations):
    #validation_scale = list(map(int, (ast.literal_eval(settings['validation_scale']))))
    validation_scale = settings.validation_scale
    #scale = float(scale)
    #saveval = int(saveval)
    
    #print("labels", labels.shape)
    #print("intensity", c1_raw.shape)
    #channel 1 and properties
    main_table = pd.DataFrame(
        measure.regionprops_table(
            labels,
            intensity_image=c1_raw,
            properties=['label', 'centroid', 'area', 'mean_intensity'], #area is volume for 3D images
        )
    )

    #rename mean intensity
    main_table.rename(columns={'mean_intensity':'C1_intensity'}, inplace=True)
    main_table.rename(columns={'centroid-0':'y'}, inplace=True)
    main_table.rename(columns={'centroid-1':'x'}, inplace=True)
    main_table.insert(loc=2, column='z', value=section+1)
    main_table = main_table.reindex(columns=["label","x","y","z","area","C1_intensity"])
    
    #rescale values if scale up has been used: -- no longer used as regions passed at image scale
    #if scale > 1:
    #    main_table['x'] = main_table['x'].div(scale).round(2)
    #    main_table['y'] = main_table['x'].div(scale).round(2)
    #    main_table['area'] = main_table['x'].div(scale).round(2)  
        
    #measure intensity of other channels - check for low res dapi

    if os.path.isdir(locations.registered_dir+"2/") and settings.DAPI_channel != 2 and settings.c2_measure == 1 or (settings.DAPI_channel == 2 and settings.full_res_DAPI == True and settings.c2_measure == 1) == True:
        main_table = measure_int_and_add_column(c2_raw, labels, main_table, "C2_raw_int")
        
    if os.path.isdir(locations.registered_dir+"3/") and int(settings.DAPI_channel) != 3 and settings.c3_measure == 1 or (settings.DAPI_channel == 3 and settings.full_res_DAPI == True and settings.c3_measure == 1) == True:
        main_table = measure_int_and_add_column(c3_raw, labels, main_table, "C3_raw_int")

    if os.path.isdir(locations.registered_dir+"4/") and int(settings.DAPI_channel) != 4 and settings.c4_measure == 1 or (settings.DAPI_channel == 4 and settings.full_res_DAPI == True and settings.c4_measure == 1) == True:
        main_table = measure_int_and_add_column(c4_raw, labels, main_table, "C4_raw_int")

    
    #measure overlap # later on - include a check to not measure overlap for own channel
    
    if settings.c1_cell_analysis == True:
        main_table = measure_int_and_add_column(c1_restore, labels, main_table, "C1_restore_int")
        main_table = measure_int_and_add_column(c1_mask, labels, main_table, "C1_mask_overlap")
        #main_table["C1_mask_overlap2"] =  main_table["C1_mask_overlap"] /  main_table["area"] 
    if settings.c2_cell_analysis == True:
        main_table = measure_int_and_add_column(c2_restore, labels, main_table, "C2_restore_int")
        main_table = measure_int_and_add_column(c2_mask, labels, main_table, "C2_mask_overlap")
        #main_table["C2_mask_overlap2"] =  main_table["C2_mask_overlap"] /  main_table["area"] 
    if settings.c3_cell_analysis == True:
        main_table = measure_int_and_add_column(c3_restore, labels, main_table, "C3_restore_int")
        main_table = measure_int_and_add_column(c3_mask, labels, main_table, "C3_mask_overlap")
        #main_table["C3_mask_overlap2"] =  main_table["C3_mask_overlap"] /  main_table["area"] 
    if settings.c4_cell_analysis == True:
        main_table = measure_int_and_add_column(c4_restore, labels, main_table, "C4_restore_int")
        main_table = measure_int_and_add_column(c4_mask, labels, main_table, "C4_mask_overlap")
        #main_table["C4_mask_overlap2"] =  main_table["C4_mask_overlap"] /  main_table["area"] 
    
    
    #* IF SLOW - could potentially only measure in objects that match filtered table
    #filter objects
    volume_min = cell_size[0] 
    volume_max = cell_size[1] 

    filtered_table = main_table[(main_table['area'] > volume_min) & (main_table['area'] < volume_max) ] 
    
    filtered_table.insert(loc=0, column='channel', value=channel)

    print("After filtering", len(filtered_table), "objects remain from total of", len(main_table))
    
    #create colored by area image - don't need to color by area
    #colored_by_area = util.map_array(
    #    labels,
    #    np.asarray(filtered_table['label']),
    #    np.asarray(filtered_table['area']).astype(float),
    #    #np.asarray(filtered_table['label']).astype(float),
    #    )
    
    #create validation images
    if saveval == True:
    #scale down 2x2 then save as jpeg
    #save restored image
        #save filtered labels colored by area - overwrites labels - area helps to see different neighboring
        #cells as different intensities, but all around same value - rather than thousands of values, potentially requiring 32bit.
        
        if settings.validation_format == "tif":
            imwrite(locations.cell_val_dir+str(channel)+"/"+str(section)+".tif", labels.astype('uint32'))
            #imwrite(locations.cell_val_dir+str(channel)+"/"+str(section)+".tif", rescale(colored_by_area, 1, anti_aliasing=False).astype('uint32'), imagej=True)
        else:
            
            cv2.imwrite(locations.cell_val_dir+str(channel)+"/"+str(section)+".jpg", labels.astype('uint16'), [cv2.IMWRITE_JPEG_QUALITY, settings.validation_jpeg_comp])   
            #cv2.imwrite(locations.cell_val_dir+str(channel)+"/"+str(section)+".jpg", rescale(colored_by_area, 1/(validation_scale[0]), anti_aliasing=False).astype('uint16'), [cv2.IMWRITE_JPEG_QUALITY, settings.validation_jpeg_comp])   
    
    return filtered_table, labels

def measure_and_create_validation_image_centroid_only(channel, section, raw_image, restored_image, labels, scale, saveval, cell_size, settings, locations):
    #validation_scale = list(map(int, (ast.literal_eval(settings['validation_scale']))))
    validation_scale = settings.validation_scale
    #scale = float(scale)
    #saveval = int(saveval)
    #print("labels", labels.shape)
    #print("intensity", c1_raw.shape)
    #channel 1 and properties
    main_table = pd.DataFrame(
        measure.regionprops_table(
            labels,
            intensity_image=c1_raw,
            properties=['label', 'centroid', 'area'], #area is volume for 3D images
        )
    )

    #rename mean intensity
    main_table.rename(columns={'centroid-0':'y'}, inplace=True)
    main_table.rename(columns={'centroid-1':'x'}, inplace=True)
    main_table.insert(loc=2, column='z', value=section+1)
    main_table = main_table.reindex(columns=["label","x","y","z","area"])
    
    #rescale values if scale up has been used: - no longer used as labels same scale as image
    #if scale > 1:
    #    main_table['x'] = main_table['x'].div(scale).round(2)
    #    main_table['y'] = main_table['x'].div(scale).round(2)
    #    main_table['area'] = main_table['x'].div(scale).round(2)  
        
    
    #* IF SLOW - could potentially only measure in objects that match filtered table
    #filter objects
    volume_min = cell_size[0] 
    volume_max = cell_size[1] 

    filtered_table = main_table[(main_table['area'] > volume_min) & (main_table['area'] < volume_max) ] 
    
    #add in channel column
    filtered_table.insert(loc=0, column='channel', value=channel)

    print("After filtering", len(filtered_table), "objects remain from total of", len(main_table))
    
    #create colored by area image - don't need to color by area
    colored_by_area = util.map_array(
        labels,
        np.asarray(filtered_table['label']),
        np.asarray(filtered_table['area']).astype(float),
        #np.asarray(filtered_table['label']).astype(float),
        )
    #save validation data
    
    if saveval == True:
    #scale down 2x2 then save as jpeg
    #save restored image
        #save filtered labels colored by area - overwrites labels - area helps to see different neighboring
        #cells as different intensities, but all around same value - rather than thousands of values, potentially requiring 32bit.
        #validation_scale = tuple(x * scale for x in validation_scale)
        
        if settings.validation_format == "tif":
            imwrite(locations.cell_val_dir+str(channel)+"/"+str(section)+".tif", rescale(colored_by_area, 1/(validation_scale[0]), anti_aliasing=False).astype('uint16'), imagej=True)
        else:
            cv2.imwrite(locations.cell_val_dir+str(channel)+"/"+str(section)+".jpg", rescale(colored_by_area, 1/(validation_scale[0]), anti_aliasing=False).astype('uint16'), [cv2.IMWRITE_JPEG_QUALITY, settings.validation_jpeg_comp])  
    
    return filtered_table, colored_by_area


##############################################################################
# Atlas Mapping Functions
##############################################################################


def transform_cells_V1(rawcells1, rawcells2, rawcells3, rawcells4, settings, locations):
    if settings.c1_cell_analysis == True:
        process, transformedcells1 = transform_cell_locations(1, rawcells1, settings, locations)
        transformedcells1.to_csv(locations.raw_measurements_dir + "transformed_cells_1.csv",index=False) #, compression='gzip')
    
    if settings.c2_cell_analysis == True:
        process, transformedcells2 = transform_cell_locations(2, rawcells2, settings, locations)
        transformedcells2.to_csv(locations.raw_measurements_dir + 'transformed_cells_2.csv',index=False) #, compression='gzip')
        
    if settings.c3_cell_analysis == True:
        process, transformedcells3 = transform_cell_locations(3, rawcells3, settings, locations)
        transformedcells3.to_csv(locations.raw_measurements_dir + 'transformed_cells_3.csv',index=False) #, compression='gzip')
        
    if settings.c4_cell_analysis == True:
        process, transformedcells4 = transform_cell_locations(4, rawcells4, settings, locations)
        transformedcells4.to_csv(locations.raw_measurements_dir + 'transformed_cells_4.csv',index=False) #, compression='gzip')
        
    return transformedcells1, transformedcells2, transformedcells3, transformedcells4


def transform_cells_V2(rawcells, settings, locations):
    process, transformedcells = transform_cell_locations_V2(rawcells, settings, locations)
        
    return transformedcells


def import_and_transform_raw_cells_V2(settings, locations):
    rawcells = pd.read_csv(locations.raw_measurements_dir + "raw_cells.csv") 
    
    process, transformedcells = transform_cell_locations_V2(rawcells, settings, locations)
    
    transformedcells.to_csv(locations.raw_measurements_dir + 'transformed_cells.csv',index=False) #, compression='gzip')
    
        
    return transformedcells

def import_and_transform_raw_cells_V1(settings, locations):
    if settings.c1_cell_analysis == True:
        rawcells1 = pd.read_csv(locations.raw_measurements_dir + "raw_cells_1.csv") 
        process, transformedcells1 = transform_cell_locations(1, rawcells1, settings, locations)
        transformedcells1.to_csv(locations.raw_measurements_dir + "transformed_cells_1.csv",index=False) #, compression='gzip')
    
    if settings.c2_cell_analysis == True:
        rawcells2 = pd.read_csv(locations.raw_measurements_dir + 'raw_cells_2.csv')
        process, transformedcells2 = transform_cell_locations(2, rawcells2, settings, locations)
        transformedcells2.to_csv(locations.raw_measurements_dir + 'transformed_cells_2.csv',index=False) #, compression='gzip')
        
    if settings.c3_cell_analysis == True:
       rawcells3 = pd.read_csv(locations.raw_measurements_dir + 'raw_cells_3.csv')
       process, transformedcells3 = transform_cell_locations(3, rawcells3, settings, locations)
       transformedcells3.to_csv(locations.raw_measurements_dir + 'transformed_cells_3.csv',index=False) #, compression='gzip')
        
    if settings.c4_cell_analysis == True:
        rawcells4= pd.read_csv(locations.raw_measurements_dir + 'raw_cells_4.csv')
        process, transformedcells4 = transform_cell_locations(4, rawcells4, settings, locations)
        transformedcells4.to_csv(locations.raw_measurements_dir + 'transformed_cells_4.csv',index=False) #, compression='gzip')
        
    return transformedcells1, transformedcells2, transformedcells3, transformedcells4

def import_transformed_cells_V2(settings, locations):

    transformedcells = pd.read_csv(locations.raw_measurements_dir + "transformed_cells.csv") 

    return transformedcells

def import_V1_transformed_cells_V2(settings, locations):
    if settings.c1_cell_analysis == True:
        transformedcells1 = pd.read_csv(locations.raw_measurements_dir + "transformed_cells_1.csv") 
        transformedcells1.insert(loc=0, column='channel', value=1)
        
    if settings.c2_cell_analysis == True:
        transformedcells2 = pd.read_csv(locations.raw_measurements_dir + 'transformed_cells_2.csv')
        transformedcells2.insert(loc=0, column='channel', value=2)
        
    if settings.c3_cell_analysis == True:
        transformedcells3 = pd.read_csv(locations.raw_measurements_dir + 'transformed_cells_3.csv')
        transformedcells3.insert(loc=0, column='channel', value=3)
        
    if settings.c4_cell_analysis == True:
        transformedcells4= pd.read_csv(locations.raw_measurements_dir + 'transformed_cells_4.csv')
        transformedcells4.insert(loc=0, column='channel', value=4)
    
    transformedcells = pd.concat([transformedcells1, transformedcells2, transformedcells3, transformedcells4])  
    
    return transformedcells
    

def import_transformed_cells(settings, locations):
    if settings.c1_cell_analysis == True:
        transformedcells1 = pd.read_csv(locations.raw_measurements_dir + "transformed_cells_1.csv") 
    
    if settings.c2_cell_analysis == True:
        transformedcells2 = pd.read_csv(locations.raw_measurements_dir + 'transformed_cells_2.csv')
        
    if settings.c3_cell_analysis == True:
        transformedcells3 = pd.read_csv(locations.raw_measurements_dir + 'transformed_cells_3.csv')
        
    if settings.c4_cell_analysis == True:
        transformedcells4= pd.read_csv(locations.raw_measurements_dir + 'transformed_cells_4.csv')
        
    return transformedcells1, transformedcells2, transformedcells3, transformedcells4
    

@Timer(name= "transform_cells", text="Transforming cells processing time: {:.1f} seconds.\n")
def transform_cell_locations(cell_channel, cell_table_input, settings, locations):
    input_res = settings.final_res
    section_thickness = settings.section_thickness
    atlas_res = settings.atlas_res
    
    total_points = cell_table_input.shape[0]
    
    cell_table = cell_table_input.copy(deep=True)
       
    cell_points_file = locations.raw_measurements_dir + "raw_cells_locations_"+str(cell_channel)+".txt"
    
    #resample points to atlas scale OriginalRes x ResampleRes
    cell_table["x"] =  cell_table["x"] * (input_res/atlas_res[0])
    cell_table["y"] =  cell_table["y"] * (input_res/atlas_res[1])
    cell_table["z"] =  cell_table["z"] * (section_thickness/atlas_res[2])

    
    # export text file of cells and prepare of transformix
    (cell_table.round(1)).to_csv(cell_points_file, sep=" ", columns=["x","y","z"], index = False, header=False)
    
    #Add point and total number of cells
    main.line_prepender(cell_points_file,"point\n"+str(total_points))
    
    #run transformix
    transformix_process = subprocess.Popen([settings.elastix_dir+"transformix.exe",
                  "-def",
                  cell_points_file,
                  "-tp", 
                  locations.transform_param,
                  "-out",
                  locations.transformed_cell_dir],
                  stdout=PIPE, stderr=PIPE
                )

    # File can take a few seconds to appear - wait below up to 20 sec for file
    time_to_wait = 1500
    time_counter = 0
    while not os.path.exists(locations.transformed_cell_dir + "outputpoints.txt"):
        time.sleep(1)
        time_counter += 1
        if time_counter > time_to_wait:break
            
            

    # read in transformed points
    transformed_cells = pd.read_csv(locations.transformed_cell_dir + "outputpoints.txt", sep=" ", header=None)
    
    while transformed_cells.shape[0] != total_points:
        time.sleep(5)
        transformed_cells = pd.read_csv(locations.transformed_cell_dir + "outputpoints.txt", sep=" ", header=None)
    
   
    #extract XYZ and update table
    #transformed_cells = transformed_cells.iloc[:,[32,33,34]] # extracts z,y,x columns as x,y,z
    cell_table["x"] = transformed_cells.iloc[:,25].values #updates x with full res transformed Z coords
    cell_table["y"] = transformed_cells.iloc[:,26].values #updates y with full res transformed y coords
    cell_table["z"] = transformed_cells.iloc[:,27].values #updates z with full res transformed x coords
    
    #transformed_cells.to_csv(transformed_cell_dir + "outputpoints_"+str(cell_channel)+".txt",index=False)
        
    os.remove(locations.transformed_cell_dir + "outputpoints.txt")
    os.remove(locations.raw_measurements_dir + "raw_cells_locations_"+str(cell_channel)+".txt")

    return transformix_process, cell_table

@Timer(name= "transform_cells", text="Transforming cells processing time: {:.1f} seconds.\n")
def transform_cell_locations_V2(cell_table_input, settings, locations):
    input_res = settings.final_res
    section_thickness = settings.section_thickness
    atlas_res = settings.atlas_res
    
    total_points = cell_table_input.shape[0]
    
    cell_table = cell_table_input.copy(deep=True)
       
    cell_points_file = locations.raw_measurements_dir + "raw_cells_locations.txt"
    
    #resample points to atlas scale OriginalRes x ResampleRes
    cell_table["x"] =  cell_table["x"] * (input_res/atlas_res[0])
    cell_table["y"] =  cell_table["y"] * (input_res/atlas_res[1])
    cell_table["z"] =  cell_table["z"] * (section_thickness/atlas_res[2])

    
    # export text file of cells and prepare of transformix
    (cell_table.round(1)).to_csv(cell_points_file, sep=" ", columns=["x","y","z"], index = False, header=False)
    
    #Add point and total number of cells
    main.line_prepender(cell_points_file,"point\n"+str(total_points))
    
    #run transformix
    transformix_process = subprocess.Popen([settings.elastix_dir+"transformix.exe",
                  "-def",
                  cell_points_file,
                  "-tp", 
                  locations.transform_param,
                  "-out",
                  locations.transformed_cell_dir],
                  stdout=PIPE, stderr=PIPE
                )

    # File can take a few seconds to appear - wait below up to 20 sec for file
    time_to_wait = 1500
    time_counter = 0
    time.sleep(5)
    while not os.path.exists(locations.transformed_cell_dir + "outputpoints.txt"):
        time.sleep(10)
        time_counter += 1
        if time_counter > time_to_wait:break
            
            

    # read in transformed points
    transformed_cells = pd.read_csv(locations.transformed_cell_dir + "outputpoints.txt", sep=" ", header=None)
    
    while transformed_cells.shape[0] != total_points:
        time.sleep(10)
        transformed_cells = pd.read_csv(locations.transformed_cell_dir + "outputpoints.txt", sep=" ", header=None)
    
   
    #extract XYZ and update table
    #transformed_cells = transformed_cells.iloc[:,[32,33,34]] # extracts z,y,x columns as x,y,z
    cell_table["x"] = transformed_cells.iloc[:,25].values #updates x with full res transformed Z coords
    cell_table["y"] = transformed_cells.iloc[:,26].values #updates y with full res transformed y coords
    cell_table["z"] = transformed_cells.iloc[:,27].values #updates z with full res transformed x coords
    
    #transformed_cells.to_csv(transformed_cell_dir + "outputpoints_"+str(cell_channel)+".txt",index=False)
        
    
    if transformed_cells.shape[0] == total_points:
        os.remove(locations.raw_measurements_dir + "raw_cells_locations.txt")
        os.remove(locations.transformed_cell_dir + "outputpoints.txt")

    return transformix_process, cell_table

def annotate_all_cells(transformedcells, settings, locations):
    #Annotate Cells with atlas location - equivalent to AnnotatePoints
    #without Dask - the cell annotation takes around 1.6 min / million cells
    
    #*** Rewrite with dask
    # note that string heavy dataframes aren't speed up much by using dask dataframes. 
    
    #include estimated time 

    #channel 1
    if settings.c1_cell_analysis == True:
        print("Creating atlas region annotated count table for channel "+str(1)+" ...")
        create_annotated_count_table(transformedcells, locations, 1)

    #channel 2
    if settings.c2_cell_analysis == True:
        print("Creating atlas region annotated count table for channel "+str(2)+" ...")
        create_annotated_count_table(transformedcells, locations, 2)
        
     #channel 3
    if settings.c3_cell_analysis == True:
        print("Creating atlas region annotated count table for channel "+str(3)+" ...")
        create_annotated_count_table(transformedcells, locations, 3)
         
     #channel 4
    if settings.c4_cell_analysis == True:
        print("Creating atlas region annotated count table for channel "+str(4)+" ...")
        create_annotated_count_table(transformedcells, locations, 4)
    

    
def create_annotated_count_table(transformedcells, locations, channel):
    # subset the transformed cells dataframe for the specified channel
    transformedcells_subset = transformedcells[transformedcells['channel'] == channel]
    transformedcells_subset = transformedcells_subset.drop('channel', axis=1)

    # run the cell annotation GPU function
    cell_locations, summary, outofbounds, outofbrain = cell_annotation_gpu(transformedcells_subset, locations)

    # save the locations and summary dataframes to CSV files
    cell_locations.to_csv(locations.cell_analysis_out_dir + f"C{channel}_Annotated_Cells.csv", index=False)
    summary.to_csv(locations.cell_analysis_out_dir + f"C{channel}_Annotated_Cells_Summary.csv")

    print(f"Created atlas region annotated count table for channel {channel}.")


#@Timer(name= "annotate_cells", text="Annotating cells processing time: {:.1f} seconds.")
def annotate_points_v3(transformed_cells, annotations, output_id_dict, acronym_dict, locations):
    
    region_info = pd.read_csv("C:/Users/Luke_H/Desktop/BrainJ Atlas/ABA_CCF_25_2017/Atlas_Regions.csv")

    transformed_cells = transformed_cells.reset_index()

    transformed_cells.insert(loc = 5, column = "hemisphere", value = 0)
    transformed_cells.insert(loc = 6, column = "id", value = 0)
    transformed_cells.insert(loc = 7, column = "acronym", value = 0)
    
    #update annotations to include 0 value for external cells
    
    # Loop over points and find intensity/region ID - currently simple version - but improve with faster optimized loop
    outofbounds = 0
    outofbrain = 0

    for cell in range(transformed_cells.shape[0]):

        #check to make sure it is in atlas bounds
        pos_z = int(transformed_cells.loc[cell,"z"])
        pos_y = int(transformed_cells.loc[cell,"y"])
        pos_x = int(transformed_cells.loc[cell,"x"])
                    
        if pos_z > 0 and pos_z < annotations.shape[0] and pos_y > 0 and pos_y < annotations.shape[1] and pos_x > 0 and pos_x < annotations.shape[2]: 

           #Read out intensity at location - remember XYZ = ZYX in python
            region_id = annotations[pos_z, pos_y, pos_x]
            if region_id >=1: 
                #update transformed_cells with id and acronym
                
                #transformed_cells.at[cell,'id'] = region_id # - if ids matched image this would be correct
                        #but atlas imaged modified for java and some intensities need to be remapped to larger values
                
                #these steps take all the time
                transformed_cells.at[cell,'id'] = output_id_dict[region_id]
                transformed_cells.at[cell,'acronym'] = acronym_dict[region_id]
                if pos_x <= annotations.shape[2]/2:
                    hemisphere= "Right"
                else:
                    hemisphere = "Left"
                # then update hemisphere
                transformed_cells.at[cell,'hemisphere'] = hemisphere
            else:
                outofbrain = outofbrain+1
                #delete row from transformed_cells
                transformed_cells = transformed_cells.drop(index=cell)
        else:
            outofbounds = outofbounds+1
            transformed_cells = transformed_cells.drop(index=cell)

    # then create summary tables 
    #drop unneeded columns and add columns for counts
    
    region_info = region_info.drop(columns=["id","red","green","blue"])
    region_info.rename(columns={'output_id':'id'}, inplace=True)    
    region_info.insert(loc = 0, column = "total_cells_left", value = 0)
    region_info.insert(loc = 0, column = "total_cells_right", value = 0)
    
    #create a summary of the counts: output is dataframe with columns named
    
    #first create a dictionary of left and right
    dict_of_hemi = dict(iter(transformed_cells.groupby('hemisphere')))
    
    if ('Right' in dict_of_hemi) == True: 
        summary_right = pd.Series.to_frame(dict_of_hemi["Right"]['id'].value_counts())
        summary_right.rename(columns={'id':'count'}, inplace=True)
        summary_right['id'] = summary_right.index
        
    if ('Left' in dict_of_hemi) == True:
        summary_left = pd.Series.to_frame(dict_of_hemi["Left"]['id'].value_counts())
        summary_left.rename(columns={'id':'count'}, inplace=True)
        summary_left['id'] = summary_left.index
    
    for region in range(region_info.shape[0]):
        # could likely do this much more efficiently at the very end of all the annotations.
        region_id = region_info.at[region,'id']
        if ('Right' in dict_of_hemi) == True:
            if region_id in summary_right["id"].values:
                region_info.at[region,'total_cells_right'] = summary_right[summary_right['id']== region_info.at[region,'id']]['count'].values[0]
        if ('Left' in dict_of_hemi) == True:
            if region_id in summary_left["id"].values:
                region_info.at[region,'total_cells_left'] = summary_left[summary_left['id']== region_info.at[region,'id']]['count'].values[0]
 
      
    #print(transformed_cells.shape[0], "cells processed.")
    # print(outofbounds, "cells were found to be out of atlas boundaries")
    # print((cell_locations['id'] == 0).sum(), "cells were annotated outside of the brain.")
    # print(cell_locations.shape[0]-(cell_locations['id'] == 0).sum(), "cells were annotated within the brain.")
     
    return transformed_cells, region_info, outofbounds, outofbrain

@Timer(name= "annotate_cells", text="Annotating cells processing time: {:0.1f} seconds.\n")
def cell_annotation_in_blocks(cells, locations):
    print("Estimated time for annotating cells:", int(round(cells.shape[0]*0.0001,-1))60,"minutes.");
    #import annotation image
    global annotations
    annotations = imread(locations.annotations_image)
    
    
    
    #split cell table into managable chunks
    split = np.ceil(cells.shape[0]/15000)
    df_split = np.array_split(cells, split)
    print(len(df_split),"dataframes, which are approximately",len(df_split[0]),"rows long, are being processed.")
 
    #Insert estimated time
    
    #create empty tables and variables
    
    locations_out = pd.DataFrame()
    summary_out = pd.read_csv(locations.annotations_table)
    #create dictionary
    output_id_dict = dict(zip(summary_out['id'],summary_out['output_id']))
    acronym_dict = dict(zip(summary_out['id'],summary_out['acronym']))
    
    outofbounds = 0
    outofbrain = 0
    #drop unneeded columns and add columns for counts
    summary_out = summary_out.drop(columns=["id","red","green","blue"])
    summary_out.rename(columns={'output_id':'id'}, inplace=True)
    summary_out.insert(loc = 0, column = "total_cells_left", value = 0)
    summary_out.insert(loc = 0, column = "total_cells_right", value = 0)

    for i in range(len(df_split)):
        locations, summary, outofbounds_block, outofbrain_block = annotate_points_v3(df_split[i],
                                                                                     annotations, 
                                                                                     output_id_dict, 
                                                                                     acronym_dict, locations)
        locations_out = pd.concat((locations_out, locations))
        summary_out['total_cells_right'] = summary_out['total_cells_right'] + summary['total_cells_right']
        summary_out['total_cells_left'] = summary_out['total_cells_left'] + summary['total_cells_left']
        outofbounds = outofbounds + outofbounds_block
        outofbrain = outofbrain + outofbrain_block
        
    print(locations_out.shape[0],"cells mapped into the brain.")
    print(outofbounds,"cells mapped out of the bounds of the atlas image.",outofbrain,"cells mapped outside of the brain.")
    
    return locations_out, summary_out, outofbounds, outofbrain
        

@Timer(name= "annotate_cells", text="Annotating cells processing time: {:0.1f} seconds.\n")
def cell_annotation_gpu(cells, locations):
        
    #import region information
    region_info = pd.read_csv("C:/Users/Luke_H/Desktop/BrainJ Atlas/ABA_CCF_25_2017/Atlas_Regions.csv")
    
    #import annotation image
    #global annotations
    annotations = imread(locations.annotations_image)
    annotations_gpu = cp.array(annotations)
    
    # measure intensity in atlas
    #convert df to xyz numpy array
    atlas_coordinates = cells[['x', 'y','z']].values
    
    #convert to cupy array
    atlas_coordinates_cp = cp.array(atlas_coordinates, dtype=cp.int32)
    #mask for valid coordinates
    valid_mask = (
      (atlas_coordinates_cp[:, 0] >= 0) & (atlas_coordinates_cp[:, 0] < annotations_gpu.shape[2]) &
      (atlas_coordinates_cp[:, 1] >= 0) & (atlas_coordinates_cp[:, 1] < annotations_gpu.shape[1]) &
      (atlas_coordinates_cp[:, 2] >= 0) & (atlas_coordinates_cp[:, 2] < annotations_gpu.shape[0])
    )

    # Get the valid coordinates - this avoids issues with cells outside image space
    valid_atlas_coordinates_cp = atlas_coordinates_cp[valid_mask]
    
    # Get the values at valid coordinates
    valid_region_ids = annotations_gpu[valid_atlas_coordinates_cp[:, 2], valid_atlas_coordinates_cp[:, 1], valid_atlas_coordinates_cp[:, 0]]

    # Initialize the result array with NaN
    region_ids = cp.full(atlas_coordinates.shape[0], np.nan, dtype=np.float32)

    # Assign the valid values to the result array
    region_ids[valid_mask] = valid_region_ids
    
    
    #measure
    #region_ids = cp.zeros(atlas_coordinates.shape[0], dtype=annotations_gpu.dtype)
    #for idx, coord in enumerate(atlas_coordinates):
    #    x, y, z = coord
    #    region_ids[idx] = annotations_gpu[z, y, x]
    
    outofbounds = int(np.sum(np.isnan(region_ids)))
    outofbrain = int(np.count_nonzero(region_ids == 0))
    
    
    #add column for region id
    cells['id'] = cp.asnumpy(region_ids)
    
    #create hemisphre columns in dataframe
    cells = cells.assign(hemisphere=np.where(cells['x'] <= annotations.shape[2]/2, "Right", "Left"))
    
    #add information to columns based on region id - such as region abbreviation etc
    region_info_subset = region_info[['id','output_id', 'acronym','parent_ID','parent_acronym']]
    
    cells = cells.merge(region_info_subset, on='id', how='left')
    
    #replace id with output_id and then drop the output_id column
    cells['id'] = cells['output_id']
    cells = cells.drop('output_id', axis=1)
    cells = cells.drop('label', axis=1)
    
    # Create a summary table:
    summary_table = cells.groupby(['id', 'hemisphere']).agg(count=('id', 'count'))
    if summary_table.shape[0] >= 1:
        summary_table = summary_table.reset_index()
        summary_table = summary_table.pivot(index='id', columns='hemisphere', values='count')
        # Replace NaN values with zeros
        summary_table = summary_table.fillna(0)
        summary_table.columns = ['total_cells_right', 'total_cells_left']
        summary_table = summary_table.reset_index()
        
        #add in region info for each row:
        summary_table = summary_table.rename(columns={'id': 'output_id'})
    
        summary_table = summary_table.merge(region_info, on='output_id', how='left')
        summary_table = summary_table.drop('id', axis=1)
        summary_table = summary_table.rename(columns={'output_id': 'id'})
        
        #add in new column "total cells"
        total_cells = summary_table['total_cells_right'] + summary_table['total_cells_left']
        summary_table.insert(3, 'total_cells', total_cells)
    
    #append outofbounds and out of pbrain
    outofbounds_row = pd.DataFrame({'id': [0],'total_cells_right': [0],'total_cells_left': [0], 'total_cells': [outofbounds], 'name': ['outside atlas image']})
    
    outofbrain_row = pd.DataFrame({'id': [0],'total_cells_right': [0],'total_cells_left': [0], 'total_cells': [outofbrain], 'name': ['mapped outside of the brain']})

    summary_table = pd.concat([summary_table, outofbounds_row], ignore_index=True)
    summary_table = pd.concat([summary_table, outofbrain_row], ignore_index=True)

          
    print(f"{cells.shape[0]:,}"," cells mapped into the brain.")
    print(outofbounds,"cells mapped out of the bounds of the atlas image.",outofbrain,"cells mapped outside of the brain.")
    
    return cells, summary_table, outofbounds, outofbrain
        

    