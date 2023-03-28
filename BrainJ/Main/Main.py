# -*- coding: utf-8 -*-
"""
Main functions
==========


"""
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__webpage__   = 'http://cellularimaging.org'
__download__  = 'http://www.github.com/lahmmond/BrainJ-Python'


import os
import sys
#import numpy as np
import tensorflow as tf
import pandas as pd
import yaml
import ast

##############################################################################
# Main Functions
##############################################################################
Locations = None
Settings = None

#create dir   
def create_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

#count dirs
def count_dirs(path):
    count = 0
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)):
            count += 1

    return count

#count files
def count_files(path):
    count = 0
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)):
            count += 1

    return count

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

def check_gpu():
  if tf.test.gpu_device_name()=='':
    print('You do not have GPU access.') 

  else:
    print('You have GPU access')
    #!nvidia-smi


def create_brainj_dirs(Settings, Locations):
    create_dir(Locations.processing_dir)
    create_dir(Locations.transformed_cell_dir)
    create_dir(Locations.raw_measurements_dir)
    create_dir(Locations.analysis_out_dir)
    create_dir(Locations.cell_analysis_out_dir)
    
    if Settings.c1_save_val_data == True or Settings.c2_save_val_data == True or Settings.c3_save_val_data == True or Settings.c4_save_val_data == True:
        create_dir(Locations.restore_val_dir)
        create_dir(Locations.cell_val_dir)

    if Settings.c1_cell_analysis == True:
        create_dir(Locations.restore_val_dir+"1/")
        create_dir(Locations.cell_val_dir+"1/")
        #rawcells1 = pd.DataFrame()
    if Settings.c2_cell_analysis == True:
        create_dir(Locations.restore_val_dir+"2/")
        create_dir(Locations.cell_val_dir+"2/")
        #rawcells2 = pd.DataFrame()
    if Settings.c3_cell_analysis == True:
        create_dir(Locations.restore_val_dir+"3/")
        create_dir(Locations.cell_val_dir+"3/")
        #rawcells3 = pd.DataFrame()
    if Settings.c4_cell_analysis == True:
        create_dir(Locations.restore_val_dir+"4/")
        create_dir(Locations.cell_val_dir+"4/")
        #rawcells4 = pd.DataFrame()
        
def initialize_brainJ(brain_dir):
    
    Settings = Create_Settings(brain_dir)
    Locations =  Create_Locations(brain_dir, Settings.atlas_dir)
    create_brainj_dirs(Settings, Locations)
    
    return Settings, Locations
    
class ConfigObject:
    def __init__(self, data):
        self.__dict__.update(data)

def initialize_brainJ1(brain_dir=None):
    # Use default filename if none is provided
    if brain_dir is None:
        brain_dir = 'none.yaml'

    try:
        with open(brain_dir, 'r') as f:
            data = yaml.safe_load(f)
            return ConfigObject(data)
    except FileNotFoundError:
        print(f"Modules loaded. Ready to load dataset.")
        return None


 
##############################################################################
# Main Classes
##############################################################################
class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)

    
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Create_Locations():
  def __init__(self, brain_dir, atlas_dir):
      self.registered_dir =  brain_dir+"3_Registered_Sections/"
      self.processing_dir = brain_dir+"4_Processed_Sections/"
      self.raw_measurements_dir = brain_dir+"4_Processed_Sections/"+"Cell_Locations_and_Intensities/"
      self.restore_val_dir = brain_dir+"4_Processed_Sections/"+"Image_Restore_Validation/"
      self.cell_val_dir = brain_dir+"4_Processed_Sections/"+"Cell_Detection_Validation/"
      self.transformed_cell_dir = brain_dir+"4_Processed_Sections/"+"Transformed_Cells/"
      self.analysis_out_dir = brain_dir+"5_Analysis_Output/"
      self.cell_analysis_out_dir = brain_dir+"5_Analysis_Output/Cell_Analysis/"

      #ATLAS FILE LOCATIONS

      self.template_brain_image = atlas_dir + "Template.tif"
      self.annotations_image = atlas_dir + "Annotation.tif"
      self.annotations_table = atlas_dir + "Atlas_Regions.csv"

      self.affine = atlas_dir + "/Registration_Parameters/MB49_Param_Affine.txt"
      self.bspline = atlas_dir + "/Registration_Parameters/MB49_Param_BSpline.txt"
      self.bspline_lenient = atlas_dir + "/Registration_Parameters/MB49_Param_BSpline_L.txt"
      self.bspline_ultra_lenient = atlas_dir + "/Registration_Parameters/MB49_Param_BSpline_UL.txt"
      self.inv_bspline = atlas_dir + "/Registration_Parameters/MB49_Param_BSpline_Inv.txt"

      # Registration File Locations

      self.align_out = brain_dir+"5_Analysis_Output/5_Analysis_Output/Temp/Template_aligned/"
      self.exp_brain_25 = brain_dir+"3_Registered_Sections/DAPI_25.tif"
      self.exp_brain_mask_25 = brain_dir+"3_Registered_Sections/DAPI_25_Mask.tif"
      self.annotation_image_out = brain_dir+"5_Analysis_Output/Temp/"

      self.transform_param = brain_dir+"5_Analysis_Output/Transform_Parameters/Cell_TransformParameters.1.txt"
      self.transform_param_mod = brain_dir+"5_Analysis_Output/Temp/Template_aligned/ModifiedTransformParameters.1.txt"
      self.inv_transparam_mod = brain_dir+"5_Analysis_Output/Transform_Parameters/ProjectionTransformParameters.txt"

      self.origin_points = brain_dir+"5_Analysis_Output/Transform_Parameters/OriginPoints/OriginPoints.txt"
      self.origin_points_out = brain_dir+"5_Analysis_Output/Transform_Parameters/OriginPoints/"

      self.aligned_result = brain_dir+"5_Analysis_Output/Temp/Template_aligned/result.1.mhd"
      self.inverse_out = brain_dir+"5_Analysis_Output/Temp/InvOut/"


class Create_Settings():

    def __init__(self, brain_dir):
        with open(brain_dir+"Analysis_Settings.yaml", 'r') as ymlfile:
            #setting = yaml.safe_load(ymlfile)
            setting = yaml.load(ymlfile, Loader = PrettySafeLoader)

            self.input_res = setting["Parameters"]["input_res"]
            self.final_res = setting["Parameters"]["final_res"]
            self.section_thickness = setting["Parameters"]["section_thickness"]
            self.tissue_background = setting["Parameters"]["tissue_background"]

            self.ref_section = setting["Analysis"]["ref_section"]
            self.full_res_DAPI = setting["Analysis"]["full_res_DAPI"]
            self.DAPI_cell_analysis = setting["Analysis"]["DAPI_cell_analysis"]
            self.DAPI_channel = setting["Analysis"]["DAPI_channel"]
            self.tiles_for_prediction = list(map(int, (ast.literal_eval(setting["Analysis"]["tiles_for_prediction"]))))
            self.stard_nms_thresh = setting["Analysis"]["stard_nms_thresh"]
            self.save_intermediate_data = setting["Analysis"]["save_intermediate_data"]
            self.validation_format = setting["Analysis"]["validation_format"]
            self.validation_scale = list(map(int, (ast.literal_eval(setting["Analysis"]["validation_scale"]))))
            self.validation_jpeg_comp = setting["Analysis"]["validation_jpeg_comp"]

            self.atlas_dir = setting["Atlas"]["atlas_dir"]
            self.atlas_res = list(map(int, (ast.literal_eval(setting["Atlas"]["atlas_res"]))))

            self.elastix_dir = setting["Elastix"]["location"]

            self.c1_cell_analysis = setting["Channel_1"]["cell_analysis"]
            self.c1_measure = setting["Channel_1"]["measure"]
            self.c1_rest_model_path = setting["Channel_1"]["rest_model_path"]
            self.c1_rest_type = ast.literal_eval(setting["Channel_1"]["rest_type"])
            self.c1_seg_model = ast.literal_eval(setting["Channel_1"]["seg_model"])
            self.c1_normalize = list(map(int, (ast.literal_eval(setting["Channel_1"]["normalize"]))))
            self.c1_scale = setting["Channel_1"]["scale"]
            self.c1_preprocess = list(map(int, (ast.literal_eval(setting["Channel_1"]["preprocess"]))))
            self.c1_prob_thresh = setting["Channel_1"]["prob_thresh"]
            self.c1_cell_size = list(map(int, (ast.literal_eval(setting["Channel_1"]["cell_size"]))))
            self.c1_intensity_filter = setting["Channel_1"]["intensity_filter"]
            self.c1_save_val_data = setting["Channel_1"]["save_val_data"]

            self.c2_cell_analysis = setting["Channel_2"]["cell_analysis"]
            self.c2_measure = setting["Channel_2"]["measure"]
            self.c2_rest_model_path = setting["Channel_2"]["rest_model_path"]
            self.c2_rest_type = ast.literal_eval(setting["Channel_2"]["rest_type"])
            self.c2_seg_model = ast.literal_eval(setting["Channel_2"]["seg_model"])
            self.c2_normalize = list(map(int, (ast.literal_eval(setting["Channel_2"]["normalize"]))))
            self.c2_scale = setting["Channel_2"]["scale"]
            self.c2_preprocess = list(map(int, (ast.literal_eval(setting["Channel_2"]["preprocess"]))))
            self.c2_prob_thresh = setting["Channel_2"]["prob_thresh"]
            self.c2_cell_size = list(map(int, (ast.literal_eval(setting["Channel_2"]["cell_size"]))))
            self.c2_intensity_filter = setting["Channel_2"]["intensity_filter"]
            self.c2_save_val_data = setting["Channel_2"]["save_val_data"]

            self.c3_cell_analysis = setting["Channel_3"]["cell_analysis"]
            self.c3_measure = setting["Channel_3"]["measure"]
            self.c3_rest_model_path = setting["Channel_3"]["rest_model_path"]
            self.c3_rest_type = ast.literal_eval(setting["Channel_3"]["rest_type"])
            self.c3_seg_model = ast.literal_eval(setting["Channel_3"]["seg_model"])
            self.c3_normalize = list(map(int, (ast.literal_eval(setting["Channel_3"]["normalize"]))))
            self.c3_scale = setting["Channel_3"]["scale"]
            self.c3_preprocess = list(map(int, (ast.literal_eval(setting["Channel_3"]["preprocess"]))))
            self.c3_prob_thresh = setting["Channel_3"]["prob_thresh"]
            self.c3_cell_size = list(map(int, (ast.literal_eval(setting["Channel_3"]["cell_size"]))))
            self.c3_intensity_filter = setting["Channel_3"]["intensity_filter"]
            self.c3_save_val_data = setting["Channel_3"]["save_val_data"]

            self.c4_cell_analysis = setting["Channel_4"]["cell_analysis"]
            self.c4_measure = setting["Channel_4"]["measure"]
            self.c4_rest_model_path = setting["Channel_4"]["rest_model_path"]
            self.c4_rest_type = ast.literal_eval(setting["Channel_4"]["rest_type"])
            self.c4_seg_model = ast.literal_eval(setting["Channel_4"]["seg_model"])
            self.c4_normalize = list(map(int, (ast.literal_eval(setting["Channel_4"]["normalize"]))))
            self.c4_scale = setting["Channel_4"]["scale"]
            self.c4_preprocess = list(map(int, (ast.literal_eval(setting["Channel_4"]["preprocess"]))))
            self.c4_prob_thresh = setting["Channel_4"]["prob_thresh"]
            self.c4_cell_size = list(map(int, (ast.literal_eval(setting["Channel_4"]["cell_size"]))))
            self.c4_intensity_filter = setting["Channel_4"]["intensity_filter"]
            self.c4_save_val_data = setting["Channel_4"]["save_val_data"]

