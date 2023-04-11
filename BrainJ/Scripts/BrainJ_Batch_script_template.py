#**BrainJ Single Brain Processing**
#
#- Currently expects registered sections from BrainJ ImageJ pipeline
#- Channels are enhanced using CARE networks trained on 1.6µm data and segmented using U-Nets and StarDist.
#- csv files of raw, transformed and mapped cells are generated in addition to optional intermediate images for data checking

__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__webpage__   = 'http://cellularimaging.org'
__download__  = 'http://www.github.com/lahmmond/BrainJ-Python'

##%%

#Add the BrainJ install directory to the path
import sys
sys.path.append('D:/Dropbox/Github/BrainJ-Python/')

#import QLEAN modules
from BrainJ.Environment import main, imgan

#Settings for Jupyter Notebook
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
#%gui qt
# other options
#plt.rcParams["figure.figsize"]=20,20 # makes inline figures full width

#Testing unet initalization and GPU access

main.check_gpu()

#%% Provide Brain dir or dirs: - convert to loop


brain_dirs = ["D:/BrainJ Datasets/Brain1/"]#,
              #"D:/BrainJ Datasets/Brain2/",
              #"D:/BrainJ Datasets/etc/"]

for brain_dir in brain_dirs:
    print("Processing brain: "+brain_dir)

    #Load in experiment parameters and analysis settings   
    settings, locations = main.initialize_brainJ(brain_dir)

    #Modify specific parameters and settings:    
    settings.save_intermediate_data = True
    locations.annotations_table = "C:/Users/Luke_H/Desktop/BrainJ Atlas/ABA_CCF_25_2017/Atlas_Regions.csv"
    settings.tiles_for_prediction = (2,2)
    settings.c1_cell_analysis = True
    
    locations.cell_analysis_out_dir

    #1: restore and detect cells
    rawcells = imgan.cell_detection(settings, locations)

    #2a: transform cells - use for normal processing - 2b or 2c are for reprocessing.
    transformedcells = imgan.transform_cells_V2(rawcells, settings, locations)
    
    
    #2b: if there was an error with elastix import raw cells using the function below, then transform them
    #transformedcells = imgan.import_and_transform_raw_cells_V2(settings, locations) 
    
    #2c: or import already created transformed cells for reannotation, for example with a different atlas
    #transformedcells = imgan.import_transformed_cells_V2(settings, locations) 
    
    
    #Annotate cells and create tables
    imgan.annotate_all_cells(transformedcells, settings, locations)

