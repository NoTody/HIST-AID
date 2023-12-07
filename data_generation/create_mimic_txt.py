# System                                                                                           
import os
import sys
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Define MIMIC IV Data Location
core_mimiciv_path = './mimic-iv-2.2/'

# Define MIMIC IV Image Data Location (usually external drive)
core_mimiciv_imgcxr_path = '/gpfs/data/denizlab/Datasets/Public/physionet.org/files/mimic-cxr/2.0.0/'

# BUILD DATAFRAME OF IMAGES AND NOTES FOR MIMIC-IV CXR
def build_mimic_cxr_jpg_dataframe(core_mimiciv_imgcxr_path, do_save=False):
    # Inputs:
    #   core_mimiciv_imgcxr_path -> Directory of CXR images and image notes
    #   do_save -> Flag to save dataframe
    #
    # Outputs:
    #   df_mimic_cxr_jpg -> CXR images and image notes Dataframe
    df_mimic_cxr_jpg = pd.DataFrame()
    mimic_cxr_jpg_dir = core_mimiciv_imgcxr_path
    
    count = 0
    #Iterate
    for subdir, dirs, files in os.walk(mimic_cxr_jpg_dir):
        for file in files:
            # Extract filename and extension to filter by CSV only
            filename, extension = os.path.splitext(file)
            if extension=='.txt':
                note = open(subdir + '/' + filename + extension, "r", errors='ignore')
                img_note_text = note.read()
                note.close()
                img_folder = subdir + '/' + filename
                
                for img_subdir, img_dirs, img_files in os.walk(img_folder):
                    for img_file in img_files:
                        # Extract filename and extension to filter by CSV only
                        img_filename, img_extension = os.path.splitext(img_file)
                        if img_extension=='.dcm':
                            img_extension='.jpg'
                            df_mimic_cxr_jpg = df_mimic_cxr_jpg.append({'Note_folder': subdir.replace(core_mimiciv_imgcxr_path,''), 'Note_file': filename + extension , 'Note': img_note_text, 'Img_Folder': img_folder.replace(core_mimiciv_imgcxr_path,''), 'Img_Filename': img_filename + img_extension, 'dicom_id': img_filename}, ignore_index=True)
        count += 1
        
    #Save
    if do_save:
        df_mimic_cxr_jpg.to_csv(core_mimiciv_path + 'mimic-cxr-2.0.0-jpeg-txt.csv')
        
    return df_mimic_cxr_jpg


build_mimic_cxr_jpg_dataframe(core_mimiciv_imgcxr_path, do_save=True)