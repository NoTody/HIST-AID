import sys
from MIMIC_IV_HAIM_API import *
import os
import multiprocessing as mp
from multiprocessing import Manager, Pool
from tqdm import tqdm
import itertools
import copy
import pandas as pd

directory = './mimic-iv-2.2/pickle'
filename = "00000000" + '.pkl'

full_path = os.path.join(directory, filename)
patient = load_patient_object(full_path)

def process_df(row):
    full_path = os.path.join(directory, row)
    patient = load_patient_object(full_path)
    patient.cxr = patient.cxr.sort_values(by=['deltacharttime'])
    start_hr = None

    hadm_id = patient.admissions.hadm_id.values[0]
    admittime = patient.admissions.admittime.values[0]
    dischtime = patient.admissions.dischtime.values[0]

    # get cxr within admission time
    df_cxr  = patient.cxr
    cur_patient = copy.deepcopy(df_cxr)
    
    # insert all last cxr record
    if not cur_patient.empty:
        df_stay_cxr = df_cxr.loc[(cur_patient['charttime'] >= admittime) & (cur_patient['charttime'] <= dischtime)]
        
        img_folders, img_filenames = [], []
        img_deltacharttimes = []
        text_deltacharttimes = []
        radnotes = []
        
        start_hr = None
        
        # end hour from last deltachartime
        if not df_stay_cxr.empty:
            for i in range(len(df_stay_cxr)):
                img_folders, img_filenames = [], []
                img_deltacharttimes = []
                text_deltacharttimes = []
                radnotes = []
                
                cur_cxr = copy.deepcopy(df_stay_cxr.iloc[i])
                end_hr = cur_cxr.iloc[-1]

                # get timebound data
                cur_patient = copy.deepcopy(patient)
                past_time = 1
                dt_patient = get_timebound_patient_icustay(cur_patient, past_time, start_hr, end_hr, include_end_hr=False)
                
                # sort
                dt_patient.cxr = dt_patient.cxr.sort_values(by=['deltacharttime'])
                dt_patient.radnotes = dt_patient.radnotes.sort_values(by=['deltacharttime'])
                
                # append cxr info
                for i in range(len(dt_patient.cxr)):
                    dt_cxr = dt_patient.cxr.iloc[i]
                    dicom_id, study_id, subject_id, split, deltacharttime = dt_cxr['dicom_id'], dt_cxr['study_id'], \
                        dt_cxr['subject_id'], dt_cxr['split'], dt_cxr['deltacharttime']
                    img_folder, img_filename = dt_cxr['Img_Folder'], dt_cxr['Img_Filename']
                    img_folders.append(img_folder)
                    img_filenames.append(img_filename)
                    img_deltacharttimes.append(deltacharttime)
                
                # append radnotes info
                for i in range(len(dt_patient.radnotes)):
                    radnotes.append(dt_patient.radnotes.iloc[i].text)
                    text_deltacharttimes.append(dt_patient.radnotes.iloc[i].deltacharttime)
                
                # get last cxr
                dicom_id, study_id, subject_id, split, deltacharttime, cxrtime = cur_cxr['dicom_id'], cur_cxr['study_id'], \
                    cur_cxr['subject_id'], cur_cxr['split'], cur_cxr['deltacharttime'], cur_cxr['cxrtime']
                img_folder, img_filename, pathologies = cur_cxr['Img_Folder'], cur_cxr['Img_Filename'], \
                    cur_cxr[4:18]
                
                # append cxr info
                img_folders.append(img_folder)
                img_filenames.append(img_filename)
                img_deltacharttimes.append(deltacharttime)

                # store current patient only if note is not empty
                if not (radnotes == []):
                    new_row = [hadm_id, study_id, subject_id, split, img_folders, img_filenames, radnotes, \
                        img_deltacharttimes, text_deltacharttimes, cxrtime]
                    new_row = list(itertools.chain(new_row, pathologies))
                    shared_res.append(new_row)

                del cur_patient
                del cur_cxr


if __name__ == '__main__':
    lst_df = os.listdir(directory)
    shared_res = []
    
    manager = Manager()
    shared_res = manager.list()
    
    print(f"Num CPU = {mp.cpu_count()}, Num Pickles = {len(lst_df)}")
    pool = Pool(processes=mp.cpu_count())
    
    for _ in tqdm(pool.imap_unordered(process_df, lst_df), total=len(lst_df)):
        pass
    
    pool.close()

    lst = list(shared_res)
    column_names = ['hadm_id', 'study_id', 'subject_id', 'split', 'img_folders', \
        'img_filenames', 'radnotes', 'img_deltacharttimes', 'text_deltacharttimes', 'cxrtime']
    column_names = list(itertools.chain(column_names, list(patient.cxr.columns[4:18])))

    df_new = pd.DataFrame(lst, columns=column_names)
    print(f"DataFrame Length: {len(df_new)}")
    df_new.to_csv("./dataset_full_1hr.csv", index=None)
