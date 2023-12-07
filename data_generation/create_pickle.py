#HAIM
import os
import sys
from MIMIC_IV_HAIM_API import *

# Define MIMIC IV Data Location
core_mimiciv_path = './mimic-iv-2.2-v2/'

# Define MIMIC IV Note Data Location
core_mimiciv_note_path = './mimic-iv-note-deidentified-free-text-clinical-notes-2.2/'

# Define MIMIC IV Image Data Location (usually external drive)
core_mimiciv_imgcxr_path = './Public/physionet.org/files/mimic-cxr-jpg/2.0.0/'

# Define mimic-cxr note location
mimic_cxr_note_path = './mimic-iv-2.2/mimic-cxr-2.0.0-jpeg-txt.csv'
df_mimic_cxr = pd.read_csv(mimic_cxr_note_path)

# create pickle file folder
pickle_folder = core_mimiciv_path + 'pickle/'

if not os.path.exists(pickle_folder):
    os.makedirs(pickle_folder)

## CORE
df_admissions = dd.read_csv(core_mimiciv_path + 'hosp/admissions.csv', assume_missing=True, dtype={'admission_location': 'object','deathtime': 'object','edouttime': 'object','edregtime': 'object'})
df_patients = dd.read_csv(core_mimiciv_path + 'hosp/patients.csv', assume_missing=True, dtype={'dod': 'object'})  
df_transfers = dd.read_csv(core_mimiciv_path + 'hosp/transfers.csv', assume_missing=True, dtype={'careunit': 'object'})

## HOSP
df_d_labitems = dd.read_csv(core_mimiciv_path + 'hosp/d_labitems.csv', assume_missing=True, dtype={'loinc_code': 'object'})
df_d_icd_procedures = dd.read_csv(core_mimiciv_path + 'hosp/d_icd_procedures.csv', assume_missing=True, dtype={'icd_code': 'object', 'icd_version': 'object'})
df_d_icd_diagnoses = dd.read_csv(core_mimiciv_path + 'hosp/d_icd_diagnoses.csv', assume_missing=True, dtype={'icd_code': 'object', 'icd_version': 'object'})
df_d_hcpcs = dd.read_csv(core_mimiciv_path + 'hosp/d_hcpcs.csv', assume_missing=True, dtype={'category': 'object', 'long_description': 'object'})
df_diagnoses_icd = dd.read_csv(core_mimiciv_path + 'hosp/diagnoses_icd.csv', assume_missing=True, dtype={'icd_code': 'object', 'icd_version': 'object'})
df_drgcodes = dd.read_csv(core_mimiciv_path + 'hosp/drgcodes.csv', assume_missing=True)
df_emar = dd.read_csv(core_mimiciv_path + 'hosp/emar.csv', assume_missing=True, dtype={'enter_provider_id': 'object'})
df_emar_detail = dd.read_csv(core_mimiciv_path + 'hosp/emar_detail.csv', assume_missing=True, low_memory=False, dtype={'completion_interval': 'object','dose_due': 'object','dose_given': 'object','infusion_complete': 'object','infusion_rate_adjustment': 'object','infusion_rate_unit': 'object','new_iv_bag_hung': 'object','product_description_other': 'object','reason_for_no_barcode': 'object','restart_interval': 'object','route': 'object','side': 'object','site': 'object','continued_infusion_in_other_location': 'object','infusion_rate': 'object','non_formulary_visual_verification': 'object','prior_infusion_rate': 'object','product_amount_given': 'object', 'infusion_rate_adjustment_amount': 'object'})
df_hcpcsevents = dd.read_csv(core_mimiciv_path + 'hosp/hcpcsevents.csv', assume_missing=True, dtype={'hcpcs_cd': 'object'})
df_labevents = dd.read_csv(core_mimiciv_path + 'hosp/labevents.csv', assume_missing=True, dtype={'storetime': 'object', 'value': 'object', 'valueuom': 'object', 'flag': 'object', 'priority': 'object', 'comments': 'object'})
df_microbiologyevents = dd.read_csv(core_mimiciv_path + 'hosp/microbiologyevents.csv', assume_missing=True, dtype={
    'comments': 'object', 'quantity': 'object', 'ab_name': 'object', 'dilution_comparison': 'object',
    'dilution_text': 'object', 'interpretation': 'object', 'org_name': 'object'})
df_poe = dd.read_csv(core_mimiciv_path + 'hosp/poe.csv', assume_missing=True, dtype={'discontinue_of_poe_id': 'object','discontinued_by_poe_id': 'object','order_status': 'object'})
df_poe_detail = dd.read_csv(core_mimiciv_path + 'hosp/poe_detail.csv', assume_missing=True)
df_prescriptions = dd.read_csv(core_mimiciv_path + 'hosp/prescriptions.csv', assume_missing=True, dtype={'form_rx': 'object','gsn': 'object', 'dose_val_rx': 'object', 'form_val_disp': 'object'})
df_procedures_icd = dd.read_csv(core_mimiciv_path + 'hosp/procedures_icd.csv', assume_missing=True, dtype={'icd_code': 'object', 'icd_version': 'object'})
df_services = dd.read_csv(core_mimiciv_path + 'hosp/services.csv', assume_missing=True, dtype={'prev_service': 'object'})

## ICU
df_d_items = dd.read_csv(core_mimiciv_path + 'icu/d_items.csv', assume_missing=True)
df_procedureevents = dd.read_csv(core_mimiciv_path + 'icu/procedureevents.csv', assume_missing=True, dtype={'value': 'object', 'secondaryordercategoryname': 'object', 'totalamountuom': 'object'})
df_outputevents = dd.read_csv(core_mimiciv_path + 'icu/outputevents.csv', assume_missing=True, dtype={'value': 'object'})
df_inputevents = dd.read_csv(core_mimiciv_path + 'icu/inputevents.csv', assume_missing=True, dtype={'value': 'object', 'secondaryordercategoryname': 'object', 'totalamountuom': 'object'})
df_icustays = dd.read_csv(core_mimiciv_path + 'icu/icustays.csv', assume_missing=True)
df_datetimeevents = dd.read_csv(core_mimiciv_path + 'icu/datetimeevents.csv', assume_missing=True, dtype={'value': 'object'})
df_chartevents = dd.read_csv(core_mimiciv_path + 'icu/chartevents.csv', assume_missing=True, low_memory=False, dtype={'value': 'object', 'valueuom': 'object'})

## CXR
df_mimic_cxr_split = dd.read_csv(core_mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-split.csv', assume_missing=True)
df_mimic_cxr_chexpert = dd.read_csv(core_mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-chexpert.csv', assume_missing=True)
try:
    df_mimic_cxr_metadata = dd.read_csv(core_mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-metadata.csv', assume_missing=True, dtype={'dicom_id': 'object'}, blocksize=None)
except:
    df_mimic_cxr_metadata = pd.read_csv(core_mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-metadata.csv', dtype={'dicom_id': 'object'})
    df_mimic_cxr_metadata = dd.from_pandas(df_mimic_cxr_metadata, npartitions=7)
df_mimic_cxr_negbio = dd.read_csv(core_mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-negbio.csv', assume_missing=True)

## NOTES
df_dsnotes = dd.from_pandas(pd.read_csv(core_mimiciv_note_path + 'note/discharge.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
df_radnotes = dd.from_pandas(pd.read_csv(core_mimiciv_note_path + 'note/radiology.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)

### Fix data type issues to allow for merging

## CORE
df_admissions['admittime'] = dd.to_datetime(df_admissions['admittime'])
df_admissions['dischtime'] = dd.to_datetime(df_admissions['dischtime'])
df_admissions['deathtime'] = dd.to_datetime(df_admissions['deathtime'])
df_admissions['edregtime'] = dd.to_datetime(df_admissions['edregtime'])
df_admissions['edouttime'] = dd.to_datetime(df_admissions['edouttime'])

df_transfers['intime'] = dd.to_datetime(df_transfers['intime'])
df_transfers['outtime'] = dd.to_datetime(df_transfers['outtime'])


## HOSP
df_diagnoses_icd.icd_code = df_diagnoses_icd.icd_code.str.strip()
df_diagnoses_icd.icd_version = df_diagnoses_icd.icd_version.str.strip()
df_d_icd_diagnoses.icd_code = df_d_icd_diagnoses.icd_code.str.strip()
df_d_icd_diagnoses.icd_version = df_d_icd_diagnoses.icd_version.str.strip()

df_procedures_icd.icd_code = df_procedures_icd.icd_code.str.strip()
df_procedures_icd.icd_version = df_procedures_icd.icd_version.str.strip()
df_d_icd_procedures.icd_code = df_d_icd_procedures.icd_code.str.strip()
df_d_icd_procedures.icd_version = df_d_icd_procedures.icd_version.str.strip()

df_hcpcsevents.hcpcs_cd = df_hcpcsevents.hcpcs_cd.str.strip()
df_d_hcpcs.code = df_d_hcpcs.code.str.strip()

df_prescriptions['starttime'] = dd.to_datetime(df_prescriptions['starttime'])
df_prescriptions['stoptime'] = dd.to_datetime(df_prescriptions['stoptime'])

df_emar['charttime'] = dd.to_datetime(df_emar['charttime'])
df_emar['scheduletime'] = dd.to_datetime(df_emar['scheduletime'])
df_emar['storetime'] = dd.to_datetime(df_emar['storetime'])

df_labevents['charttime'] = dd.to_datetime(df_labevents['charttime'])
df_labevents['storetime'] = dd.to_datetime(df_labevents['storetime'])

df_microbiologyevents['chartdate'] = dd.to_datetime(df_microbiologyevents['chartdate'])
df_microbiologyevents['charttime'] = dd.to_datetime(df_microbiologyevents['charttime'])
df_microbiologyevents['storedate'] = dd.to_datetime(df_microbiologyevents['storedate'])
df_microbiologyevents['storetime'] = dd.to_datetime(df_microbiologyevents['storetime'])

df_poe['ordertime'] = dd.to_datetime(df_poe['ordertime'])
df_services['transfertime'] = dd.to_datetime(df_services['transfertime'])


## ICU
df_procedureevents['starttime'] = dd.to_datetime(df_procedureevents['starttime'])
df_procedureevents['endtime'] = dd.to_datetime(df_procedureevents['endtime'])
df_procedureevents['storetime'] = dd.to_datetime(df_procedureevents['storetime'])

df_outputevents['charttime'] = dd.to_datetime(df_outputevents['charttime'])
df_outputevents['storetime'] = dd.to_datetime(df_outputevents['storetime'])

df_inputevents['starttime'] = dd.to_datetime(df_inputevents['starttime'])
df_inputevents['endtime'] = dd.to_datetime(df_inputevents['endtime'])
df_inputevents['storetime'] = dd.to_datetime(df_inputevents['storetime'])

df_icustays['intime'] = dd.to_datetime(df_icustays['intime'])
df_icustays['outtime'] = dd.to_datetime(df_icustays['outtime'])

df_datetimeevents['charttime'] = dd.to_datetime(df_datetimeevents['charttime'])
df_datetimeevents['storetime'] = dd.to_datetime(df_datetimeevents['storetime'])

df_chartevents['charttime'] = dd.to_datetime(df_chartevents['charttime'])
df_chartevents['storetime'] = dd.to_datetime(df_chartevents['storetime'])

## CXR
if (not 'cxrtime' in df_mimic_cxr_metadata.columns) or (not 'Img_Filename' in df_mimic_cxr_metadata.columns):
    # Create CXRTime variable if it does not exist already
    print("Processing CXRtime stamps")
    df_cxr = df_mimic_cxr_metadata.compute()
    df_cxr['StudyDateForm'] = pd.to_datetime(df_cxr['StudyDate'], format='%Y%m%d')
    df_cxr['StudyTimeForm'] = df_cxr.apply(lambda x : '%#010.3f' % x['StudyTime'] ,1)
    df_cxr['StudyTimeForm'] = pd.to_datetime(df_cxr['StudyTimeForm'], format='%H%M%S.%f').dt.time
    df_cxr['cxrtime'] = df_cxr.apply(lambda r : dt.datetime.combine(r['StudyDateForm'],r['StudyTimeForm']),1)
    # Add paths and info to images in cxr
    df_mimic_cxr_jpg =pd.read_csv(core_mimiciv_path + 'mimic-cxr-2.0.0-jpeg-txt.csv')
    df_cxr = pd.merge(df_mimic_cxr_jpg, df_cxr, on='dicom_id')
    # Save
    df_cxr.to_csv(core_mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-metadata.csv', index=False)
    #Read back the dataframe
    try:
        df_mimic_cxr_metadata = dd.read_csv(core_mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-metadata.csv', assume_missing=True, dtype={'dicom_id': 'object', 'Note': 'object'}, blocksize=None)
    except:
        df_mimic_cxr_metadata = pd.read_csv(core_mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-metadata.csv', dtype={'dicom_id': 'object', 'Note': 'object'})
        df_mimic_cxr_metadata = dd.from_pandas(df_mimic_cxr_metadata, npartitions=7)
df_mimic_cxr_metadata['cxrtime'] = dd.to_datetime(df_mimic_cxr_metadata['cxrtime'])


## NOTES
df_dsnotes['charttime'] = dd.to_datetime(df_dsnotes['charttime'])
df_dsnotes['storetime'] = dd.to_datetime(df_dsnotes['storetime'])

df_radnotes['charttime'] = dd.to_datetime(df_radnotes['charttime'])
df_radnotes['storetime'] = dd.to_datetime(df_radnotes['storetime'])

# -> SORT data
## CORE
print('PROCESSING "CORE" DB...')
df_admissions = df_admissions.compute().sort_values(by=['subject_id','hadm_id'])
df_patients = df_patients.compute().sort_values(by=['subject_id'])
df_transfers = df_transfers.compute().sort_values(by=['subject_id','hadm_id'])


# ## HOSP
print('PROCESSING "HOSP" DB...')
df_diagnoses_icd = df_diagnoses_icd.compute().sort_values(by=['subject_id'])
df_drgcodes = df_drgcodes.compute().sort_values(by=['subject_id','hadm_id'])
df_emar = df_emar.compute().sort_values(by=['subject_id','hadm_id'])
df_emar_detail = df_emar_detail.compute().sort_values(by=['subject_id'])

df_hcpcsevents = df_hcpcsevents.compute().sort_values(by=['subject_id','hadm_id'])
df_labevents = df_labevents.compute().sort_values(by=['subject_id','hadm_id'])
df_microbiologyevents = df_microbiologyevents.compute().sort_values(by=['subject_id','hadm_id'])
df_poe = df_poe.compute().sort_values(by=['subject_id','hadm_id'])
df_poe_detail = df_poe_detail.compute().sort_values(by=['subject_id'])
df_prescriptions = df_prescriptions.compute().sort_values(by=['subject_id','hadm_id'])
df_procedures_icd = df_procedures_icd.compute().sort_values(by=['subject_id','hadm_id'])
df_services = df_services.compute().sort_values(by=['subject_id','hadm_id'])
#--> Unwrap dictionaries
df_d_icd_diagnoses = df_d_icd_diagnoses.compute()
df_d_icd_procedures = df_d_icd_procedures.compute()
df_d_hcpcs = df_d_hcpcs.compute()
df_d_labitems = df_d_labitems.compute()


## ICU
print('PROCESSING "ICU" DB...')
df_procedureevents = df_procedureevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
df_outputevents = df_outputevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
df_inputevents = df_inputevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
df_icustays = df_icustays.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
df_datetimeevents = df_datetimeevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
df_chartevents = df_chartevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
#--> Unwrap dictionaries
df_d_items = df_d_items.compute()


## CXR
print('PROCESSING "CXR" DB...')
df_mimic_cxr_split = df_mimic_cxr_split.compute().sort_values(by=['subject_id'])
df_mimic_cxr_chexpert = df_mimic_cxr_chexpert.compute().sort_values(by=['subject_id'])
df_mimic_cxr_metadata = df_mimic_cxr_metadata.compute().sort_values(by=['subject_id'])
df_mimic_cxr_negbio = df_mimic_cxr_negbio.compute().sort_values(by=['subject_id'])


## NOTES
print('PROCESSING "NOTES" DB...')
df_dsnotes = df_dsnotes.compute().sort_values(by=['subject_id','hadm_id'])
df_radnotes = df_radnotes.compute().sort_values(by=['subject_id','hadm_id'])

# -> MASTER DICTIONARY of health items
# Generate dictionary for chartevents, labevents and HCPCS
df_patientevents_categorylabels_dict = pd.DataFrame(columns = ['eventtype', 'category', 'label'])

# Get Chartevent items with labels & category
df = df_d_items
for category_idx, category in enumerate(sorted((df.category.astype(str).unique()))):
    #print(category)
    category_list = df[df['category']==category]
    for item_idx, item in enumerate(sorted(category_list.label.astype(str).unique())):
        df_patientevents_categorylabels_dict = df_patientevents_categorylabels_dict.append({'eventtype': 'chart', 'category': category, 'label': item}, ignore_index=True)

# Get Lab items with labels & category
df = df_d_labitems
for category_idx, category in enumerate(sorted((df.category.astype(str).unique()))):
    #print(category)
    category_list = df[df['category']==category]
    for item_idx, item in enumerate(sorted(category_list.label.astype(str).unique())):
        df_patientevents_categorylabels_dict = df_patientevents_categorylabels_dict.append({'eventtype': 'lab', 'category': category, 'label': item}, ignore_index=True)
        
# Get HCPCS items with labels & category
df = df_d_hcpcs
for category_idx, category in enumerate(sorted((df.category.astype(str).unique()))):
    category_list = df[df['category']==category]
    for item_idx, item in enumerate(sorted(category_list.long_description.astype(str).unique())):
        df_patientevents_categorylabels_dict = df_patientevents_categorylabels_dict.append({'eventtype': 'hcpcs', 'category': category, 'label': item}, ignore_index=True)

df_base_core = df_admissions.merge(df_patients, how='left').merge(df_transfers, how='left')
df_base_core.to_csv(core_mimiciv_path + 'hosp/core.csv')

# Get Unique Subject/HospAdmission/Stay Combinations
df_ids = pd.concat([pd.DataFrame(), df_procedureevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_outputevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_inputevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_icustays[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_datetimeevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_chartevents[['subject_id','hadm_id','stay_id']]], sort=True).drop_duplicates()

# Get Unique Subjects with Chest Xrays
df_cxr_ids = pd.concat([pd.DataFrame(), df_mimic_cxr_chexpert[['subject_id']]], sort=True).drop_duplicates()

# Get Unique Subject/HospAdmission/Stay Combinations with Chest Xrays
df_haim_ids = df_ids[df_ids['subject_id'].isin(df_cxr_ids['subject_id'].unique())] 

# Save Unique Subject/HospAdmission/Stay Combinations with Chest Xrays    
df_haim_ids.to_csv(core_mimiciv_path + 'haim_mimiciv_key_ids.csv', index=False)

print('Unique Subjects: ' + str(len(df_patients['subject_id'].unique())))
print('Unique Subjects/HospAdmissions/Stays Combinations: ' + str(len(df_ids)))
print('Unique Subjects with Chest Xrays Available: ' + str(len(df_cxr_ids)))

# Save Unique Subject/HospAdmission/Stay Combinations with Chest Xrays    
df_haim_ids = pd.read_csv(core_mimiciv_path + 'haim_mimiciv_key_ids.csv')
print('Unique HAIM Records Available: ' + str(len(df_haim_ids)))


# GET FULL MIMIC IV PATIENT RECORD USING DATABASE KEYS
def get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id):
    # Inputs:
    #   key_subject_id -> subject_id is unique to a patient
    #   key_hadm_id    -> hadm_id is unique to a patient hospital stay
    #   key_stay_id    -> stay_id is unique to a patient ward stay
    #   
    #   NOTES: Identifiers which specify the patient. More information about 
    #   these identifiers is available at https://mimic-iv.mit.edu/basics/identifiers

    # Outputs:
    #   Patient_ICUstay -> ICU patient stay structure

    #-> FILTER data
    ##-> CORE
    f_df_base_core = df_base_core[(df_base_core.subject_id == key_subject_id) & (df_base_core.hadm_id == key_hadm_id)]
    f_df_admissions = df_admissions[(df_admissions.subject_id == key_subject_id) & (df_admissions.hadm_id == key_hadm_id)]
    f_df_patients = df_patients[(df_patients.subject_id == key_subject_id)]
    f_df_transfers = df_transfers[(df_transfers.subject_id == key_subject_id) & (df_transfers.hadm_id == key_hadm_id)]
    ###-> Merge data into single patient structure
    f_df_core = f_df_base_core
    f_df_core = f_df_core.merge(f_df_admissions, how='left')
    f_df_core = f_df_core.merge(f_df_patients, how='left')
    f_df_core = f_df_core.merge(f_df_transfers, how='left')

    ##-> HOSP
    f_df_diagnoses_icd = df_diagnoses_icd[(df_diagnoses_icd.subject_id == key_subject_id)]
    f_df_drgcodes = df_drgcodes[(df_drgcodes.subject_id == key_subject_id) & (df_drgcodes.hadm_id == key_hadm_id)]
    f_df_emar = df_emar[(df_emar.subject_id == key_subject_id) & (df_emar.hadm_id == key_hadm_id)]
    f_df_emar_detail = df_emar_detail[(df_emar_detail.subject_id == key_subject_id)]
    f_df_hcpcsevents = df_hcpcsevents[(df_hcpcsevents.subject_id == key_subject_id) & (df_hcpcsevents.hadm_id == key_hadm_id)]
    f_df_labevents = df_labevents[(df_labevents.subject_id == key_subject_id) & (df_labevents.hadm_id == key_hadm_id)]
    f_df_microbiologyevents = df_microbiologyevents[(df_microbiologyevents.subject_id == key_subject_id) & (df_microbiologyevents.hadm_id == key_hadm_id)]
    f_df_poe = df_poe[(df_poe.subject_id == key_subject_id) & (df_poe.hadm_id == key_hadm_id)]
    f_df_poe_detail = df_poe_detail[(df_poe_detail.subject_id == key_subject_id)]
    f_df_prescriptions = df_prescriptions[(df_prescriptions.subject_id == key_subject_id) & (df_prescriptions.hadm_id == key_hadm_id)]
    f_df_procedures_icd = df_procedures_icd[(df_procedures_icd.subject_id == key_subject_id) & (df_procedures_icd.hadm_id == key_hadm_id)]
    f_df_services = df_services[(df_services.subject_id == key_subject_id) & (df_services.hadm_id == key_hadm_id)]
    ###-> Merge content from dictionaries
    f_df_diagnoses_icd = f_df_diagnoses_icd.merge(df_d_icd_diagnoses, how='left') 
    f_df_procedures_icd = f_df_procedures_icd.merge(df_d_icd_procedures, how='left')
    f_df_hcpcsevents = f_df_hcpcsevents.merge(df_d_hcpcs, how='left')
    f_df_labevents = f_df_labevents.merge(df_d_labitems, how='left')

    ##-> ICU
    f_df_procedureevents = df_procedureevents[(df_procedureevents.subject_id == key_subject_id) & (df_procedureevents.hadm_id == key_hadm_id) & (df_procedureevents.stay_id == key_stay_id)]
    f_df_outputevents = df_outputevents[(df_outputevents.subject_id == key_subject_id) & (df_outputevents.hadm_id == key_hadm_id) & (df_outputevents.stay_id == key_stay_id)]
    f_df_inputevents = df_inputevents[(df_inputevents.subject_id == key_subject_id) & (df_inputevents.hadm_id == key_hadm_id) & (df_inputevents.stay_id == key_stay_id)]
    f_df_icustays = df_icustays[(df_icustays.subject_id == key_subject_id) & (df_icustays.hadm_id == key_hadm_id) & (df_icustays.stay_id == key_stay_id)]
    f_df_datetimeevents = df_datetimeevents[(df_datetimeevents.subject_id == key_subject_id) & (df_datetimeevents.hadm_id == key_hadm_id) & (df_datetimeevents.stay_id == key_stay_id)]
    f_df_chartevents = df_chartevents[(df_chartevents.subject_id == key_subject_id) & (df_chartevents.hadm_id == key_hadm_id) & (df_chartevents.stay_id == key_stay_id)]
    ###-> Merge content from dictionaries
    f_df_procedureevents = f_df_procedureevents.merge(df_d_items, how='left')
    f_df_outputevents = f_df_outputevents.merge(df_d_items, how='left')
    f_df_inputevents = f_df_inputevents.merge(df_d_items, how='left')
    f_df_datetimeevents = f_df_datetimeevents.merge(df_d_items, how='left')
    f_df_chartevents = f_df_chartevents.merge(df_d_items, how='left')

    ##-> CXR
    f_df_mimic_cxr_split = df_mimic_cxr_split[(df_mimic_cxr_split.subject_id == key_subject_id)]
    f_df_mimic_cxr_chexpert = df_mimic_cxr_chexpert[(df_mimic_cxr_chexpert.subject_id == key_subject_id)]
    f_df_mimic_cxr_metadata = df_mimic_cxr_metadata[(df_mimic_cxr_metadata.subject_id == key_subject_id)]
    f_df_mimic_cxr_negbio = df_mimic_cxr_negbio[(df_mimic_cxr_negbio.subject_id == key_subject_id)]
    ###-> Merge data into single patient structure
    f_df_cxr = f_df_mimic_cxr_split
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_chexpert, how='left')
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_metadata, how='left')
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_negbio, how='left')
    ###-> Get images of that timebound patient
    f_df_imcxr = []
    for img_idx, img_row in f_df_cxr.iterrows():
        img_path = core_mimiciv_imgcxr_path + str(img_row['Img_Folder']) + '/' + str(img_row['Img_Filename'])
        img_cxr_shape = [224, 224]
        img_cxr = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (img_cxr_shape[0], img_cxr_shape[1]))
        f_df_imcxr.append(np.array(img_cxr))
    
    # Find paired note for from mimic-cxr
    radnotes = []
    for index, row in f_df_cxr.iterrows():
        dicom_id = row['dicom_id']
        radnote = df_mimic_cxr[df_mimic_cxr['dicom_id']==dicom_id]['Note'].iloc[0]
        radnotes.append(radnote)
    f_df_cxr['radnote'] = radnotes

    ##-> NOTES from mimic-iv
    f_df_dsnotes = df_dsnotes[(df_dsnotes.subject_id == key_subject_id) & (df_dsnotes.hadm_id == key_hadm_id)]
    f_df_radnotes = df_radnotes[(df_radnotes.subject_id == key_subject_id) & (df_radnotes.hadm_id == key_hadm_id)]
    
    ###-> Merge data into single patient structure
    #--None

    # -> Create & Populate patient structure
    ## CORE
    admissions = f_df_admissions
    demographics = f_df_patients
    transfers = f_df_transfers
    core = f_df_core

    ## HOSP
    diagnoses_icd = f_df_diagnoses_icd
    drgcodes = f_df_diagnoses_icd
    emar = f_df_emar
    emar_detail = f_df_emar_detail
    hcpcsevents = f_df_hcpcsevents
    labevents = f_df_labevents
    microbiologyevents = f_df_microbiologyevents
    poe = f_df_poe
    poe_detail = f_df_poe_detail
    prescriptions = f_df_prescriptions
    procedures_icd = f_df_procedures_icd
    services = f_df_services

    ## ICU
    procedureevents = f_df_procedureevents
    outputevents = f_df_outputevents
    inputevents = f_df_inputevents
    icustays = f_df_icustays
    datetimeevents = f_df_datetimeevents
    chartevents = f_df_chartevents

    ## CXR
    cxr = f_df_cxr 
    imcxr = f_df_imcxr

    ## NOTES
    dsnotes = f_df_dsnotes
    radnotes = f_df_radnotes
    
    # Create patient object and return
    Patient_ICUstay = Patient_ICU(admissions, demographics, transfers, core, \
        diagnoses_icd, drgcodes, emar, emar_detail, hcpcsevents, \
        labevents, microbiologyevents, poe, poe_detail, \
        prescriptions, procedures_icd, services, procedureevents, \
        outputevents, inputevents, icustays, datetimeevents, \
        chartevents, cxr, imcxr, dsnotes, radnotes)

    return Patient_ICUstay


# EXTRACT ALL INFO OF A SINGLE PATIENT FROM MIMIC-IV DATASET USING HAIM ID
def extract_single_patient_records_mimiciv(haim_patient_idx, df_haim_ids, start_hr, end_hr):
    # Inputs:
    #   haim_patient_idx -> Ordered number of HAIM patient
    #   df_haim_ids -> Dataframe with all unique available HAIM_MIMICIV records by key identifiers
    #   start_hr -> start_hr indicates the first valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #   end_hr -> end_hr indicates the last valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #
    # Outputs:
    #   key_subject_id -> MIMIC-IV Subject ID of selected patient
    #   key_hadm_id -> MIMIC-IV Hospital Admission ID of selected patient
    #   key_stay_id -> MIMIC-IV ICU Stay ID of selected patient
    #   patient -> Full ICU patient ICU stay structure
    #   dt_patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    
    # Extract information for patient
    key_subject_id = df_haim_ids.iloc[haim_patient_idx].subject_id
    key_hadm_id = df_haim_ids.iloc[haim_patient_idx].hadm_id
    key_stay_id = df_haim_ids.iloc[haim_patient_idx].stay_id
    start_hr = start_hr # Select timestamps
    end_hr = end_hr   # Select timestamps
    patient = get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id)
    dt_patient = get_timebound_patient_icustay(patient, start_hr , end_hr)
    
    return key_subject_id, key_hadm_id, key_stay_id, patient, dt_patient


# GENERATE ALL SINGLE PATIENT ICU STAY RECORDS FOR ENTIRE MIMIC-IV DATABASE
def generate_all_mimiciv_patient_object(df_haim_ids, core_mimiciv_path):
    # Inputs:
    #   df_haim_ids -> Dataframe with all unique available HAIM_MIMICIV records by key identifiers
    #   core_mimiciv_path -> Path to structured MIMIC IV databases in CSV files
    #
    # Outputs:
    #   nfiles -> Number of single patient HAIM files produced
    
    # Clean out
    sys.stdout.flush()
    
    # Extract information for patient
    nfiles = len(df_haim_ids)
    with tqdm(total = nfiles) as pbar:
        # Update process bar
        nbase= 0
        pbar.update(nbase)
        #Iterate through all patients
        for haim_patient_idx in range(nbase, nfiles):
            # Let's select each single patient and extract patient object
            start_hr = None # Select timestamps
            end_hr = None   # Select timestamps
            key_subject_id, key_hadm_id, key_stay_id, patient, dt_patient = extract_single_patient_records_mimiciv(haim_patient_idx, df_haim_ids, start_hr, end_hr)
            
            # Save
            filename = f"{haim_patient_idx:08d}" + '.pkl'
            save_patient_object(dt_patient, core_mimiciv_path + 'pickle/' + filename)
            # Update process bar
            pbar.update(1)
    return nfiles

# GENERATE ALL SINGLE PATIENT ICU STAY RECORDS FOR ENTIRE MIMIC-IV DATABASE
nfiles = generate_all_mimiciv_patient_object(df_haim_ids, core_mimiciv_path)
print(f"Saved {len(nfiles)} files")