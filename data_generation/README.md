## Generate Dataset

0 - Download and unzip mimic-cxr-jpg \url{https://www.physionet.org/content/, mimic-iv-2.2 \url{https://physionet.org/content/mimiciv/2.2/}, mimic-cxr-jpg/2.0.0/} and mimic-iv-note \url{https://physionet.org/content/mimiciv/2.2/}
1 - Change core_mimiciv_path, core_mimiciv_note_path, core_mimiciv_imgcxr_path in create_pickle.py and create_mimic_txt.py to dataset saved path
2 - Create mimic text file
``
python create_mimic_txt.py
``
3 - Create patient pickle object
``
python create_pickle.py
``
4 - Create original dataset from patient pickle files
``
python create_img_text_csv.py
``
5 - Separate sections for radiology report
``
python create_sections.py
``
6 - Split dataset to train/val/test
``
Run Notebook split_dataset.ipynb
``