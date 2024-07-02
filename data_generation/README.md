## Generate Dataset

0 - Download and unzip [mimic-cxr-jpg](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/), [mimic-iv-2.2](https://physionet.org/content/mimiciv/2.2/) and [mimic-iv-note](https://www.physionet.org/content/mimic-iv-note/2.2/) (PhysioNet Authorization required)
<br>
1 - Change core_mimiciv_path, core_mimiciv_note_path, core_mimiciv_imgcxr_path in create_pickle.py and create_mimic_txt.py to dataset saved path 
<br>
2 - Create mimic text file
```
python create_mimic_txt.py
```
3 - Create patient pickle object
```
python create_pickle.py
``` 
4 - Create original dataset from patient pickle files 
```
python create_img_text_csv.py
```
5 - Separate sections for radiology report
```
python create_sections.py
``` 
6 - Split dataset to train/val/test <br>
Run Notebook split_dataset.ipynb