## Generate Dataset

0 - Download and unzip mimic-cxr-jpg [URL](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/), mimic-iv-2.2 [URL](https://physionet.org/content/mimiciv/2.2/) and mimic-iv-note [URL](https://physionet.org/content/mimiciv/2.2/)
<br>
1 - Change core_mimiciv_path, core_mimiciv_note_path, core_mimiciv_imgcxr_path in create_pickle.py and create_mimic_txt.py to dataset saved path 
<br>
2 - Create mimic text file
```
python create_mimic_txt.py
```
<br>
3 - Create patient pickle object
```
python create_pickle.py
``` 
<br>
4 - Create original dataset from patient pickle files 
```
python create_img_text_csv.py
``` 
<br>
5 - Separate sections for radiology report
```
python create_sections.py
``` 
<br>
6 - Split dataset to train/val/test
```
Run Notebook split_dataset.ipynb
```