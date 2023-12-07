from tqdm import tqdm
import itertools
import pandas as pd
import re
from ast import literal_eval
from section_parser import section_text


def list_rindex(l, s):
    """Helper function: *last* matching element in a list"""
    return len(l) - l[-1::-1].index(s) - 1


def process_df(row, shared_res):
    sns = ['impression', 'findings', 'last_paragraph', 
           'comparison', 'history', 'indication']
    radnotes = row[6]
    # list of lists
    study_sections = [[] for i in range(6)]
    for radnote in radnotes:
        radnote = '\n ' + radnote.replace('\n', '\n ')
        
        for sn in sns:
            compiled = re.compile(re.escape(sn), re.IGNORECASE)
            radnote = compiled.sub(sn.upper(), radnote)

        # split text into sections
        sections, section_names, section_idx = section_text(
            radnote
        )

        # grab the *last* section with the given title
        # prioritizes impression > findings, etc.

        # "last_paragraph" is text up to the end of the report
        # many reports are simple, and have a single section
        # header followed by a few paragraphs
        # these paragraphs are grouped into section "last_paragraph"

        # note also comparison seems unusual but if no other sections
        # exist the radiologist has usually written the report
        # in the comparison section
        idx = -1
        for sn in sns:
            if sn in section_names:
                idx = list_rindex(section_names, sn)
                break

        for i, sn in enumerate(sns):
            if sn in section_names:
                idx = list_rindex(section_names, sn)
                study_sections[i].append(sections[idx].strip())
            else:
                study_sections[i].append(None)
    
    if study_sections != [[] for i in range(6)]:
        new_row = list(itertools.chain(row, study_sections))
        
    shared_res.append(new_row)
    
    return shared_res


if __name__ =='__main__':
    df = pd.read_csv("./dataset_full_1hr.csv")
    df['radnotes'] = df['radnotes'].apply(lambda x: literal_eval(x) if "[" in x else x)
    df['img_folders'] = df['img_folders'].apply(lambda x: literal_eval(x) if "[" in x else x)
    df['img_filenames'] = df['img_filenames'].apply(lambda x: literal_eval(x) if "[" in x else x)
    df['img_deltacharttimes'] = df['img_deltacharttimes'].apply(lambda x: literal_eval(x) if "[" in x else x)
    df['text_deltacharttimes'] = df['text_deltacharttimes'].apply(lambda x: literal_eval(x) if "[" in x else x)
    lst_df = df.values.tolist()

    shared_res = []

    for df_cxr in tqdm(lst_df):
        shared_res = process_df(df_cxr, shared_res)

    lst = list(shared_res)
    sns = ['impression', 'findings',
           'last_paragraph', 'comparison', 'history', 'indication']
    column_names = df.columns.tolist()
    column_names = list(itertools.chain(column_names, sns))

    df_new = pd.DataFrame(lst, columns=column_names)
    df_new.to_csv("./dataset_full_1hr_sectioned.csv", index=False)
