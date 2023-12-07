import os
import csv

from pathlib import Path
from PIL import Image
from ast import literal_eval

import torch
from torch.utils.data import Dataset

class MIMICCXRDataSet(Dataset):
    def __init__(self, args, data_PATH, img_time_series=False, transform = None, policy = "ones"):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels.
        """
        self.args = args
        self.img_time_series = img_time_series
        image_paths = []
        texts = []
        labels = []
        image_times = []
        text_times = []

        with open(data_PATH, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None) # skip the header
            for line in enumerate(csvReader):
                line = line[1]
                img_folders = line[3]
                img_fnames = line[4]
                
                img_folders = literal_eval(img_folders) if (type(img_folders)!=float and "[" in img_folders) else img_folders
                img_fnames = literal_eval(img_fnames) if (type(img_fnames)!=float and "[" in img_fnames) else img_fnames
                
                img_deltacharttimes = line[6]
                text_deltacharttimes = line[7]
                
                img_deltacharttimes = literal_eval(img_deltacharttimes) if (type(img_deltacharttimes)!=float and "[" in img_deltacharttimes) else img_deltacharttimes
                text_deltacharttimes = literal_eval(text_deltacharttimes) if (type(text_deltacharttimes)!=float and "[" in text_deltacharttimes) else text_deltacharttimes

                image_path = [os.path.join(img_folders[i], img_fnames[i]) for i in range(len(img_folders))]
                label = line[8:21]
                
                # impression = 21, finding = 22, indication = 26
                if args.section == 'all':
                    # with impression + finding
                    cur_texts_imp = line[22]
                    cur_texts_find = line[23]
                    cur_texts_imp = literal_eval(cur_texts_imp) if (type(cur_texts_imp)!=float and "[" in cur_texts_imp) else cur_texts_imp
                    cur_texts_find = literal_eval(cur_texts_find) if (type(cur_texts_find)!=float and "[" in cur_texts_find) else cur_texts_find
                    # replace None
                    cur_texts_imp = ['None' if x is None else x for x in cur_texts_imp]
                    cur_texts_find = ['None' if x is None else x for x in cur_texts_find]
                    
                    # concatenate impression and finding
                    cur_texts = [f'Impression: {imp}' + f"\t Finding: {ind}" for imp, ind in zip(cur_texts_imp, cur_texts_find)]
                    #cur_texts = line[5]
                elif args.section == 'impression':
                    cur_texts = line[22]
                elif args.section == 'finding':
                    cur_texts = line[23]
                elif args.section == 'indication':
                    cur_texts = line[27]
                else:
                    raise ValueError("Invalid section")
                
                if args.section != 'all':
                    cur_texts = literal_eval(cur_texts) if (type(cur_texts)!=float and "[" in cur_texts) else cur_texts
                    cur_texts = ['None' if x is None else x for x in cur_texts]
                
                for i in range(13):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                
                image_paths.append(image_path)
                image_times.append(img_deltacharttimes)
                text_times.append(text_deltacharttimes)
                texts.append(cur_texts)
                labels.append(label)

        # image root 
        self.root = Path('/gpfs/data/denizlab/Datasets/Public/physionet.org/files/mimic-cxr-jpg/2.0.0/')
        self.image_paths = image_paths
        self.image_times = image_times
        self.text_times = text_times
        self.texts = texts
        self.labels = labels
        self.transform = transform
        self.args = args

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        # read image
        if self.img_time_series:
            image_path = self.image_paths[index][-self.args.img_max_len:]
            image = torch.Tensor()
            
            for img_path in image_path:
                file_path = self.root / img_path
                PIL_image = Image.open(file_path).convert("RGB")
                # transform
                if self.transform is not None:
                    image_tensor = self.transform(PIL_image)
                image = torch.concat((image, image_tensor.unsqueeze(0)), dim=0)
                
            image_time = self.image_times[index][-self.args.img_max_len:]
            # normalize offset text_time to avoid too large values (max-min normalize)
            image_time = [1.0 if max(image_time) - min(image_time) == 0 \
                else (time - min(image_time)) / (max(image_time) - min(image_time)) for time in image_time]
        else:
            image_path = self.image_paths[index][-1]
            file_path = self.root / image_path
            PIL_image = Image.open(file_path).convert("RGB")
            # transform
            if self.transform is not None:
                image = self.transform(PIL_image)
            image_time = self.image_times[index][-1]
        # read label
        label = self.labels[index]
        
        # read text
        text = self.texts[index][-self.args.text_max_len:]
        text_time = self.text_times[index][-self.args.text_max_len:]
        
        # normalize offset text_time to avoid too large values (max-min normalize)
        text_time = [1.0 if max(self.image_times[index]) - min(text_time) == 0 \
            else (time - min(text_time)) / (max(self.image_times[index]) - min(text_time)) for time in text_time]
        
        return image, text, torch.FloatTensor(label), image_time, text_time

    def __len__(self):
        return len(self.image_paths)