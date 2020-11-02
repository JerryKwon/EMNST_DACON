# dataset_loader.py
# Manipulating EMNST Dataset
# 1. Save images one by one
# 2. Spliting Train by n_folds

import os
import platform
import warnings
import re

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

class DatasetLoader:
    def __init__(self):

        warnings.filterwarnings("ignore")

        os_env = platform.system()

        if os_env == 'Linux':
            self.data_path = os.path.abspath(os.path.dirname(os.path.abspath('__file__'))) + '/input/data'
            self.result_path = os.path.abspath((os.path.dirname(os.path.abspath('__file__')))) + '/output/results'

        elif os_env == 'Windows':
            self.data_path = os.path.abspath(os.path.dirname(os.path.abspath('__file__'))) + '\\input\\data'
            self.result_path = os.path.abspath((os.path.dirname(os.path.abspath('__file__')))) + '\\output\\results'

        self.train_path = os.path.join(self.data_path,'train')
        self.test_path = os.path.join(self.data_path,'test')

    def load(self, file_name):
        data_path = self.data_path
        dataset = pd.read_csv(os.path.join(data_path,file_name))
        return dataset

    def split(self, n_folds=5):

        df_train = self.load("train.csv")
        df_test = self.load("test.csv")

        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=101)

        for fold, (trn_idx, val_idx) in enumerate(kfold.split(X=df_train,y=df_train.digit)):
            df_train.loc[val_idx,"fold"] = fold

        df_train.loc[:,"fold"]= df_train.fold.astype("int")

        self.make_folder(self.train_path)
        self.make_folder(self.test_path)

        df_train_imgs = df_train.loc[:,df_train.columns[3:-1]].values
        df_test_imgs = df_test.loc[:,df_test.columns[2:]].values

        for fold in range(n_folds):
            fold_path = os.path.join(self.train_path,str(fold))
            self.make_folder(fold_path)

        print("Generating Local Train Dataset per image by fold", '*'*20)

        for idx, id in tqdm(enumerate(df_train.id.values)):
            img_id = id
            fold = df_train.loc[df_train.id==img_id, "fold"].values[0]
            digit = df_train.loc[df_train.id==img_id, "digit"].values[0]
            letter = df_train.loc[df_train.id==img_id, "letter"].values[0]
            img = df_train_imgs[idx,:]

            target_path = os.path.join(os.path.join(self.train_path,str(fold)),"%d_%d_%c.pkl"%(img_id,digit,letter))

            with open(target_path,'wb') as f:
                pickle.dump(img,f)

        print("Generating Local Test Dataset per image", '*' * 20)

        for idx,id in tqdm(enumerate(df_test.id.values)):
            img_id = id
            letter = df_test.loc[df_test.id==img_id,"letter"].values[0]
            img = df_test_imgs[idx,:]

            target_path = os.path.join(self.test_path,"%d_%c.pkl"%(img_id, letter))

            with open(target_path,"wb") as f:
                pickle.dump(img,f)

    def load_split(self,n_folds=5):
        np.random.seed(101)
        val_fold = np.random.randint(5)
        trn_folds = [idx for idx in range(5) if idx != val_fold]
        print(f"Valid fold is {val_fold} among {list(range(n_folds))}")

        trn_pattern = '[0-9]+_[0-9]_[A-Z]'

        trn_dict = dict()
        val_dict = dict()
        idx = 0

        print(f"Loading Train Dataset w/o No.{val_fold} fold",'*'*20)

        for trn_fold in trn_folds:
            fold_path = os.path.join(self.train_path,str(trn_fold))
            for dir_name, _, file_names in os.walk(fold_path):
                for file_name in tqdm(file_names):
                    img_str = re.search(trn_pattern,file_name).group()
                    img_id, img_digit, img_letter = img_str.split("_")
                    img_path = os.path.join(fold_path,file_name)
                    with open(img_path,'rb') as f:
                        img = pickle.load(f)
                        trn_dict[idx] = {"img_id":img_id, "digit":img_digit, "letter":img_letter, "img":img}
                        idx += 1

        idx = 0

        print(f"Loading Valid Dataset No.{val_fold} fold",'*'*20)

        fold_path = os.path.join(self.train_path,str(val_fold))
        for dir_nam, _, file_names in os.walk(fold_path):
            for file_name in tqdm(file_names):
                img_str = re.search(trn_pattern, file_name).group()
                img_id, img_digit, img_letter = img_str.split("_")
                img_path = os.path.join(fold_path,file_name)
                with open(img_path, "rb") as f:
                    img = pickle.load(f)
                    val_dict[idx] = {"img_id":img_id, "digit":img_digit, "letter":img_letter,"img":img}
                    idx+=1

        test_pattern = '[0-9]+_[A-Z]'
        test_dict = dict()
        idx = 0

        print(f"Loading Test Dataset",'*'*20)

        for dir_name, _, file_names in os.walk(self.test_path):
            for file_name in tqdm(file_names):
                img_str = re.search(test_pattern ,file_name).group()
                img_id, img_letter = img_str.split("_")
                img_path = os.path.join(self.test_path,file_name)
                with open(img_path, "rb") as f:
                    img = pickle.load(f)
                    test_dict[idx] = {"img_id":img_id,"letter":img_letter,"img":img}
                    idx+=1

        return trn_dict, val_dict, test_dict

    def submit(self, result_dict, filename,output_type="test"):
        base_path = self.get_basepath()
        result_path = os.path.join(base_path, 'output', 'results')
        result_file_path = os.path.join(result_path, f'{filename}')

        if output_type == "valid":
            submission = pd.DataFrame.from_dict(result_dict,orient='index')
            submission = submission.reset_index()
            submission.columns = ["id","digit"]
            submission.loc[:,"id"] = submission.id.astype("int")
            submission = submission.sort_values(by="id")

        else:
            submission = pd.read_csv(os.path.join(self.data_path,'submission.csv'))
            submission.loc[:,"digit"] = submission.id.apply(lambda x: result_dict[x])

        submission.to_csv((result_file_path), index=False)

        print(f"submission file {filename} is created At {result_file_path}")

    def make_folder(self, path):
        is_dir = os.path.isdir(path)
        if not is_dir:
            os.mkdir(path)
            print(path, " is made.")
        else:
            print(path, " is already existed.")

    def get_basepath(self):
        return os.path.abspath(os.path.dirname(os.path.abspath('__file__')))
