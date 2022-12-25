import os
import logging

import pandas as pd
from PIL import Image
import numpy as np
from scipy import rand

import torch
from torch import Tensor
from torch.utils.data import Dataset

import albumentations as albu

logger = logging.getLogger()

VAL_SET = [
    ["OL", "LL", "E2", "EW", "GM", "S3"],
    ['E2', 'EW', 'HN', 'D8', 'AC', 'AQ'],
    ['BH', 'EW', 'LL', 'GI', 'A1', 'A7'],
    ['E9', 'BH', 'A8', 'AR', 'EW', 'LL'],
    ['D8', 'AQ', 'AR', 'C8', 'OL', 'A7']
]

class Bcss_dataset(Dataset):
    def __init__(self, data_path, transforms, frac=1, threshold=0.1, fold=0) -> None:
        super().__init__()

        self.data_path = data_path
        self.csv_path = data_path + "/data.csv"
        self.transforms = transforms
        self.frac = frac
        self.threshold = threshold
        self.fold = fold

        self._prepare()
        

    def __len__(self) -> int:
        return len(self.data_df)
        
    def __getitem__(self, index: int) -> Tensor:
        img_path = os.path.join(self.data_path, self.filename_imgs[index])
        img = np.asarray(Image.open(img_path))
        
        mask_path = os.path.join(self.data_path, self.filename_masks[index])
        mask = np.asarray(Image.open(mask_path))
        
        if self.transforms:
            sample = self.transforms(image=img, mask=mask)

        return sample

    def _prepare(self) -> None:
        data_df = pd.read_csv(self.csv_path)
        logger.info(f"Reading {len(data_df)} files in {self.csv_path}...")

        data_df = data_df[~data_df["filename"].str.split("-").str[1].isin(VAL_SET[self.fold])].reset_index(drop=True)
        logger.info(f"Using fold {self.fold} and keep {len(data_df)} train files only...")
        
        logger.info(f"Removing images with threshold of {self.threshold}...")
        data_df = data_df[data_df['ratio_masked_area']>=self.threshold].reset_index(drop=True)
        logger.info(f"Create train set with {len(data_df)} files...")

        self.data_df = data_df.sample(frac=self.frac, replace=False, random_state=1).reset_index(drop=True)
        logger.info(f"Use {self.frac} percent of data to train: {len(self.data_df)}!")

        self.filename_imgs = self.data_df['filename_img'].to_numpy()
        self.filename_masks = self.data_df['filename_mask'].to_numpy()


class Bcss_dataset_val(Dataset):
    def __init__(self, data_path, transforms, threshold=0.1, fold=0) -> None:
        super().__init__()

        self.data_path = data_path
        self.csv_path = data_path + "/val_data.csv"
        self.transforms = transforms
        self.threshold = threshold
        self.fold = fold

        self._prepare()
        

    def __len__(self) -> int:
        return len(self.files)
        
    def __getitem__(self, index: int) -> Tensor:
        filename = self.files[index]
        df = self.data_df[self.data_df['filename']==filename].reset_index(drop=True)

        target_imgs = []
        target_masks = []
        for img_name, mask_name in zip(df["filename_img"], df["filename_mask"]):
            img_path = os.path.join(self.data_path, img_name)
            img = np.asarray(Image.open(img_path))
            
            mask_path = os.path.join(self.data_path, mask_name)
            mask = np.asarray(Image.open(mask_path))
            
            if self.transforms:
                sample = self.transforms(image=img, mask=mask)

            target_imgs.append(sample["image"])
            target_masks.append(sample["mask"])

        target_imgs = torch.stack(target_imgs, axis=0)
        target_masks = torch.stack(target_masks, axis=0)
        sample = {"image": target_imgs, "mask": target_masks}

        return sample

    def _prepare(self) -> None:
        data_df = pd.read_csv(self.csv_path)
        logger.info(f"Reading {len(data_df)} files in {self.csv_path}...")

        data_df = data_df[data_df["filename"].str.split("-").str[1].isin(VAL_SET[self.fold])].reset_index(drop=True)
        data_df = data_df[~data_df["filename"].str.contains("shift")].reset_index(drop=True)
        logger.info(f"Using fold {self.fold} and keep {len(data_df)} val files only...")
        
        logger.info(f"Removing images with threshold of {self.threshold}...")
        data_df = data_df[data_df['ratio_masked_area']>=self.threshold].reset_index(drop=True)
        logger.info(f"Create val set with {len(data_df)} files...")

        self.files = data_df["filename"].unique()
        self.data_df = data_df