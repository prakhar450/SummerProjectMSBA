import torch
import os
from skimage import io
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform = None):
        self.df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = os.path.join(self.img_dir, self.df.iloc[index, 0])
        image = io.imread(img_path)
        score = self.df.iloc[index, 1]
        actual_score = self.df.iloc[index, 5]

        sample = {'image': image, 'score': score, 'actual_score': actual_score}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        
        return sample