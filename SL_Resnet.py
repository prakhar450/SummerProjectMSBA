# Import Statements
import numpy as np
import torch
from skimage import io
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()


# Global Variables
np.random.seed(100)
nrows = 400
ncolumns = 300
batch_size = 4
img_dir = '/Users/prakharsrivastav/Summer2023/place-pulse-2.0/images'
image_score_df = '/Users/prakharsrivastav/Summer2023/place-pulse-2.0/smaller_image_score_table.csv'


# Utility Classes
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

        img_path = os.path.join(self.img_dir, self.df.iloc[index, 1])
        image = io.imread(img_path)
        score = self.df.iloc[index, 2]

        sample = {'image': image, 'score': score}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        
        return sample



# Sample Data Loader

image_dataset = CustomDataset(csv_path=image_score_df, img_dir=img_dir)


## Plot the first 4 images to visualise
fig = plt.figure()
for i, sample in enumerate(image_dataset):
    print(i, sample['image'].shape, sample['score'])

    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample['image'])
    
    if i == 3:
        #plt.show(block = True)
        break

# Data Loaders
tsfrm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )
])

transformed_image_dataset = CustomDataset(csv_path=image_score_df, img_dir=img_dir, transform=tsfrm)

for i, sample in enumerate(transformed_image_dataset):
    print(i, sample['image'].size(), sample['score'])
    if i ==3:
        break

dataloader = DataLoader(transformed_image_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# Helper function to show a batch
def show_batch(sample_batched):
    """Show image for a batch of samples."""
    images_batch = sample_batched['image']

    grid = utils.make_grid(images_batch)
    plt.title('Batch from dataloader')
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

# if you are using Windows, uncomment the next line and indent the for loop.
# you might need to go back and change ``num_workers`` to 0.

# if __name__ == '__main__':
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['score'])

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break


# Model Implementation






# Model Training







# Model Testing






# Examples to demo



