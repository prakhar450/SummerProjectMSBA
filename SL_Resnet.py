# Import Statements
import numpy as np
import torch
from skimage import io
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
import time
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()   # interactive mode


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()


# Global Variables

np.random.seed(100)
nrows = 400
ncolumns = 300
batch_size = 4
num_workers = 0

img_dir = '/Users/prakharsrivastav/Summer2023/place-pulse-2.0/images'
image_score_df_train = '/Users/prakharsrivastav/Summer2023/place-pulse-2.0/smaller_image_score_table_train.csv'
image_score_df_val = '/Users/prakharsrivastav/Summer2023/place-pulse-2.0/smaller_image_score_table_val.csv'
csv_paths = {'train': image_score_df_train, 'val': image_score_df_val}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

# Data Loaders
data_transforms = {
    'train' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256)),
        transforms.CenterCrop(224)
        ]),
    'val' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256)),
        transforms.CenterCrop(224)
        ])
}

transformed_image_datasets = {x: CustomDataset(csv_path=csv_paths[x], img_dir=img_dir, transform=data_transforms[x]) 
                              for x in ['train', 'val'] }


dataloaders = { x: DataLoader(transformed_image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
               for x in ['train', 'val'] }

dataset_sizes = {x: len(transformed_image_datasets[x]) for x in ['train', 'val']}

# Helper function to show a batch
def show_batch(sample_batched):
    """Show image for a batch of samples."""
    images_batch = sample_batched['image']

    grid = utils.make_grid(images_batch)
    plt.title('Batch from dataloader')
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

# if __name__ == '__main__':
for i_batch, sample_batched in enumerate(dataloaders['train']):
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

# Helper Functions for Model Training

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_loss = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                # Iterate over data.
                for inputs, scores in dataloaders[phase]:
                    inputs = inputs.to(device)
                    scores = scores.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, scores)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Loss: {best_loss:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model



# Model Definition




# Model Training and Evaluation






# Examples to demo



