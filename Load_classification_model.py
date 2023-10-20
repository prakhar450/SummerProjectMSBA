# Import Statements
import numpy as np
import torch
from skimage import io
import os
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import time
from tempfile import TemporaryDirectory
import sys

cudnn.benchmark = True
plt.ion()   # interactive mode


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()

"""
Arguments to this code

1. model name to be loaded
2. folder name (beautiful etc)
3. US/OECD
4. full/small
5. output csv file name
6. batch_size
"""

# Global Variables

np.random.seed(100)
num_workers = 0
model = "../SummerProjectMSBA/saved_models/oct16/unbalanced_custom_resnet50.pt"
batch_size = 32

img_dir = '../images'

image_score_df_train = '../Dataset_wo_vip/safe/full_balanced_df_train_1_US.csv'
image_score_df_val = '../Dataset_wo_vip/safe/full_unbalanced_df_valid_1_US.csv'

csv_paths = {'train': image_score_df_train, 'val': image_score_df_val}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Utility Classes
class CustomDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform = None):
        self.df = pd.read_csv(csv_path, index_col=0)
        #self.df = pd.read_csv(csv_path)
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

# Data Loaders
data_transforms = {
    'train' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        ]),
    'val' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        ])
}

transformed_image_datasets = {x: CustomDataset(csv_path=csv_paths[x], img_dir=img_dir, transform=data_transforms[x]) 
                              for x in ['train', 'val'] }

dataloaders = { x: DataLoader(transformed_image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
               for x in ['train', 'val'] }

dataset_sizes = {x: len(transformed_image_datasets[x]) for x in ['train', 'val']}

total_classes = transformed_image_datasets['train'].df['trueskill.score'].max() + 1

print(dataset_sizes)

for i_batch, sample_batched in enumerate(dataloaders['train']):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['score'])

    # observe 4th batch and stop.
    if i_batch == 3:
       break



# Visualising model with some images

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    #fig = plt.figure()
    num_images = 200
    df_output = pd.DataFrame(columns=['predicted', 'score'])

    with open("df_output.txt", "a") as file:
        with torch.no_grad():
            for sample_batch in dataloaders['val']:
                inputs = sample_batch['image'].to(device)
                scores = sample_batch['score'].to(device)

                outputs = model(inputs)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    line = f'predicted: {outputs[j]} and score: {scores[j]}\n'
                    file.writelines(line)
                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
        model.train(mode=was_training)

# Store the outputs in a csv
def run_model_and_store_outputs(model):
    print("start storing predicted values")
    was_training = model.training
    model.eval()
    df_output = pd.DataFrame(columns=['actual_score', 'rounded_score', 'predicted'])

    with torch.no_grad():
        for sample_batch in dataloaders['val']:
            inputs = sample_batch['image'].to(device)
            scores = sample_batch['score']
            actual_scores = sample_batch['actual_score']
            
            outputs = model(inputs)
            outputs = outputs.cpu()
            for j in range(inputs.size()[0]):
                index_val = outputs[j].data.numpy().argmax()
                df_output.loc[len(df_output.index)] = [actual_scores[j], scores[j], index_val]
    name_of_csv_file = "../SummerProjectMSBA/outputs/safe/predictedVsActualScores_US.csv"
    df_output.to_csv(name_of_csv_file)
    print("done")

# Store all the probability values in a csv
def run_model_and_store_all_probabilities(model):
    print("start storing probabilities")
    was_training = model.training
    model.eval()
    df_output = pd.DataFrame(columns=['actual_score', 'rounded_score', 'predicted_probabilites'])

    with torch.no_grad():
        for sample_batch in dataloaders['val']:
            inputs = sample_batch['image'].to(device)
            scores = sample_batch['score']
            actual_scores = sample_batch['actual_score']
            
            outputs = model(inputs)
            p = torch.nn.functional.softmax(outputs, dim=1)
            p = p.cpu()
            for j in range(inputs.size()[0]):
                df_output.loc[len(df_output.index)] = [actual_scores[j], scores[j], p[j]]
    name_of_csv_file = "../SummerProjectMSBA/outputs/safe/probabilitiesVsActualScores_US.csv"
    df_output.to_csv(name_of_csv_file)
    print("done")

def run_combined_model_and_store_outputs(model):
    print("start storing predicted values")
    was_training = model.training
    model.eval()
    df_output = pd.DataFrame(columns=['actual_score', 'rounded_score', 'predicted', 'predicted_probabilites'])

    with torch.no_grad():
        for sample_batch in dataloaders['val']:
            inputs = sample_batch['image'].to(device)
            scores = sample_batch['score']
            actual_scores = sample_batch['actual_score']
            
            outputs = model(inputs)
            outputs = outputs.cpu()
            p = torch.nn.functional.softmax(outputs, dim=1)
            p = p.cpu()
            for j in range(inputs.size()[0]):
                index_val = outputs[j].data.numpy().argmax()
                df_output.loc[len(df_output.index)] = [actual_scores[j], scores[j], index_val, p[j]]
    name_of_csv_file = "../SummerProjectMSBA/outputs/safe/predictedVsActualScores_US.csv"
    df_output.to_csv(name_of_csv_file)
    print("done")



# Model Definition

total_classes = transformed_image_datasets['train'].df['trueskill.score'].max() + 1

#model_ft = resnet50()
#num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, total_classes)

# Define a custom model by inheriting nn.Module
class CustomResNet(nn.Module):
    def __init__(self, num_classes=36):
        super(CustomResNet, self).__init__()
        # Load pretrained ResNet-50
        self.resnet = models.resnet50(pretrained=True)
        # Modify the fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Instantiate the custom model
num_classes = 36
model_ft = CustomResNet(num_classes).cuda()



PATH = "../SummerProjectMSBA/saved_models/oct16/unbalanced_custom_resnet50.pt"
model_ft.load_state_dict(torch.load(PATH))

model_ft.to(device)
run_combined_model_and_store_outputs(model_ft)
