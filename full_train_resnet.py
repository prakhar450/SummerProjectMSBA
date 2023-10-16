#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms, models
import pandas as pd
import os
from glob import glob
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tempfile import TemporaryDirectory
if not os.path.exists('./outputs'):
    os.mkdir('./outputs')


# In[2]:
import warnings
warnings.filterwarnings("ignore")

# Global parameters

# If USE_CUDA is True, computations will be done using the GPU (may not work in all systems)
# This will make the calculations happen faster
USE_CUDA = torch.cuda.is_available()

DATASET_PATH = '../images'

BATCH_SIZE = 32 # Number of images that are used for calculating gradients at each step

NUM_EPOCHS = 25 # Number of times we will go through all the training images. Do not go over 25

LEARNING_RATE = 0.001 # Controls the step size
MOMENTUM = 0.9 # Momentum for the gradient descent
WEIGHT_DECAY = 0.0001
num_workers = 2
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# In[ ]:


image_score_df_train = '../Dataset_wo_vip/safe/full_unbalanced_df_train_1_US.csv'


image_score_df_val = '../Dataset_wo_vip/safe/full_unbalanced_df_valid_1_US.csv'

csv_paths = {'train': image_score_df_train, 'val': image_score_df_val}


# In[ ]:


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


# In[9]:


# Create datasets and data loaders
# Transformations

data_transforms_1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
data_transforms_2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        #transforms.RandomResizedCrop((224, 224)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        #transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


transformed_image_datasets = {x: CustomDataset(csv_path=csv_paths[x], img_dir=DATASET_PATH, transform=data_transforms_1) 
                              for x in ['train', 'val'] }

dataloaders = { x: DataLoader(transformed_image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
               for x in ['train', 'val'] }

dataset_sizes = {x: len(transformed_image_datasets[x]) for x in ['train', 'val']}

total_classes = transformed_image_datasets['train'].df['trueskill.score'].max() + 1

print(dataset_sizes)

#train_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, 'train'), data_transforms)
train_loader = DataLoader(transformed_image_datasets['train'], BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)


#test_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, 'test'), data_transforms)
test_loader = DataLoader(transformed_image_datasets['val'], BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)



print('Dataloaders OK')
#test_loader





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
model = CustomResNet(num_classes).cuda()


# Instantiate Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(beta1, beta2), eps=epsilon, weight_decay=WEIGHT_DECAY)
#optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()


# # Training with ResNet

# In[157]:


# Main loop
train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []
epochs = []


with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS+1):

        print(f'\n\nRunning epoch {epoch} of {NUM_EPOCHS}...\n')
        epochs.append(epoch)

        #-------------------------Train-------------------------
        
        #Reset these below variables to 0 at the begining of every epoch
        correct = 0
        iterations = 0
        iter_loss = 0.0
        
        model.train()  # Put the network into training mode
        
        for i, sample_batch in enumerate(train_loader):
            if USE_CUDA:
                inputs = sample_batch['image'].cuda()
                scores = sample_batch['score'].cuda()        
                
            outputs = model(inputs)
#            print(outputs.shape)
#            print(scores.shape)
            loss = criterion(outputs, scores)
            iter_loss += loss.item()  # Accumulate the loss
            optimizer.zero_grad() # Clear off the gradient in (w = w - gradient)
            loss.backward()   # Backpropagation 
            optimizer.step()  # Update the weights
            
            # Record the correct predictions for training data 
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == scores).sum()
            iterations += 1
            
        scheduler.step()
            
        # Record the training loss
        train_loss.append(iter_loss/iterations)
        # Record the training accuracy
        train_accuracy.append((100 * correct / dataset_sizes['train']))   
         
        #-------------------------Test--------------------------
        
        correct = 0
        iterations = 0
        testing_loss = 0.0
        
        model.eval()  # Put the network into evaluation mode
        
        for i, sample_batch in enumerate(train_loader):

            if USE_CUDA:
                inputs = inputs.cuda()
                scores = sample_batch['score'].cuda()
            
            outputs = model(inputs)     
            loss = criterion(outputs, scores) # Calculate the loss
            testing_loss += loss.item()
            # Record the correct predictions for training data
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == scores).sum()
            
            iterations += 1

        # Record the Testing loss
        test_loss.append(testing_loss/iterations)
        # Record the Testing accuracy
        test_accuracy.append((100 * correct / dataset_sizes['val']))

        if test_accuracy[-1] > best_acc:
            best_acc = test_accuracy[-1]
            print(f'\nUpdating best accuracy model with accuracy {test_accuracy[-1]}\n')
            torch.save(model.state_dict(), best_model_params_path)
        
        print(f'\nEpoch {epoch} validation results: Loss={test_loss[-1]} | Accuracy={test_accuracy[-1]}\n')

    model.load_state_dict(torch.load(best_model_params_path))


# # Results

# In[158]:
name_of_model_file = "./saved_models/oct16/unbalanced_resnet50.pt"
torch.save(model.state_dict(), name_of_model_file)

print(f'Final train loss: {train_loss[-1]}')
print(f'Final test loss: {test_loss[-1]}')
print(f'Final train accuracy: {train_accuracy[-1]}')
print(f'Final test accuracy: {test_accuracy[-1]}')

