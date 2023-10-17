# Import Statements
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import sys

import CustomDataset
import CustomModel
import TrainModel

cudnn.benchmark = True

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


batch_size = 32
num_epochs = 10
num_workers = 2
total_classes = 36
model_name = "resnet50"
criterion = nn.CrossEntropyLoss()
optimizer_name = "Adam"
learning_rate = 0.01
momentum = 0.9
weight_decay = 0.0001
step_size = 10
gamma = 0.3

image_score_df_train = './dataset/safe/full_balanced_df_train_1_US.csv'
image_score_df_val = './dataset/safe/full_unbalanced_df_valid_1_US.csv'

csv_paths = {'train': image_score_df_train, 'val': image_score_df_val}

img_dir = '../../images'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_dataset = CustomDataset(csv_path=csv_paths['train'], img_dir=img_dir)
test_dataset = CustomDataset(csv_path=csv_paths['val'], img_dir=img_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

dataset_sizes = {'train': len(train_dataset), 'val': len(test_dataset)}
print(dataset_sizes)

custom_model = CustomModel(model_name=model_name, total_classes=total_classes).retrieveModel()

model = custom_model['model']
preprocess = custom_model['preprocess']

model = model.to(device)

if optimizer_name == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

if optimizer_name == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma)

training_step = TrainModel(model, criterion, optimizer, exp_lr_scheduler, num_epochs, preprocess, train_loader, test_loader, device, dataset_sizes)

trained_model = training_step.train_model()

trained_model_file = "./saved_models/{}".format(sys.argv[1])
torch.save(trained_model.state_dict(), trained_model_file)









