import torch
import time
from tempfile import TemporaryDirectory
import os

class TrainModel():
    def __init__(self, model, criterion, optimizer, exp_lr_scheduler, num_epochs, preprocess, train_loader, test_loader, device, dataset_sizes):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = exp_lr_scheduler
        self.num_epochs = num_epochs
        self.preprocess = preprocess
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.dataset_sizes = dataset_sizes

    def trainModel(self):
        since = time.time()
        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            torch.save(self.model.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(self.num_epochs):
                print(f'Epoch {epoch}/{self.num_epochs - 1}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train','val']:
                    if phase == 'train':
                        self.model.train()  # Set model to training mode
                        dataloader = self.train_loader
                    else:
                        self.model.eval()   # Set model to evaluate mode
                        dataloader = self.test_loader
                    
                    running_loss = 0.0
                    running_corrects = 0
                    
                    # Iterate over data.
                    for sample_batch in dataloader:
                        inputs = sample_batch['image'].to(self.device)
                        scores = sample_batch['score'].to(self.device)
                        for i, img in enumerate(inputs):
                           inputs[i] = self.preprocess(img).unsqueeze(0) 
                            
                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs.float())
                            _, preds = torch.max(outputs, 1)
                            loss = self.criterion(outputs, scores)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == scores.data)
                    if phase == 'train':
                        self.scheduler.step()

                    epoch_loss = running_loss / self.dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(self.model.state_dict(), best_model_params_path)

                print()

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:4f}')

            # load best model weights
            self.model.load_state_dict(torch.load(best_model_params_path))
        return self.model
