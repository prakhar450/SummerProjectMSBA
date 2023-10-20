from torchvision.models import resnet50, ResNet50_Weights
from tempfile import TemporaryDirectory
import torch.nn as nn
import time

class CustomResnet():
    def __init__(self, total_classes):
        self.total_classes = total_classes
        self.model = None
        self.weights = None
        self.preprocess = None
    
    def createModel(self):
        self.model = resnet50(weights=None)
        self.weights = ResNet50_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.total_classes)
   
    def trainModel(self, criterion, optimizer, scheduler, num_epochs, train_loader, test_loader, device, dataset_sizes):
        since = time.time()
        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            torch.save(self.model.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(num_epochs):
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train','val']:
                    if phase == 'train':
                        self.model.train()  # Set model to training mode
                        dataloader = train_loader
                    else:
                        self.model.eval()   # Set model to evaluate mode
                        dataloader = test_loader

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for sample_batch in dataloader:
                        inputs = sample_batch['image'].to(device)
                        scores = sample_batch['score'].to(device)
                        for i, img in enumerate(inputs):
                           inputs[i] = self.preprocess(img).unsqueeze(0)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs.float())
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, scores)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == scores.data)
                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

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

    def getModel(self):
        self.createModel()
        return self.model

        
