import torch.nn as nn
import time
import torch
import torchvision.models as models
from torchvision import transforms
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from tempfile import TemporaryDirectory

'''
# Transform to preprocess input images
	preprocess = transforms.Compose([
	    transforms.ToPILImage(),  # Convert to PIL Image
	    transforms.Resize(256),
	    transforms.CenterCrop(224),
	    transforms.ToTensor(),
	])
	

'''

class CustomAlexnetSVM():
	def __init__(self, total_classes):
        self.total_classes = total_classes
        self.model = None
        self.weights = None
        self.preprocess = None
    
    def createModel(self):
        self.model = models.alexnet(weights='DEFAULT')
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.weights = models.AlexNet_Weights.IMAGENET1K_V1
        self.preprocess = self.weights.transforms()
        
        self.svm_classifier = svm.SVC(kernel='linear')
        
        self.classifier = cls()
        self.classifier.alexnet = self.model
        self.classifier.svm_classifier = self.svm_classifier
    
    def load_model_state(path):
        checkpoint = torch.load(filename)
        
        classifier = cls()
        classifier.alexnet.load_state_dict(checkpoint['alexnet_state_dict'])
        classifier.svm_classifier = checkpoint['svm_classifier']
        
        return classifier
    
    def getModel(self):
        self.createModel()
        return self.classifier
	    
	def extract_features(img):
	    with torch.no_grad():
	        features = self.model(img.unsqueeze(0))
	        features = features.view(features.size(0), -1)
	    return features
    
    def trainModel(self, criterion, optimizer, scheduler, num_epochs, train_loader, test_loader, device, dataset_sizes):
        since = time.time()
        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            torch.save(self.model.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(num_epochs):
                print(f'Epoch {epoch}/{self.num_epochs - 1}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train','val']:
                    if phase == 'train':
                        dataloader = train_loader
                    else:
                        dataloader = test_loader

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for sample_batch in dataloader:
                        inputs = sample_batch['image'].to(device)
                        scores = sample_batch['score'].to(device)
                        
                        feature_batch = []
                        for i, img in enumerate(inputs):
                           inputs[i] = self.preprocess(img).unsqueeze(0)
                           features = self.extract_features(inputs[i])
                           feature_batch.append(features.numpy())
                        
                        feature_batch = np.stack(feature_batch)
                        
                        if phase == 'train':
                            self.svm_classifier.train(features, scores)
                        else:
                            accuracy = svm_classifier.evaluate(features, scores)
                            print("Accuracy: ", accuracy)

                    # deep copy the model
                    if phase == 'val' and accuracy > best_acc:
                        best_acc = accuracy
                        checkpoint = {
                            'alexnet_state_dict': self.model.state_dict(),
                            'svm_classifier': self.svm_classifier,
                        }
                        torch.save(checkpoint, best_model_params_path)

                print()

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:4f}')

            # load best model weights
            return self.load_model_state(best_model_params_path)
