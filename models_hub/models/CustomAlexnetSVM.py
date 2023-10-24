import torch.nn as nn
from cuml import SVC
from cuml.svm import LinearSVC
import os
import time
import torch
import torchvision.models as models
from torchvision import transforms
from sklearn import svm
from sklearn.svm import NuSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
from tempfile import TemporaryDirectory
from tqdm import tqdm

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
        self.classifier = {}
    
    def create_model(self):
        #self.model = models.alexnet(weights='DEFAULT')
        # load the pre-trained weights
        arch = "alexnet"
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        self.model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-2])
        self.model.eval()

        self.weights = models.AlexNet_Weights.IMAGENET1K_V1
        self.preprocess = self.weights.transforms()

        self.svm_classifier = make_pipeline(StandardScaler(), NuSVR(nu=0.5, C = 0.1, kernel='rbf'))
        #self.svm_classifier = SVC(kernel='linear', gamma='auto')
        #self.svm_classifier = LinearSVC(loss='squared_hinge', penalty='l2', C=1)

        self.classifier['alexnet'] = self.model
        self.classifier['svm_classifier'] = self.svm_classifier

    def load_model_state(path):
        checkpoint = torch.load(filename)

        classifier = {}
        classifier['alexnet'] = alexnet.load_state_dict(checkpoint['alexnet_state_dict'])
        classifier['svm_classifier'] = checkpoint['svm_classifier']

        return classifier

    def get_model(self):
        self.create_model()
        return self.classifier
	    
    def extract_features(self, img):
        with torch.no_grad():
            features = self.model(img.cuda().float())
            features = features.view(features.size(0), -1)
        return features
    
    def train_model(self, criterion, optimizer, scheduler, num_epochs, train_loader, test_loader, device, dataset_sizes):
        since = time.time()
        self.model.to(device)

        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            best_acc = 0.0

            for epoch in range(num_epochs):
                all_predicted = []
                all_scores = []
                time_now = time.strftime("%H:%M:%S", time.localtime())
                print("training start time: ", time_now)
        
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train','val']:
                    if phase == 'train':
                        dataloader = train_loader
                    else:
                        dataloader = test_loader

                    running_loss = 0.0
                    running_corrects = 0
                    batch_iterator = iter(dataloader)
                    num_batches = len(dataloader)
                    # Iterate over data.
                    for batch_idx in tqdm(range(num_batches), leave=False):
                        sample_batch = next(batch_iterator)
                        

                        inputs = sample_batch['image'].to(device)
                        scores = sample_batch['score'].numpy()
                        
                        feature_batch = []
                        for i, img in enumerate(inputs):
                           new_img = self.preprocess(img).unsqueeze(0)
                           inputs[i] = new_img
                           features = self.extract_features(new_img)
                           features = features.squeeze(0)
                           feature_batch.append(features.cpu().numpy())
                        
                        feature_batch = np.stack(feature_batch)
                        if phase == 'train':
                            self.svm_classifier.fit(feature_batch, scores)
                        else:
                            #predicted = self.svm_classifier.predict(feature_batch)
                            #accuracy = accuracy_score(scores, predicted)
                            predicted = self.svm_classifier.predict(feature_batch)
                            all_predicted.append(predicted)
                            all_scores.append(scores)


                    
                print("predicted type : ",type(predicted))
                print("scores type: ", type(scores))
                all_pred = np.concatenate(all_predicted)
                all_sc = np.concatenate(all_scores)
                accuracy = r2_score(all_sc, all_pred)
                print("accuracy: ", accuracy)
                # deep copy the model
                if accuracy > best_acc:
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
