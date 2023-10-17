from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn


class CustomResnet():
    def __init__(self, total_classes):
        self.total_classes = total_classes
        self.model = None
        self.weights = None
        self.preprocess = None
    
    def createModel(self):
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.weights = ResNet50_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.total_classes)
    
    def getModel(self):
        self.createModel()
        return {'model': self.model, 'preprocess': self.preprocess}

        