from models.CustomResnet import CustomResnet
from models.CustomAlexnetSVM import CustomAlexnetSVM

class CustomModel():
    def __init__(self, model_name, total_classes):
        self.model_name = model_name
        self.total_classes = total_classes
    
    def retrieveModel(self):
        if self.model_name == "resnet50":
            return CustomResnet(total_classes=self.total_classes)
        elif self.model_name == "alexnetSVM":

            return CustomAlexnetSVM(total_classes=self.total_classes)
        else:
            return 'No model found\n'
