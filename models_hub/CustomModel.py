from CustomResnet import CustomResnet

class CustomModel():
    def __init__(self, model_name, total_classes):
        self.model_name = model_name
        self.total_classes = total_classes
    
    def retrieveModel(self):
        if self.model_name == "resnet50":
            return CustomResnet(total_classes=self.total_classes).getModel()
