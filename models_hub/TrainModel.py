
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
        return self.model.trainModel(self.criterion, self.optimizer, self.exp_lr_scheduler, self.num_epochs, self.train_loader, self.test_loader, self.device, self.dataset_sizes)
