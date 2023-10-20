
class TrainModel():
    def __init__(self, model, num_epochs, train_loader, test_loader, device, dataset_sizes, criterion=None, optimizer=None, exp_lr_scheduler=None) :
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = exp_lr_scheduler
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.dataset_sizes = dataset_sizes

    def trainModel(self):
        return self.model.train_model(self.criterion, self.optimizer, self.scheduler, self.num_epochs, self.train_loader, self.test_loader, self.device, self.dataset_sizes)
