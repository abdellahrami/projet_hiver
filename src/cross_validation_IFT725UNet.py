from CNNTrainTestManager import CNNTrainTestManager, optimizer_setup
from models.IFT725UNet import IFT725UNet
import torch.optim as optim
from HDF5Dataset import HDF5Dataset
import torchvision.transforms as transforms
import torch.nn as nn


hdf5_file = '../data/hdf5/ift725_acdc.hdf5'
acdc_base_transform = transforms.Compose([
        transforms.ToTensor()
])
train_set = HDF5Dataset('train', hdf5_file, transform=acdc_base_transform)
test_set = HDF5Dataset('test', hdf5_file, transform=acdc_base_transform)

residuts = [
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,1,0,1,0,1,0,1,0,1,0],
            [1,1,1,1,1,1,0,0,0,0,0,0]
            ]
            
num_epochs = 10
best_val_loss = 10
best_residut = [1,1,1,1,1,1,1,1,1,1,1,1]
best_lr = 0
for lr in [0.001, 0.005]:
    for residut in residuts:
    model = IFT725UNet(num_classes=4, residuts=residut)
    optimizer_factory = optimizer_setup(optim.Adam, lr=0.001)
    model_trainer = CNNTrainTestManager(model=model,
                                trainset=train_set,
                                testset=test_set,
                                batch_size=20,
                                loss_fn=nn.CrossEntropyLoss(),
                                optimizer_factory=optimizer_factory,
                                validation=0.1,
                                use_cuda=True)
    model_trainer.train(num_epochs)
    val_loss = model_trainer.metric_values['val_loss'][-1]
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_residut = residut
        best_lr = lr
print("best_lr, best_residut:",best_lr, best_residut)