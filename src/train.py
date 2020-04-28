import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from CNNTrainTestManager import CNNTrainTestManager, optimizer_setup
from HDF5Dataset import HDF5Dataset
from ProjetHiver_DataSet import ProjetHiver_DataSet
from Highway_DataSet import Highway_DataSet
from models.AlexNet import AlexNet
from models.CNNVanilla import CnnVanilla
from models.IFT725Net import IFT725Net
from models.IFT725UNet import IFT725UNet
from models.ResNet import ResNet
from models.UNet import UNet
from models.VggNet import VggNet
from torchvision import datasets
import os

def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [hyper_parameters]'
                                           '\n python3 train.py --model UNet [hyper_parameters]'
                                           '\n python3 train.py --model UNet --predict',
                                     description="This program allows to train different models of classification on"
                                                 " different datasets. Be aware that when using UNet model there is no"
                                                 " need to provide a dataset since UNet model only train "
                                                 "on acdc dataset.")
    parser.add_argument('--model', type=str, default="CnnVanilla",
                        choices=["CnnVanilla", "VggNet", "AlexNet", "ResNet", "IFT725Net", "UNet", "IFT725UNet"])
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "svhn","highway",'projetSession'])
    parser.add_argument('--batch_size', type=int, default=20,
                        help='The size of the training batch')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"],
                        help="The optimizer to use for training the model")
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='The number of epochs')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--data_aug', action='store_true',
                        help="Data augmentation")
    parser.add_argument('--predict', action='store_true',
                        help="Use UNet model to predict the mask of a randomly selected image from the test set")
    parser.add_argument('--continue_train', action='store_true',
                        help="continue training using last training weights")
    return parser.parse_args()


def load_checkpoint(model, model_trainer, filename='UNet_state'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model_trainer.epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model_trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        model_trainer.metric_values = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return


if __name__ == "__main__":

    args = argument_parser()
    continue_train = args.continue_train
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_set = args.validation
    learning_rate = args.lr
    data_augment = args.data_aug
    if data_augment:
        print('Data augmentation activated!')
    else:
        print('Data augmentation NOT activated!!')

    # set hdf5 path according your hdf5 file location
    hdf5_file = '../data/hdf5/ift725_acdc.hdf5'

    # set highway path 
    highway_file = '../data/highway'

    # set projetSession path
    ProjetSession_file = '../data/projetSession'

    # Transform is used to normalize data among others
    acdc_base_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    acdc_train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    highway_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    projetSession_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

#################################################################################
#################################################################################
#################################################################################   
    if data_augment : 

        transform_train_cifar = transforms.Compose([
        transforms.RandomRotation(3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]   )

        transform_train_svhn = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomResizedCrop( 32,  scale=(0.8, 1.0), ratio=(
                0.75, 4/3)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        acdc_train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ToTensor()
        ])
    if args.dataset == 'cifar10':
        # Download the train and test set and apply transform on it
        if data_augment : 
                     train_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train_cifar)
                     test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
        else :              
            train_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=base_transform)
            test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=base_transform)
    elif args.dataset == 'svhn':
        # Download the train and test set and apply transform on it
        if data_augment : 
                     train_set = datasets.SVHN(root='../data', split='train', download=True, transform=transform_train_svhn)
                     test_set = datasets.SVHN(root='../data', split='test', download=True, transform=transform_test)
        else :              
            train_set = datasets.SVHN(root='../data', split='train', download=True, transform=base_transform)
            test_set = datasets.SVHN(root='../data', split='test', download=True, transform=base_transform)

    if args.optimizer == 'SGD':
        optimizer_factory = optimizer_setup(torch.optim.SGD, lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

    if args.model == 'CnnVanilla':
        model = CnnVanilla(num_classes=10)
    elif args.model == 'AlexNet':
        model = AlexNet(num_classes=10)
    elif args.model == 'VggNet':
        model = VggNet(num_classes=10)
    elif args.model == 'ResNet':
        model = ResNet(num_classes=10)
    elif args.model == 'IFT725Net':
        model = IFT725Net(num_classes=10)
    elif args.model == 'IFT725UNet':
        model = IFT725UNet(num_classes=4)
        args.dataset = 'acdc'

        train_set = HDF5Dataset(
            'train', hdf5_file, transform=acdc_base_transform)
        test_set = HDF5Dataset(
            'test', hdf5_file, transform=acdc_base_transform)
    elif args.model == 'UNet':
        if args.dataset == 'projetSession':
            model = UNet(num_classes=2, in_channels=3)
            train_set = ProjetHiver_DataSet('train', ProjetSession_file, transform=highway_transform)
            test_set = ProjetHiver_DataSet('test', ProjetSession_file, transform=highway_transform)
        elif args.dataset == 'highway':
            model = UNet(num_classes=4, in_channels=3)
            train_set = Highway_DataSet('train', highway_file, transform=highway_transform)
            test_set = Highway_DataSet('test', highway_file, transform=highway_transform)
        else :
            model = UNet(num_classes=4, in_channels=1)
            args.dataset = 'acdc'

            train_set = HDF5Dataset('train', hdf5_file, transform=acdc_base_transform)
            test_set = HDF5Dataset('test', hdf5_file, transform=acdc_base_transform)

    model_trainer = CNNTrainTestManager(model=model,
                                        trainset=train_set,
                                        testset=test_set,
                                        batch_size=batch_size,
                                        loss_fn=nn.CrossEntropyLoss(),
                                        optimizer_factory=optimizer_factory,
                                        validation=val_set,
                                        use_cuda=True)
    if args.predict:
        model.load_weights('UNet.pt')
        # model_trainer.evaluate_on_test_set()
        print("predicting the mask of a randomly selected image from test set")
        model_trainer.plot_image_mask_prediction()
    else:
        if isinstance(model,UNet) and continue_train :
            model.load_weights('UNet.pt')
            load_checkpoint(model,model_trainer,filename='UNet_state')
        print("Training {} on {} for {} epochs".format(model.__class__.__name__, args.dataset, args.num_epochs))
        model_trainer.train(num_epochs)
        model_trainer.evaluate_on_test_set()
        if isinstance(model, UNet):
            model.save()  # save the model's weights for prediction (see help for more details)
            state = {'epoch': model_trainer.epoch , 'state_dict': model.state_dict(),
                     'optimizer': model_trainer.optimizer.state_dict(), 'losslogger': model_trainer.metric_values, }
            torch.save(state, "UNet_state") 
            model_trainer.plot_image_mask_prediction()
        model_trainer.plot_metrics()


