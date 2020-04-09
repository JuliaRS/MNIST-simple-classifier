import torchvision
from torch.utils.data import DataLoader
import logging

def get_data():
    data_train = torchvision.datasets.MNIST('./data', download=True, train=True, transform=torchvision.transforms.ToTensor())
    data_test = torchvision.datasets.MNIST('./data', train=False, transform=torchvision.transforms.ToTensor())
    train_loader = DataLoader(data_train, num_workers=8, batch_size=32)
    test_loader = DataLoader(data_test, num_workers=4, batch_size=16)
    logging.debug("DATA loaded successful")
    return train_loader, test_loader
