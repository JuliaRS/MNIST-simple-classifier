import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x) :
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return 'MLP'


class CNN(nn.Module):
    def __init__(self, num_clases):
        super(CNN, self).__init__()
        self.layer1 = nn.Conv2d(1, 20, 5, 1)
        self.layer2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_clases)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.layer2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return 'CNN'
