import torch.nn as nn
import torch.nn.functional as F


# Define an example network
class Net(nn.Module):
    def __init__(self, num_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = F.sigmoid(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        out = self.fc3(out)
        out = F.sigmoid(out)
        out = self.fc4(out)
        return out

