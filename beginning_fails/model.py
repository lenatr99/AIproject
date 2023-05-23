import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, fc3_units = 64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer
            
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # self.fc1 = nn.Linear(state_size, fc1_units)
        # self.bn1 = nn.BatchNorm1d(fc1_units)
        # self.act1 = nn.ReLU()
        # self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.bn2 = nn.BatchNorm1d(fc2_units)
        # self.act2 = nn.ReLU()
        # self.fc3 = nn.Linear(fc2_units, fc3_units)
        # self.bn3 = nn.BatchNorm1d(fc3_units)
        # self.act3 = nn.ReLU()
        # self.fc4 = nn.Linear(fc3_units, action_size)

        self.conv_depth_1 = 256
        self.conv_depth_2 = 256
        self.conv_kernel_sizes = [(1,2), (2,1)]
        self.hidden_units = 256
        self.output_units = 4  # action size
        self.input_shape = (16, 4, 4)  # (channel, height, width) - state size

        # Define layers
        # first convolutional layer
        self.conv1 = nn.ModuleList([nn.Conv2d(self.input_shape[0], self.conv_depth_1, kernel_size) for kernel_size in self.conv_kernel_sizes])
        self.bn1 = nn.BatchNorm2d(self.conv_depth_1)
        
        # second convolutional layer
        self.conv2 = nn.ModuleList([nn.Conv2d(self.conv_depth_1, self.conv_depth_2, kernel_size) for kernel_size in self.conv_kernel_sizes])
        self.bn2 = nn.BatchNorm2d(self.conv_depth_2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_depth_2 * 2 * 2, self.hidden_units)  # assuming that the output of the conv2d layers will be 2*2, change if necessary
        self.fc2 = nn.Linear(self.hidden_units, self.output_units)



    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        # x = self.fc1(state)
        # x = self.act1(x)
        # x = self.bn1(x)
        # x = self.fc2(x)
        # x = self.act2(x)
        # x = self.bn2(x)
        # x = self.fc3(x)
        # x = self.act3(x)
        # return self.fc4(x)
    
        x = F.relu(self.bn1(sum([conv(state) for conv in self.conv1])))
        x = F.relu(self.bn2(sum([conv(x) for conv in self.conv2])))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))

        return self.fc2(x)
