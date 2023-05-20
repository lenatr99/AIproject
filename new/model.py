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

        self.input = 4
        self.conv_depth_1 = 128
        self.conv_depth_2 = 128
        self.conv_kernel_sizes = [(1,2), (2,1)]
        self.hidden_units = 256
        self.output_units = 4  # action size
        self.input_shape = (16, 4, 4)  # (channel, height, width) - state size

        # Define layers
        # first convolutional layer
        self.conv11 = nn.Conv2d(self.input_shape[0],self.conv_depth_1, self.conv_kernel_sizes[0])
        self.conv12 = nn.Conv2d(self.input_shape[0],self.conv_depth_1, self.conv_kernel_sizes[1])
        self.bn1 = nn.BatchNorm2d(self.conv_depth_1)
        
        # second convolutional layer
        self.conv21 = nn.Conv2d(self.conv_depth_1, self.conv_depth_2, self.conv_kernel_sizes[0])
        self.conv22 = nn.Conv2d(self.conv_depth_1, self.conv_depth_2, self.conv_kernel_sizes[1])
        self.bn2 = nn.BatchNorm2d(self.conv_depth_2)
        

        self.expand_size = 2* 4* self.conv_depth_2 * 2  + 3*3*self.conv_depth_2*2+ 4*3*self.conv_depth_1*2
        # Fully connected layers
        self.fc1 = nn.Linear(self.expand_size, self.hidden_units)  # assuming that the output of the conv2d layers will be 2*2, change if necessary
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
        state = state.permute(0, 3, 1, 2)
        x1 = F.relu(self.bn1(self.conv11(state)))
        x2 = F.relu(self.bn1(self.conv12(state)))
        x1 = F.relu(self.bn1(x1))
        x2 = x = F.relu(self.bn1(x2))

        x11 = F.relu(self.bn2(self.conv21(x1)))
        x12 = F.relu(self.bn2(self.conv21(x2)))
        x21 = F.relu(self.bn2(self.conv22(x1)))
        x22 = F.relu(self.bn2(self.conv22(x2)))

        x1 = torch.flatten(x1, start_dim=1)
        x2 = torch.flatten(x2, start_dim=1)

        x11 = torch.flatten(x11, start_dim=1)
        x12 = torch.flatten(x12, start_dim=1)
        x21 = torch.flatten(x21, start_dim=1)
        x22 = torch.flatten(x22, start_dim=1)

        x = torch.cat((x1, x2, x11, x12, x21, x22), dim=1)
        x = F.relu(self.fc1(x))

        return self.fc2(x)
