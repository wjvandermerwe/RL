from gym import spaces
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert (
            type(observation_space) == spaces.Box
        ), "observation_space must be of type Box"
        assert (
            len(observation_space.shape) == 3
        ), "observation space must have the form channels x width x height"
        assert (
            type(action_space) == spaces.Discrete
        ), "action_space must be of type Discrete"

        # TODO Implement DQN Network

        self.num_actions = action_space.n

        # Input channels (number of stacked frames)
        c, h, w = observation_space.shape

        # Define the network layers as per the Nature DQN paper
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Calculate the size of the feature map after the convolutional layers
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64  # 64 is the number of filters in conv3

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, self.num_actions)

    def forward(self, x):
        x = x / 255.0  # Scale images to [0, 1] range

        # Pass through convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))  # Output shape: (batch_size, 32, convh1, convw1)
        x = F.relu(self.conv2(x))  # Output shape: (batch_size, 64, convh2, convw2)
        x = F.relu(self.conv3(x))  # Output shape: (batch_size, 64, convh3, convw3)

        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
