# Q Network Model
# # Path: DQN_Mountain_Car\Q_Network_Model.py
# Reference: https://github.com/Bduz/intro_pytorch/blob/main/intro_rl/dqn/model.py
# Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Date: 2023/05/23
# Author: Doğacan Dolunay Acar
# ----------------------------------------------
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

     # Initialize parameters and build model.
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, fc3_units=256): # 12, 21, 0, 128, 128
         """Initialize parameters and build model.
         Params
         ======
             state_size (int): Dimension of each state. In this case, it is 2.
             action_size (int): Dimension of each action. In this case, it is 3.
             seed (int): Random seed.
             fc1_units (int): Number of nodes in first hidden layer. In this case, it is 64.
             fc2_units (int): Number of nodes in second hidden layer. In this case, it is 64.
         """
         # Initialize the parent class
         super(QNetwork, self).__init__()
         # Set the seed
         self.seed = torch.manual_seed(seed)
         # Create the first fully connected layer
         self.fc1 = nn.Linear(state_size, fc1_units)
         # Create the second fully connected layer
         self.fc2 = nn.Linear(fc1_units, fc2_units)
         # Create the third fully connected layer
         self.fc3 = nn.Linear(fc2_units, fc3_units)
         # Create the fourth fully connected layer
         self.fc4 = nn.Linear(fc3_units, action_size)

    # Build a network that maps state -> action values.
    def forward(self, state):
        """Build a network that maps state -> action values.
        Params
          ======
              state (array_like): Input state.
          """
        # Pass the state through the first fully connected layer
        x = F.relu(self.fc1(state))
        # Pass the state through the second fully connected layer
        x = F.relu(self.fc2(x))
        # Pass the state through the third fully connected layer
        x = F.relu(self.fc3(x))
        # Pass the state through the fourth fully connected layer
        return self.fc4(x)
