# Actor - Critic Network Model
# Path: DRL_Deep_Reinforcement_Learning\DRL_DQN\DDPG_Mountain_Car_Continuous\Actor_Critic_NN_model.py
# Reference: https://github.com/Bduz/intro_pytorch/blob/main/intro_rl/ddpg/model.py
# Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Date: 2023/06/03
# Author: Doğacan Dolunay Acar
# ----------------------------------------------
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# hidden_init
def hidden_init(layer):
    """Initialize the weights."""
    # Get the fan_in of the layer (number of input units)
    fan_in = layer.weight.data.size()[0] # 2 (state_size) for the first layer
    # Calculate the limit for the uniform distribution
    lim = 1. / np.sqrt(fan_in) #  fan_in = 2 (state_size) 
    # Return the limit for the uniform distribution
    return (-lim, lim)

class ActorNetwork(nn.Module):
    """Actor (Policy) Model."""

     # Initialize parameters and build model.
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300): # 2, 3, 0, 64, 64
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
         super(ActorNetwork, self).__init__()
         # Set the seed
         self.seed = torch.manual_seed(seed)
         # Create the first fully connected layer
         self.fc1 = nn.Linear(state_size, fc1_units)
         # Create the second fully connected layer
         self.fc2 = nn.Linear(fc1_units, fc2_units)
         # Create the third fully connected layer
         self.fc3 = nn.Linear(fc2_units, action_size)
         # reset_parameters
         self.reset_parameters()

    # reset_parameters
    def reset_parameters(self):
        # Initialize the weights
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        # Initialize the weights
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # Initialize the weights
        self.fc3.weight.data.uniform_(-3e-3, 3e-3) # Why?


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
        return F.tanh(self.fc3(x))

# CriticNetwork
class CriticNetwork(nn.Module):
    """Critic (Value) Model."""

    # Initialize parameters and build model.
    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300): # 2, 3, 0, 64, 64
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state. In this case, it is 2.
            action_size (int): Dimension of each action. In this case, it is 3.
            seed (int): Random seed.
            fcs1_units (int): Number of nodes in the first hidden layer. In this case, it is 64.
            fc2_units (int): Number of nodes in the second hidden layer. In this case, it is 64.
        """
        # Initialize the parent class
        super(CriticNetwork, self).__init__()
        # Set the seed
        self.seed = torch.manual_seed(seed)
        # Create the first fully connected layer
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        # Create the second fully connected layer
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        # Create the third fully connected layer
        self.fc3 = nn.Linear(fc2_units, 1)
        # reset_parameters
        self.reset_parameters()

    # reset_parameters
    def reset_parameters(self):
        # Initialize the weights
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        # Initialize the weights
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # Initialize the weights
        self.fc3.weight.data.uniform_(-3e-3, 3e-3) # Why?

    # Build a network that maps state -> action values.
    def forward(self, state, action):
        """Build a network that maps state, action pairs -> Q-values.
        Params
          ======
              state (array_like): Input state.
              action (array_like): Input action.
          """
        # Forward formula: Q(s,a) = fcs1(s) + fc2(s,a)
        # Pass the state through the first fully connected layer
        xs = F.relu(self.fcs1(state))
        # Concatenate the state and action
        x = torch.cat((xs, action), dim=1) # dim=1 means concatenate along the columns (horizontally). Example output: tensor([[1, 2, 3, 4]]) tensor([[5, 6, 7, 8]]) -> tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        # Pass the state through the second fully connected layer
        x = F.relu(self.fc2(x))
        # Pass the state through the third fully connected layer
        return self.fc3(x)