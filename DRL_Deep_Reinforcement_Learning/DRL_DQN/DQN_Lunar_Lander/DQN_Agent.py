# Referenced https://github.com/Bduz/intro_pytorch/blob/main/intro_rl/dqn/dqn_agent.py
# Referenced https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# DQN Agent class
# Date: 2021/05/20
# Author: Doğacan Dolunay Acar
# ----------------------------------------------
# Import libraries
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from Q_Network_Model import QNetwork

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set the hyperparameters
BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 64         # Minibatch size
GAMMA = 0.99            # Discount factor
TAU = 1e-3              # For soft update of target parameters
LR = 5e-4               # Learning rate
UPDATE_EVERY = 4        # How often to update the network

# Set the seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Define the DQN_Agent class 
class DQN_Agent():
    """Interacts with and learns from the environment."""

    # Initialize the DQN_Agent class
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): Dimension of each state. In this case, it is 37.
            action_size (int): Dimension of each action. In this case, it is 4.
            seed (int): Random seed.
        """
        # Set the seed
        self.seed = random.seed(seed)
        # Set the state size
        self.state_size = state_size
        # Set the action size
        self.action_size = action_size
        # Create the Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        # Create the target Q-Network
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        # Set the optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Set the replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize the time step
        self.t_step = 0

    # Select an action
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): Current state. In this case, it is 37.
            eps (float): Epsilon, for epsilon-greedy action selection. In this case, it is 0.
        """
        # Get the state as a tensor
        state = torch.from_numpy(state).unsqueeze(0).to(device)
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Set the network to evaluation mode
        self.qnetwork_local.eval()
        # Get the action values
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        # Set the network to training mode
        self.qnetwork_local.train()
        # Epsilon-greedy action selection
        if random.random() > eps:
            # Return the action with the highest Q-value
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Return a random action
            return random.choice(np.arange(self.action_size))

    # Save the experience in replay memory
    def step(self, state, action, reward, next_state, done):
        """Save the experience in replay memory, and use random sample from buffer to learn.
        Params
        ======
            state (array_like): Current state. In this case, it is 37.
            action (array_like): Action taken. In this case, it is 4.
            reward (array_like): Reward received. In this case, it is 1.
            next_state (array_like): Next state. In this case, it is 37.
            done (array_like): Whether the episode is done or not. In this case, it is 1.
        """
        # Save the experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # If it is time to learn
        if self.t_step == 0:
            # If enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                # Sample a random batch of experiences from memory
                experiences = self.memory.sample()
                # Learn from the experiences
                self.learn(experiences, GAMMA)

    # Learn from the experiences
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (array_like): Tuple of (s, a, r, s', done) tuples. In this case, it is 64.
            gamma (float): Discount factor. In this case, it is 0.99.
        """
        # Get the states, actions, rewards, next states, and dones from the experiences
        states, actions, rewards, next_states, dones = experiences
        # Get the Q-values for the next states
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute the Q-targets for the current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get the expected Q-values from the local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # Compute the loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update the target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    # Update the target network
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): Weights will be copied from. In this case, it is the local model.
            target_model (PyTorch model): Weights will be copied to. In this case, it is the target model.
            tau (float): Interpolation parameter. In this case, it is 0.001.
        """
        # For each parameter in the local model
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # Update the target parameter
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# Define the ReplayBuffer class
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    # Initialize the ReplayBuffer class
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): Dimension of each action. In this case, it is 4.
            buffer_size (int): Maximum size of buffer. In this case, it is 100000.
            batch_size (int): Size of each training batch. In this case, it is 64.
            seed (int): Random seed.
        """
        # Set the seed
        self.seed = random.seed(seed)
        # Set the batch size
        self.batch_size = batch_size
        # Create the memory
        self.memory = deque(maxlen=buffer_size)
        # Create the experience tuple
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    # Add a new experience to memory.
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.
        Params
        ======
            state (array_like): Current state. In this case, it is 37.
            action (array_like): Action taken. In this case, it is 4.
            reward (array_like): Reward received. In this case, it is 1.
            next_state (array_like): Next state. In this case, it is 37.
            done (array_like): Whether the episode is done or not. In this case, it is 1.
        """
        # Create the experience tuple
        e = self.experience(state, action, reward, next_state, done)
        # Append the experience tuple to memory
        self.memory.append(e)

    # Sample a batch of experiences from memory.
    def sample(self):
        """Sample a batch of experiences from memory."""
        # Sample experiences from memory
        experiences = random.sample(self.memory, k=self.batch_size)
        # Create the states
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # Create the actions
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        # Create the rewards
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # Create the next states
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        # Create the dones
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        # Return the states, actions, rewards, next states, and dones
        return (states, actions, rewards, next_states, dones)
    
    # Return the size of the memory
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    


