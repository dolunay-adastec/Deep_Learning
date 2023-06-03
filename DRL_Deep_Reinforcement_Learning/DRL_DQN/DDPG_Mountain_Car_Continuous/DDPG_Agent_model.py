# Deep Deterministic Policy Gradient (DDPG) Agent
# Path: DRL_Deep_Reinforcement_Learning\DRL_DQN\DDPG_Mountain_Car_Continuous\DDPG_Agent.py
# Reference: https://github.com/Bduz/intro_pytorch/blob/main/intro_rl/ddpg/ddpg_agent.py
# Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Date: 2023/06/03
# Author: Doğacan Dolunay Acar
# ----------------------------------------------
# Import libraries
import numpy as np
import random
import copy
from collections import namedtuple, deque

from Actor_Critic_NN_model import ActorNetwork, CriticNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
BUFFER_SIZE = int(1e5)  # replay buffer size 
BATCH_SIZE = 128        # minibatch size (128)
GAMMA = 0.99            # discount factor (0.99)
TAU = 1e-3              # for soft update of target parameters (1e-3)
LR_ACTOR = 1e-4         # learning rate of the actor (1e-4)
LR_CRITIC = 1e-3        # learning rate of the critic (1e-4)
WEIGHT_DECAY = 0        # L2 weight decay (regularization)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set the seed
random.seed(0) # set the seed to 0 for reproducibility purposes
np.random.seed(0) # set the seed to 0 for reproducibility purposes
torch.manual_seed(0) # set the seed to 0 for reproducibility purposes

class DDPG_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): Dimension of each state. In this case, it is 2.
            action_size (int): Dimension of each action. In this case, it is 1.
            random_seed (int): Random seed.
        """
        # Initialize the random seed
        self.seed = random.seed(random_seed)
        # Initialize the state size
        self.state_size = state_size
        # Initialize the action size
        self.action_size = action_size

        # Initialize the Actor Network
        self.actor_local = ActorNetwork(state_size, action_size, random_seed).to(device)
        # Initialize the Actor Network
        self.actor_target = ActorNetwork(state_size, action_size, random_seed).to(device)
        
        # Initialize the Critic Network
        self.critic_local = CriticNetwork(state_size, action_size, random_seed).to(device)
        # Initialize the Critic Network
        self.critic_target = CriticNetwork(state_size, action_size, random_seed).to(device)
        
        # Initialize the optimizer for the Actor Network
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        # Initialize the optimizer for the Critic Network
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Initialize the Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        # Initialize the time step
        self.t_step = 0

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn, if enough samples are available in memory
        # If the memory size is greater than the batch size
        if len(self.memory) > BATCH_SIZE:
            # Sample a random batch from the replay memory
            experiences = self.memory.sample()
            # Learn from the sampled batch
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        # Convert the state to a torch tensor
        state = torch.from_numpy(state).float().to(device)
        # Set the actor network to evaluation mode
        self.actor_local.eval()
        # Disable the gradient calculation
        with torch.no_grad():
            # Get the action from the actor network
            action = self.actor_local(state).cpu().data.numpy()
        # Set the actor network to training mode
        self.actor_local.train()
        # Add noise to the action
        if add_noise:
            action += self.noise.sample()
        # Return the action
        return np.clip(action, -1, 1) # All the actions are clipped between -1 and 1. "Clip" is a numpy function that limits the values in an array between a minimum and maximum value.
    
    def reset(self):
        """Reset the noise"""
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): Tuple of (s, a, r, s', done) tuples.
            gamma (float): Discount factor.
        """        
        # Get the states, actions, rewards, next_states, and dones from the experiences
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Formula: Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        # Update Rule: critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Logic is to minimize the loss between the Q targets and the Q expected
        # Critic network is responsible for evaluating the optimal actions and the actor network is responsible for generating the optimal actions
        # ______________________________________________________________________ #
        # Get the next actions from the target actor network
        next_actions = self.actor_target(next_states)
        # Get the next Q values from the target critic network
        Q_targets_next = self.critic_target(next_states, next_actions) # Q_targets_next is the Q values from the target critic network
        # Compute the Q targets for the current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) # if done then Q_targets = reward else Q_targets = reward + gamma * Q_targets_next
        # Compute the expected Q values from the local critic network
        Q_expected = self.critic_local(states, actions) # Q_expected is the Q values from the local critic network
        # Compute the loss between the Q targets and the Q expected
        critic_loss = F.mse_loss(Q_expected, Q_targets) # The loss function is the mean squared error loss function
        # Minimize the loss
        self.critic_optimizer.zero_grad() # Clear the gradients
        # Backpropagate the loss
        critic_loss.backward()
        # Clip the gradient to a maximum of 1
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # Gradient clipping is a technique to prevent exploding gradients in very deep networks
        # Update the weights of the local critic network
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Formula: actor_loss = -critic_local(states, actor_local(states)).mean()
        # Update Rule: actor_loss.backward()
        # Logic is to maximize the loss between the Q values and the actions, which is the same as minimizing the loss between the Q values and the actions with a negative sign
        # ______________________________________________________________________ #
        # Compute the loss between the Q values and the actions
        actor_loss = -self.critic_local(states, self.actor_local(states)).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad() # Clear the gradients
        # Backpropagate the loss
        actor_loss.backward() 
        # Update the weights of the local actor network
        self.actor_optimizer.step() 

        # ----------------------- update target networks ----------------------- #
        # Update the target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters. θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): Weights will be copied from
            target_model (PyTorch model): Weights will be copied to
            tau (float): Interpolation parameter 
        """
        # Interpolation parameter (tau) is used to control the amount of interpolation between the local and target networks. 
        # Update the target model parameters
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()): # zip() function returns a list of tuples, where the i-th tuple contains the i-th element from each of the argument sequences or iterables
            # Update the target model parameters
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data) # θ_target = τ*θ_local + (1 - τ)*θ_target

# Ornstein-Uhlenbeck process
class OUNoise:
    """Ornstein-Uhlenbeck process.
    Source: 1. https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
            2. https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    """
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        # Initialize the parameters
        self.mu = mu * np.ones(size)    # mu is the mean
        self.theta = theta              # theta is the rate of mean reversion. It is the rate at which the process pulls towards the mean
        self.sigma = sigma              # sigma is the volatility parameter. It is the standard deviation of the process
        self.seed = random.seed(seed)   # seed is the random seed
        # Initialize the noise process
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        # Reset the internal state
        self.state = copy.copy(self.mu) # state is the internal state of the process

    def sample(self):
        """Update internal state and return it as a noise sample."""
        # Update the internal state
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        # dx: the change in the noise variable at each time step
        # self.theta:  is the parameter that determines how fast the noise variable reverts to the mean (self.mu).
        # (self.mu - x): represents the difference between the mean and the current state of the noise variable.
        # self.sigma: is the standard deviation of the process. It determines the amount of variation in the noise variable.
        # np.array([random.random() for i in range(len(x))]): is the random component of the process. It is a random variable that is drawn from a normal distribution with mean 0 and standard deviation 1.
        # Generates an array of random numbers drawn from a uniform distribution between 0 and 1. The length of this array matches the length of the input x, which is typically the number of dimensions in the action space.
        # Return the noise sample
        # self.sigma * np.array([random.random() for i in range(len(x))]) scales the random array by the standard deviation (self.sigma).
        # 1 - The difference between the mean (self.mu) and the current state (x), scaled by the parameter self.theta. This component represents the tendency of the noise variable to revert towards the mean.
        # 2 - A random array scaled by the standard deviation (self.sigma). This component introduces random fluctuations to the noise variable.
        # By adjusting the values of self.theta and self.sigma, you can control the properties of the generated noise, such as its speed of reversion to the mean and its overall magnitude.
        self.state = x + dx
        return self.state
    
# Define the replay buffer
class ReplayBuffer:
        """Fixed-size buffer to store experience tuples."""

        def __init__(self, action_size, buffer_size, batch_size, seed):
            """Initialize a ReplayBuffer object.
            Params
            ======
                action_size (int): dimension of each action
                buffer_size (int): maximum size of buffer
                batch_size (int): size of each training batch
                seed (int): random seed
            """
            # Initialize the class variables
            self.action_size = action_size
            # Create the replay buffer
            self.memory = deque(maxlen=buffer_size) # deque is a double-ended queue. It can be used to add or remove elements from both ends.
            # Create the batch size
            self.batch_size = batch_size
            # Initialize the random seed
            self.seed = random.seed(seed)
            # Initialize the experience
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        def add(self, state, action, reward, next_state, done):
            """Add a new experience to memory."""
            # Add a new experience to memory
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

        def sample(self):
            """Randomly sample a batch of experiences from memory."""
            # Randomly sample a batch of experiences from memory
            experiences = random.sample(self.memory, k=self.batch_size)

            # Convert the batch of experiences into a batch of Torch tensors
            # torch.from_numpy() converts the NumPy array into a Torch tensor. It also copies the data into GPU memory, if a GPU is available.
            # np.vstack() function stacks arrays in sequence vertically (row wise). The arrays must have the same shape along all but the first axis.
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device) # The output: torch.Size([64, 37])
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

            return (states, actions, rewards, next_states, dones)
    
        def __len__(self):
            """Return the current size of internal memory."""
            # Return the current size of internal memory
            return len(self.memory)