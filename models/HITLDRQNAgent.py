
import math
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from models.BaseAgent import BaseAgent

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def reset(self):
        self.memory = deque([], maxlen=self.capacity)

# NOTE: Resetting hidden state every episode begins results in higher rewards as compared to not restarting hidden states
# TODO: Bootstrapped continuous updates - train using sampling at any point -> yes hidden state is passed from step to step, but batch_input used for optimize_model() is non sequential
class DRNN_QN(nn.Module):
    """Referencing https://github.com/marload/DeepRL-TensorFlow2/blob/master/DRQN/DRQN_Discrete.py"""
    def __init__(self, n_observations, n_actions):
        super(DRNN_QN, self).__init__()
        self.n_observation = n_observations
        self.n_actions = n_actions
        self.hidden_state = None

        self.rnn = nn.RNN(self.n_observation, 32, nonlinearity='tanh')
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, self.n_actions)


    def forward(self, x):
        x, h_t = self.rnn(x, self.hidden_state) # observation -> 32 + hidden state
        self.hidden_state = h_t.detach()
        x = F.relu(self.fc1(x)) # 32 -> 16 -> relu 16
        x = self.fc2(x) # 16 -> actions
        return x
    
    def reset_hidden_state(self):
        self.hidden_state = None

class HITLDRQNAgent(BaseAgent):
    def __init__(self, n_observations, n_actions, device, memory_size=10000, batch_size=64, gamma=0.99, 
                 eps_start=0.9, eps_end=0.05, eps_decay=1000, tau=0.005, lr=1e-4):
        """
        Initialize the base agent with common parameters.

        Parameters:
        - n_observations: Number of observations in the state space.
        - n_actions: Number of actions in the action space.
        - device: Device to run the computations on (CPU or GPU).
        - memory_size: Size of the replay memory.
        - batch_size: Size of the batches sampled from the replay memory.
        - gamma: Discount factor for future rewards.
        - eps_start: Initial value of epsilon for epsilon-greedy policy.
        - eps_end: Final value of epsilon for epsilon-greedy policy.
        - eps_decay: Decay rate of epsilon.
        - tau: Update rate of the target network with reference to policy network \theta <- \tau \theta_{policy} + (1-\tau)\theta_{target}
        - lr: Learning rate for the optimizer.
        """
        self.is_exploring = True
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr

        self.device = device
        self.memory = ReplayMemory(memory_size)

        self.policy_net = DRNN_QN(n_observations, n_actions).to(device)
        self.target_net = DRNN_QN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.steps_done = 0 # number od steps done using the same replay memory

    def select_action(self, state):
        """Select an action based on the current state."""        
        if self.is_exploring:
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay) # calculated decayed epsilon threshold to compare with sample value
            self.steps_done += 1
            if sample > eps_threshold: # e-greedy threshold
                with torch.no_grad(): # choose action of highest probability greedily
                    return self.policy_net(state).max(1).indices.view(1, 1)
            else: # random choice
                return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        else:
            return self.policy_net(state).max(1).indices.view(1, 1)

    def optimize_model(self):
        """Optimize the model by sampling from the replay memory and performing a gradient descent step."""
        if len(self.memory) < self.batch_size:  # Only update once steps saved > self.batch_size
            return

        # Sample List[Transition objects] of len batch_size
        transitions = self.memory.sample(self.batch_size)

        # Group all (states, action, next_state, rewards) transition objects into tensors
        # (1 tensor for each transition object attribute of len <batch_size> examples * dimen of attribute)
        batch = Transition(*zip(*transitions))

        # Mask to identify non-final states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), 
            device=self.device, 
            dtype=torch.bool
        )
        
        # Concatenate all non-final next states
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Separate batch (a dict of tensors) -> to individual tensors
        # shape = batch_size * state_dimen
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # policy_net_out: tensor - size batch_size * action_space => Q value of each action (+ current state) pair
        policy_net_out = self.policy_net(state_batch)  # Current state -- policy --> ??
        
        # Take the Q value for the action taken at current state
        # Calculate the Q(s,a) values of the (action actually taken + state s)
        state_action_values = policy_net_out.gather(1, action_batch)

        # Initialize next_state_values tensor
        next_state_values = torch.zeros(self.batch_size, device=self.device)  # Why do we need next state values to be initialized?

        # We do not need the gradient because we are not backpropagating, just inference
        with torch.no_grad(): 
            # Get the next state values using the bootstrapped method (estimated using the target network)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        # Calculating bootstrapped future expected rewards r+\gamma*max_aQ(s',a')
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Calculate loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Perform backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Update target network parameters with more stable update using tau
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        
        # Load updated parameters into target_net
        self.target_net.load_state_dict(target_net_state_dict)


    def reset_replay_memory(self):
        """Reset replay memory used to optimize the model, when you want to remove all previous experiences"""
        self.memory.reset()
        self.steps_done = 0
    
    def reset_hidden_state(self):
        self.target_net.reset_hidden_state()
        self.policy_net.reset_hidden_state()

    def memorize(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def save_model(self, filepath):
        """Save the model parameters to a file."""
        torch.save(self.policy_net.state_dict(), filepath)

    def load_model(self, filepath):
        """Load the model parameters from a file."""
        state_dict = torch.load(filepath)
        self.policy_net.load_state_dict(state_dict)