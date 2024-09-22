from gym import spaces
import numpy as np
import torch
from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer

device = "cuda"


class DQNAgent:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
    ):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_buffer = replay_buffer
        self.use_double_dqn = use_double_dqn
        self.num_actions = action_space.n
        self.policy_network = DQN(observation_space, action_space).to(self.device)
        self.target_network = DQN(observation_space, action_space).to(self.device)
        self.update_target_network()
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(lr=lr, params=self.policy_network.parameters())
        self.loss_fn = torch.nn.MSELoss()

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        # TODO
        #   Optimise the TD-error over a single minibatch of transitions
        #   Sample the minibatch from the replay-memory
        #   using done (as a float) instead of if statement
        #   return loss
        if len(self.replay_buffer) < self.batch_size:
            return 0
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.from_numpy(states).to(self.device)  # Shape: (batch_size, C, H, W)
        actions = torch.from_numpy(actions).long().to(self.device)  # Shape: (batch_size,)
        rewards = torch.from_numpy(rewards).float().to(self.device)  # Shape: (batch_size,)
        next_states = torch.from_numpy(next_states).to(self.device)  # Shape: (batch_size, C, H, W)
        dones = torch.from_numpy(dones).float().to(self.device)

        q_values = self.policy_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values_target = self.target_network(next_states)
        next_q_values, _ = next_q_values_target.max(1)

        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        # update target_network parameters with policy_network parameters
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def act(self, state: np.ndarray, eps: float = 0.0):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        # Select action greedily from the Q-network given the state
        if np.random.rand() < eps:
            # Exploration: choose a random action
            return np.random.randint(self.num_actions)
        else:
            # Exploitation: choose the best action according to the policy network
            state = torch.from_numpy(state).unsqueeze(0).to(self.device)  # Add batch dimension

            with torch.no_grad():
                q_values = self.policy_network(state)
                action = q_values.max(1)[1].item()  # Get the index of the max Q-value

            return action
