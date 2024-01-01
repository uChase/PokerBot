import random
import torch
from collections import deque
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device) 

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

#implement attention maybe? maybe shape state data into convolutions?

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.attention = nn.Linear(256, 256)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        attention_weights = F.softmax(self.attention(x), dim=-1)
        x = x * attention_weights
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class Trainer:
    def __init__(self, model, replay_buffer, learning_rate, gamma):
        self.model = model
        self.replay_buffer = replay_buffer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = gamma

    def train_model(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert states, actions, rewards, next_states, and dones to tensors if they aren't already
        states = torch.stack(states) if not isinstance(states[0], torch.Tensor) else states
        actions = torch.tensor(actions, dtype=torch.int64) if not isinstance(actions, torch.Tensor) else actions
        rewards = torch.tensor(rewards, dtype=torch.float32) if not isinstance(rewards, torch.Tensor) else rewards
        next_states = torch.stack(next_states) if not isinstance(next_states[0], torch.Tensor) else next_states
        dones = torch.tensor(dones, dtype=torch.float32) if not isinstance(dones, torch.Tensor) else dones

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        q_values = self.model(states)
        q_values = q_values.squeeze(1) if q_values.dim() == 3 else q_values
        actions = actions.unsqueeze(1) if actions.dim() == 1 else actions
        q_values = q_values.gather(1, actions).squeeze(1)

        next_q_values = self.model(next_states)
        next_q_values = next_q_values.squeeze()
        next_q_values = torch.max(next_q_values, dim=1)[0]        

        rewards = rewards.squeeze() if rewards.dim() > 1 else rewards
        dones = dones.squeeze() if dones.dim() > 1 else dones

        # Calculate expected Q-values
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute the loss
        loss = self.criterion(q_values, expected_q_values.detach())

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



class PokerAgent:
    def __init__(self, input_dim, output_dim, capacity, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay):
        self.replay_buffer = ReplayBuffer(capacity)
        self.model = DQN(input_dim, output_dim).to(device)
        self.trainer = Trainer(self.model, self.replay_buffer, learning_rate, gamma)
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def decay(self):
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            return self.epsilon

    def play(self, state_tensor):
        state_tensor = state_tensor.to(device)
        if random.random() < self.epsilon:
            action = random.randint(0, 2)
        else:
            with torch.no_grad():
                q_values = self.model(state_tensor)
                action = torch.argmax(q_values).item()
        return action

    def update_replay_buffer(self, state, action, next_state, done, end_of_game_reward=None):
        # If the game is over, use the end-of-game reward

        if done and end_of_game_reward is not None:
            reward = end_of_game_reward
        else:
            reward = 0  # Use a default or placeholder reward for non-final states

        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self, batch_size):
        if len(self.replay_buffer) > batch_size:
            self.trainer.train_model(batch_size)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

#NOTE, 215 input to DQN