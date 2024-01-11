import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A3CBuffer:
    def __init__(self):
        self.buffer = deque()  # Unlimited capacity

    def push(
        self,
        log_prob,
        reward,
        value,
    ):
        experience = (log_prob, reward, value)
        self.buffer.append(experience)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    @property
    def experiences(self):
        return list(self.buffer)


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU())
        self.actorLSTM = nn.LSTM(512, 256, num_layers, batch_first=True)
        self.criticLSTM = nn.LSTM(512, 256, num_layers, batch_first=True)
        self.actorlayer = nn.Linear(256, output_dim)
        self.criticlayer = nn.Linear(256, 1)
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=0.25)
        self.actorAttention = nn.Linear(512, 512)
        self.criticAttention = nn.Linear(512, 512)

    def _init_or_adjust_state(self, state, batch_size):
        if state is None:
            return torch.zeros(self.num_layers, batch_size, 256).to(device)
        else:
            return state.view(self.num_layers, batch_size, -1)

    def forward(self, x, ah0=None, ac0=None, ch0=None, cc0=None):
        x = self.shared_layers(x)
        x = self.dropout(x)
        actorAttention_weights = torch.softmax(self.actorAttention(x), dim=-1)
        criticAttention_weights = torch.softmax(self.criticAttention(x), dim=-1)

        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension: [seq_len, features] -> [1, seq_len, features]

        # Initialize hidden states correctly based on batch size
        batch_size = x.size(0)
        
        if ah0 is None:
            ah0 = torch.zeros(self.num_layers, batch_size, 256).to(device)
        if ac0 is None:
            ac0 = torch.zeros(self.num_layers, batch_size, 256).to(device)
        if ch0 is None:
            ch0 = torch.zeros(self.num_layers, batch_size, 256).to(device)
        if cc0 is None:
            cc0 = torch.zeros(self.num_layers, batch_size, 256).to(device)

        actorX = x * actorAttention_weights
        criticX = x * criticAttention_weights

        actorLSTMO, (actorLSTMh, actorLSTMc) = self.actorLSTM(actorX, (ah0, ac0))
        criticLSTMO, (criticLSTMh, criticLSTMc) = self.criticLSTM(criticX, (ch0, cc0))
        policy = torch.softmax(self.actorlayer(actorLSTMO[:, -1, :]), dim=-1)
        value = self.criticlayer(criticLSTMO[:, -1, :])

        return policy, value, (actorLSTMh, actorLSTMc), (criticLSTMh, criticLSTMc)


class Worker:
    def __init__(
        self,
        global_model,
        gamma,
        input_dim,
        output_dim,
        num_layers,
        optimizer=None,
    ):
        self.local_model = ActorCritic(input_dim, output_dim, num_layers).to(device)
        self.local_model.load_state_dict(global_model.state_dict())
        self.buffer = A3CBuffer()
        self.global_model = global_model
        self.l1_loss = nn.SmoothL1Loss()
        self.gamma = gamma
        self.optimizer = (
            optimizer if optimizer else optim.Adam(global_model.parameters(), lr=0.001)
        )

    def choose_action(self, state, ah0=None, ac0=None, ch0=None, cc0=None):
            policy, value, (actorLSTMh, actorLSTMc), (criticLSTMh, criticLSTMc) = self.local_model(state, ah0, ac0, ch0, cc0)
            action = torch.multinomial(policy, 1).item()
            return action, policy, (actorLSTMh, actorLSTMc), (criticLSTMh, criticLSTMc), value

    def update_buffer(self, log_prob, reward, value):
        self.buffer.push(log_prob, reward, value)

    def compute_loss(self, log_probs, values, rewards):
        returns = []
        R = 0
        # loops backwards
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device)

        values = torch.tensor(values).to(device)  # Convert values to a tensor
        log_probs = torch.stack(log_probs)  # Convert log_probs to a tensor of tensors

        advantage = returns - values
        log_probs = log_probs.squeeze(1)
        advantage_expanded = advantage.unsqueeze(-1).expand(-1, 3)
        policy_loss = -log_probs * advantage_expanded.detach()
        value_loss = self.l1_loss(torch.cat((values,), dim=0), returns)

        return policy_loss.sum() + value_loss.sum()

    def update_global(self, values, log_probs, rewards):
        # Compute loss
        loss = self.compute_loss(log_probs, values, rewards)

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        for local_param, global_param in zip(
            self.local_model.parameters(), self.global_model.parameters()
        ):
            global_param._grad = local_param.grad
        self.optimizer.step()

        # Update local model
        self.local_model.load_state_dict(self.global_model.state_dict())

    def train(self):
        log_probs, values, rewards = [], [], []
        for experience in self.buffer.experiences:
            log_prob, reward, value = experience

            # SINCE WE TECHNICALLY DID NOT LOG IT BEFORE
            log_prob = torch.log(torch.tensor(log_prob, dtype=torch.float).to(device).clone().detach().requires_grad_(True))
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

        self.update_global(values, log_probs, rewards)

        self.buffer.clear()


class GlobalModel:
    def __init__(self, input_dim, output_dim, num_layers):
        self.model = ActorCritic(input_dim, output_dim, num_layers).to(device)

    def get_model(self):
        return self.model

    def choose_action(self, state, ah0=None, ac0=None, ch0=None, cc0=None):
        (
            policy,
            _,
            (actorLSTMh, actorLSTMc),
            (criticLSTMh, criticLSTMc),
        ) = self.model(state, ah0, ac0, ch0, cc0)
        action = torch.multinomial(policy, 1).item()
        return action, policy, (actorLSTMh, actorLSTMc), (criticLSTMh, criticLSTMc)

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
