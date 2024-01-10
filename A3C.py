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

    def forward(self, x, ah0=None, ac0=None, ch0=None, cc0=None):
        x = self.shared_layers(x)
        if ah0 is None or ac0 is None:
            ah0 = torch.zeros(self.num_layers, x.size(0), 256).to(device)
            ac0 = torch.zeros(self.num_layers, x.size(0), 256).to(device)

        if ch0 is None or cc0 is None:
            ch0 = torch.zeros(self.num_layers, x.size(0), 256).to(device)
            cc0 = torch.zeros(self.num_layers, x.size(0), 256).to(device)
        actorLSTMO, (actorLSTMh, actorLSTMc) = self.actorLSTM(x, (ah0, ac0))
        criticLSTMO, (criticLSTMh, criticLSTMc) = self.criticLSTM(x, (ch0, cc0))
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
        (
            policy,
            value,
            (actorLSTMh, actorLSTMc),
            (criticLSTMh, criticLSTMc),
        ) = self.local_model(state, (ah0, ac0, ch0, cc0))
        action = torch.multinomial(policy, 1).item()
        return (
            action,
            policy,
            (actorLSTMh, actorLSTMc),
            (criticLSTMh, criticLSTMc),
            value,
        )

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

        advantage = returns - values
        policy_loss = -log_probs * advantage.detach()
        value_loss = self.l1_loss(torch.cat(values), returns)

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
            log_prob = torch.log(torch.tensor(log_prob, dtype=torch.float).to(device))
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
        ) = self.local_model(state, (ah0, ac0, ch0, cc0))
        action = torch.multinomial(policy, 1).item()
        return action, policy, (actorLSTMh, actorLSTMc), (criticLSTMh, criticLSTMc)
