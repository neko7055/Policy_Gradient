import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyContinuous(nn.Module):

    def __init__(self, input_size, output_size, action_range, learning_rate=0.01):
        super(PolicyContinuous, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 256)
        self.active1 = nn.PReLU()
        self.fc_mu = nn.Linear(256, 128)
        self.active_mu = nn.PReLU()
        self.mu = nn.Linear(128, output_size)
        self.fc_sigma = nn.Linear(256, 128)
        self.active_sigma = nn.Tanh()
        self.sigma = nn.Linear(128, output_size)
        self.action_range = torch.from_numpy(action_range)
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        self.float()

    def forward(self, x):
        x = self.active1(self.fc1(x))
        mu, sigma = self.active_mu(self.fc_mu(x)), self.active_sigma(self.fc_sigma(x))
        return torch.tanh(self.mu(mu)) * self.action_range, torch.exp(self.sigma(sigma))

    def update(self, S, A, R):
        n = len(R)
        loss = torch.zeros(1)
        for s, a, r in zip(S, A, R):
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()
            r = torch.from_numpy(r).float()
            mu, sigma = self(s)
            loss += (torch.sum(r @ torch.log(sigma)) + torch.sum(
                r @ (torch.square(mu - a) / (2 * torch.square(sigma)))))
        loss /= n
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def sample_action(self, x):
        x = torch.from_numpy(x).float()
        mean, sigma = self(x)
        mean, sigma = mean.numpy(), sigma.numpy()
        return np.random.randn(self.output_size) * sigma[0] + mean[0]

    @torch.no_grad()
    def predict(self, x):
        x = torch.from_numpy(x).float()
        mean, _ = self(x)
        return mean.numpy()


class PolicyDiscrete(nn.Module):

    def __init__(self, layers_size, learning_rate=0.01):
        super(PolicyDiscrete, self).__init__()
        self.input_size = layers_size[0]
        self.output_size = layers_size[-1]
        self.layers = nn.ModuleList(
            [nn.Linear(input_size, output_size) for input_size, output_size in zip(layers_size[:-1], layers_size[1:])])

        self.actives = nn.ModuleList([nn.PReLU() for _ in range(len(layers_size) - 1)])
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        self.float()

    def forward(self, x):
        for layer, active in zip(self.layers[:-1], self.actives):
            x = active(layer(x))
        return F.softmax(self.layers[-1](x), dim=-1)

    def update(self, S, A, R):
        n = len(R)
        loss = torch.zeros(1)
        for s, a, r in zip(S, A, R):
            s = torch.from_numpy(s).float()
            a = F.one_hot(torch.from_numpy(a).long(), num_classes=self.output_size)
            r = torch.from_numpy(r).float()
            a_pro = self(s)
            loss -= torch.sum(r @ (a * torch.log(a_pro)))
        loss /= n
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def sample_action(self, x):
        x = torch.from_numpy(x).float()
        action_pro = self(x).numpy()
        return np.random.choice(range(self.output_size), size=1, p=action_pro[0])[0]

    @torch.no_grad()
    def predict(self, x):
        x = torch.from_numpy(x).float()
        action_pro = self(x).numpy()
        return np.argmax(action_pro,axis=1)
