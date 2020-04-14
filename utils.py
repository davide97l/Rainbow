from torch import nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init  # used to initialize weight_sigma and bias_sigma

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Generate new noise and make epsilon matrices."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add noise to the parameters to perform the forward step"""
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Generate noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))
        return x.sign().mul(x.abs().sqrt())


class DenseNet(nn.Module):
    """
    Attributes:
        in_dim (int): input size
        out_dim (int): output size
        atom_size (int): atom size, used to compute categorical distribution
        support (torch.Tensor): discrete support z
        hidden_size (int): hidden size
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            atom_size: int,
            support: torch.Tensor,
            hidden_size: int,
            # options
            no_dueling=False,
            no_noise=False
    ):
        """Initialization."""
        super(DenseNet, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size
        self.hidden_size = hidden_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, self.hidden_size),
            nn.ReLU(),
        )

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(self.hidden_size, self.hidden_size)
        self.advantage_layer = NoisyLinear(self.hidden_size, out_dim * atom_size)  # output one distribution per action
        # set value layer
        self.value_hidden_layer = NoisyLinear(self.hidden_size, self.hidden_size)
        self.value_layer = NoisyLinear(self.hidden_size, atom_size)

        # options
        self.no_dueling = no_dueling
        self.no_noise = no_noise
        if no_noise:
            # use linear standard layers
            self.advantage_hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
            self.advantage_layer = nn.Linear(self.hidden_size, out_dim * atom_size)
            self.value_hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
            self.value_layer = nn.Linear(self.hidden_size, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation, return one Q-value for each action."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms, one distribution for each action."""
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        if not self.no_dueling:
            value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            # disable dueling network, ignore value layer and advantage formula
            q_atoms = advantage

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


class ConvNet(nn.Module):
    """
    Attributes:
        in_dim (int): input size
        out_dim (int): output size
        atom_size (int): atom size, used to compute categorical distribution
        support (torch.Tensor): discrete support z
    """
    def __init__(
            self,
            in_dim: List[int],  # (h, w, c)
            out_dim: int,
            atom_size: int,
            support: torch.Tensor,
            # options
            no_dueling=False,
            no_noise=False
    ):
        """Initialization."""
        super(ConvNet, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # conv layer (input at least 1 x 1 x 40 x 40)
        self.conv1 = nn.Conv2d(in_dim[2], 16, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=8, stride=4):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(in_dim[1], 8, 4), 4, 2), 3, 1)
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(in_dim[0], 8, 4), 4, 2), 3, 1)
        self.hidden_size = conv_w * conv_h * 32

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(self.hidden_size, self.hidden_size)
        self.advantage_layer = NoisyLinear(self.hidden_size, out_dim * atom_size)  # output one distribution per action
        # set value layer
        self.value_hidden_layer = NoisyLinear(self.hidden_size, self.hidden_size)
        self.value_layer = NoisyLinear(self.hidden_size, atom_size)

        # options
        self.no_dueling = no_dueling
        self.no_noise = no_noise
        if no_noise:
            # use linear standard layers
            self.advantage_hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
            self.advantage_layer = nn.Linear(self.hidden_size, out_dim * atom_size)
            self.value_hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
            self.value_layer = nn.Linear(self.hidden_size, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation, return one Q-value for each action."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms, one distribution for each action."""
        x = x.permute(0, 3, 1, 2)  # adapt to pytorch format
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        feature = x.reshape(x.shape[0], -1)

        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        if not self.no_dueling:
            value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            # disable dueling network, ignore value layer and advantage formula
            q_atoms = advantage

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


if __name__ == '__main__':
    dims = [1, 100, 100, 3]
    s1 = torch.FloatTensor(np.random.uniform(size=dims))  # state
    atom_size = 51
    support = torch.linspace(0, 200, atom_size)
    m = ConvNet(dims[1:], 4, atom_size, support)
    o = m(s1)
    print(o)

    dims = 4
    s1 = torch.FloatTensor(np.random.uniform(size=dims))  # state
    atom_size = 51
    support = torch.linspace(0, 200, atom_size)
    m = DenseNet(4, 4, atom_size, support, 100)
    o = m(s1.unsqueeze(0))
    print(o)

