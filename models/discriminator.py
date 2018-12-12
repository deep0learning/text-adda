"""Discriminator model for ADDA."""

from torch import nn
from params import model_param as mp


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(mp.d_input_dims, mp.d_hidden_dims),
            nn.ReLU(),
            nn.Linear(mp.d_hidden_dims, mp.d_hidden_dims),
            nn.ReLU(),
            nn.Linear(mp.d_hidden_dims, mp.d_output_dims)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
