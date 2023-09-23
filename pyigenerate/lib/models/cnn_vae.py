import torch
import torch.nn as nn

class CnnVae(nn.Module):
    def __init__(self, in_channels=3, latent_dim=5):
        super().__init__()

        self.in_channels    = in_channels
        self.latent_dim     = 5

        # Encoding Layers
        # 3 --> 8 --> 16 --> 32 --> 64
        self.encoding_layer_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=8,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.encoding_layer_2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.encoding_layer_3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.encoding_layer_4 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=0
        )

        # Fully connected layers for learning representation
        self.fully_connected_encoder    = nn.Linear(64, 128)
        self.fully_connected_mu         = nn.Linear(128, self.latent_dim)
        self.fully_connected_log_var    = nn.Linear(128, self.latent_dim)
        self.fully_connected_decoder    = nn.Linear(self.latent_dim, 64)

        # Decoding Layers
        # 64 --> 32 --> 16 --> 8 --> 3
        self.decoding_layer_1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=1,
            padding=0
        )

        self.decoding_layer_2 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.decoding_layer_3 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=8,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.decoding_layer_4 = nn.ConvTranspose2d(
            in_channels=8,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.adaptive_average_pool_2d = nn.AdaptiveAvgPool2d(1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)    # standard deviation
        eps = torch.rand_like(std)      # same size as std
        sample = mu + (std * eps)

        return sample

    def encode(self, x):
        # Encoding
        x = self.encoding_layer_1(x)
        x = self.relu(x)

        x = self.encoding_layer_2(x)
        x = self.relu(x)

        x = self.encoding_layer_3(x)
        x = self.relu(x)

        x = self.encoding_layer_4(x)
        x = self.relu(x)

        # Average pool and flatten
        batch, _, _, _ = x.shape
        x = self.adaptive_average_pool_2d(x)
        x = x.reshape(batch, -1)

        hidden = self.fully_connected_encoder(x)

        # Get latent dimensions for mu and log_var
        mu      = self.fully_connected_mu(hidden)
        log_var = self.fully_connected_log_var(hidden)

        # Get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fully_connected_decoder(z)

        z = z.view(-1, 64, 1, 1)

        return z, mu, log_var

    def sample(self, mu, log_var):
        z = self.reparameterize(mu, log_var)
        z = self.fully_connected_decoder(z)

        z = z.view(-1, 64, 1, 1)

        return self.decode(z)

    def decode(self, z):
        y = self.decoding_layer_1(z)
        y = self.relu(y)

        y = self.decoding_layer_2(y)
        y = self.relu(y)

        y = self.decoding_layer_3(y)
        y = self.relu(y)

        y = self.decoding_layer_4(y)
        y = self.sigmoid(y)

        return y

    def forward(self, x):
        z, mu, log_var = self.encode(x)
        y = self.decode(z)

        return y, mu, log_var
