import configparser as cp

import torch
import torch.nn as nn
import torch.nn.functional as F


# define a Conv VAE
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):

    def __init__(
        self, 
        kernel_size: int = 4,
        init_channels: int = 8,
        image_channels: int = 3,
        latent_dim: int = 128,
    ):
        super(ConvVAE, self).__init__()

        # Encoder
        self.enc1 = nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1)   # 32 -> 16
        self.enc2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)  # 16 -> 8
        self.enc3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1) # 8 -> 4
        self.enc4 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # 4 -> 2

        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, 64 * 2 * 2)

        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),      # 2 → 4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),      # 4 → 8
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),      # 8 → 16
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),      # 16 → 32
            nn.Conv2d(16, 3, kernel_size=3, padding=1)         # final output
        )




    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        #print("enc1 before", x.shape)
        x = F.relu(self.enc1(x))
        #print("enc1", x.shape)
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))  # output shape: (B, 64, 2, 2)

        # Global average pooling
        batch_size = x.size(0)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)  # (B, 64)

        hidden = self.fc1(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        z = self.reparameterize(mu, log_var)

        # Decoder
        z = self.fc2(z)
        z = z.view(-1, 64, 2, 2)

        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = self.dec4(x)  # no sigmoid, use MSELoss

        return reconstruction, mu, log_var
