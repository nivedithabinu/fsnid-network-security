import torch
import torch.nn as nn

class MinorityClassGenerator(nn.Module):
    """
    Generator for producing synthetic NSL-KDD minority-class samples.

    Input:
        noise: Random latent vectors of shape (batch_size, 100)

    Output:
        Synthetic samples of shape (batch_size, 41),
        with values bounded between -1 and 1.
    """
    def __init__(self, noise_dimensions=100, output_dimensions=41):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(noise_dimensions,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64,output_dimensions),
            nn.Tanh()
        )
    
    def forward(self, noise):
        return self.network(noise)

class U2RGenerator(MinorityClassGenerator):
    """Generator dedicated to U2R attack samples."""
    pass

class R2LGenerator(MinorityClassGenerator):
    """Generator dedicated to R2L attack samples."""
    pass