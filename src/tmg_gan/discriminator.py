import torch
import torch.nn as nn

class SharedDiscriminatorClassifier(nn.Module):
    """
    Shared discriminator-classifier architecture for TMG-GAN.

    Both the Wasserstein critic and the five-class classifier use the same learned feature representation.

    Input:
        samples: Tensor of shape (batch_size, 41)

    Outputs:
        critic_score: Tensor of shape (batch_size, 1)
        class_logits: Tensor of shape (batch_size, 5)
        features: Shared representation of shape (batch_size, 64)
    """
    def __init__(self, input_dimensions=41, hidden_dim1=128, hidden_dim2=64, num_classes=5, dropout=0.3):
        super().__init__()
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dimensions, hidden_dim1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.utils.spectral_norm(nn.Linear(hidden_dim1, hidden_dim2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        # Wasserstein critic head
        # No sigmoid: the critic produces unrestricted real-valued scores.
        self.critic_head = nn.utils.spectral_norm(nn.Linear(hidden_dim2, 1))

        # Five-class classification head
        # Returns logits; softmax is applied only when probabilities are needed.
        self.classifier_head = nn.Linear(hidden_dim2, num_classes)
    
    def forward(self, samples):
        features = self.shared_layers(samples)
        
        critic_score = self.critic_head(features)
        class_logits = self.classifier_head(features)
        
        return critic_score, class_logits, features