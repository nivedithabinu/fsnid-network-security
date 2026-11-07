import torch
import torch.nn as nn

from src.tmg_gan.losses import (wasserstein_critic_loss, gradientPenalty, classification_loss, cosine_similarity_loss)

def test_tmg_losses():
    torch.manual_seed(42)
    
    batch_size = 16
    feature_dimensions = 41
    num_classes = 5
    
    # 1. Wasserstein critic loss
    realScores = torch.full((batch_size,1), 2.0)
    fakeScores = torch.full((batch_size,1), -1.0)
    
    criticLoss = wasserstein_critic_loss(realScores, fakeScores)
    
    assert torch.isclose(criticLoss, torch.tensor(-3.0))
    
    # 2. Gradient penalty
    critic = nn.Sequential(
        nn.Linear(feature_dimensions, 64),
        nn.LeakyReLU(0.2),
        nn.Linear(64, 1)
    )
    
    realSamples = torch.randn(batch_size, feature_dimensions)
    fakeSamples = torch.randn(batch_size, feature_dimensions)
    
    gp = gradientPenalty(critic, realSamples, fakeSamples)
    
    assert gp.ndim == 0
    assert torch.isfinite(gp)
    assert gp.item() >= 0
    
    # Verify gradient penalty supports backward propagation
    gp.backward()
    assert any(i.grad is not None for i in critic.parameters())
    
    # 3. Classification Loss
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    cls_loss = classification_loss(logits, labels)
    
    assert cls_loss.ndim == 0
    assert torch.isfinite(cls_loss)
    
    # 4. Cosine similarity loss
    generated_features = torch.randn(batch_size, 64, requires_grad=True)
    real_same_class = torch.randn(batch_size, 64)
    generated_other_class = torch.randn(batch_size, 64)
    cos_loss = cosine_similarity_loss(generated_features, real_same_class, generated_other_class)
    
    assert cos_loss.ndim == 0
    assert torch.isfinite(cos_loss)
    
    cos_loss.backward()
    assert generated_features.grad is not None
    
    print("Wasserstein critic loss: ", criticLoss.item())
    print("Gradient penalty: ", gp.item())
    print("Classification Loss: ", cls_loss.item())
    print("Cosine similarity loss: ", cos_loss.item())
    
    print("\nAll TMG-GAN loss tests passed!")

if __name__ == "__main__":
    test_tmg_losses()