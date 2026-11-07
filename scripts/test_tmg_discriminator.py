import torch
from src.tmg_gan.discriminator import SharedDiscriminatorClassifier

def test_shared_discriminator_classifier():
    torch.manual_seed(42)

    batch_size = 16
    input_dimensions = 41
    num_classes = 5

    model = SharedDiscriminatorClassifier()
    samples = torch.randn(batch_size, input_dimensions)
    critic_scores, class_logits, features = model(samples)

    # Verify output dimensions
    assert critic_scores.shape == (batch_size, 1)
    assert class_logits.shape == (batch_size, num_classes)
    assert features.shape == (batch_size, 64)

    # Verify numerical stability
    assert torch.isfinite(critic_scores).all()
    assert torch.isfinite(class_logits).all()
    assert torch.isfinite(features).all()

    # Verify that both heads use the same feature representation
    assert model.critic_head.in_features == model.classifier_head.in_features

    # Verify gradient flow through shared layers
    loss = critic_scores.mean() + class_logits.mean()
    loss.backward()

    for i in model.shared_layers.parameters():
        assert i.grad is not None

    print("Critic output shape:", critic_scores.shape)
    print("Classifier output shape:", class_logits.shape)
    print("Shared feature shape:", features.shape)

    print("\nShared architecture and gradient tests passed!")

if __name__ == "__main__":
    test_shared_discriminator_classifier()