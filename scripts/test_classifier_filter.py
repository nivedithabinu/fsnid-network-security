import torch

from src.tmg_gan.discriminator import SharedDiscriminatorClassifier
from src.tmg_gan.classifier_filter import ClassifierBasedFilter

def test_classifier_filter():
    torch.manual_seed(42)
    discriminator = SharedDiscriminatorClassifier()
    sample_filter = ClassifierBasedFilter(discriminator)
    generated_samples = torch.rand(32, 41) * 2 - 1

    accepted_samples, accepted_indices, confidence_scores = (
        sample_filter.filter_samples(generated_samples, target_class=4)
    )

    print("Generated samples:", generated_samples.shape)
    print("Accepted samples:", accepted_samples.shape)
    print("Accepted indices:", accepted_indices.shape)
    print("Confidence scores:", confidence_scores.shape)

    assert accepted_samples.shape[1] == 41
    assert accepted_indices.ndim == 1
    assert confidence_scores.shape[0] == accepted_samples.shape[0]

    assert torch.all((confidence_scores >= 0) & (confidence_scores <= 1))
    assert torch.all(accepted_indices >= 0)
    assert torch.all(accepted_indices < generated_samples.shape[0])

    print("\nClassifier filtering test passed!")

if __name__ == "__main__":
    test_classifier_filter()