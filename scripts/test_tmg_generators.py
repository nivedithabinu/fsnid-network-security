import torch
from src.tmg_gan.generator import U2RGenerator, R2LGenerator

def test_generators():
    torch.manual_seed(42)

    batch_size = 16
    noise_dimensions = 100
    feature_dimensions = 41

    u2r_generator = U2RGenerator()
    r2l_generator = R2LGenerator()

    noise_u2r = torch.randn(batch_size, noise_dimensions)
    noise_r2l = torch.randn(batch_size, noise_dimensions)

    generated_u2r = u2r_generator(noise_u2r)
    generated_r2l = r2l_generator(noise_r2l)

    assert generated_u2r.shape == (batch_size, feature_dimensions)
    assert generated_r2l.shape == (batch_size, feature_dimensions)

    assert torch.isfinite(generated_u2r).all()
    assert torch.isfinite(generated_r2l).all()

    assert generated_u2r.min() >= -1
    assert generated_u2r.max() <= 1

    assert generated_r2l.min() >= -1
    assert generated_r2l.max() <= 1

    print("U2R output shape:", generated_u2r.shape)
    print("R2L output shape:", generated_r2l.shape)

    print("U2R output range:", generated_u2r.min().item(), generated_u2r.max().item())
    print("R2L output range:", generated_r2l.min().item(), generated_r2l.max().item())

    print("\nBoth generators passed all tests!")

if __name__ == "__main__":
    test_generators()