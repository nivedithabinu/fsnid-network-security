import torch

from src.tmg_gan.generator import U2RGenerator, R2LGenerator
from src.tmg_gan.discriminator import SharedDiscriminatorClassifier
from src.tmg_gan.trainer import TMGGANTrainer

def test_tmg_trainer():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    u2r_generator = U2RGenerator()
    r2l_generator = R2LGenerator()
    discriminator = SharedDiscriminatorClassifier()

    generator_optimizer_u2r = torch.optim.Adam(u2r_generator.parameters(), lr=0.0001)
    generator_optimizer_r2l = torch.optim.Adam(r2l_generator.parameters(), lr=0.0001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    # Temporary coefficients for integration testing only.
    trainer = TMGGANTrainer(
        u2r_generator=u2r_generator,
        r2l_generator=r2l_generator,
        discriminator=discriminator,
        generator_optimizer_u2r=generator_optimizer_u2r,
        generator_optimizer_r2l=generator_optimizer_r2l,
        discriminator_optimizer=discriminator_optimizer,
        classification_weight=1.0,
        cosine_weight=1.0,
        device=device
    )

    batch_size = 8

    real_samples = torch.rand(batch_size, 41)
    real_labels = torch.randint(0, 5, (batch_size,))

    u2r_real_samples = torch.rand(batch_size, 41)
    r2l_real_samples = torch.rand(batch_size, 41)

    # Save initial parameters to verify that models update.
    initial_u2r = next(u2r_generator.parameters()).detach().clone()
    initial_r2l = next(r2l_generator.parameters()).detach().clone()
    initial_discriminator = next(discriminator.parameters()).detach().clone()

    losses = trainer.train_step(real_samples, real_labels, u2r_real_samples, r2l_real_samples)

    print("\nTraining-step losses:")
    for i, j in losses.items():
        print(f"{i}: {j:.6f}")
        assert torch.isfinite(torch.tensor(j)), (
            f"Non-finite loss detected: {i}"
        )

    # Verify that all three models were updated.
    assert not torch.equal(initial_u2r, next(u2r_generator.parameters()).detach()), "U2R generator was not updated."
    assert not torch.equal(initial_r2l, next(r2l_generator.parameters()).detach()), "R2L generator was not updated."
    assert not torch.equal(initial_discriminator, next(discriminator.parameters()).detach()), "Discriminator was not updated."

    print("\nAll three models updated successfully!")
    print("TMG-GAN integration test passed!")

if __name__ == "__main__":
    test_tmg_trainer()