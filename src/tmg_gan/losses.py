import torch
import torch.nn as nn
import torch.nn.functional as F 

def wasserstein_critic_loss(realScores, fakeScores):
    """
    Wasserstein critic loss.

    Minimise: E[D(fake)] - E[D(real)]

    The critic should assign higher scores to real samples.
    """
    return fakeScores.mean()-realScores.mean()

def gradientPenalty(critic, realSamples, fakeSamples):
    """
    WGAN-GP gradient penalty.

    Interpolates between real and generated samples and penalises gradient norms that deviate from 1.
    """
    batch_size = realSamples.size(0)
    device = realSamples.device
    
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(realSamples)
    
    interpolated = (alpha*realSamples+(1-alpha)*fakeSamples).requires_grad_(True)
    interpolatedScores = critic(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=interpolatedScores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolatedScores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.reshape(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    
    return ((gradient_norm-1)**2).mean()

def classification_loss(class_logits, target_labels):
    """
    Five-class categorical cross-entropy.
    Expects raw logits and integer class labels.
    """
    return F.cross_entropy(class_logits, target_labels)

def cosine_similarity_loss(generated_features, real_same_class_features, generated_other_class_features):
    """
    Class-aware cosine similarity objective.
    Encourages generated samples to resemble real samples from their intended class while discouraging similarity to generated samples from another class.
    Minimisation objective: similarity(other class) - similarity(same class)
    """
    same_similarity = F.cosine_similarity(generated_features, real_same_class_features, dim=1)
    other_similarity = F.cosine_similarity(generated_features, generated_other_class_features, dim=1)

    return (other_similarity-same_similarity).mean()