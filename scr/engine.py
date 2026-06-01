import torch
import torch.nn.functional as F
from typing import Tuple

class EmpiricalStabilizationEngine:
    def __init__(self, lambda_base: float = 0.75, gamma_reg: float = 0.3, epsilon: float = 1e-6):
        """
        Operationalizes the Feedback Operator (F_t) to measure and correct 
        trajectory deviations on high-dimensional unit hyperspheres.
        """
        self.lambda_base = lambda_base
        self.gamma_reg = gamma_reg
        self.epsilon = epsilon
        self.S_t: torch.Tensor = None

    def initialize_anchor(self, text_embeddings: torch.Tensor):
        """Maps Sensory Grounding (S_t) onto a unit hypersphere constraint manifold."""
        self.S_t = F.normalize(text_embeddings.detach().mean(dim=1), p=2, dim=-1)

    def evaluate_trajectory_step(self, latents: torch.Tensor, disable_entropy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the joint error function balancing spatial entropy proxy against geometric distance.
        """
        batch_size = latents.shape[0]
        
        if disable_entropy:
            f_t_score = torch.ones(batch_size, device=latents.device)
        else:
            visual_variance = latents.detach().view(batch_size, -1).var(dim=1)
            f_t_score = torch.clamp(1.0 / (visual_variance + self.epsilon), 0.0, 1.0)

        # Map current Generative State (G_t) to the shared unit sphere geometry
        G_t_flat = latents.detach().view(batch_size, -1)
        target_dim = self.S_t.shape[-1]
        G_t_projected = F.adaptive_avg_pool1d(G_t_flat.unsqueeze(1), target_dim).squeeze(1)
        G_t = F.normalize(G_t_projected, p=2, dim=-1)

        # Continuous tracking distance verification
        grounding_distance = torch.norm(G_t - self.S_t, p=2, dim=-1)
        lambda_t = 1.0 / (grounding_distance + self.epsilon)
        
        # Unified Feedback execution value
        effective_lambda = lambda_t * ((1.0 - self.gamma_reg) + self.gamma_reg * f_t_score)
        
        return grounding_distance, effective_lambda
