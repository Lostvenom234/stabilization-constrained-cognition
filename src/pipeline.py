import torch
from typing import Dict, Any, List
from src.engine import EmpiricalStabilizationEngine

class InterceptiveControlledPipeline:
    def __init__(self, base_pipeline, engine: EmpiricalStabilizationEngine):
        self.pipeline = base_pipeline
        self.engine = engine

    @torch.no_grad()
    def generate_batch(self, prompts: List[str], num_inference_steps: int = 30, use_control: bool = True, disable_entropy: bool = False) -> Dict[str, Any]:
        device = self.pipeline.device
        dtype = self.pipeline.text_encoder.dtype
        batch_size = len(prompts)

        # 1. Initialize Sensory Grounding (S_t)
        max_length = self.pipeline.tokenizer.model_max_length
        text_inputs = self.pipeline.tokenizer(prompts, padding="max_length", max_length=max_length, return_tensors="pt").to(device)
        text_embeddings = self.pipeline.text_encoder(text_inputs.input_ids)[0]
        self.engine.initialize_anchor(text_embeddings)

        self.pipeline.scheduler.set_timesteps(num_inference_steps)
        
        # Initialize early state coordinates
        latents = torch.randn((batch_size, self.pipeline.unet.config.in_channels, 64, 64), dtype=dtype, device=device)
        telemetry = {"distances": [], "lambdas": []}

        # Main Dynamical State Update Loop
        for step, t in enumerate(self.pipeline.scheduler.timesteps):
            noise_pred = self.pipeline.unet(latents, t, encoder_hidden_states=text_embeddings).sample

            if use_control:
                # Intercept loop to extract real-time Feedback vector updates (F_t)
                dist, eff_lambda = self.engine.evaluate_trajectory_step(latents, disable_entropy=disable_entropy)
                
                # Apply piece-wise control law to actuate stochastic parameters
                eta_tensor = torch.where(eff_lambda < self.engine.lambda_base, 0.0, 0.5)
                eta_step = float(eta_tensor.mean().item())
                
                telemetry["distances"].append(dist.mean().item())
                telemetry["lambdas"].append(eff_lambda.mean().item())
            else:
                eta_step = 0.5 # Default open-loop DDIM trajectory sampling behavior

            # Step incorporates memory context history tracking vectors (M_t) via scheduler sample steps
            scheduler_output = self.pipeline.scheduler.step(noise_pred, t, latents, eta=eta_step)
            latents = scheduler_output.prev_sample

        images = self.pipeline.numpy_to_pil(self.pipeline.decode_latents(latents))
        return {"images": images, "telemetry": telemetry}
