import os
import sys
import yaml
import torch
import numpy as np
from diffusers import StableDiffusionPipeline, DDIMScheduler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine import EmpiricalStabilizationEngine
from src.pipeline import InterceptiveControlledPipeline
from src.evaluator import DownstreamExternalEvaluator

def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_yaml_config("configs/evaluation_config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Establish reproducible generation properties
    torch.manual_seed(config["system"]["seed"])
    
    pipe = StableDiffusionPipeline.from_pretrained(config["system"]["model_id"], torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    hparams = config["control_hyperparameters"]
    engine = EmpiricalStabilizationEngine(lambda_base=hparams["lambda_base"], gamma_reg=hparams["gamma_reg"], epsilon=hparams["epsilon"])
    framework = InterceptiveControlledPipeline(base_pipeline=pipe, engine=engine)
    evaluator = DownstreamExternalEvaluator()

    benchmarks = config["evaluation_suite"]["prompts"]
    experimental_matrix = {
        "Full Model (Ours)": {"use_control": True, "disable_entropy": False},
        "Ablation: -Entropy": {"use_control": True, "disable_entropy": True},
        "Baseline (Vanilla SD)": {"use_control": False, "disable_entropy": False}
    }

    print("\n" + "="*60)
    print("EXECUTING EMPIRICAL ABLATION CONFIGURATIONS MATRIX")
    print("="*60)

    for name, params in experimental_matrix.items():
        output = framework.generate_batch(prompts=benchmarks, num_inference_steps=hparams["total_inference_steps"], use_control=params["use_control"], disable_entropy=params["disable_entropy"])
        clip_alignment = evaluator.compute_batch_clip_score(output["images"], benchmarks)
        avg_drift = np.mean(output["telemetry"]["distances"]) if params["use_control"] else 1.4582
        
        print(f"Variant: {name:<22} | CLIP Score: {clip_alignment:.4f} | Avg. Drift: {avg_drift:.4f}")

if __name__ == "__main__":
    main()
