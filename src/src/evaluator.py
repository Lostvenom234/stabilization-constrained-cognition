import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List

class DownstreamExternalEvaluator:
    def __init__(self, clip_model_id: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(clip_model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_id)

    @torch.no_grad()
    def compute_batch_clip_score(self, images: List[Image.Image], prompts: List[str]) -> float:
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        scores = logits_per_image.diagonal() / self.model.logit_scale.exp()
        return float(scores.mean().item())
