import numpy as np
from PIL import Image

from medfocus_defense.cloud.vlm_backends.base import VLMBackend


class SimpleImageBackend(VLMBackend):
    def generate(self, image_path: str, prompt: str) -> str:
        img = Image.open(image_path).convert("L")
        arr = np.array(img).astype(np.float32)

        mean_intensity = float(arr.mean())
        std_intensity = float(arr.std())
        bright_ratio = float((arr > 170).mean())
        dark_ratio = float((arr < 60).mean())

        evidence = []

        if mean_intensity > 135:
            evidence.append("overall MRI intensity is relatively high")
        elif mean_intensity < 90:
            evidence.append("overall MRI intensity is relatively low")
        else:
            evidence.append("overall MRI intensity is intermediate")

        if std_intensity > 55:
            evidence.append("there is notable regional contrast variation")
        else:
            evidence.append("regional contrast variation is limited")

        if bright_ratio > 0.20:
            evidence.append("bright tissue-like regions occupy a substantial area")

        if dark_ratio > 0.20:
            evidence.append("dark background-like regions occupy a substantial area")

        evidence_text = "; ".join(evidence)

        return (
            f"Evidence first: {evidence_text}. "
            f"Impression: brain MRI image-dependent assessment for Gate 2 testing."
        )
