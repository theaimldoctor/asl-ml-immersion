import numpy as np
from PIL import ImageFilter


class PerturbationMonitor:
    def high_frequency_score(self, image, background_mask=None):
        gray = image.convert("L")
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))

        arr = np.asarray(gray).astype(np.float32) / 255.0
        blur = np.asarray(blurred).astype(np.float32) / 255.0
        high_freq = np.abs(arr - blur)

        if background_mask is not None:
            mask = background_mask.astype(np.float32)
            denom = mask.sum() + 1e-8
            mean = float((high_freq * mask).sum() / denom)
            std = float(np.sqrt((((high_freq - mean) ** 2) * mask).sum() / denom))
        else:
            mean = float(high_freq.mean())
            std = float(high_freq.std())

        return float(mean + 2.0 * std)

    def clean_adv_delta_score(self, clean_image, adv_image, background_mask=None):
        clean = np.asarray(clean_image.convert("RGB")).astype(np.float32) / 255.0
        adv = np.asarray(adv_image.convert("RGB")).astype(np.float32) / 255.0

        diff = np.abs(adv - clean).mean(axis=2)

        if background_mask is not None:
            mask = background_mask.astype(np.float32)
            return float((diff * mask).sum() / (mask.sum() + 1e-8))

        return float(diff.mean())
