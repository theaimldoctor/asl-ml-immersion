import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class ForegroundSegmenter:
    """
    Placeholder foreground segmenter.

    Current version:
    - makes an elliptical foreground mask centered in the image
    - useful for testing the MedFocusGuard pipeline

    Later replacement:
    - MedSAM
    - SAM-Med2D
    - modality-specific organ/lesion segmentation
    """

    def __init__(self, blur_radius=5):
        self.blur_radius = blur_radius

    def segment(self, image):
        image = image.convert("RGB")
        w, h = image.size

        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        x0 = int(0.20 * w)
        y0 = int(0.15 * h)
        x1 = int(0.80 * w)
        y1 = int(0.85 * h)

        draw.ellipse((x0, y0, x1, y1), fill=255)

        if self.blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        m_fg = np.asarray(mask).astype(np.float32) / 255.0
        m_bg = 1.0 - m_fg

        return {
            "m_fg": m_fg,
            "m_bg": m_bg,
            "mask_image": mask,
        }
