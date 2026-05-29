import numpy as np
from PIL import Image


def softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=np.float32)
    x = x / max(temperature, 1e-8)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-8)


def make_masked_image(image, mask, fill_value=0):
    """
    Keeps mask region and suppresses the rest.
    mask shape: H x W, values in [0, 1]
    """
    image = image.convert("RGB")
    arr = np.asarray(image).astype(np.float32)

    if mask.shape[:2] != arr.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {arr.shape}")

    mask3 = mask[..., None].astype(np.float32)
    out = arr * mask3 + fill_value * (1.0 - mask3)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def normalize_cosine_to_01(sim):
    """
    Cosine can theoretically be [-1, 1].
    Convert to [0, 1] so mismatch is easier to interpret.
    """
    return float(max(0.0, min(1.0, (sim + 1.0) / 2.0)))


class ClipGroundingMonitor:
    """
    Patch-level surrogate grounding using UniMed-CLIP.

    Estimates:
    - B_CLIP: background grounding risk
    - R_CLIP: foreground grounding
    - D_align: foreground image-text mismatch
    """

    def __init__(self, clip_model, patch_size=56, stride=56, temperature=0.07):
        self.clip_model = clip_model
        self.patch_size = patch_size
        self.stride = stride
        self.temperature = temperature

    def iter_patches(self, image):
        image = image.convert("RGB")
        w, h = image.size

        patches = []

        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                crop = image.crop((x, y, x + self.patch_size, y + self.patch_size))
                patches.append({
                    "patch": crop,
                    "box": (x, y, x + self.patch_size, y + self.patch_size),
                    "cx": x + self.patch_size / 2,
                    "cy": y + self.patch_size / 2,
                })

        return patches

    def patch_mask_value(self, mask, box):
        x0, y0, x1, y1 = box
        region = mask[int(y0):int(y1), int(x0):int(x1)]
        if region.size == 0:
            return 0.0
        return float(region.mean())

    def foreground_alignment(self, image, evidence_text, m_fg):
        fg_image = make_masked_image(image, m_fg, fill_value=0)

        sim_fg = self.clip_model.similarity(fg_image, evidence_text)
        align_01 = normalize_cosine_to_01(sim_fg)
        d_align = 1.0 - align_01

        return {
            "foreground_similarity_raw": float(sim_fg),
            "foreground_alignment_01": float(align_01),
            "D_align": float(d_align),
        }

    def compute(self, image, evidence_text, m_fg, m_bg):
        patches = self.iter_patches(image)

        similarities = []

        for item in patches:
            sim = self.clip_model.similarity(item["patch"], evidence_text)
            similarities.append(sim)

        relevance = softmax(similarities, temperature=self.temperature)

        fg_mass = 0.0
        bg_mass = 0.0

        patch_rows = []

        for idx, item in enumerate(patches):
            box = item["box"]
            r = float(relevance[idx])

            fg_value = self.patch_mask_value(m_fg, box)
            bg_value = self.patch_mask_value(m_bg, box)

            fg_contrib = r * fg_value
            bg_contrib = r * bg_value

            fg_mass += fg_contrib
            bg_mass += bg_contrib

            patch_rows.append({
                "patch_id": idx,
                "box": box,
                "similarity": float(similarities[idx]),
                "relevance": r,
                "fg_value": float(fg_value),
                "bg_value": float(bg_value),
                "fg_contrib": float(fg_contrib),
                "bg_contrib": float(bg_contrib),
            })

        denom = fg_mass + bg_mass + 1e-8

        b_clip = bg_mass / denom
        r_clip = fg_mass / denom

        align = self.foreground_alignment(
            image=image,
            evidence_text=evidence_text,
            m_fg=m_fg,
        )

        return {
            "A_fg_CLIP": float(fg_mass),
            "A_bg_CLIP": float(bg_mass),
            "B_CLIP": float(b_clip),
            "R_CLIP": float(r_clip),
            "D_align": float(align["D_align"]),
            "foreground_similarity_raw": float(align["foreground_similarity_raw"]),
            "foreground_alignment_01": float(align["foreground_alignment_01"]),
            "patch_rows": patch_rows,
        }
