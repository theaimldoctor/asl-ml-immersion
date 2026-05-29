import numpy as np


class AttentionMonitor:
    def background_attention_ratio(self, attention_map, background_mask):
        if attention_map is None or background_mask is None:
            return None

        attn = np.asarray(attention_map).astype(np.float32)
        mask = np.asarray(background_mask).astype(np.float32)

        if attn.shape != mask.shape:
            raise ValueError(f"attention shape {attn.shape} != mask shape {mask.shape}")

        attn = np.maximum(attn, 0)
        attn = attn / (attn.sum() + 1e-8)

        bg = float((attn * mask).sum())
        fg = float((attn * (1.0 - mask)).sum())
        ratio = bg / (bg + fg + 1e-8)

        return {
            "background_attention": bg,
            "foreground_attention": fg,
            "background_ratio": float(ratio),
        }

    def fake_center_attention(self, image_size):
        w, h = image_size
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2, h / 2
        dist = ((xx - cx) ** 2 + (yy - cy) ** 2) ** 0.5
        attn = 1.0 / (dist + 1.0)
        attn = attn / attn.sum()
        return attn.astype(np.float32)
