import numpy as np
from PIL import Image, ImageFilter


def neutralize_patch(image, box, mode="blur"):
    image = image.convert("RGB")
    out = image.copy()

    x0, y0, x1, y1 = [int(v) for v in box]
    patch = out.crop((x0, y0, x1, y1))

    if mode == "blur":
        patch_new = patch.filter(ImageFilter.GaussianBlur(radius=8))

    elif mode == "mean":
        arr = np.asarray(patch).astype(np.float32)
        mean_color = tuple(np.clip(arr.mean(axis=(0, 1)), 0, 255).astype(np.uint8))
        patch_new = Image.new("RGB", patch.size, mean_color)

    else:
        raise ValueError(f"Unknown neutralization mode: {mode}")

    out.paste(patch_new, (x0, y0))
    return out


def simple_text_distance(a, b):
    a_tokens = set(str(a).lower().split())
    b_tokens = set(str(b).lower().split())

    if not a_tokens and not b_tokens:
        return 0.0

    jaccard = len(a_tokens & b_tokens) / (len(a_tokens | b_tokens) + 1e-8)
    return float(1.0 - jaccard)


class CausalPatchTester:
    """
    Gate 2 causal patch tester.

    Computes:
    C_bg = mean output change after suspicious background patch neutralization
    C_fg = mean output change after foreground control patch neutralization
    R_bg_causal = C_bg / (C_bg + C_fg + eps)
    """

    def __init__(
        self,
        victim_generate,
        prompt_builder,
        top_k=3,
        k_fg=1,
        neutralize_mode="blur",
    ):
        self.victim_generate = victim_generate
        self.prompt_builder = prompt_builder
        self.top_k = top_k
        self.k_fg = k_fg
        self.neutralize_mode = neutralize_mode

    def select_top_background_patches(self, patch_rows):
        bg_patches = [
            p for p in patch_rows
            if p["bg_value"] >= p["fg_value"]
        ]

        bg_patches = sorted(
            bg_patches,
            key=lambda p: p["relevance"] * p["bg_value"],
            reverse=True,
        )

        return bg_patches[:self.top_k]

    def select_foreground_control_patches(self, patch_rows):
        fg_patches = [
            p for p in patch_rows
            if p["fg_value"] > p["bg_value"]
        ]

        fg_patches = sorted(
            fg_patches,
            key=lambda p: p["relevance"] * p["fg_value"],
            reverse=True,
        )

        return fg_patches[:self.k_fg]

    def _run_patch_set(self, image, prompt, y0, selected_patches):
        results = []

        for item in selected_patches:
            patched_image = neutralize_patch(
                image=image,
                box=item["box"],
                mode=self.neutralize_mode,
            )

            yj = self.victim_generate(patched_image, prompt)
            delta = simple_text_distance(y0, yj)

            results.append({
                "patch_id": item["patch_id"],
                "box": item["box"],
                "original_relevance": item["relevance"],
                "fg_value": item["fg_value"],
                "bg_value": item["bg_value"],
                "patched_output": yj,
                "delta": float(delta),
            })

        return results

    def run(self, image, user_prompt, y0, patch_rows):
        prompt = self.prompt_builder(user_prompt)

        bg_selected = self.select_top_background_patches(patch_rows)
        fg_selected = self.select_foreground_control_patches(patch_rows)

        bg_results = self._run_patch_set(
            image=image,
            prompt=prompt,
            y0=y0,
            selected_patches=bg_selected,
        )

        fg_results = self._run_patch_set(
            image=image,
            prompt=prompt,
            y0=y0,
            selected_patches=fg_selected,
        )

        c_bg = float(np.mean([p["delta"] for p in bg_results])) if bg_results else 0.0
        c_fg = float(np.mean([p["delta"] for p in fg_results])) if fg_results else 0.0

        r_bg_causal = c_bg / (c_bg + c_fg + 1e-8)

        return {
            "selected_background_patches": bg_selected,
            "selected_foreground_patches": fg_selected,
            "background_patch_results": bg_results,
            "foreground_patch_results": fg_results,
            "C_bg": c_bg,
            "C_fg": c_fg,
            "R_bg_causal": float(r_bg_causal),
        }
