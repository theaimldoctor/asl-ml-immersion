import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


META_PATH = Path("medfocus_defense/data/brain_tumor_mri/metadata/samples.jsonl")
MASK_DIR = Path("medfocus_defense/data/brain_tumor_mri/masks")
OUT_META_PATH = Path("medfocus_defense/data/brain_tumor_mri/metadata/samples_with_masks.jsonl")


def make_simple_foreground_mask(image_path: Path) -> Image.Image:
    img = Image.open(image_path).convert("L")
    arr = np.array(img).astype(np.float32)

    threshold = max(20.0, float(arr.mean() * 0.55))
    mask = (arr > threshold).astype(np.uint8) * 255

    mask_img = Image.fromarray(mask, mode="L")
    mask_img = mask_img.filter(ImageFilter.MedianFilter(size=5))

    return mask_img


def main() -> None:
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_META_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    updated = []
    skipped = 0

    for row in rows:
        image_path = Path(row["image_path"])

        if not image_path.exists():
            skipped += 1
            continue

        mask_name = f"{row['sample_id']}_mask.png"
        mask_path = MASK_DIR / mask_name

        if not mask_path.exists():
            mask = make_simple_foreground_mask(image_path)
            mask.save(mask_path)

        row["mask_path"] = str(mask_path)
        row["mask_type"] = "simple_intensity_foreground_mask"
        updated.append(row)

    with OUT_META_PATH.open("w", encoding="utf-8") as f:
        for row in updated:
            f.write(json.dumps(row) + "\n")

    print(f"Input rows: {len(rows)}")
    print(f"Updated rows: {len(updated)}")
    print(f"Skipped missing images: {skipped}")
    print(f"New metadata: {OUT_META_PATH}")
    print(f"Mask directory: {MASK_DIR}")


if __name__ == "__main__":
    main()
