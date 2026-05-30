import json
from pathlib import Path
from PIL import Image


RAW_ROOT = Path("medfocus_defense/data/raw/brain_tumor_classification_mri")
OUT_ROOT = Path("medfocus_defense/data/brain_tumor_mri")
META_DIR = OUT_ROOT / "metadata"
IMAGE_OUT_DIR = OUT_ROOT / "images"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


CLASS_TO_FINDING = {
    "glioma_tumor": "Brain MRI finding suggestive of glioma tumor.",
    "meningioma_tumor": "Brain MRI finding suggestive of meningioma tumor.",
    "pituitary_tumor": "Brain MRI finding suggestive of pituitary tumor.",
    "no_tumor": "Brain MRI with no tumor finding.",
}


def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def class_to_ground_truth(label: str) -> str:
    key = normalize_label(label)
    return CLASS_TO_FINDING.get(key, f"Brain MRI class: {label}.")


def find_images() -> list[tuple[Path, str, str]]:
    found = []

    for split_name in ["Training", "Testing"]:
        split_dir = RAW_ROOT / split_name
        if not split_dir.exists():
            continue

        split = "train" if split_name == "Training" else "test"

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            label = class_dir.name

            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in VALID_EXTS:
                    found.append((img_path, split, label))

    return found


def verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def main() -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = find_images()
    print(f"Found {len(images)} candidate images.")

    rows = []
    skipped = 0

    for idx, (src_path, split, label) in enumerate(images):
        if not verify_image(src_path):
            skipped += 1
            continue

        clean_label = normalize_label(label)
        sample_id = f"brain_mri_{idx:06d}_{clean_label}"
        out_name = f"{sample_id}{src_path.suffix.lower()}"
        dst_path = IMAGE_OUT_DIR / out_name

        if not dst_path.exists():
            dst_path.write_bytes(src_path.read_bytes())

        row = {
            "sample_id": sample_id,
            "modality": "brain_mri",
            "image_path": str(dst_path),
            "mask_path": None,
            "prompt": "Analyze this brain MRI and provide the most likely tumor-related finding.",
            "ground_truth": class_to_ground_truth(label),
            "class_label": label,
            "split": split,
            "source_dataset": "sartajbhuvaji/brain-tumor-classification-mri",
        }

        rows.append(row)

    metadata_path = META_DIR / "samples.jsonl"
    with metadata_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Valid samples prepared: {len(rows)}")
    print(f"Skipped invalid images: {skipped}")
    print(f"Metadata written to: {metadata_path}")
    print(f"Images written to: {IMAGE_OUT_DIR}")


if __name__ == "__main__":
    main()
