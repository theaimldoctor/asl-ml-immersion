from medfocus_defense.cloud.vlm_backends.simple_image_backend import SimpleImageBackend


def get_vlm_backend(backend_name: str):
    if backend_name == "simple_image":
        return SimpleImageBackend()

    if backend_name == "qwen2vl":
        return Qwen2VLBackend(**kwargs)

    raise ValueError(
        f"Unknown VLM backend: {backend_name}. "
        f"Available backends: simple_image,qwen2vl"
    )
