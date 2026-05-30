import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from medfocus_defense.cloud.vlm_backends.base import VLMBackend


class Qwen2VLBackend(VLMBackend):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        max_new_tokens: int = 128,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"Loading Qwen2-VL backend: {model_name}")
        print(f"Device: {self.device}")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name)

    def generate(self, image_path: str, prompt: str) -> str:
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()
