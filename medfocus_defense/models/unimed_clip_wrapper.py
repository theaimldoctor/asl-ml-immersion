import sys
import torch
import torch.nn.functional as F


class UniMedCLIPWrapper:
    def __init__(
        self,
        repo_path="/home/jupyter/my_projects/asl-ml-immersion/clip_~modality/UniMed-CLIP",
        weight_path="/home/jupyter/my_projects/asl-ml-immersion/clip_~modality/UniMed-CLIP/weights/unimed_clip_vit_b16.pt",
        model_name="ViT-B-16",
        text_encoder_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        device="cpu",
    ):
        self.repo_path = repo_path
        self.weight_path = weight_path
        self.model_name = model_name
        self.text_encoder_name = text_encoder_name
        self.device = device

        self.model = None
        self.preprocess = None
        self.tokenizer = None

    def _load_tokenizer(self, open_clip):
        """
        UniMed-CLIP's local open_clip fork may not expose get_tokenizer
        at top-level. Try multiple known locations.
        """
        if hasattr(open_clip, "get_tokenizer"):
            return open_clip.get_tokenizer(self.model_name, self.text_encoder_name)

        try:
            from open_clip.factory import get_tokenizer
            return get_tokenizer(self.model_name, self.text_encoder_name)
        except Exception:
            pass

        try:
            from open_clip.tokenizer import HFTokenizer
            return HFTokenizer(self.text_encoder_name)
        except Exception as e:
            raise RuntimeError(
                "Could not create UniMed-CLIP tokenizer. "
                "Tried open_clip.get_tokenizer, open_clip.factory.get_tokenizer, "
                "and open_clip.tokenizer.HFTokenizer."
            ) from e

    def load(self):
        src_path = f"{self.repo_path}/src"

        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.weight_path,
            device=self.device,
            text_encoder_name=self.text_encoder_name,
        )

        self.tokenizer = self._load_tokenizer(open_clip)
        self.model.eval()

        print(f"[UniMedCLIPWrapper] loaded model: {self.model_name}")
        print(f"[UniMedCLIPWrapper] text encoder: {self.text_encoder_name}")
        print(f"[UniMedCLIPWrapper] weights: {self.weight_path}")
        print(f"[UniMedCLIPWrapper] tokenizer: {type(self.tokenizer)}")

    @torch.no_grad()
    def encode_image(self, image):
        if self.model is None:
            self.load()

        x = self.preprocess(image).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def encode_text(self, text):
        if self.model is None:
            self.load()

        tokens = self.tokenizer([text])

        if isinstance(tokens, dict):
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
        else:
            tokens = tokens.to(self.device)

        feat = self.model.encode_text(tokens)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def similarity(self, image, text):
        image_feat = self.encode_image(image)
        text_feat = self.encode_text(text)
        return float((image_feat @ text_feat.T).item())
