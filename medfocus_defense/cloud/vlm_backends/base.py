from abc import ABC, abstractmethod


class VLMBackend(ABC):
    @abstractmethod
    def generate(self, image_path: str, prompt: str) -> str:
        raise NotImplementedError
