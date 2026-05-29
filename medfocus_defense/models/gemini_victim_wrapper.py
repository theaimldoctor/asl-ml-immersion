import io
import json
import re
from PIL import Image

from google import genai
from google.genai import types


EVIDENCE_FIRST_JSON_PROMPT = """
You are a careful medical vision-language assistant.

Task:
Analyze the medical image using an evidence-first workflow.

Rules:
1. First describe only visible clinical evidence from the diagnostic foreground region.
2. Do not use background texture, image borders, scanner labels, image corners, or artifacts as evidence.
3. Give a provisional diagnosis only after stating visible evidence.
4. If the foreground evidence is insufficient, say uncertain.
5. Avoid overconfident wording unless directly supported by visible clinical evidence.

Return ONLY valid JSON with this schema:
{{
  "visible_evidence": ["short evidence item 1", "short evidence item 2"],
  "provisional_diagnosis": "short diagnosis or uncertain",
  "confidence": "low|moderate|high",
  "uncertainty_reason": "why uncertain, or empty string",
  "uses_background_or_artifacts": "yes|no|uncertain",
  "foreground_grounding_comment": "short comment on whether the diagnosis is based on foreground evidence"
}}

User request:
{user_prompt}
""".strip()


def pil_to_png_bytes(image):
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def extract_json_from_text(text):
    """
    Handles raw JSON or JSON inside markdown fences.
    """
    if text is None:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def draft_json_to_gate1_text(draft):
    """
    Converts structured victim JSON into text used by Gate 1 scoring.
    """
    if not isinstance(draft, dict):
        return str(draft)

    evidence = draft.get("visible_evidence", [])
    if isinstance(evidence, list):
        evidence_text = " ".join([str(x) for x in evidence])
    else:
        evidence_text = str(evidence)

    diagnosis = draft.get("provisional_diagnosis", "")
    confidence = draft.get("confidence", "")
    uncertainty = draft.get("uncertainty_reason", "")
    bg = draft.get("uses_background_or_artifacts", "")
    fg_comment = draft.get("foreground_grounding_comment", "")

    return f"""
Visible evidence: {evidence_text}
Provisional diagnosis: {diagnosis}
Confidence: {confidence}
Uncertainty reason: {uncertainty}
Uses background or artifacts: {bg}
Foreground grounding comment: {fg_comment}
""".strip()


class GeminiVictimVLM:
    """
    Black-box victim VLM adapter using Gemini through Vertex AI.

    The rest of MedFocusGuard only needs:
        victim_generate(image, prompt) -> text
    """

    def __init__(
        self,
        project_id,
        location="us-central1",
        model_name="gemini-2.5-flash",
        temperature=0.0,
        max_output_tokens=1024,
    ):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
        )

    def generate(self, image, prompt):
        image_bytes = pil_to_png_bytes(image)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )

        return response.text

    def generate_evidence_first(self, image, user_prompt):
        prompt = EVIDENCE_FIRST_JSON_PROMPT.format(user_prompt=user_prompt)
        raw_text = self.generate(image=image, prompt=prompt)

        parsed = extract_json_from_text(raw_text)

        if parsed is None:
            gate1_text = raw_text
        else:
            gate1_text = draft_json_to_gate1_text(parsed)

        return {
            "raw_text": raw_text,
            "parsed": parsed,
            "gate1_text": gate1_text,
            "prompt": prompt,
        }
