from defense.segmentation import ForegroundSegmenter
from defense.clinical_awareness import clinical_awareness_score
from defense.clip_grounding import ClipGroundingMonitor
from defense.prompts import build_evidence_first_prompt, build_defended_regen_prompt
from defense.causal_patch import CausalPatchTester
from defense.prompt_router import PromptRouter


class MedFocusGuard:
    """
    MedFocusGuard:
    Foreground-grounded causal defense for MedFocusLeak-style attacks.

    Gate 1:
    R0 = lambda_bg * B_CLIP
       + lambda_evid * (1 - S_evid)
       + lambda_overconf * C_overconf
       + lambda_mismatch * C_mismatch
       + lambda_delay * D_delay
       + lambda_align * D_align

    Gate 2:
    R_bg_causal = C_bg / (C_bg + C_fg + eps)
    """

    def __init__(
        self,
        clip_model,
        victim_generate=None,
        tau0=0.55,
        tauc=0.50,

        lambda_bg=0.32,
        lambda_align=0.22,
        lambda_delay=0.18,
        lambda_mismatch=0.12,
        lambda_overconf=0.08,
        lambda_evid=0.08,

        top_k=3,
    ):
        self.clip_model = clip_model
        self.victim_generate = victim_generate

        self.tau0 = tau0
        self.tauc = tauc

        self.lambda_bg = lambda_bg
        self.lambda_align = lambda_align
        self.lambda_delay = lambda_delay
        self.lambda_mismatch = lambda_mismatch
        self.lambda_overconf = lambda_overconf
        self.lambda_evid = lambda_evid

        self.segmenter = ForegroundSegmenter()
        self.grounder = ClipGroundingMonitor(clip_model=clip_model)
        self.prompt_router = PromptRouter()

        if victim_generate is not None:
            self.causal_tester = CausalPatchTester(
                victim_generate=victim_generate,
                prompt_builder=build_evidence_first_prompt,
                top_k=top_k,
            )
        else:
            self.causal_tester = None

    def build_risk_vector(self, clinical_scores, grounding_scores):
        return {
            "B_CLIP": float(grounding_scores["B_CLIP"]),
            "S_evid": float(clinical_scores["S_evid"]),
            "C_overconf": float(clinical_scores["C_overconf"]),
            "C_mismatch": float(clinical_scores["C_mismatch"]),
            "D_delay": float(clinical_scores["D_delay"]),
            "D_align": float(grounding_scores["D_align"]),
        }

    def gate1_risk(self, q):
        r0 = (
            self.lambda_bg * q["B_CLIP"]
            + self.lambda_evid * (1.0 - q["S_evid"])
            + self.lambda_overconf * q["C_overconf"]
            + self.lambda_mismatch * q["C_mismatch"]
            + self.lambda_delay * q["D_delay"]
            + self.lambda_align * q["D_align"]
        )

        return float(max(0.0, min(1.0, r0)))

    def build_routed_defense_prompt(self, user_prompt, prompt_route):
        base_prompt = build_defended_regen_prompt(user_prompt)
        routed = prompt_route.get("routed_prompt", "")

        return f"""
{base_prompt}

Additional routed safety instructions selected by MedFocusGuard:
{routed}
""".strip()

    def run_gate1_with_existing_draft(self, image, evidence_text, y0=None):
        masks = self.segmenter.segment(image)

        clinical = clinical_awareness_score(evidence_text)

        grounding = self.grounder.compute(
            image=image,
            evidence_text=evidence_text,
            m_fg=masks["m_fg"],
            m_bg=masks["m_bg"],
        )

        q = self.build_risk_vector(
            clinical_scores=clinical,
            grounding_scores=grounding,
        )

        r0 = self.gate1_risk(q)
        routed = self.prompt_router.route(q)

        decision = "low_risk_accept" if r0 <= self.tau0 else "suspicious_run_gate2"

        return {
            "decision": decision,
            "R0": r0,
            "q": q,
            "clinical": clinical,
            "grounding": grounding,
            "prompt_route": routed,
            "masks": masks,
            "y0": y0,
            "evidence_text": evidence_text,
        }

    def run_full(self, image, user_prompt):
        if self.victim_generate is None:
            raise ValueError("run_full requires victim_generate(image, prompt).")

        evidence_prompt = build_evidence_first_prompt(user_prompt)
        draft = self.victim_generate(image, evidence_prompt)

        gate1 = self.run_gate1_with_existing_draft(
            image=image,
            evidence_text=draft,
            y0=draft,
        )

        if gate1["R0"] <= self.tau0:
            gate1["final_action"] = "accept_y0"
            gate1["final_answer"] = draft
            return gate1

        if self.causal_tester is None:
            gate1["final_action"] = "gate2_needed_but_no_victim_tester"
            gate1["final_answer"] = draft
            gate1["defense_prompt_preview"] = self.build_routed_defense_prompt(
                user_prompt=user_prompt,
                prompt_route=gate1["prompt_route"],
            )
            return gate1

        gate2 = self.causal_tester.run(
            image=image,
            user_prompt=user_prompt,
            y0=draft,
            patch_rows=gate1["grounding"]["patch_rows"],
        )

        gate1["gate2"] = gate2

        if gate2["R_bg_causal"] <= self.tauc:
            gate1["final_action"] = "accept_cautiously"
            gate1["final_answer"] = draft
            return gate1

        defended_prompt = self.build_routed_defense_prompt(
            user_prompt=user_prompt,
            prompt_route=gate1["prompt_route"],
        )

        defended = self.victim_generate(image, defended_prompt)

        gate1["final_action"] = "defended_regeneration"
        gate1["final_answer"] = defended
        gate1["defense_prompt_used"] = defended_prompt

        return gate1
