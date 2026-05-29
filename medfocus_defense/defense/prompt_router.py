ROUTER_INSTRUCTION_BANK = {
    "background_ignore": """
Background-grounding risk is high.
Strictly ignore image background, borders, corners, scanner labels, artifacts, and non-diagnostic regions.
Do not let background texture influence the diagnosis.
""".strip(),

    "foreground_evidence": """
Clinical evidence is weak or missing.
Before any diagnosis, explicitly describe visible diagnostic foreground anatomy and findings.
If no clear foreground evidence supports a diagnosis, say so.
""".strip(),

    "overconfidence_control": """
Overconfidence or confidence-evidence mismatch risk is high.
Remove definitive wording unless directly supported by visible foreground clinical evidence.
Prefer cautious wording such as "suggestive of", "possible", or "uncertain" when evidence is limited.
""".strip(),

    "delayed_awareness_control": """
Delayed clinical-awareness risk is high.
Do not diagnose first and justify later.
Use this order only: foreground evidence → provisional diagnosis → confidence or uncertainty.
""".strip(),

    "alignment_check": """
Foreground image-text alignment risk is high.
Check whether each stated finding is visually supported by the diagnostic foreground.
Delete or weaken any claim that is not grounded in the foreground.
""".strip(),
}


class PromptRouter:
    """
    AMPT-inspired rule-based prompt router.

    q = {
        B_CLIP,
        S_evid,
        C_overconf,
        C_mismatch,
        D_delay,
        D_align
    }
    """

    def __init__(
        self,
        bg_threshold=0.45,
        weak_evidence_threshold=0.35,
        overconf_threshold=0.20,
        delay_threshold=0.35,
        align_threshold=0.35,
    ):
        self.bg_threshold = bg_threshold
        self.weak_evidence_threshold = weak_evidence_threshold
        self.overconf_threshold = overconf_threshold
        self.delay_threshold = delay_threshold
        self.align_threshold = align_threshold

    def route(self, q):
        selected = []
        weights = {}

        b_clip = float(q.get("B_CLIP", 0.0))
        s_evid = float(q.get("S_evid", 1.0))
        weak_evidence = 1.0 - s_evid
        c_overconf = float(q.get("C_overconf", 0.0))
        c_mismatch = float(q.get("C_mismatch", c_overconf * weak_evidence))
        d_delay = float(q.get("D_delay", 0.0))
        d_align = float(q.get("D_align", 0.0))

        if b_clip >= self.bg_threshold:
            selected.append("background_ignore")
            weights["background_ignore"] = b_clip

        if weak_evidence >= self.weak_evidence_threshold:
            selected.append("foreground_evidence")
            weights["foreground_evidence"] = weak_evidence

        if c_overconf >= self.overconf_threshold or c_mismatch >= self.overconf_threshold:
            selected.append("overconfidence_control")
            weights["overconfidence_control"] = max(c_overconf, c_mismatch)

        if d_delay >= self.delay_threshold:
            selected.append("delayed_awareness_control")
            weights["delayed_awareness_control"] = d_delay

        if d_align >= self.align_threshold:
            selected.append("alignment_check")
            weights["alignment_check"] = d_align

        if not selected:
            selected = ["foreground_evidence"]
            weights["foreground_evidence"] = 1.0

        total = sum(weights.values()) + 1e-8
        prompt_weights = {k: float(v / total) for k, v in weights.items()}

        routed_text = "\n\n".join(
            ROUTER_INSTRUCTION_BANK[name] for name in selected
        )

        return {
            "selected_prompts": selected,
            "prompt_weights": prompt_weights,
            "routed_prompt": routed_text,
            "routed_instructions": routed_text,
        }
