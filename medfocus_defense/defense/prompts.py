EVIDENCE_FIRST_PROMPT = """
First list visible clinical evidence from the diagnostic foreground region.
Then provide a provisional diagnosis.
Do not rely on image borders, background texture, scanner labels, corners, artifacts, or non-diagnostic regions.
If foreground evidence is insufficient, state uncertainty.

User request:
{user_prompt}
""".strip()


DEFENDED_REGEN_PROMPT = """
Regenerate the report using only clinically relevant foreground anatomy and visible pathological evidence.
Ignore background texture, borders, scanner labels, image corners, artifacts, and non-diagnostic regions.
If foreground evidence is insufficient or inconsistent, state uncertainty and recommend human review.

User request:
{user_prompt}
""".strip()


# These are the actual defense prompt blocks used by PromptRouter.
# They are short, composable, and aligned with MedFocusGuard.
DEFENSE_PROMPT_BANK = {
    "background_suppression": """
Ignore non-diagnostic background regions, borders, scanner labels, corners, image artifacts, and background texture.
Do not use background appearance as evidence for diagnosis.
""".strip(),

    "foreground_grounding": """
Base the report only on visible clinical foreground anatomy and foreground pathological evidence.
Explicitly state the foreground evidence before giving any diagnosis.
""".strip(),

    "evidence_strength_check": """
If the visible foreground evidence is weak, incomplete, ambiguous, or not localizable, state uncertainty.
Do not force a diagnosis when the foreground evidence is insufficient.
""".strip(),

    "overconfidence_control": """
Avoid definitive language unless the diagnosis is directly supported by foreground evidence.
Use cautious clinical wording when evidence is limited.
""".strip(),

    "foreground_alignment_check": """
Check whether each stated finding is visually supported by the diagnostic foreground.
Remove or soften any finding that is not grounded in the foreground region.
""".strip(),

    "human_review_trigger": """
If the image appears corrupted, adversarially manipulated, or inconsistent with the stated diagnosis, recommend human review.
""".strip(),
}


def build_evidence_first_prompt(user_prompt):
    return EVIDENCE_FIRST_PROMPT.format(user_prompt=user_prompt)


def build_defended_regen_prompt(user_prompt):
    return DEFENDED_REGEN_PROMPT.format(user_prompt=user_prompt)


def build_routed_defense_prompt(user_prompt, routed_prompt):
    return f"""
{DEFENDED_REGEN_PROMPT.format(user_prompt=user_prompt)}

Additional routed defense instructions:
{routed_prompt}
""".strip()
