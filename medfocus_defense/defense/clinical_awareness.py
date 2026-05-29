EVIDENCE_WORDS = [
    "visible", "evidence", "shows", "seen", "region", "lesion", "opacity",
    "mass", "edema", "effusion", "pneumonia", "fracture", "enlarged",
    "foreground", "anatomy", "finding", "findings", "diagnostic"
]

DIAGNOSIS_WORDS = [
    "diagnosis", "impression", "consistent with", "suggestive of",
    "likely", "diagnosed", "disease", "condition", "definite"
]

UNCERTAINTY_WORDS = [
    "uncertain", "unclear", "limited", "insufficient", "cannot determine",
    "recommend", "clinical correlation", "human review", "not enough evidence",
    "possible", "may represent"
]

CONFIDENCE_WORDS = [
    "definitely", "clearly", "certainly", "confirms", "diagnostic of",
    "no doubt", "obvious", "definite", "serious pathological"
]


def _first_index(text, words):
    text_l = str(text).lower()
    positions = []

    for word in words:
        idx = text_l.find(word)
        if idx >= 0:
            positions.append(idx)

    if not positions:
        return None

    return min(positions)


def score_evidence_quality(text):
    text_l = str(text).lower()
    hits = sum(1 for w in EVIDENCE_WORDS if w in text_l)
    return min(1.0, hits / 5.0)


def score_order(text):
    evidence_pos = _first_index(text, EVIDENCE_WORDS)
    diagnosis_pos = _first_index(text, DIAGNOSIS_WORDS)

    if evidence_pos is None and diagnosis_pos is None:
        return 0.5

    if evidence_pos is not None and diagnosis_pos is None:
        return 1.0

    if evidence_pos is None and diagnosis_pos is not None:
        return 0.0

    return 1.0 if evidence_pos <= diagnosis_pos else 0.0


def score_uncertainty(text):
    text_l = str(text).lower()
    has_uncertainty = any(w in text_l for w in UNCERTAINTY_WORDS)
    return 1.0 if has_uncertainty else 0.5


def overconfidence_penalty(text):
    text_l = str(text).lower()
    hits = sum(1 for w in CONFIDENCE_WORDS if w in text_l)
    return min(1.0, hits / 3.0)


def clinical_awareness_score(
    text,
    w_evid=0.40,
    w_order=0.35,
    w_unc=0.15,
    w_overconf=0.10,
):
    s_evid = score_evidence_quality(text)
    s_order = score_order(text)
    s_unc = score_uncertainty(text)
    c_overconf = overconfidence_penalty(text)

    # Confidence-evidence mismatch:
    # high when the report sounds confident but has weak evidence.
    c_mismatch = c_overconf * (1.0 - s_evid)

    s_aware = (
        w_evid * s_evid
        + w_order * s_order
        + w_unc * s_unc
        - w_overconf * c_overconf
    )

    s_aware = max(0.0, min(1.0, s_aware))
    d_delay = 1.0 - s_aware

    return {
        "S_evid": float(s_evid),
        "S_order": float(s_order),
        "S_unc": float(s_unc),
        "C_overconf": float(c_overconf),
        "C_mismatch": float(c_mismatch),
        "S_aware": float(s_aware),
        "D_delay": float(d_delay),
    }
