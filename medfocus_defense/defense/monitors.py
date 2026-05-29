from defense.perturbation_monitor import PerturbationMonitor
from defense.attention_monitor import AttentionMonitor


class CombinedMonitors:
    def __init__(self, unimed_clip=None, chexone=None):
        self.unimed_clip = unimed_clip
        self.chexone = chexone
        self.perturb = PerturbationMonitor()
        self.attention = AttentionMonitor()

    def compute_all(
        self,
        image,
        report_text,
        background_mask=None,
        attention_map=None,
        clean_image=None,
        thresholds=None,
    ):
        thresholds = thresholds or {}
        result = {}

        result["perturbation_score"] = self.perturb.high_frequency_score(
            image=image,
            background_mask=background_mask,
        )

        if clean_image is not None:
            result["paired_delta_score"] = self.perturb.clean_adv_delta_score(
                clean_image=clean_image,
                adv_image=image,
                background_mask=background_mask,
            )
        else:
            result["paired_delta_score"] = None

        if attention_map is not None and background_mask is not None:
            result["attention"] = self.attention.background_attention_ratio(
                attention_map=attention_map,
                background_mask=background_mask,
            )
            result["background_attention_ratio"] = result["attention"]["background_ratio"]
        else:
            result["attention"] = None
            result["background_attention_ratio"] = None

        if self.unimed_clip is not None:
            try:
                result["clip_similarity"] = self.unimed_clip.similarity(image, report_text)
            except Exception as e:
                result["clip_similarity"] = None
                result["clip_error"] = str(e)
        else:
            result["clip_similarity"] = None

        result["risk_score"] = self.compute_risk_score(result, thresholds)
        return result

    def compute_risk_score(self, monitor_result, thresholds):
        scores = []

        clip_similarity = monitor_result.get("clip_similarity")
        if clip_similarity is not None:
            min_sim = thresholds.get("clip_similarity_min", 0.22)
            scores.append(max(0.0, min(1.0, (min_sim - clip_similarity) / max(min_sim, 1e-8))))

        perturb = monitor_result.get("perturbation_score")
        if perturb is not None:
            max_p = thresholds.get("perturbation_score_max", 0.30)
            scores.append(max(0.0, min(1.0, perturb / max(max_p, 1e-8))))

        bg_ratio = monitor_result.get("background_attention_ratio")
        if bg_ratio is not None:
            max_bg = thresholds.get("attention_background_max", 0.45)
            scores.append(max(0.0, min(1.0, bg_ratio / max(max_bg, 1e-8))))

        paired_delta = monitor_result.get("paired_delta_score")
        if paired_delta is not None:
            scores.append(max(0.0, min(1.0, paired_delta / 0.05)))

        if not scores:
            return 0.0

        return float(sum(scores) / len(scores))
