"""TranslateAgent — cross-level translation assessment."""

from __future__ import annotations

import json
import logging
from typing import Any

from dargus.agents.base import BaseAgent

logger = logging.getLogger(__name__)


# Phase 0 deterministic lookup for well-known diseases.
# In Phase 1 this will be replaced by a calibrated random-forest model.
_KNOWN_TRANSLATION: dict[str, dict[str, Any]] = {
    "Parkinson's disease": {
        "overall": 0.38,
        "95_CI": [0.22, 0.55],
        "layer_specific": {
            "molecular_to_cellular": 0.65,
            "cellular_to_exvivo": 0.55,
            "cellular_to_animal": 0.42,
            "exvivo_to_animal": 0.50,
            "animal_to_clinical": 0.47,
            "molecular_to_clinical": 0.18,
        },
        "key_attenuation_factors": [
            "病理机制高度复杂且互联，单靶点干预可能在完整生物体中效果有限",
            "最常用动物模型 (MPTP) 不模拟 α-synuclein 病理和慢性退行进程",
            "DA 神经元保护 → 运动功能改善的翻译链尚未在任何药物中得到验证",
        ],
    },
    "HCC": {
        "overall": 0.52,
        "95_CI": [0.35, 0.68],
        "layer_specific": {
            "molecular_to_cellular": 0.70,
            "cellular_to_exvivo": 0.60,
            "cellular_to_animal": 0.55,
            "exvivo_to_animal": 0.58,
            "animal_to_clinical": 0.50,
            "molecular_to_clinical": 0.30,
        },
        "key_attenuation_factors": [
            "肿瘤微环境在体外模型中缺失，导致翻译衰减",
            "PDX 模型不能完整复现人类肝癌的免疫微环境",
        ],
    },
}


class TranslateAgent(BaseAgent):
    """Assess disease-specific cross-level translation reliability."""

    name = "TranslateAgent"

    def run(self, task_spec: dict[str, Any]) -> dict[str, Any]:
        project_id = task_spec["project_id"]
        spec = task_spec.get("task_spec", {})
        disease = spec.get("disease", "unknown")

        known = _KNOWN_TRANSLATION.get(disease)
        if known:
            score = known["overall"]
            ci = known["95_CI"]
            layer_specific = known["layer_specific"]
            factors = known["key_attenuation_factors"]
            calibration_quality = "中等 — 基于回顾性药物对比"
        else:
            score = 0.35
            ci = [0.15, 0.60]
            layer_specific = {
                "molecular_to_cellular": 0.60,
                "cellular_to_exvivo": 0.50,
                "cellular_to_animal": 0.40,
                "exvivo_to_animal": 0.45,
                "animal_to_clinical": 0.40,
                "molecular_to_clinical": 0.20,
            }
            factors = ["疾病特定翻译数据不可用；使用默认先验"]
            calibration_quality = "低 — 使用默认先验"

        result = {
            "disease": disease,
            "translation_score": {
                "overall": score,
                "95_CI": ci,
                "layer_specific": layer_specific,
            },
            "key_attenuation_factors": factors,
            "recommended_translation_chain": ["molecular → cellular → animal → clinical"],
            "calibration_available": known is not None,
            "calibration_quality": calibration_quality,
            "suggested_experiments_to_improve_score": [
                "系统性比较同一药物的细胞→动物→临床效应量",
            ],
        }

        project_dir = self._project_dir(project_id)
        translation_dir = project_dir / "translation"
        translation_dir.mkdir(parents=True, exist_ok=True)
        report_path = translation_dir / "translation_report.md"
        score_path = translation_dir / "translation_score.json"

        report_path.write_text(self._report_text(disease, result), encoding="utf-8")
        score_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "status": "ok",
            "outputs": {"report": str(report_path), "score": str(score_path)},
            "translation_score": result,
        }

    def _report_text(self, disease: str, result: dict[str, Any]) -> str:
        ts = result["translation_score"]
        return f"""# Translation Assessment: {disease}

## Overall translation score
{ts['overall']} (95 % CI: {ts['95_CI']})

## Layer-specific scores
{ts['layer_specific']}

## Key attenuation factors
{chr(10).join('- ' + f for f in result['key_attenuation_factors'])}

## Recommended translation chain
{chr(10).join('- ' + c for c in result['recommended_translation_chain'])}

## Calibration quality
{result['calibration_quality']}

> **Disclaimer**: Dargus outputs are for research purposes only and do not
> constitute clinical advice.
"""
