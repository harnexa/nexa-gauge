"""
Giskard Node — runs adversarial vulnerability probes via giskard.scan().

Only activated when EvalJobConfig.adversarial=True.
If giskard is not installed, returns a result with giskard_available=False
and a GISKARD_NOT_AVAILABLE warning — does not raise.

TODO: Map Giskard scan categories to the LumisEval adversarial probe enum.
"""

import logging
from typing import Optional

from lumiseval_core.types import GiskardScanResult, GiskardVulnerability, Severity

logger = logging.getLogger(__name__)

DEFAULT_PROBE_CATEGORIES = [
    "prompt_injection",
    "pii_leakage",
    "jailbreak",
    "stereotype",
]


def run(
    generation: str,
    probe_categories: Optional[list[str]] = None,
) -> GiskardScanResult:
    """Run Giskard adversarial probes against ``generation``.

    Args:
        generation: The LLM-generated text to test.
        probe_categories: Subset of Giskard vulnerability types to probe.
            Defaults to DEFAULT_PROBE_CATEGORIES.

    Returns:
        GiskardScanResult. If giskard is not installed, returns with
        ``giskard_available=False``.
    """
    probe_categories = probe_categories or DEFAULT_PROBE_CATEGORIES

    try:
        import giskard  # noqa: F401
    except ImportError:
        logger.warning(
            "Giskard is not installed. Adversarial probe coverage is reduced. "
            "Install with: uv add giskard"
        )
        return GiskardScanResult(
            giskard_available=False,
            error="GISKARD_NOT_AVAILABLE",
        )

    try:
        import giskard

        def _predict(df):
            import pandas as pd

            return pd.Series([generation] * len(df))

        model = giskard.Model(
            model=_predict,
            model_type="text_generation",
            name="lumiseval_probe_target",
            description="LumisEval adversarial probe target",
            feature_names=["input"],
        )

        scan_results = giskard.scan(model, only=probe_categories)
        issues = scan_results.issues if hasattr(scan_results, "issues") else []

        vulnerabilities = []
        for issue in issues:
            vulnerabilities.append(
                GiskardVulnerability(
                    probe_type=str(getattr(issue, "group", "unknown")),
                    severity=_map_severity(getattr(issue, "level", "MEDIUM")),
                    description=str(getattr(issue, "description", "")),
                    reproduction_details=str(getattr(issue, "examples", "")),
                )
            )

        return GiskardScanResult(vulnerabilities=vulnerabilities)

    except Exception as exc:
        logger.error("Giskard scan failed: %s", exc)
        return GiskardScanResult(error=str(exc))


def _map_severity(level: str) -> Severity:
    mapping = {
        "low": Severity.LOW,
        "medium": Severity.MEDIUM,
        "high": Severity.HIGH,
        "critical": Severity.CRITICAL,
        "major": Severity.HIGH,
        "minor": Severity.LOW,
    }
    return mapping.get(str(level).lower(), Severity.MEDIUM)
