"""
Giskard Node — runs adversarial vulnerability probes via giskard.scan().

Only activated when EvalJobConfig.adversarial=True.
If giskard is not installed, returns a result with giskard_available=False
and a GISKARD_NOT_AVAILABLE warning — does not raise.

TODO: Map Giskard scan categories to the LumisEval adversarial probe enum.
"""

from typing import Optional
import giskard  # noqa: F401
from lumiseval_core.types import GiskardScanResult, GiskardVulnerability, Severity

from lumiseval_agent.log import get_node_logger

log = get_node_logger("giskard")

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

    # try:
    
    # except ImportError:
    #     log.warning("Giskard not installed — adversarial coverage reduced. Install: uv add giskard")
    #     return GiskardScanResult(
    #         giskard_available=False,
    #         error="GISKARD_NOT_AVAILABLE",
    #     )

    log.info(f"Probing categories: {probe_categories}")
    # try:

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
    log.info(f"Scan complete — {len(issues)} issue(s) found")

    vulnerabilities = []
    for issue in issues:
        vuln = GiskardVulnerability(
            probe_type=str(getattr(issue, "group", "unknown")),
            severity=_map_severity(getattr(issue, "level", "MEDIUM")),
            description=str(getattr(issue, "description", "")),
            reproduction_details=str(getattr(issue, "examples", "")),
        )
        vulnerabilities.append(vuln)
        log.info(f"  [{vuln.severity.value}] {vuln.probe_type}: {vuln.description[:80]}")

    return GiskardScanResult(vulnerabilities=vulnerabilities)

    # except Exception as exc:
    #     log.error(f"Giskard scan failed: {exc}")
    #     return GiskardScanResult(error=str(exc))


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
