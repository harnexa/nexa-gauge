"""
LumisEval REST API — FastAPI application.

Endpoints:
  POST /jobs          — create and run an evaluation job
  GET  /jobs/{id}     — get job status (TODO: async job queue)
  GET  /jobs/{id}/report — retrieve the EvalReport

TODO:
  - Async job submission via TaskIQ for long-running batch evaluations.
  - SQLite persistence layer (SQLModel) for job records and reports.
  - Stream job progress via Server-Sent Events.
"""

import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lumiseval_agent.graph import run_graph
from lumiseval_core.types import EvalJobConfig, EvalReport, RubricRule

app = FastAPI(
    title="LumisEval API",
    description="Agentic LLM evaluation pipeline — REST interface",
    version="0.1.0",
)


class EvalJobRequest(BaseModel):
    generation: str
    question: str | None = None
    ground_truth: str | None = None
    rubric_rules: list[RubricRule] = []
    reference_files: list[str] = []
    judge_model: str = "gpt-4o-mini"
    web_search: bool = False
    adversarial: bool = False
    enable_ragas: bool = True
    enable_deepeval: bool = True
    enable_giskard: bool = False
    enable_rubric_eval: bool = False
    evidence_threshold: float = 0.75
    budget_cap_usd: float | None = None
    acknowledge_cost: bool = False


@app.post("/jobs", response_model=EvalReport)
def create_job(request: EvalJobRequest) -> EvalReport:
    """Create and synchronously execute an evaluation job.

    Set ``acknowledge_cost=true`` to confirm the pre-run cost estimate and proceed.
    """
    job_id = str(uuid.uuid4())
    job_config = EvalJobConfig(
        job_id=job_id,
        judge_model=request.judge_model,
        enable_ragas=request.enable_ragas,
        enable_deepeval=request.enable_deepeval,
        enable_giskard=request.enable_giskard,
        enable_rubric_eval=request.enable_rubric_eval,
        web_search=request.web_search,
        adversarial=request.adversarial,
        evidence_threshold=request.evidence_threshold,
        budget_cap_usd=request.budget_cap_usd,
    )

    try:
        report = run_graph(
            generation=request.generation,
            job_config=job_config,
            question=request.question,
            ground_truth=request.ground_truth,
            rubric_rules=request.rubric_rules or None,
            reference_files=request.reference_files or None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return report


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}
