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
from typing import Union

from fastapi import FastAPI
from lumiseval_core.types import EvalJobConfig, EvalReport, Rubric
from lumiseval_graph.graph import run_graph
from pydantic import BaseModel

app = FastAPI(
    title="LumisEval API",
    description="Agentic LLM evaluation pipeline — REST interface",
    version="0.1.0",
)


class EvalJobRequest(BaseModel):
    generation: str
    question: str | None = None
    ground_truth: str | None = None
    context: list[str] = []
    rubric: list[Rubric] = []
    reference_files: list[str] = []
    judge_model: str = "gpt-4o-mini"
    web_search: bool = False
    enable_hallucination: bool = True
    enable_faithfulness: bool = True
    enable_answer_relevancy: bool = True
    enable_adversarial: bool = False
    enable_rubric: bool = False
    evidence_threshold: float = 0.75
    budget_cap_usd: float | None = None
    acknowledge_cost: bool = False


def _run_one(request: EvalJobRequest) -> EvalReport:
    job_id = str(uuid.uuid4())
    job_config = EvalJobConfig(
        job_id=job_id,
        judge_model=request.judge_model,
        enable_hallucination=request.enable_hallucination,
        enable_faithfulness=request.enable_faithfulness,
        enable_answer_relevancy=request.enable_answer_relevancy,
        enable_adversarial=request.enable_adversarial,
        enable_rubric=request.enable_rubric,
        web_search=request.web_search,
        evidence_threshold=request.evidence_threshold,
        budget_cap_usd=request.budget_cap_usd,
    )

    report = run_graph(
        generation=request.generation,
        job_config=job_config,
        question=request.question,
        ground_truth=request.ground_truth,
        context=request.context or None,
        rubric=request.rubric or None,
        reference_files=request.reference_files or None,
    )
    return report


@app.post("/jobs", response_model=Union[EvalReport, list[EvalReport]])
def create_job(
    request: Union[EvalJobRequest, list[EvalJobRequest]],
) -> Union[EvalReport, list[EvalReport]]:
    """Create and synchronously execute one or many evaluation jobs.

    Accepts either:
      - a single EvalJobRequest object
      - a JSON array of EvalJobRequest objects

    Set ``acknowledge_cost=true`` in each request payload to indicate pre-run cost acknowledgement.
    """
    if isinstance(request, list):
        return [_run_one(item) for item in request]
    return _run_one(request)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}
