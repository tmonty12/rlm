"""
RLM Orchestrator HTTP server for Dynamo deployment.

FastAPI wrapper around RLM.completion() that receives requests via HTTP and
delegates LLM inference to the Frontend service and code execution to the
SandboxPool service, both reachable via K8s DNS.
"""

import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rlm import RLM
from rlm.logger import RLMLogger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

INFERENCE_URL = os.environ.get("RLM_INFERENCE_URL", "http://rlm-frontend:8000/v1")
MODEL_NAME = os.environ.get("RLM_MODEL_NAME", "Qwen/Qwen3-0.6B")
SANDBOX_URL = os.environ.get("RLM_SANDBOX_URL", "http://rlm-sandboxpool:9090")
MAX_ITERATIONS = int(os.environ.get("RLM_MAX_ITERATIONS", "30"))
LOG_ENABLED = os.environ.get("RLM_LOG_ENABLED", "false").lower() == "true"
LOG_DIR = os.environ.get("RLM_LOG_DIR", "./logs")

# ---------------------------------------------------------------------------
# RLM instance
# ---------------------------------------------------------------------------

# Create logger if enabled
rlm_logger = None
if LOG_ENABLED:
    logger.info(f"RLM logging enabled. Writing to {LOG_DIR}")
    rlm_logger = RLMLogger(log_dir=LOG_DIR, file_name="rlm")

rlm_instance = RLM(
    backend="vllm",
    backend_kwargs={
        "model_name": MODEL_NAME,
        "base_url": INFERENCE_URL,
        "api_key": "not-needed",
    },
    environment="kubernetes",
    environment_kwargs={
        "sandbox_url": SANDBOX_URL,
        "inference_url": INFERENCE_URL,
        "model_name": MODEL_NAME,
    },
    max_iterations=MAX_ITERATIONS,
    logger=rlm_logger,
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="RLM Orchestrator")


class CompletionRequest(BaseModel):
    prompt: str
    root_prompt: str | None = None


class CompletionResponse(BaseModel):
    response: str
    root_model: str
    execution_time: float


@app.post("/completion")
def completion(request: CompletionRequest) -> CompletionResponse:
    try:
        result = rlm_instance.completion(
            prompt=request.prompt,
            root_prompt=request.root_prompt,
        )
        return CompletionResponse(
            response=result.response,
            root_model=result.root_model,
            execution_time=result.execution_time,
        )
    except Exception as e:
        logger.exception("Completion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
