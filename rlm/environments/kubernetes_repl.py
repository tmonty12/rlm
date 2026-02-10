import logging
import time
from typing import Any

import requests

from rlm.core.types import REPLResult
from rlm.environments.base_env import IsolatedEnv

logger = logging.getLogger(__name__)


class KubernetesREPL(IsolatedEnv):
    """
    Executes code in a remote sandbox pod via HTTP.

    Unlike other isolated environments (Modal, E2B) that need a broker pattern for
    LLM calls, in Kubernetes all pods can reach all services via DNS. So sandbox code
    calling llm_query() simply makes an HTTP call to the Frontend service directly.
    """

    def __init__(
        self,
        sandbox_url: str,
        inference_url: str,
        model_name: str,
        context_payload: dict | list | str | None = None,
        persistent: bool = False,
        timeout: int = 300,
        **kwargs,
    ):
        if persistent:
            raise NotImplementedError(
                "Persistent REPLs are currently not supported for environment: KubernetesREPL"
            )
        # Pop lm_handler_address if passed by RLM core -- we don't use it since
        # sandbox code calls the Frontend service directly over the K8s network.
        kwargs.pop("lm_handler_address", None)

        super().__init__(persistent=persistent, **kwargs)

        self.sandbox_url = sandbox_url.rstrip("/")
        self.inference_url = inference_url
        self.model_name = model_name
        self.timeout = timeout
        self.session_id: str | None = None

        self.setup()

        if context_payload is not None:
            self.load_context(context_payload)

    def setup(self):
        """Create a sandbox session on the remote sandbox pool."""
        resp = requests.post(
            f"{self.sandbox_url}/sessions",
            json={
                "inference_url": self.inference_url,
                "model_name": self.model_name,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        self.session_id = resp.json()["session_id"]
        logger.info(f"Created sandbox session: {self.session_id}")

    def load_context(self, context_payload: dict | list | str):
        """Send context to the sandbox session."""
        resp = requests.post(
            f"{self.sandbox_url}/sessions/{self.session_id}/context",
            json={"payload": context_payload},
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def execute_code(self, code: str) -> REPLResult:
        """Execute code in the remote sandbox session and return the result."""
        start_time = time.perf_counter()

        resp = requests.post(
            f"{self.sandbox_url}/sessions/{self.session_id}/execute",
            json={"code": code},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        execution_time = time.perf_counter() - start_time

        return REPLResult(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            locals=data.get("locals", {}),
            execution_time=execution_time,
            rlm_calls=[],
        )

    def cleanup(self):
        """Delete the sandbox session."""
        if self.session_id is not None:
            try:
                requests.delete(
                    f"{self.sandbox_url}/sessions/{self.session_id}",
                    timeout=self.timeout,
                )
                logger.info(f"Cleaned up sandbox session: {self.session_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup sandbox session {self.session_id}: {e}")
            self.session_id = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        self.cleanup()
