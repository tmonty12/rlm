"""
RLM Sandbox Pool HTTP server for Dynamo deployment.

Manages sandboxed code execution sessions. Each session maintains in-memory
Python state (globals/locals) similar to LocalREPL. Code executed in a session
can call llm_query() which makes HTTP requests directly to the Frontend
service over the K8s network — no broker pattern needed.

Endpoints:
    POST   /sessions                     → create session, return session_id
    POST   /sessions/{id}/context        → load context into session namespace
    POST   /sessions/{id}/execute        → exec(code) in session, return stdout/stderr/locals
    DELETE /sessions/{id}                → cleanup session
    GET    /health                       → health check
"""

import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from typing import Any

import requests as http_requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe builtins — mirrors rlm/environments/local_repl.py
# ---------------------------------------------------------------------------

_SAFE_BUILTINS = {
    # Core types and functions
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "__import__": __import__,
    "open": open,
    # Exceptions
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    # Blocked
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}


# ---------------------------------------------------------------------------
# Serialization helper — mirrors rlm/core/types.py::_serialize_value
# ---------------------------------------------------------------------------

def _serialize_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if callable(value):
        return f"<{type(value).__name__} '{getattr(value, '__name__', repr(value))}'>"
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


# ---------------------------------------------------------------------------
# Session — one per RLM completion(), holds in-memory Python state
# ---------------------------------------------------------------------------

class Session:
    """Sandboxed Python execution session with persistent namespace."""

    def __init__(self, session_id: str, inference_url: str, model_name: str):
        self.session_id = session_id
        self.inference_url = inference_url.rstrip("/")
        self.model_name = model_name
        self.temp_dir = tempfile.mkdtemp(prefix=f"sandbox_{session_id}_")
        self._lock = threading.Lock()

        # Sandboxed namespace
        self.globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }
        self.locals: dict[str, Any] = {}
        self._context_count = 0

        # Inject helper functions
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["llm_query"] = self._llm_query

    # -- Helper functions injected into the sandbox namespace ---------------

    def _final_var(self, variable_name: str) -> str:
        variable_name = variable_name.strip().strip("\"'")
        if variable_name in self.locals:
            return str(self.locals[variable_name])
        available = [k for k in self.locals.keys() if not k.startswith("_")]
        if available:
            return (
                f"Error: Variable '{variable_name}' not found. "
                f"Available variables: {available}. "
                f"You must create and assign a variable BEFORE calling FINAL_VAR on it."
            )
        return (
            f"Error: Variable '{variable_name}' not found. "
            f"No variables have been created yet. "
            f"You must create and assign a variable in a REPL block BEFORE calling FINAL_VAR on it."
        )

    def _show_vars(self) -> str:
        available = {k: type(v).__name__ for k, v in self.locals.items() if not k.startswith("_")}
        if not available:
            return "No variables created yet. Use ```repl``` blocks to create variables."
        return f"Available variables: {available}"

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        """Query the LLM via HTTP to the Frontend service."""
        try:
            resp = http_requests.post(
                f"{self.inference_url}/chat/completions",
                json={
                    "model": model or self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: LM query failed - {e}"

    # -- Context loading ----------------------------------------------------

    def load_context(self, payload: dict | list | str):
        """Load a context payload into the session namespace."""
        idx = self._context_count
        var_name = f"context_{idx}"

        if isinstance(payload, str):
            path = os.path.join(self.temp_dir, f"context_{idx}.txt")
            with open(path, "w") as f:
                f.write(payload)
            self.execute(f"with open(r'{path}', 'r') as f:\n    {var_name} = f.read()")
        else:
            path = os.path.join(self.temp_dir, f"context_{idx}.json")
            with open(path, "w") as f:
                json.dump(payload, f)
            self.execute(
                f"import json\nwith open(r'{path}', 'r') as f:\n    {var_name} = json.load(f)"
            )

        if idx == 0:
            self.execute(f"context = {var_name}")

        self._context_count = idx + 1

    # -- Code execution -----------------------------------------------------

    def execute(self, code: str) -> dict:
        """Execute code in the session namespace and return results."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            old_cwd = os.getcwd()

            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                os.chdir(self.temp_dir)

                combined = {**self.globals, **self.locals}
                exec(code, combined, combined)

                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value

                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()
            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
                os.chdir(old_cwd)

        serialized_locals = {
            k: _serialize_value(v) for k, v in self.locals.items() if not k.startswith("_")
        }

        return {
            "stdout": stdout,
            "stderr": stderr,
            "locals": serialized_locals,
        }

    # -- Cleanup ------------------------------------------------------------

    def cleanup(self):
        import shutil

        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        self.globals.clear()
        self.locals.clear()


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

_sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="RLM Sandbox Pool")


class CreateSessionRequest(BaseModel):
    inference_url: str
    model_name: str


class CreateSessionResponse(BaseModel):
    session_id: str


class ContextRequest(BaseModel):
    payload: dict | list | str


class ExecuteRequest(BaseModel):
    code: str


class ExecuteResponse(BaseModel):
    stdout: str
    stderr: str
    locals: dict


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(request: CreateSessionRequest):
    session_id = str(uuid.uuid4())
    session = Session(
        session_id=session_id,
        inference_url=request.inference_url,
        model_name=request.model_name,
    )
    with _sessions_lock:
        _sessions[session_id] = session
    logger.info(f"Created session {session_id}")
    return CreateSessionResponse(session_id=session_id)


def _get_session(session_id: str) -> Session:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session


@app.post("/sessions/{session_id}/context")
def load_context(session_id: str, request: ContextRequest):
    session = _get_session(session_id)
    session.load_context(request.payload)
    return {"status": "ok"}


@app.post("/sessions/{session_id}/execute", response_model=ExecuteResponse)
def execute_code(session_id: str, request: ExecuteRequest):
    session = _get_session(session_id)
    result = session.execute(request.code)
    return ExecuteResponse(**result)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    session.cleanup()
    logger.info(f"Deleted session {session_id}")
    return {"status": "ok"}


@app.get("/health")
def health():
    with _sessions_lock:
        count = len(_sessions)
    return {"status": "ok", "active_sessions": count}
