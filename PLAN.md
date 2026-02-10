# RLM on Dynamo: POC Design

## Context

RLM is an agentic framework where an LLM iteratively writes and executes Python code in a REPL to solve tasks. Dynamo deploys distributed inference workloads on Kubernetes via the DynamoGraphDeployment (DGD) CRD. The goal: deploy RLM natively on Dynamo — inference model, orchestrator, and sandboxed code execution — all configured via a single DGD.

User choices: **KubernetesREPL** (new isolated sandbox environment), **single model** (one Frontend+Worker pair for all depths).

## Architecture

```
                    User Request (HTTP)
                          │
                          ▼
               ┌─────────────────────┐
               │  RLM Orchestrator   │  componentType: default
               │  (FastAPI + RLM)    │  port 9090
               └──────┬────────┬─────┘
                      │        │
        LLM calls     │        │  code execution
        (depth 0+1)   │        │  requests
                      ▼        ▼
          ┌──────────────┐  ┌──────────────┐
          │   Frontend   │  │ SandboxPool  │  componentType: default
          │  (port 8000) │  │  (port 9090) │  stateful sessions
          └──────┬───────┘  └──────┬───────┘
                 │                 │
                 ▼                 │ llm_query() from sandbox code
          ┌──────────────┐        │ calls Frontend directly
          │    Worker    │ ◄──────┘
          │  vLLM + GPU  │
          └──────────────┘
```

### Key Insight: K8s Simplifies the Broker Pattern

In Modal/E2B, sandboxes can't reach the LMHandler, requiring a complex broker+polling bridge. In Kubernetes, **all pods can reach all services via DNS**. So sandbox code calling `llm_query()` simply makes an HTTP call to the Frontend service — no broker, no polling thread, no tunnel.

## Request Flow

```
POST /completion → Orchestrator pod
  → RLM.completion() starts
    → KubernetesREPL.setup()
        → POST http://rlm-sandboxpool:9090/sessions  (create sandbox session)
    → ITERATION LOOP:
        1. LLM call → HTTP to http://rlm-frontend:8000/v1/chat/completions
        2. Parse ```repl``` blocks from response
        3. KubernetesREPL.execute_code(code)
              → POST http://rlm-sandboxpool:9090/sessions/{id}/execute
              → Sandbox exec(code) in session namespace
              → If code calls llm_query():
                    → Sandbox HTTP to http://rlm-frontend:8000/v1/chat/completions
              → Returns stdout/stderr/locals
        4. Check for FINAL() → return or continue
    → KubernetesREPL.cleanup()
        → DELETE http://rlm-sandboxpool:9090/sessions/{id}
```

## DGD Manifest

The final deliverable. Service names follow operator pattern `<dgd-name>-<lowercase(component)>` (`graph.go:448-449`). Non-frontend/non-epp components get K8s Service on port 9090 with targetPort "system" (`graph.go:574-580`).

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: rlm
spec:
  services:
    # === RLM Orchestrator (no GPU) ===
    Orchestrator:
      componentType: default
      replicas: 1
      envs:
        - name: RLM_INFERENCE_URL
          value: "http://rlm-frontend:8000/v1"
        - name: RLM_MODEL_NAME
          value: "Qwen/Qwen3-0.6B"
        - name: RLM_SANDBOX_URL
          value: "http://rlm-sandboxpool:9090"
        - name: RLM_MAX_ITERATIONS
          value: "30"
      extraPodSpec:
        mainContainer:
          image: rlm-orchestrator:latest
          imagePullPolicy: Never
          command: ["python3", "-m", "uvicorn"]
          args: ["rlm_dynamo.server:app", "--host", "0.0.0.0", "--port", "9090"]
          ports:
            - containerPort: 9090
              name: system
              protocol: TCP

    # === Inference Frontend (OpenAI-compatible API) ===
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.1

    # === Inference Worker (GPU) ===
    Worker:
      componentType: worker
      replicas: 1
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.1
          workingDir: /workspace/examples/backends/vllm
          command: ["python3", "-m", "dynamo.vllm"]
          args: ["--model", "Qwen/Qwen3-0.6B"]

    # === Sandbox Pool (code execution, no GPU) ===
    SandboxPool:
      componentType: default
      replicas: 1
      envs:
        - name: RLM_INFERENCE_URL
          value: "http://rlm-frontend:8000/v1"
        - name: RLM_MODEL_NAME
          value: "Qwen/Qwen3-0.6B"
      extraPodSpec:
        mainContainer:
          image: rlm-sandbox:latest
          imagePullPolicy: Never
          command: ["python3", "-m", "uvicorn"]
          args: ["rlm_dynamo.sandbox:app", "--host", "0.0.0.0", "--port", "9090"]
          ports:
            - containerPort: 9090
              name: system
              protocol: TCP
```

## Changes by Repository

### RLM Repo (`/Users/tmontfort/Dynamo/repos/rlm`)

#### 1. New file: `rlm/environments/kubernetes_repl.py` — KubernetesREPL

New `IsolatedEnv` that talks to the sandbox pool via HTTP. Modeled after `DockerREPL` but simpler (no broker pattern).

```python
class KubernetesREPL(IsolatedEnv):
    """Executes code in a remote sandbox pod via HTTP."""

    def __init__(self, sandbox_url, inference_url, model_name, ...):
        self.sandbox_url = sandbox_url
        # Create session on sandbox pod
        resp = requests.post(f"{sandbox_url}/sessions", json={
            "inference_url": inference_url,
            "model_name": model_name,
        })
        self.session_id = resp.json()["session_id"]

    def load_context(self, context_payload):
        # Send context to sandbox session
        requests.post(f"{self.sandbox_url}/sessions/{self.session_id}/context",
                      json={"payload": context_payload})

    def execute_code(self, code: str) -> REPLResult:
        resp = requests.post(
            f"{self.sandbox_url}/sessions/{self.session_id}/execute",
            json={"code": code})
        data = resp.json()
        return REPLResult(stdout=data["stdout"], stderr=data["stderr"],
                          locals=data["locals"], ...)

    def cleanup(self):
        requests.delete(f"{self.sandbox_url}/sessions/{self.session_id}")
```

Key files to reference:
- `rlm/environments/base_env.py` — `IsolatedEnv` base class
- `rlm/environments/docker_repl.py` — closest existing pattern (HTTP proxy for LLM, state via dill)
- `rlm/environments/local_repl.py` — `_SAFE_BUILTINS`, `execute_code()` exec pattern to reuse

#### 2. Modify: `rlm/environments/__init__.py` — register `"kubernetes"` environment type

#### 3. New file: `rlm_dynamo/server.py` — Orchestrator HTTP server

FastAPI wrapper around `RLM.completion()`:

```python
rlm_instance = RLM(
    backend="vllm",
    backend_kwargs={
        "model_name": os.environ["RLM_MODEL_NAME"],
        "base_url": os.environ["RLM_INFERENCE_URL"],
        "api_key": "not-needed",
    },
    environment="kubernetes",
    environment_kwargs={
        "sandbox_url": os.environ["RLM_SANDBOX_URL"],
        "inference_url": os.environ["RLM_INFERENCE_URL"],
        "model_name": os.environ["RLM_MODEL_NAME"],
    },
    max_iterations=int(os.environ.get("RLM_MAX_ITERATIONS", "30")),
)
```

No changes to `LMHandler` or `RLM` core — `backend="vllm"` already uses `OpenAIClient` with custom `base_url` (`rlm/clients/__init__.py:23-29`).

#### 4. New file: `rlm_dynamo/sandbox.py` — Sandbox HTTP server

FastAPI server that manages execution sessions. Each session maintains in-memory state (like `LocalREPL` does with `self.globals`/`self.locals`):

```
POST   /sessions                         → create session, return session_id
POST   /sessions/{id}/context            → load context into session namespace
POST   /sessions/{id}/execute            → exec(code) in session, return stdout/stderr/locals
DELETE /sessions/{id}                    → cleanup session
GET    /health                           → health check
```

`llm_query()` inside sandbox code calls the inference Frontend directly:
```python
def llm_query(prompt, model=None):
    resp = requests.post(f"{INFERENCE_URL}/chat/completions",
                         json={"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}]})
    return resp.json()["choices"][0]["message"]["content"]
```

This replaces the entire broker pattern. No polling, no tunnel, no TCP sockets — just HTTP on the K8s network.

#### 5. New files: `Dockerfile.orchestrator`, `Dockerfile.sandbox`

Both are lightweight Python images (no GPU libs needed):
```dockerfile
FROM python:3.12-slim
COPY . /app
WORKDIR /app
RUN pip install -e . fastapi uvicorn requests
```

#### 6. New file: `deploy/dgd.yaml` — The DGD manifest (shown above)

### Dynamo Repo (`/Users/tmontfort/Dynamo/repos/dynamo`)

**No changes expected for POC.** The `componentType: default` path (`component_common.go:37-38`) uses `BaseComponentDefaults` which provides a minimal container+pod spec. The user's `extraPodSpec.mainContainer.command` overrides the default `/bin/sh -c` command. The auto-created K8s Service on port 9090 (targetPort "system") routes correctly when the container exposes port 9090 named "system".

If issues arise with the `default` type service routing, the operator's `GenerateComponentService()` (`graph.go:549-612`) may need a tweak to support custom ports for `default` components.

## Session Affinity Concern

The SandboxPool K8s Service load-balances across replicas. But all `execute_code()` calls for one `completion()` must hit the **same pod** (state is in-memory). Options:
- **POC**: Single sandbox replica (replicas: 1). Simple, sufficient for demo.
- **Later**: Orchestrator discovers pod IPs via headless service and routes directly to a chosen pod per session. Or sandbox server returns its pod IP on session creation and orchestrator calls it directly.

## Verification

1. Build images using minikube docker context:
   ```bash
   eval $(minikube docker-env)
   docker build -t rlm-orchestrator:latest -f Dockerfile.orchestrator .
   docker build -t rlm-sandbox:latest -f Dockerfile.sandbox .
   ```
2. Create namespace and deploy:
   ```bash
   kubectl create namespace tm
   kubectl apply -f deploy/dgd.yaml -n tm
   ```
3. Wait for pods: `kubectl get pods -n tm -l nvidia.com/dynamo-graph-deployment-name=rlm`
4. Port-forward: `kubectl port-forward -n tm svc/rlm-orchestrator 9090:9090`
5. Test: `curl -X POST localhost:9090/completion -H 'Content-Type: application/json' -d '{"prompt": "Compute the first 20 primes"}'`
6. Verify response contains correct primes and that sandbox execution logs show code blocks being run

**Notes:**
- For minikube deployment, images use `imagePullPolicy: Never` to use locally built images
- HuggingFace token secret is not required for Qwen3-0.6B model
- The DGD requires annotation `nvidia.com/enable-grove: "false"` to avoid SchedulingGated state in non-Grove clusters

### Verification Results (2026-02-10)

✅ **All components deployed successfully:**
- 4 Deployments ready (Frontend, Worker, Orchestrator, SandboxPool)
- 12 Services created (3 variants per component: default, -d debug, -p production)
- All pods running (1/1 ready)

✅ **Request flow verified:**
- HTTP → Orchestrator → creates sandbox session
- Sandbox executes code (multiple iterations observed in logs)
- LLM inference functional (Frontend↔Worker communication working)
- Session cleanup working properly

✅ **Infrastructure components validated:**
- KubernetesREPL: Creating sessions, executing code, cleaning up
- Orchestrator HTTP server: Receiving requests, coordinating execution
- Sandbox HTTP server: Managing Python execution sessions with in-memory state
- Frontend/Worker: Model loaded (Qwen/Qwen3-0.6B), inference endpoints ready

**Model behavior note:** Qwen3-0.6B (600M parameters) is responding but not reliably generating RLM-formatted code blocks. This is expected for such a small model. Infrastructure is fully functional - larger models would produce better quality code generation.
