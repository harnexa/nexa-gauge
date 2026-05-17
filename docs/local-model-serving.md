# Local Model Serving Guide

This guide explains which local model server to use with nexa-gauge under
different hardware and operator constraints. nexa-gauge calls LLMs through
LiteLLM, so the preferred integration point is an OpenAI-compatible local
endpoint plus a LiteLLM model string.

## Quick Recommendation

| Situation | Recommended backend | Why |
| --- | --- | --- |
| NVIDIA GPU, Linux, high-throughput evaluation | vLLM | Best default for fast batched serving, Hugging Face models, concurrency, and production-style eval runs. |
| CPU-only machine | llama.cpp / llama-server | Best CPU and quantized GGUF path; predictable on laptops and small servers. |
| Apple Silicon Mac | llama.cpp, Ollama, or LM Studio | Metal-backed local inference and easy GGUF workflows; Ollama/LM Studio improve onboarding. |
| AMD GPU or non-NVIDIA GPU | llama.cpp first, vLLM/SGLang only for expert Linux setups | ROCm/Vulkan paths exist, but compatibility varies by card, OS, and driver. |
| User wants the easiest install | Ollama | Simple model pull and local server UX. |
| User wants desktop model management | LM Studio | Friendly GUI, model browser, and OpenAI-compatible local server. |
| Team wants Hugging Face production serving | vLLM or TGI | vLLM is usually more flexible for eval workloads; TGI is strong in HF-native deployments. |
| Team wants advanced structured decoding and serving research features | SGLang | Strong structured output and high-performance serving, but a more advanced operator choice. |

## Product Stance

nexa-gauge should support local inference as a provider family, not as a single
hard dependency:

```text
local model support = LiteLLM route + model name + endpoint URL + optional API key
```

The recommended documentation path should be:

1. vLLM for serious GPU evaluation.
2. llama.cpp / llama-server for CPU and portable quantized evaluation.
3. Ollama for quick local setup.
4. LM Studio for desktop users.
5. SGLang and TGI as advanced alternatives.

This keeps the CLI stable while letting users choose the runtime that fits their
hardware.

## Hardware-Based Decision Guide

### NVIDIA GPU

Use vLLM by default.

vLLM is the best fit when the user has CUDA-capable hardware and wants to run
many judge calls across cases, chunks, claims, and metrics. It is designed for
throughput, batching, and OpenAI-compatible serving.

Good for:

- Large benchmark runs.
- Concurrent claim extraction.
- Repeated grounding and relevance calls.
- Hugging Face transformer models.
- Server deployments with one or more GPUs.

Example:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

Then route nexa-gauge through LiteLLM:

```bash
export HOSTED_VLLM_API_BASE=http://localhost:8000/v1
export LLM_MODEL=hosted_vllm/Qwen/Qwen2.5-7B-Instruct
```

For per-node routing:

```bash
export LLM_CLAIMS_MODEL=hosted_vllm/Qwen/Qwen2.5-7B-Instruct
export LLM_GROUNDING_MODEL=hosted_vllm/Qwen/Qwen2.5-7B-Instruct
export LLM_RELEVANCE_MODEL=hosted_vllm/Qwen/Qwen2.5-7B-Instruct
```

Recommended nexa-gauge runtime profile:

```bash
export CLAIMS_MAX_WORKERS=8
nexagauge run eval --input sample.json --llm-concurrency 8 --max-in-flight 4
```

Tune those numbers down for smaller GPUs.

### CPU Only

Use llama.cpp / llama-server by default.

CPU inference is slower, but it is valuable for privacy-sensitive local eval,
offline testing, CI smoke tests, and users without GPUs. The important move is
to use smaller quantized GGUF models and low concurrency.

Good for:

- CPU-only laptops and servers.
- Offline local testing.
- Small models, 3B to 8B.
- Quantized GGUF models, such as Q4_K_M or Q5_K_M.
- Conservative eval runs where speed is less important than portability.

Example:

```bash
llama-server -hf bartowski/Qwen2.5-7B-Instruct-GGUF -c 4096 --port 8080
```

Route through the CLI host-model shortcut:

```bash
nexagauge run eval \
  --input sample.json \
  --host-model-url http://localhost:8080/v1 \
  --llm-concurrency 1 \
  --max-in-flight 1
```

Recommended nexa-gauge runtime profile:

```bash
export CLAIMS_MAX_WORKERS=1
nexagauge run eval --input sample.json --llm-concurrency 1 --max-in-flight 1
```

If a run produces many chunks, use a chunk/refiner budget so CPU mode does not
turn into an accidental all-night job.

### Apple Silicon

Use llama.cpp, Ollama, or LM Studio depending on the user's comfort level.

Apple Silicon is a special local-inference environment: users often want easy
model management, Metal acceleration, and GGUF or MLX model formats. vLLM is not
the natural first recommendation here.

Recommended choices:

| User type | Backend |
| --- | --- |
| CLI-first user who wants control | llama.cpp / llama-server |
| User who wants the easiest setup | Ollama |
| Desktop user who wants model browsing | LM Studio |
| Advanced Apple-only model workflows | MLX-based tools, as an optional future profile |

Ollama example:

```bash
ollama pull llama3.1
ollama serve
```

LiteLLM route:

```bash
export LLM_MODEL=ollama_chat/llama3.1
```

llama.cpp example:

```bash
llama-server -hf bartowski/Qwen2.5-7B-Instruct-GGUF -c 4096 --port 8080
```

Recommended nexa-gauge runtime profile:

```bash
nexagauge run eval --input sample.json --llm-concurrency 2 --max-in-flight 1
```

Use higher concurrency only after checking memory pressure.

### AMD GPU and Other GPUs

Start with llama.cpp unless the user is comfortable managing ROCm or Vulkan
serving stacks.

AMD GPU support can be powerful, but it is less uniform than NVIDIA CUDA. The
best backend depends heavily on OS, card generation, drivers, and available
wheels or containers.

Recommended choices:

| Situation | Backend |
| --- | --- |
| User wants reliability | llama.cpp with CPU/GPU offload where available |
| Linux user with supported ROCm stack | vLLM or SGLang can be considered |
| User wants simple local UX | Ollama, if their platform build supports the hardware well |
| Production eval | Validate vLLM/SGLang on the exact GPU before recommending |

For product docs, avoid promising universal AMD acceleration. Say it is
supported through selected backends and should be verified on the target
machine.

## Backend Comparison

| Backend | Model ecosystem | CPU | NVIDIA GPU | Apple Silicon | AMD/other GPU | LiteLLM fit | nexa-gauge role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vLLM | Hugging Face transformers | Poor | Excellent | Not preferred | Expert/ROCm-dependent | Excellent via `hosted_vllm/` | Recommended GPU backend |
| llama.cpp / llama-server | GGUF models | Excellent | Good | Excellent | Good but hardware-dependent | Good via OpenAI-compatible route | Recommended CPU/portable backend |
| Ollama | Ollama library plus custom Modelfiles | Good | Good | Good | Platform-dependent | Good via `ollama_chat/` | Easiest local onboarding |
| LM Studio | GUI-managed local models, GGUF/MLX | Good | Good | Excellent | Platform-dependent | Good via OpenAI-compatible route | Desktop-friendly option |
| SGLang | Hugging Face transformers | Poor | Excellent | Not preferred | Expert/ROCm-dependent | Good via OpenAI-compatible route | Advanced high-performance option |
| TGI | Hugging Face transformers | Poor | Excellent | Not preferred | Limited/expert | Good via OpenAI-compatible route | HF production serving option |

## Structured Output Considerations

nexa-gauge uses structured JSON outputs for nodes such as claim extraction,
grounding, relevance, redteam, and GEval. Local model serving should therefore
be judged not just by raw speed, but by how reliably it returns valid JSON.

Practical guidance:

- Prefer instruction-tuned models with strong JSON-following behavior.
- Keep temperature near `0.0` for evaluation nodes.
- Prefer servers with JSON schema or constrained decoding support when possible.
- Keep fallback models configured for important evaluation runs.
- Treat parsing failures as a backend/model quality signal, not only a prompt
  problem.

Backend notes:

- vLLM supports structured outputs through its OpenAI-compatible server.
- Ollama supports JSON mode and `response_format` through its OpenAI-compatible
  endpoint, depending on model capability.
- llama.cpp and llama-cpp-python support OpenAI-compatible serving and JSON
  schema workflows.
- SGLang has strong structured output support and is a good advanced option.

## Recommended nexa-gauge Profiles

### `local-gpu`

Best for NVIDIA GPU and vLLM.

```bash
export HOSTED_VLLM_API_BASE=http://localhost:8000/v1
export LLM_MODEL=hosted_vllm/Qwen/Qwen2.5-7B-Instruct

nexagauge run eval --input sample.json --llm-concurrency 8 --max-in-flight 4
```

### `local-cpu`

Best for CPU and llama.cpp.

```bash
export CLAIMS_MAX_WORKERS=1

nexagauge run eval \
  --input sample.json \
  --host-model-url http://localhost:8080/v1 \
  --llm-concurrency 1 \
  --max-in-flight 1
```

### `local-easy`

Best for Ollama.

```bash
ollama pull llama3.1
ollama serve

export LLM_MODEL=ollama_chat/llama3.1
nexagauge run eval --input sample.json --llm-concurrency 2 --max-in-flight 1
```

### `local-desktop`

Best for LM Studio.

1. Start the LM Studio local server.
2. Load a model.
3. Point LiteLLM to the OpenAI-compatible endpoint.

```bash
nexagauge run eval \
  --input sample.json \
  --host-model-url http://localhost:1234/v1 \
  --llm-concurrency 2 \
  --max-in-flight 1
```

## Endpoint Routing (Implemented)

Current nexa-gauge model routing already supports global and per-node model
selection:

```bash
export LLM_MODEL=...
export LLM_CLAIMS_MODEL=...
export LLM_GROUNDING_MODEL=...
export LLM_RELEVANCE_MODEL=...
```

nexa-gauge now supports first-class endpoint routing alongside model routing:

```bash
nexagauge run eval \
  --input sample.json \
  --host-model-url http://localhost:8080/v1
```

The resolved `api_base` and `api_key` are passed into `litellm.completion`.
For local endpoints (`localhost`, `127.0.0.1`, `::1`), nexa-gauge auto-uses
`api_key=local` when no key is provided.

Also consider adding named profiles:

```text
local-gpu      -> vLLM defaults, higher concurrency
local-cpu      -> llama.cpp defaults, low concurrency
local-easy     -> Ollama defaults
local-desktop  -> LM Studio defaults
```

## Final Recommendation

Use this as the public recommendation:

```text
If you have an NVIDIA GPU, use vLLM.
If you need CPU support, use llama.cpp / llama-server.
If you want the easiest setup, use Ollama.
If you are on Apple Silicon and prefer a desktop app, use LM Studio.
If you are an advanced serving user, evaluate SGLang or TGI.
```

That gives nexa-gauge a clear default without blocking users who already have a
preferred local inference stack.

## References

- LiteLLM provider list and unified API: https://docs.litellm.ai/
- LiteLLM vLLM provider: https://docs.litellm.ai/docs/providers/vllm
- vLLM LiteLLM integration: https://docs.vllm.ai/en/stable/deployment/frameworks/litellm/
- vLLM structured outputs: https://docs.vllm.ai/en/stable/features/structured_outputs/
- LiteLLM Ollama provider: https://docs.litellm.ai/docs/providers/ollama
- Ollama OpenAI compatibility: https://docs.ollama.com/api/openai-compatibility
- llama.cpp server docs: https://www.mintlify.com/ggml-org/llama.cpp/inference/server
- llama-cpp-python server docs: https://llama-cpp-python.readthedocs.io/en/latest/server/
- Hugging Face TGI Messages API: https://huggingface.co/docs/text-generation-inference/messages_api
- SGLang structured outputs: https://docs.sglang.ai/backend/structured_outputs.html
- LM Studio API docs: https://lmstudio.ai/docs/api
