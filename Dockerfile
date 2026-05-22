# Dockerfile for the Hugging Face Spaces deployment.
#
# This image is small enough to fit comfortably in HF's free CPU tier
# (16 GB RAM, 2 vCPU): CPU-only torch + pre-downloaded Qwen2.5-0.5B.
#
# HF Spaces convention: listen on port 7860, bound to 0.0.0.0.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/.cache/huggingface \
    TRANSFORMERS_CACHE=/tmp/.cache/huggingface/transformers \
    TINY_VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
    TINY_VLLM_DEVICE=cpu \
    TINY_VLLM_DTYPE=float32

WORKDIR /app

# Minimal system deps: curl for healthcheck, ca-certs for HTTPS.
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than the default GPU build).
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of our deps (skip torch — already done).
COPY requirements.txt .
RUN grep -v '^torch' requirements.txt > requirements.no-torch.txt \
 && pip install -r requirements.no-torch.txt

# Pre-download the model so cold-start latency is just engine warmup.
# Failing this step at build time is better than failing on first request.
RUN python -c "import os; m=os.environ['TINY_VLLM_MODEL']; \
    from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained(m); \
    AutoModelForCausalLM.from_pretrained(m); \
    print(f'pre-fetched {m}')"

# Now copy the source (placed AFTER the heavy deps so layer cache helps reruns).
COPY tiny_vllm/ ./tiny_vllm/
COPY web/ ./web/
COPY README.md LICENSE pyproject.toml ./

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD curl -fsS http://localhost:7860/health || exit 1

# Conservative resource settings — HF free CPU is small.
CMD ["python", "-m", "tiny_vllm.server", \
     "--host", "0.0.0.0", "--port", "7860", \
     "--block-size", "16", "--num-blocks", "128", \
     "--max-num-seqs", "4", "--max-num-batched-tokens", "256", \
     "--max-model-len", "1024"]
