# Serving layer — decision matrix (reference answers)

Companion to [nvidia-stack.md §4](nvidia-stack.md#4-serving-layer--decision-matrix). **H / M / L** = High / Medium / Low for that criterion. For *Hardware lock-in*, H is bad; for everything else, H is good.

> Snapshot as of Sep 2026. Pricing and project status change, so re-check NIM licensing and TGI's maintenance status before quoting them.

## Matrix


| Criteria                         | TensorRT-LLM (direct)                                       | Triton                                                                                         | Dynamo                                                   | NIM                                                                                   | vLLM                                                                   | SGLang                                                                | TGI                                       | llama.cpp / Ollama                        |
| -------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------- |
| **Peak throughput / latency**    | **H**: fastest on NVIDIA, esp. FP8/FP4 on Hopper/Blackwell  | **H**, but only as fast as its backend (TRT-LLM / vLLM)                                        | **H+ at scale**: prefill/decode split + KV-aware routing | **H**: pre-tuned TRT-LLM / vLLM / SGLang profiles                                     | **H**: close to TRT-LLM, sometimes level                               | **H**: best with shared prefixes / structured output (RadixAttention) | **M**                                     | **L**: single user, quantized             |
| **Time-to-first-deploy**         | **L–M**: engine build/config per GPU (`trtllm-serve` helps) | **L–M**: model repo + `config.pbtxt`                                                           | **L**: Kubernetes, etcd/NATS, operator                   | **H**: one `docker run`, OpenAI-compatible API                                        | **H**: `vllm serve <model>`                                            | **H**                                                                 | **H**                                     | **H+**: minutes                           |
| **Multi-model / non-LLM models** | **L**: LLMs (+ some VLMs) only                              | **H**: its reason to exist; ONNX, TensorRT, PyTorch, Python, ensembles, many models per server | **L–M**: LLM-focused                                     | **M**: broad catalog (embeddings, rerankers, speech, vision), one model per container | **M**: LLM, VLM, embeddings; one model per server                      | **M**                                                                 | **L–M**                                   | **M**: Ollama hot-swaps models; GGUF only |
| **Multi-node / disaggregated**   | **M**: TP/PP across nodes via MPI; some disaggregation      | **L–M**: not an orchestrator                                                                   | **H**: built for this                                    | **M**: multi-node NIMs for large models; disaggregation via Dynamo                    | **M**: Ray multi-node; disaggregation via connectors / llm-d / LMCache | **H**: mature P/D split, large expert-parallel deployments            | **L**                                     | **L**: experimental RPC only              |
| **Hardware lock-in**             | **H**: NVIDIA only; engines tied to GPU arch                | **M**: CPU + non-NVIDIA backends exist, but NVIDIA-first                                       | **M–H**: OSS, but NIXL/tuning NVIDIA-centric             | **H**: NVIDIA GPUs + NGC                                                              | **L**: NVIDIA, AMD, TPU, Gaudi, CPU, …                                 | **L–M**: NVIDIA, AMD; TPU emerging                                    | **L–M**                                   | **L**: runs almost anywhere               |
| **On-prem / air-gapped fit**     | **H**                                                       | **H**: proven in enterprises, incl. banks                                                      | **M**: OSS but heavy stack                               | **H** with NVIDIA AI Enterprise; offline model cache documented                       | **H**                                                                  | **H**                                                                 | **M**: maintenance mode = risk for a bank | **H dev / L prod**                        |
| **License / cost**               | Apache 2.0, free                                            | BSD-3, free (paid support via NVAIE)                                                           | Apache 2.0, free                                         | Free for dev/test; prod needs NVAIE (~$4.5k/GPU/yr list, verify)                      | Apache 2.0                                                             | Apache 2.0                                                            | Apache 2.0                                | MIT                                       |
| **Runs on my MacBook?**          | ❌ needs CUDA                                                | ⚠️ CPU-only in Docker, clunky                                                                  | ❌                                                        | ❌ locally; ✅ hosted NIMs at build.nvidia.com                                          | ⚠️ CPU / experimental Metal; fine for learning the API                 | ❌ in practice                                                         | ❌                                         | ✅ Metal, best Mac option (or MLX)         |




## How the pieces relate

- **TensorRT-LLM** is the *engine*. **Triton** (or `trtllm-serve`) is the *server* that exposes it.
- **Dynamo** is the *orchestration layer* above an engine (vLLM, SGLang or TRT-LLM): disaggregated prefill/decode, KV-aware routing, KV-cache transfer between GPUs.
- **NIM** is the *packaged, supported product*: you pay for NVIDIA AI Enterprise instead of tuning it yourself.



## Decision rules I'd defend in an interview

1. If it's **one LLM and I need portability across GPU vendors**, use **vLLM** (or **SGLang** for prefix-heavy / agentic workloads) because they're H on throughput and L on lock-in.
2. If I need the **lowest cost per token on NVIDIA at large scale**, use **TRT-LLM behind Dynamo** because the prefill/decode split and KV-aware routing raise GPU utilization beyond what a single engine gets.
3. If I'm serving a **mixed fleet (fraud-scoring models, ONNX/XGBoost, embeddings and LLMs together)**, use **Triton** because it's the only option rated H on multi-model / non-LLM.
4. If the enterprise wants **a vendor contract, support SLAs and fast time-to-value**, use **NIM** because it's H on time-to-first-deploy and on-prem fit, and the license buys the support.
5. If it's a **new production system at a regulated bank**, **don't** use **TGI** because it's in maintenance mode and long-term patching is uncertain.



## My lab environment

- **MacBook:** Ollama / llama.cpp for local work; hosted NIM endpoints (build.nvidia.com) to learn the NIM API.
- **RunPod (L4 default, H100 for FP8 days):** TRT-LLM, Triton, Dynamo, self-hosted NIM, vLLM and SGLang labs (`03-tensorrt-llm`, `04-triton-dynamo`, `05-nim`).

