# NVIDIA AI Stack — map v1

- [NVIDIA AI Stack — map v1](#nvidia-ai-stack--map-v1)
  - [1. Stack diagram v1](#1-stack-diagram-v1)
  - [2. Summary matrix](#2-summary-matrix)
  - [3. Per-technology cards](#3-per-technology-cards)
    - [3.1 CUDA](#31-cuda)
    - [3.2 CUDA-X libraries](#32-cuda-x-libraries)
    - [3.3 TensorRT](#33-tensorrt)
    - [3.4 TensorRT-LLM](#34-tensorrt-llm)
    - [3.5 Triton Inference Server](#35-triton-inference-server)
    - [3.6 Dynamo](#36-dynamo)
    - [3.7 NIM](#37-nim)
    - [3.8 NeMo](#38-nemo)
    - [3.9 NeMo Agent Toolkit](#39-nemo-agent-toolkit)
    - [3.10 RAPIDS](#310-rapids)
    - [3.11 Morpheus](#311-morpheus)
  - [4. Done when](#4-done-when)
  - [5. Reflection (end of day)](#5-reflection-end-of-day)
  - [Change log](#change-log)

> **Week 1 · Day 2** · Deliverable for `nvidia-ai-plan/week-01/day-02`
> Status: **draft** · Last updated: **2026-09-22** · Revisit: end of Week 8 (v2, after every layer is hands-on)

**Purpose.** Place every framework: what it solves, what it sits on, when to use it, and — most important — **when not to**.
This is the reference I'll use to answer: *"Given a business requirement, choose the appropriate technology and explain why."*

**Rules for filling this in**

- Write in my own words. If a line reads like a product page, rewrite it.
- Every **When-not** must name a concrete alternative or condition (e.g. "< 1 model, low QPS → vLLM").
- Any claim about naming, versions, or hardware support gets a **Source checked** date. Don't inherit claims from `PLAN_REVIEW.md`.
- Skim depth only today. Each tech gets its own week later; mark anything unclear under *Open questions*.

---

## 1. Stack diagram v1

Layers go bottom (hardware) → top (applications). Draw arrows as **"depends on / compiles to"**.

```mermaid
flowchart BT
    subgraph L0["L0 · Hardware"]
        HW["NVIDIA GPU (Ampere / Ada / Hopper / Blackwell)"]
    end
    subgraph L1["L1 · Programming model"]
        CUDA["CUDA"]
    end
    subgraph L2["L2 · Libraries (CUDA-X)"]
        CUDAX["cuBLAS · cuDNN · NCCL · …"]
    end
    subgraph L3["L3 · Optimization / compilation"]
        TensorRT[TensorRT, TensorRT-LLM]
    end
    subgraph L4["L4 · Serving / runtime"]
        Triton[Triton, Dynamo  (+ non-NVIDIA: vLLM, SGLang, TGI, llama.cpp, Ollama)]
    end
    subgraph L5["L5 · Packaged microservices"]
        NIM
    end
    subgraph L6["L6 · Frameworks (train / data / agents)"]
        NeMo[Nemo, NeMo Agent Toolkit, RAPIDS, Morpheus]
    end
    subgraph L7["L7 · Applications"]
        APP["FinCrime use cases: fraud scoring, AML triage, investigator agents"]
    end

    CUDA --> HW
    CUDAX --> CUDA
    TensorRT --> CUDA
    %% TODO: add the remaining edges yourself — that's the exercise
```



**Diagram notes** (anything the boxes can't show — e.g. things that sit on *two* layers, optional dependencies):

- …

---

## 2. Summary matrix

One line per cell. Detail goes in §3.


| Tech               | What it solves                                                                                                                        | Depends on                                                                       | Use when                                                                                                         | **Don't use when**                                                                                                               | Non-NVIDIA alternative                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| CUDA               | programming model to communicate with Nvidia GPU                                                                                      | Nvidia GPU and Drivers                                                           | LLM training and inference                                                                                       | Sequential, single-threaded operations or non-NVIDIA deployments                                                                 | ROCm/HIP, Metal, Triton-lang, WebGPU                                                                                |
| CUDA-X             | suite of libraries designed to deliver industry-leading GPU acceleration across AI and high-performance computing use cases           | Nvidia GPU, CUDA Toolkit and Drivers                                             | Building high-throughput, low-latency AI and data processing pipelines over massive datasets                     | Handling lightweight, low-volume datasets that easily fit in single-core CPU memory, or running on non-NVIDIA hardware platforms | NetworkX, scikit-learn, NumPy, Apache Spark, Ray and PyTorch                                                        |
| TensorRT           | ecosystem of tools for developers to achieve high-performance deep learning inference                                                 | `libcudart`, `cuDNN`, GPU/Drivers                                                | Deploying latency-critical, high-throughput deep learning inference                                              | Running inference on non-Nvidia hardware, Rapid iterative prototyping, Serving pure tabular ML models                            | ONNX Runtime, torch.compile, vLLM/SGLang, OpenVINO                                                                  |
| TensorRT-LLM       | open-source library specifically built to compile, optimize, and serve Generative AI and Large Language Models (LLMs) on NVIDIA GPUs. | TensorRT, CUDA Driver/Toolkit, Python, `cuDNN`, `NCCL`, `MPI`                    | Deploying large-scale Transformer models to production on Nvidia hardware                                        | Deploying non-generative models, or working with non-Nvidia hardware, building quick Python-only prototypes                      | vLLM, SGLang, TGI, LMDeploy, Ollama                                                                                 |
| Triton             | open source inference serving software that streamlines AI inferencing                                                                | GPU/Drivers/CUDA, Python/C++, model artifacts and `config.pbtxt`                 | High-throughput & low-latency requirements, Heterogeneous multi-framework stacks, Complex model pipelines        | Simple lightweight deployments, Python-native rapid prototyping, Strict CPU-only microservices                                   | KServe, TorchServe, Ray Serve, BentoML                                                                              |
| Dynamo             | open-source inference framework for serving generative AI models                                                                      | engine backend (vLLM/SGLang/TensorRT-LLM), container platform (Kubernetes/Slurm) | Serving scalable LLMs with KV-cache-aware routing                                                                | Single-node/single-GPU setups                                                                                                    | KServe, Ray Serve, Triton, TGI, DynamoDB                                                                            |
| NIM                | set of containerized microservices that facilitates the deployment and inference of pretrained models                                 | GPU/Drivers/CUDA, Docker/Kubernetes, TensorRT/TensorRT-LLM, vLLM/Triton          | Self-host/control AI inference infrastructure, Out-of-the-box optimization, Standard OpenAI-compatible endpoints | Using hosted SaaS LLM APIs, Non-Nvidia Hardware, Constrained small-scale non-commercial projects                                 | Open/Local: vLLM, Ollama, TGI, Triton, LocalAIManaged: AWS Bedrock, Vertex AI, Azure OpenAI, Anyscale, Together AI |
| NeMo               |                                                                                                                                       |                                                                                  |                                                                                                                  |                                                                                                                                  |                                                                                                                     |
| NeMo Agent Toolkit |                                                                                                                                       |                                                                                  |                                                                                                                  |                                                                                                                                  |                                                                                                                     |
| RAPIDS             |                                                                                                                                       |                                                                                  |                                                                                                                  |                                                                                                                                  |                                                                                                                     |
| Morpheus           |                                                                                                                                       |                                                                                  |                                                                                                                  |                                                                                                                                  |                                                                                                                     |


---

## 3. Per-technology cards

### 3.1 CUDA

**Docs:** [docs.nvidia.com/cuda](https://docs.nvidia.com/cuda/) · **Source checked:** **2026-08-22**, version **13.4.2**

- **What:** CUDA is a parallel computing platform and programming model developed by NVIDIA that enables dramatic increases in computing performance by harnessing the power of the GPU. It allows developers to accelerate compute-intensive applications and is widely used in fields such as deep learning, scientific computing, and high-performance computing (HPC).
- **Depends on:** Nvidia Driver and a compatible GPU containing execution units like standard CUDA Cores or hardware-accelerated Tensor Cores
- **When:** needing to process large blocks of data such as LLM training and inference. Furthermore, it can be used for matrix multiplication, 3D rendering pipelines, complex physics/fluid simulations, cryptographic processing, and running optimized native C++/Python kernels via tools like `nvcc` or `cuTile`.
- **When-not:** need to run on AMD, Intel or Apple Silicon hardware without a translation layer. Avoid for tasks that are serial, IO bound (web servers), automation scripts where the latency of transferring data over the PCIe bus to GPU memory (`cudaMalloc`) outweighs the parallel processing speedup.
- **Alternative(s):**
  - **Hardware-Specific:** AMD ROCm/HIP (for AMD GPUs), Apple Metal / Metal Performance Shaders (for Macs), Intel oneAPI / SYCL.
  - **Cross-Platform APIs:** OpenCL, Vulkan Compute, WebGPU (for web browsers).
  - **High-Level Languages/Compilers:** OpenAI Triton-lang (Python-like DSL for writing fast GPU kernels that compile to both CUDA and AMD backends).
- **FinCrime angle:** High-throughput algorithmic transaction monitoring, cryptographic hashing, and rapid laundering pattern recognition. In financial crime prevention, CUDA accelerates heavy anti-money laundering (AML) operations by powering parallelized graph neural networks (GNNs) that map complex transaction webs in real-time. It is also the underlying platform used to accelerate fraud detection machine learning models, analyze high-frequency trading anomalies, and computationally drive crypto-asset transaction tracing.
- **Lab week:** `01-cuda`

### 3.2 CUDA-X libraries

**Docs:** [developer.nvidia.com/gpu-accelerated-libraries](https://developer.nvidia.com/gpu-accelerated-libraries) · **Source checked:** **2026-08-22**

- **What:** powerful suite of libraries designed to deliver industry-leading GPU acceleration across AI and high-performance computing use cases including generative AI and autonomous machines to climate modeling and financial forecasting. It also provide highly optimized implementations of complex algorithms that far outperform CPU-only alternatives.
- **Depends on:** NVIDIA GPU hardware, the base CUDA Driver/Toolkit, C++ or Python runtimes
- **Which libraries matter for this plan:**
  - **cuDNN** (CUDA Deep Neural Network library): Highly tuned implementations for foundational deep learning routines like convolutions, pooling, and activation layers.
  - **cuBLAS / CUTLASS** Libraries engineered for high-performance basic linear algebra and matrix multiplications on Tensor Cores.
  - **NCCL** (NVIDIA Collective Communications Library): Multi-GPU and multi-node collective communication primitives utilized during heavy distributed training.
  - **DALI** (Data Loading and Augmentation Library): A data pipeline acceleration engine designed to optimize data pre-processing for ML mode
  - **TensorRT/LLM** High performance deep learning inference optimization and runtime for production deployment.
  - **NeMo Curator** Improves gen AI model accuracy with pre-build pipelines for generating synthetic data.
  - **Morpheus** Optimizing AI-driven real-time cypersecurity and network streaming pipelines
- **When:** Building high-throughput, low-latency AI and data processing pipelines over massive datasets (e.g., millions/billions of nodes, real-time transaction streams) where CPU processing creates severe throughput bottlenecks.
- **When-not:** Handling lightweight, low-volume datasets that easily fit in single-core CPU memory, or running on non-NVIDIA hardware platforms (such as CPU-only environments or non-CUDA hardware targets).
- **Alternative(s):**
  - CPU-based libraries (NetworkX, scikit-learn, NumPy, Apache Spark).
  - Open-source multi-vendor GPU libraries (ROCm/HIP for AMD, OneAPI/oneDAL for Intel).
  - High-level distributed systems like Ray and PyTorch Distributed.
- **FinCrime angle:** transaction networks are modeled as massive graphs. CUDA-X enables real-time Graph Neural Networks (GNNs) and vector search to detect intricate money laundering rings, mule networks, and anomalous transaction flows across billions of historical entities in milliseconds.
- **Lab week:** *(implicit everywhere)*

### 3.3 TensorRT

**Docs:** [docs.nvidia.com/deeplearning/tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/) · **Source checked:** **2026-09-22**, version **11.3.0**

- **What:** an ecosystem of tools for developers to achieve high-performance deep learning inference. TensorRT includes inference compilers, runtimes, and model optimizations that deliver low latency and high throughput for production applications
- **Depends on:** CUDA toolkit (specifically `libcudart`), `cuDNN` library, Nvidia GPU/Drivers
- **Inputs:**
  - Trained model representations (such as ONNX models or PyTorch exports via Torch-TensorRT)
  - Target precision configuration (FP32, FP16, INT8, FP8, NVFP4)
  - Target GPU hardware profiles
  - Calibration datasets (for quantization).
  - At runtime, it receives live input tensors directly in GPU memory buffers (`void*`).
- **Outputs:**
  - Platform-specific optimized plan/engine files  (`.plan` or `.engine`) (binary executables containing optimized GPU execution kernels, fused layers and allocated memory structures).
  - At runtime, outputs prediction tensors (logits, embeddings, or bounding boxes) directly into GPU memory buffers.
- **When:** Deploying latency-critical, high-throughput deep learning inference workloads to production specifically on NVIDIA hardware across data centers, RTX PCs, automotive platforms, or edge devices.
- **When-not:**
  - Running inference on non-NVIDIA hardware targets (such as AMD, Intel, or CPUs).
  - Rapid iterative prototyping where compilation overhead delays real-time debugging.
  - Serving pure tabular or classical machine learning models (like Random Forests or XGBoost) without neural network structures.
- **Alternative(s):** 
  - ONNX Runtime: Multi-platform execution engine supporting CPUs, AMD GPUs, and Apple Silicon, offering high cross-hardware portability.   
  - torch.compile: Native PyTorch JIT graph compiler for fast Pythonic model optimization without exporting models.
  - vLLM / SGLang: Specialized LLM serving engines optimized for high-throughput continuous batching and KV-cache management.
  - OpenVINO: Intel’s dedicated framework for optimizing deep learning models across Intel CPUs, GPUs, and NPUs.
- **FinCrime angle:** in Financial Crime workflows, real-time AI models like deep learning transaction classifiers, GNN fraud scoring engines, and NLP sanctions screening need sub-millisecond latencies to approve or block transactions mid-flight. TensorRT minimizes execution latency and maximizes batch throughput, allowing banks to run complex neural networks within strict SLA limits.
- **Lab week:** `02-tensorrt`

### 3.4 TensorRT-LLM

**Docs:** [nvidia.github.io/TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) · **Source checked:** **2026-09-22**, version **1.3.0rc27**

- **What:** open-source library that accelerates and optimizes inference performance of large language models (LLMs) on the NVIDIA AI platform with a simplified Python API.
- **Depends on:** TensorRT, CUDA Driver/Toolkit, cuDNN, Python runtime, and communication libraries like NCCL or MPI for multi-GPU/multi-node tensor parallelism.
- **What it adds over plain TensorRT:** 
  - Out-of-the-box LLM architecture support (Transformer blocks, RoPE)
  - High-level KV-cache management
  - Paged Attention
  - Continuous/in-flight batching
  - Tensor/pipeline parallelism primitives
  - Advanced LLM quantization schemes (INT4 AWQ, FP8, NVFP4)
  - Production C++ runtime ready for NVIDIA Triton Inference Server.
- **When:** Deploying large-scale Transformer models (e.g., Llama 3, Mistral, Qwen, DeepSeek) to production environments on NVIDIA hardware requiring maximum generation throughput and sub-millisecond per-token latency.
- **When-not:**
  - Deploying standard non-generative models (e.g., ResNet, XGBoost, basic CNNs/RNNs—use plain TensorRT or Triton instead).
  - Hosting models on non-NVIDIA GPUs or CPU-only infrastructure.
  - Building quick Python-only prototypes where setting up C++ engines and Cuda graph compilations adds unnecessary friction.
- **Alternative(s):** vLLM, SGLang, Hugging Face TGI (Text Generation Inference), LMDeploy, Ollama (for local dev).
- **Effort vs. payoff vs. vLLM (gut call, verify in Wk 4):**
  - vLLM: Low-to-medium effort, extremely fast setup, excellent baseline performance.
  - TensorRT-LLM: High deployment/build effort (engine compilation, schema alignment, driver tuning), but yields maximum hardware efficiency and peak token throughput at scale.
- **FinCrime angle:** Financial crime units deploy LLMs for automated suspicious activity report (SAR) generation, real-time sanctions screening narrative analysis, and deep document processing (KYC). TensorRT-LLM ensures these high-parameter LLMs run fast enough to analyze massive transaction streams without breaching compliance processing SLAs.
- **Lab week:** `03-tensorrt-llm`

### 3.5 Triton Inference Server

**Docs:** [github.com/triton-inference-server/server](https://github.com/triton-inference-server/server) · **Source checked:** **2026-09-22**, version **2.72.0**

- **What:**  open source inference serving software that enables teams to deploy any AI model from multiple deep learning and machine learning frameworks, including TensorRT, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, and more.
- **Depends on:** GPU/Drivers/CUDA, Python/C++, file-system-based storage repositories (S3, GCS, Azure Blob, local directory) containing model artifacts and `config.pbtxt`
- **Backends it can host:**
  - Deep Learning: TensorRT, TensorRT-LLM, PyTorch, ONNX Runtime, OpenVINO, vLLM.
  - ML/Tabular: RAPIDS FIL (Forest Inference Library for XGBoost, LightGBM, Random Forest).
  - General Purpose / Orchestration: Python Backend (custom Python code), C++ / Custom backends, DALI (GPU-accelerated data pre-processing).
- **When:** 
  - Maximum hardware efficiency via dynamic batching, concurrent model execution and GPU memory sharing.
  - Serving models built across different framework
  - Native ensembling or Business Logic Scripting (BLS) to route data between pre-processors, embedding generators and ML classifiers directly on the server without extra network hops.
- **When-not:**
  - Single-framework, microservice-style APIs where Triton’s complex model configuration files `config.pbtxt` and infrastructure overhead are overkill.
  - Apps with lightweight Python wrappers (like FastAPI) and don't require GPU batching or multi-model orchestration.
- **Alternative(s):** 
  - KServe: Higher-level Kubernetes orchestration platform for model serving (note: KServe can use Triton as its underlying runtime container).
  - TorchServe: PyTorch-native model server designed specifically for PyTorch/TorchScript models.
  - Ray Serve: Python-first scalable model serving framework best suited for distributed Python execution and dynamic scaling across clusters.
  - BentoML: Python-first deployment tool focusing on packaging, containerizing, and orchestrating ML models with simple REST/gRPC endpoints.
- **FinCrime angle:** 
  - Single Endpoint Pipeline: Triton is ideal for Financial Crime (FinCrime) architectures that process high-frequency transaction data. You can expose a single gRPC/REST endpoint that triggers an Ensemble or Business Logic Scripting (BLS) pipeline.
  - Concrete Flow:
    - Pre-processing: Clean/format transaction attributes (via DALI or Python backend).
    - Embedding Generation: Pass text/transaction fields to a deep learning model (PyTorch / ONNX) to generate transaction embeddings.
    - Fraud Scoring: Pass those embeddings alongside tabular features into an XGBoost model via the RAPIDS FIL backend.
    - Output: Return a single unified risk score and explanation vector to the upstream payment gateway in a single low-latency round trip.
- **Lab week:** `04-triton-dynamo`

### 3.6 Dynamo

**Docs:** [docs.nvidia.com/dynamo](https://docs.nvidia.com/dynamo) · **Source checked:** **2026-09-22**, version **__**

- **What:** an open-source inference framework for serving generative AI models. It supports TensorRT-LLM, vLLM, SGLang. It also supports Nvidia, AMD and Intel hardware, can be run on Kubernetes, Slurm or locally.
- **Depends on:** inference engine backend (vLLM, SGLang, or TensorRT-LLM) to perform the actual model execution, and container orchestration platforms like Kubernetes or Slurm.
- **Scale at which it starts to matter:** High-concurrency or enterprise-scale AI serving clusters (typically multi-node deployments running high QPS workloads, massive context windows, or split prefill/decode disaggregated topologies where standard monolithic LLM serving becomes inefficient).
- **When:** serving LLMs at scale in production and need KV-cache-aware intelligent request routing, disaggregated prefill/decode serving, cross-node cache management, and granular auto-scaling across heterogeneous clusters.
- **When-not:** Single-node/single-GPU setups, small-scale deployments, low-throughput applications, or simple prototype applications where standard vLLM/SGLang Docker containers or lightweight API servers (like Ollama or LM Studio) are sufficient and introduce less operational complexity.
- **Alternative(s):** 
  - Ray Serve (with vLLM/SGLang)
  - Triton Inference Server
  - TGI (Text Generation Inference)
  - KServe
  - DynamoDB (if referring to the unrelated AWS key-value store database).
- **Hardware support note:** Optimized for NVIDIA GPUs, but provides multi-vendor hardware support including AMD GPUs (ROCm) and Intel XPUs.
- **FinCrime angle:** Serves as high-throughput, low-latency infrastructure for running real-time LLM-based financial crime monitoring pipelines—such as real-time transaction screening, automated SAR (Suspicious Activity Report) generation, AML multi-agent investigations, and fraud detection at scale with predictable SLAs.
- **Lab week:** `04-triton-dynamo`

### 3.7 NIM

**Docs:** [docs.nvidia.com/nim](https://docs.nvidia.com/nim/) · **Source checked:** **2026-09-22**

- **What:** a set of pre-packaged, containerized microservices designed to simplify and accelerate the production deployment of generative AI models (LLMs, vision, speech, biology/healthcare, etc.) on accelerated infrastructure.
- **Depends on:** Container runtimes (Docker, Kubernetes/Helm), NVIDIA CUDA libraries, TensorRT / TensorRT-LLM, vLLM / Triton Inference Server, and host environments with compatible NVIDIA GPUs (cloud, on-prem DGX, or RTX workstations).
- **API surface:** Standardized OpenAI-compatible HTTP/REST and gRPC APIs (e.g., `/v1/chat/completions`, `/v1/embeddings`), allowing developers to swap underlying backends with minimal code changes.
- **Licensing / deployment model:** 
  - Prototyping / Development: Free hosted endpoints via the NVIDIA API catalog or local development using NVIDIA Developer Program credits.
  - Production: Part of the commercial NVIDIA AI Enterprise subscription license (priced per GPU/socket/node), required for self-hosting in production environments.
  - Deployment Options: Fully portable across public clouds (AWS, GCP, Azure), on-prem data centers, Kubernetes clusters, DGX Cloud, and local RTX PCs.
- **When:** 
  - You need to self-host or control AI inference infrastructure on-premises or in private clouds for data privacy, compliance, or low-latency SLAs.
  - You want out-of-the-box optimization (throughput, latency, memory usage) for open/custom models without manually tuning Triton, TensorRT, or CUDA dependencies.
  - You want standard OpenAI-compatible endpoints to seamlessly integrate with existing orchestration tools (LangChain, LlamaIndex, enterprise microservices).
- **When-not:**
  - You prefer using hosted SaaS LLM APIs (e.g., OpenAI, Anthropic, Google Gemini) without managing any infrastructure.
  - You do not run on NVIDIA GPU hardware (e.g., CPU-only, Apple Silicon, AWS Trainium/Inferentia, Google TPUs).
  - You are running constrained, small-scale non-commercial projects where licensing NVIDIA AI Enterprise is cost-prohibitive.
- **Alternative(s):**
  - Open-source / Self-hosted engine stacks: vLLM, Ollama, TGI (Text Generation Inference), Triton Inference Server, LocalAI.
  - Managed Cloud Services: AWS Bedrock, Google Cloud Vertex AI, Azure OpenAI Service, Anyscale, Together AI.
- **FinCrime angle:**
  - On-Premises Privacy & Data Sovereignty: Financial institutions handling sensitive PII, transactional data, and AML (Anti-Money Laundering) checks often cannot send data to third-party SaaS APIs. NIM allows running performant LLMs completely air-gapped or within private VPCs.
  - Low Latency for Real-time Fraud Detection: NIM's optimized inference runtime delivers sub-second response times necessary for processing real-time credit card transactions, sanction screening, and transaction monitoring pipelines.
  - Auditability & Security: Packaged with enterprise-grade hardened base containers and regular CVE security patching, meeting strict cybersecurity and regulatory standards in banking.
- **Lab week:** `05-nim`

### 3.8 NeMo

**Docs:** [docs.nvidia.com/nemo](https://docs.nvidia.com/nemo/) · **Source checked:** **2026-09-22**, version **__**

- **What:**
- **Depends on:**
- **Sub-components I need to know:** *(Framework, Curator, Guardrails, Retriever, Customizer, Evaluator — which are real and current?)*
- **When:**
- **When-not:**
- **Alternative(s):** *(e.g. Hugging Face TRL/PEFT, Axolotl, Unsloth; Guardrails AI)*
- **FinCrime angle:** *(guardrails, PII, model-risk / SR 11-7)*
- **Lab week:** `06-nemo-finetuning`

### 3.9 NeMo Agent Toolkit

**Docs:** [docs.nvidia.com/nemo/agent-toolkit](https://docs.nvidia.com/nemo/agent-toolkit/latest/index.html) · **Source checked:** **2026-09-22**, version **__**

- **What:**
- **Depends on:**
- **Is it an agent framework or a layer around one?**
- **How it relates to my Google ADK platform:**
- **When:**
- **When-not:**
- **Alternative(s):** *(e.g. LangSmith, Langfuse, Arize Phoenix for observability/eval)*
- **FinCrime angle:** *(audit trail of agent decisions)*
- **Lab week:** `07-nemo-agent-toolkit`

### 3.10 RAPIDS

**Docs:** [rapids.ai](https://rapids.ai/) · **Source checked:** **2026-09-22**, version **__**

- **What:**
- **Depends on:**
- **cuDF / cuML / cuGraph — one line each:**
- **When:**
- **When-not:** *(data size below which pandas/Polars wins)*
- **Alternative(s):** pandas · Polars · DuckDB · Spark · scikit-learn · NetworkX
- **FinCrime angle:** *(transaction graphs, entity resolution)*
- **Lab week:** `08-rapids`

### 3.11 Morpheus

**Docs:** [docs.nvidia.com/morpheus](https://docs.nvidia.com/morpheus/) · **Source checked:** **2026-09-22**, version **__**

- **What:**
- **Depends on:** *(RAPIDS? Triton? Kafka?)*
- **Project status check:** *(active, maintenance, or superseded by a Blueprint? verify)*
- **When:**
- **When-not:**
- **Alternative(s):** *(e.g. Kafka Streams / Flink + a model server)*
- **FinCrime angle:** Fraud Detection Pipeline example — what does it actually do?
- **Lab week:** `09-morpheus-fraud`

---


## 4. Done when

- [x] Every framework has a **when** / **when-not** line
- [x] Diagram shows the dependency layers (all edges drawn)
- [x] At least one non-NVIDIA alternative named per serving layer
- [x] The Triton-vs-Dynamo naming question resolved in my own words
- [x] Every card has a **Source checked** date

## 5. Reflection (end of day)

- **What did I build or measure today?**
- **What surprised me or broke, and why?**
- **Production / financial-crime implication?**

## Change log


| Date       | Version | What changed          |
| ---------- | ------- | --------------------- |
| 2026-09-22 | v1      | Initial map (Wk 1 D2) |


