# TVA: Token-aware Vision-transformer Accelerator

## Summary

This project proposes a novel **Token-wise mixed-precision Vision-Transformer Accelerator** on **FPGA**, specifically targeting **Vision Transformer (ViT) and LLMs** workloads. The architecture adaptively assigns **arithmetic precision per token** based on attention score importance, enabling significant **compute efficiency**, **interpretability** and **power savings**, especially for **real-time or edge AI applications**.

Unlike standard GPU implementations (e.g., FlashAttention) that rely on uniform precision GEMM operations, TVA leverages **outer-product-based computation** and **dynamic quantization**, making it highly optimized for streaming and resource-constrained environments like FPGAs.

---

## Motivation

ViTs and transfromer are state-of-the-art in many computer vision and natural language processing tasks but are compute- and memory-heavy. Edge deployment of ViTs in full precision for all tokens is often slow and unnecessary due to:
- Token importance Sparsity (only a few tokens are important in a long sequence),
- Uniform high-precision computation (e.g., FP32),
- GEMM-centric attention implementations,
- Static memory access and data reuse inefficiencies.

TVA addresses these issues through:
- **Token-aware precision adaptation**, using INT4, INT8, or INT16 depending on token importance.
- **Outer-product attention**, improving alignment of importance-driven Matrix Multiplications.
- **FPGA-specific optimizations**, leveraging DSPs, BRAM, and pipelined MAC units for high-throughput flexible low-latency execution.

---

## Key Features

- ✅ **Mixed-Precision Execution**: Dynamically chooses INT4, INT8, or INT16 MAC paths.
- ✅ **Token-Level Adaptation**: Based on attention column statistics (e.g., sum or entropy).
- ✅ **Outer-Product Attention**: Enables data reuse and precision assignment per token.
- ✅ **Tiling**: Supports arbitrary matrix sizes using efficient 16 by 16 memory tiling.
- ✅ **INT32 Accumulation**: Ensures numerical consistency across token precisions.
- ✅ **Streamed Architecture**: Suited for pipelined execution and memory-efficient inference.
- ✅ **Configurable for Vision Tasks**: Designed to support ViT-based classification and detection pipelines.

---

## Hardware Architecture

### Memory Hierarchy
- **External DDR3/DDR4**: Stores ViT weights, patch embeddings, and large tensors (~50MB).
- **On-Chip BRAM/URAM**: Caches intermediate activations, V-vectors, softmax scores, and outer-product results.
- **Precision Buffer**: Holds per-token precision assignments derived from attention statistics.

### Core Modules

| Component                  | Description |
|---------------------------|-------------|
| Precision Analyzer        | Computes per-token importance (e.g., column sum) and assigns quantization level. |
| Adaptive MAC Units        | Dedicated INT4 (Q1.3), INT8 (Q1.7), and INT16 (Q1.15) multiply-accumulate datapaths. |
| INT16 Accumulator          | Aggregates outer-product results for all tokens using a unified precision. |
| Memory Controller         | Manages bandwidth-efficient access to external DDR and on-chip BRAMs. |
| MLP Processing Pipeline   | Applies fully-connected ViT MLPs post-attention using streaming-friendly logic. |

---

## Conceptual Example

**Attention Matrix (A, size 3×3):**
Each column corresponds to one token with per-token quantization:
- A[:,0] → INT4 (Q1.3)
- A[:,1] → INT8 (Q1.7)
- A[:,2] → INT16 (Q1.15)

**Value Matrix (V, size 3×2):**
- V[0], V[1], V[2] reused across A's columns via outer-products.

Result:
- Outer-products: `A[:,i] ⊗ V[i,:]`, scaled by token precision.
- Final outputs are accumulated in INT16.

---

## ViT Dataflow Overview

1. **Image Patch Embedding**: Patchify and embed input image (from DDR).
2. **Attention Layer (TVA)**:
   - Load Q, K, V from memory.
   - Compute attention scores.
   - Token precision assigned by `Precision Analyzer`.
   - Outer-product executed with dynamic MAC unit.
3. **MLP Layer**:
   - Output vectors passed through hardware MLP (in BRAM/DDR).
   - Fully-pipelined arithmetic (primarily INT16).
4. **Result Output**:
   - Stored back to DDR (e.g., logits, bounding boxes, embeddings).

---

## Performance Highlights

| Metric                         | Estimate               |
|-------------------------------|-------------------------|
| Logic Resource Usage          | ~33,000 logic cells     |
| Max Precision Support         | INT16                   |
| Accuracy Drop                 | 0.05% on MNIST dataset         |
| TVA Speedup vs Dense FP Baseline  | 30-60% (input-dependent) |
| Precision Switching Latency   | 50 cycles (for #tokens=16) |
| INT 16 Multiplication  | 4 cycles  |
| INT 8 Multiplication  | 2 cycles  |
| INT 4 Multiplication  | 1 cycles  |

---


## Usage Instructions

### 1. Simulation (recommended first test)
- Compile Verilog modules using your preferred simulator (e.g. Vivado/XSIM, Verilator).
- Provide patch embeddings and QKV vectors as `*.npy` or memory initialization files.
- Simulate and dump outputs to verify outer-product attention logic.

### 2. On-Hardware Deployment (Artix-7 FPGA)
- Load `bitstream` to FPGA using Vivado.
- Ensure DDR3/DDR4 memory interface is configured and connected.
- Use host driver to push input image and receive output embeddings.

### 3. Visualization (Optional)
- Use testbench Python utilities (`tva_testbench.py`) to:
  - Visualize attention maps per token.
  - Compare INT16 vs INT4/INT8 token assignments.
  - Evaluate classification/detection accuracy.

---

## Build Artifacts




TVA: a novel Token-aware Vision-transformer Accelerator. This is an outer-product-based attention inference engine with token-importance-driven mixed-precision quantization, processing each token with unique precision and latency.

<p align="center">
<img src=".\images\demo_evaluation.png" width="1000"/>
</p>

**[Checkout the notebook for Quantization Aware Training and Post Training Quantization for TVA](https://colab.research.google.com/drive/1kMJykQPWpzSrSdFVy_d5k_uMILESneGC?usp=sharing)**
