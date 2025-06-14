# TVA: Token-aware Vision-transformer Accelerator

## Summary

This research project proposes a novel **Token-wise mixed-precision Vision-Transformer Accelerator** on **FPGA**, specifically targeting **Vision Transformer (ViT) and LLMs** workloads. The architecture adaptively assigns **arithmetic precision per token** based on attention score importance, enabling significant **compute efficiency**, **interpretability** and **power savings**, especially for **real-time or edge AI applications**.

Unlike standard GPU implementations (e.g., FlashAttention) that rely on uniform precision GEMM operations, TVA leverages **outer-product-based computation** and **dynamic quantization**, making it highly optimized for processing long sequences or large images in resource-constrained environments like FPGAs. This prototype can be further extended to all layers in the modern transformer-based model for faster inference.

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
| Adaptive MAC Units (Mul16.sv)       | Dedicated INT4 (Q1.3), INT8 (Q1.7), and INT16 (Q1.15) multiply-accumulate datapaths with various latency. |
| INT32 Accumulator          | Aggregates outer-product results for all tokens using a unified precision. |
| Memory Controller (TBD)         | Manages bandwidth-efficient access to external DDR and on-chip BRAMs. |
| Relu-like Softmax   | Applies Relu-style Softmax module to avoid expensive exponential computation. |
| MLP Processing Pipeline   | Applies fully-connected ViT MLPs post-attention using streaming-friendly logic. |

---

## Conceptual Example

**Attention Matrix (A, size 3×3):**
Each column corresponds to one token with per-token quantization:
- A[:,0] → INT4 (Q1.3)
- A[:,1] → INT8 (Q1.7)
- A[:,2] → INT16 (Q1.15)

**Value Matrix (V, size 3×2):**
- V[0, :], V[1, :], V[2, :] reused across A's columns via outer-products.

Result:
- Outer-products: `A[:,i] ⊗ V[i,:]`, scaled by token precision.
- Final outputs are accumulated in INT32 (Q2.30) and downcast back to INT16.

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
   - 
<p align="center">
<img src=".\images\TVA_architect.png" width="1000"/>
</p>
---

## Performance Highlights

| Metric                         | Estimate               |
|-------------------------------|-------------------------|
| Logic Resource Usage          | ~33,000 logic cells     |
| Max Precision Support         | INT16                   |
| Accuracy Drop                 | 0.05% on MNIST dataset         |
| TVA Speedup vs Dense FP Baseline  | 30-60% (input-dependent) |
| Precision Switching Latency   | 50 cycles (negligible for full implementation) |
| INT 16 Multiplication  | 4 cycles  |
| INT 8 Multiplication  | 2 cycles  |
| INT 4 Multiplication  | 1 cycles  |

---


## Usage Instructions

### 0. Check the Original ViT Performance 
- **[Checkout the notebook for Quantization Aware Training and Post Training Quantization for TVA](https://colab.research.google.com/drive/1kMJykQPWpzSrSdFVy_d5k_uMILESneGC?usp=sharing)**
- Run the first block to get the original ViT performance on MNIST dataset with full percision -> Accuracy = 99.91%
<p align="center">
<img src=".\images\fullP_accuracy.png" width="400"/>
</p>

### 1. Quantization Aware Training and Evaluation of TVA
- Run the 2.1 block (QAT) to get preprocessed image/weight input (in verilog_inputs folder) for TVA, this might takes a while for training.
- Run the 2.2 block to evaluation TVA on MNIST: Accuracy = 99.86% (-0.05% drop)

<p align="center">
<img src=".\images\demo_accuracy.png" width="700"/>
</p>

### 2. TVA Self-Attention Hardware Simulation 
- Run on colab
  
      !zip -r verilog_inputs.zip verilog_inputs
- download the zip file, and unzip locally at TVA/testbench
- Install required module

      pip install cocotb numpy
      apt install verilator # or use your preferred package manager to install your preferred simulator

- Modify the testbench/makefile file to match the file sources, test module top-level, and testbench filename
- Compile Verilog modules using your preferred simulator (e.g. Verilator)
  
      cd TVA/testbench
      make clean
      make
      make

- You will see the verilog_outputs folder, zip it and upload it back to colab for output verification.
- Run on colab
  
      !unzip -q verilog_outputs.zip
- Run block 2.3 to evaluate the hardware output: Accuracy = 100%
  
<p align="center">
<img src=".\images\demo_evaluation.png" width="1000"/>
</p>

### 3. TVA Speed Up Ratio
- To see how dynamic precision affects the inference speed, we could run test_attention_av_multiply_demo as an example. Note: this acceleration approach can be applied to all layers in the future except for QKV generator for larger speedup
- Change the testbench/makefile to include the attention av multiply as TOPLEVEL and MODULE
- Compile Verilog modules using your preferred simulator (e.g. Verilator)
  
      cd TVA/testbench
      make clean
      make
      make
- You will see the inference time for different precision like:
  
                                                          Test Case 1: All INT4
          30.00ns INFO     cocotb.attention_av_multiply       Inputs assigned successfully!
        6770.00ns INFO     cocotb.attention_av_multiply       Test passed: Output matches expected.
      Test case 1 passed!
        6800.00ns INFO     cocotb.attention_av_multiply       
                                                              Test Case 2: All INT8
        6800.00ns INFO     cocotb.attention_av_multiply       Inputs assigned successfully!
       14820.00ns INFO     cocotb.attention_av_multiply       Test passed: Output matches expected.
      Test case 2 passed!
       14850.00ns INFO     cocotb.attention_av_multiply       
                                                              Test Case 3: All INT16
       14850.00ns INFO     cocotb.attention_av_multiply       Inputs assigned successfully!
       25430.00ns INFO     cocotb.attention_av_multiply       Test passed: Output matches expected.
      Test case 3 passed!
       25460.00ns INFO     cocotb.attention_av_multiply       
                                                              Test Case 4: Mixed Precision
       25460.00ns INFO     cocotb.attention_av_multiply       Inputs assigned successfully!
       33640.00ns INFO     cocotb.attention_av_multiply       Test passed: Output matches expected.

- If you want to test the Post Training Quantization accuracy, feel free to run block 3 in the colab like before, the PTQ accuracy drop is minimum for this module but it can increases as you add more TVA modules, thus QAT is more recommended!
  
### 4. On-Hardware Deployment (Optional: Artix-7 FPGA)
- Load `bitstream` to FPGA using Vivado.
- Ensure DDR3/DDR4 memory interface is configured and connected.
- Use host driver to push input image and receive output embeddings.

### 5. Visualization (Optional)
- Use testbench Python utilities (`tva_testbench.py`) to:
  - Visualize attention maps per token.
  - Compare INT16 vs INT4/INT8 token assignments.
  - Evaluate classification/detection accuracy.

---

## Future Work


TVA is a novel Token-aware Vision-transformer Accelerator. This is an outer-product-based attention inference engine with token-importance-driven mixed-precision quantization, processing each token with unique precision and latency. This approach can be extended to all layers to increase speedup ratio (instead assign each attention row with a unique precision based on corresponding column sum for each token then do mixed precision inference for MLP). 



## Acknowledgement
This research is conducted at California Institute of Technology (Caltech). We thank Professor Glen George for FPGA resources support.
