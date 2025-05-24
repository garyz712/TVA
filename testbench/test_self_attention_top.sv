import cocotb
from cocotb.triggers import RisingEdge, Timer, First
from cocotb.clock import Clock
import numpy as np
import random

# Parameters matching the SystemVerilog module
DATA_WIDTH = 16
SEQ_LEN = 8
EMB_DIM = 8
N_HEADS = 1

# Fixed-point conversion helpers
def to_fixed_point(x, data_width=DATA_WIDTH):
    """Convert float to Q1.15 fixed-point (16-bit)."""
    scale = 2 ** (data_width - 1)
    x = np.clip(x, -1.0, 1.0 - 1/scale)
    return np.round(x * scale).astype(np.int16)

def from_fixed_point(x, data_width=DATA_WIDTH):
    """Convert Q1.15 fixed-point to float."""
    scale = 2 ** (data_width - 1)
    return x / scale

def softmax_approx(x):
    """Approximate softmax to mimic softmax_approx module (LUT-based, 8 fractional bits)."""
    # Simplified: Apply max subtraction and approximate exp with quantization
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(np.clip(x - x_max, -10, 10))  # Prevent overflow
    # Quantize to 8 fractional bits (mimicking LUT)
    exp_x = np.round(exp_x * (2**8)) / (2**8)
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / np.maximum(sum_exp_x, 1e-10)  # Avoid division by zero

def quantize_value(v, precision):
    """Quantize value based on precision (INT4, INT8, FP16)."""
    if precision == 0:  # INT4
        scale = 2**4
        v = np.round(v * scale) / scale
    elif precision == 1:  # INT8
        scale = 2**8
        v = np.round(v * scale) / scale
    # precision == 2: FP16, no quantization needed
    return v

def compute_precision(A):
    """Mimic precision_assigner: Assign INT4, INT8, or FP16 based on attention sum."""
    # Sum attention scores per token (row sum of A)
    token_sums = np.sum(np.abs(A), axis=-1)
    # Thresholds based on typical precision_assigner logic (adjust as needed)
    threshold_int4 = 0.5
    threshold_int8 = 1.5
    precisions = np.zeros(SEQ_LEN, dtype=np.int32)
    for i in range(SEQ_LEN):
        if token_sums[i] < threshold_int4:
            precisions[i] = 0  # INT4
        elif token_sums[i] < threshold_int8:
            precisions[i] = 1  # INT8
        else:
            precisions[i] = 2  # FP16
    return precisions

async def reset_dut(dut):
    """Reset the DUT."""
    dut.rst_n.value = 0
    await Timer(20, units="ns")
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

async def wait_for_done(dut, timeout_cycles=1000):
    """Wait for done signal with timeout."""
    timeout = Timer(timeout_cycles * 10, units="ns")
    done_event = RisingEdge(dut.clk)
    result = await First(done_event.until(dut.done.value == 1), timeout)
    if result == timeout:
        raise cocotb.result.TestFailure("Test timed out waiting for done signal")

@cocotb.test()
async def test_self_attention_random(dut):
    """Testbench for self_attention module with random inputs."""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Initialize signals
    dut.rst_n.value = 1
    dut.start.value = 0
    dut.x_in.value = [0] * (SEQ_LEN * EMB_DIM)
    dut.WQ_in.value = [0] * (EMB_DIM * EMB_DIM)
    dut.WK_in.value = [0] * (EMB_DIM * EMB_DIM)
    dut.WV_in.value = [0] * (EMB_DIM * EMB_DIM)
    dut.WO_in.value = [0] * (EMB_DIM * EMB_DIM)
    
    # Reset the DUT
    await reset_dut(dut)
    
    # Generate random input matrices
    x_np = np.random.uniform(-0.9, 0.9, size=(SEQ_LEN, EMB_DIM)).astype(np.float32)
    WQ_np = np.random.uniform(-0.9, 0.9, size=(EMB_DIM, EMB_DIM)).astype(np.float32)
    WK_np = np.random.uniform(-0.9, 0.9, size=(EMB_DIM, EMB_DIM)).astype(np.float32)
    WV_np = np.random.uniform(-0.9, 0.9, size=(EMB_DIM, EMB_DIM)).astype(np.float32)
    WO_np = np.random.uniform(-0.9, 0.9, size=(EMB_DIM, EMB_DIM)).astype(np.float32)
    
    # Convert to fixed-point
    x_fp = to_fixed_point(x_np)
    WQ_fp = to_fixed_point(WQ_np)
    WK_fp = to_fixed_point(WK_np)
    WV_fp = to_fixed_point(WV_np)
    WO_fp = to_fixed_point(WO_np)
    
    # Flatten and assign to DUT inputs
    dut.x_in.value = x_fp.flatten().tolist()
    dut.WQ_in.value = WQ_fp.flatten().tolist()
    dut.WK_in.value = WK_fp.flatten().tolist()
    dut.WV_in.value = WV_fp.flatten().tolist()
    dut.WO_in.value = WO_fp.flatten().tolist()
    
    # Drive inputs
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for computation to complete
    await wait_for_done(dut)
    
    # Read output
    y_np = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.int16)
    y_out_val = dut.out.value
    for i in range(SEQ_LEN):
        for j in range(EMB_DIM):
            idx = i * EMB_DIM + j
            y_np[i, j] = np.int16(y_out_val[idx].integer)
    
    # Compute reference output with mixed-precision approximation
    Q = x_np @ WQ_np  # (L, E)
    K = x_np @ WK_np  # (L, E)
    V = x_np @ WV_np  # (L, E)
    A = Q @ K.T / np.sqrt(EMB_DIM)  # (L, L)
    A = softmax_approx(A)  # Approximate softmax
    precisions = compute_precision(A)  # Assign precisions
    AV = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.float32)
    for i in range(SEQ_LEN):
        for j in range(EMB_DIM):
            for k in range(SEQ_LEN):
                v_quant = quantize_value(V[k, j], precisions[k])
                AV[i, j] += A[i, k] * v_quant
    y_ref = AV @ WO_np  # (L, E)
    y_ref_fp = to_fixed_point(y_ref)
    
    # Debug: Print inputs and outputs
    print(f"x_np:\n{x_np}")
    print(f"WQ_np:\n{WQ_np}")
    print(f"WK_np:\n{WK_np}")
    print(f"WV_np:\n{WV_np}")
    print(f"WO_np:\n{WO_np}")
    print(f"Precisions: {precisions}")
    print(f"y_np (DUT output):\n{y_np}")
    print(f"y_ref_fp (Expected):\n{y_ref_fp}")
    
    # Verify results (with tolerance for mixed-precision errors)
    assert np.allclose(y_np, y_ref_fp, atol=4), f"Output mismatch!\nExpected:\n{y_ref_fp}\nGot:\n{y_np}"
    assert dut.out_valid.value == 1, "out_valid not set"
    assert dut.done.value == 1, "done not set"
    
    # Test reset during computation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await Timer(50, units="ns")  # Wait a few cycles
    await reset_dut(dut)
    assert dut.out_valid.value == 0, "out_valid not cleared after reset"
    assert dut.done.value == 0, "done not cleared after reset"

@cocotb.test()
async def test_self_attention_fixed(dut):
    """Testbench for self_attention module with fixed inputs."""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Initialize signals
    dut.rst_n.value = 1
    dut.start.value = 0
    dut.x_in.value = [0] * (SEQ_LEN * EMB_DIM)
    dut.WQ_in.value = [0] * (EMB_DIM * EMB_DIM)
    dut.WK_in.value = [0] * (EMB_DIM * EMB_DIM)
    dut.WV_in.value = [0] * (EMB_DIM * EMB_DIM)
    dut.WO_in.value = [0] * (EMB_DIM * EMB_DIM)
    
    # Reset the DUT
    await reset_dut(dut)
    
    # Fixed input matrices
    x_np = np.array([
        [ 0.1, -0.2,  0.3, -0.4,  0.1, -0.2,  0.3, -0.4],
        [-0.2,  0.3, -0.4,  0.1, -0.2,  0.3, -0.4,  0.1],
        [ 0.3, -0.4,  0.1, -0.2,  0.3, -0.4,  0.1, -0.2],
        [-0.4,  0.1, -0.2,  0.3, -0.4,  0.1, -0.2,  0.3],
        [ 0.1, -0.2,  0.3, -0.4,  0.1, -0.2,  0.3, -0.4],
        [-0.2,  0.3, -0.4,  0.1, -0.2,  0.3, -0.4,  0.1],
        [ 0.3, -0.4,  0.1, -0.2,  0.3, -0.4,  0.1, -0.2],
        [-0.4,  0.1, -0.2,  0.3, -0.4,  0.1, -0.2,  0.3]
    ], dtype=np.float32)
    WQ_np = np.eye(EMB_DIM, dtype=np.float32) * 0.5
    WK_np = np.eye(EMB_DIM, dtype=np.float32) * 0.5
    WV_np = np.eye(EMB_DIM, dtype=np.float32) * 0.5
    WO_np = np.eye(EMB_DIM, dtype=np.float32) * 0.5
    
    # Convert to fixed-point
    x_fp = to_fixed_point(x_np)
    WQ_fp = to_fixed_point(WQ_np)
    WK_fp = to_fixed_point(WK_np)
    WV_fp = to_fixed_point(WV_np)
    WO_fp = to_fixed_point(WO_np)
    
    # Flatten and assign to DUT inputs
    dut.x_in.value = x_fp.flatten().tolist()
    dut.WQ_in.value = WQ_fp.flatten().tolist()
    dut.WK_in.value = WK_fp.flatten().tolist()
    dut.WV_in.value = WV_fp.flatten().tolist()
    dut.WO_in.value = WO_fp.flatten().tolist()
    
    # Drive inputs
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for computation to complete
    await wait_for_done(dut)
    
    # Read output
    y_np = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.int16)
    y_out_val = dut.out.value
    for i in range(SEQ_LEN):
        for j in range(EMB_DIM):
            idx = i * EMB_DIM + j
            y_np[i, j] = np.int16(y_out_val[idx].integer)
    
    # Compute reference output with mixed-precision approximation
    Q = x_np @ WQ_np
    K = x_np @ WK_np
    V = x_np @ WV_np
    A = Q @ K.T / np.sqrt(EMB_DIM)
    A = softmax_approx(A)
    precisions = compute_precision(A)
    AV = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.float32)
    for i in range(SEQ_LEN):
        for j in range(EMB_DIM):
            for k in range(SEQ_LEN):
                v_quant = quantize_value(V[k, j], precisions[k])
                AV[i, j] += A[i, k] * v_quant
    y_ref = AV @ WO_np
    y_ref_fp = to_fixed_point(y_ref)
    
    # Debug: Print inputs and outputs
    print(f"x_np:\n{x_np}")
    print(f"WQ_np:\n{WQ_np}")
    print(f"WK_np:\n{WK_np}")
    print(f"WV_np:\n{WV_np}")
    print(f"WO_np:\n{WO_np}")
    print(f"Precisions: {precisions}")
    print(f"y_np (DUT output):\n{y_np}")
    print(f"y_ref_fp (Expected):\n{y_ref_fp}")
    
    # Verify results
    assert np.allclose(y_np, y_ref_fp, atol=4), f"Output mismatch!\nExpected:\n{y_ref_fp}\nGot:\n{y_np}"
    assert dut.out_valid.value == 1, "out_valid not set"
    assert dut.done.value == 1, "done not set"
    
    # Test reset during computation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await Timer(50, units="ns")
    await reset_dut(dut)
    assert dut.out_valid.value == 0, "out_valid not cleared after reset"
    assert dut.done.value == 0, "done not cleared after reset"

@cocotb.test()
async def test_self_attention_edge_cases(dut):
    """Testbench for self_attention module with edge case inputs."""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Initialize signals
    dut.rst_n.value = 1
    dut.start.value = 0
    dut.x_in.value = [0] * (SEQ_LEN * EMB_DIM)
    dut.WQ_in.value = [0] * (EMB_DIM * EMB_DIM)
    dut.WK_in.value = [0] * (EMB_DIM * EMB_DIM)
    dut.WV_in.value = [0] * (EMB_DIM * EMB_DIM)
    dut.WO_in.value = [0] * (EMB_DIM * EMB_DIM)
    
    # Reset the DUT
    await reset_dut(dut)
    
    # Edge case 1: Maximum and minimum values
    max_val = 1.0 - 1/(2**(DATA_WIDTH-1))
    min_val = -1.0
    x_np = np.array([
        [max_val, min_val, max_val, min_val, max_val, min_val, max_val, min_val],
        [min_val, max_val, min_val, max_val, min_val, max_val, min_val, max_val],
        [max_val, min_val, max_val, min_val, max_val, min_val, max_val, min_val],
        [min_val, max_val, min_val, max_val, min_val, max_val, min_val, max_val],
        [max_val, min_val, max_val, min_val, max_val, min_val, max_val, min_val],
        [min_val, max_val, min_val, max_val, min_val, max_val, min_val, max_val],
        [max_val, min_val, max_val, min_val, max_val, min_val, max_val, min_val],
        [min_val, max_val, min_val, max_val, min_val, max_val, min_val, max_val]
    ], dtype=np.float32)
    WQ_np = np.eye(EMB_DIM, dtype=np.float32) * 0.5
    WK_np = np.eye(EMB_DIM, dtype=np.float32) * 0.5
    WV_np = np.eye(EMB_DIM, dtype=np.float32) * 0.5
    WO_np = np.eye(EMB_DIM, dtype=np.float32) * 0.5
    
    # Convert to fixed-point
    x_fp = to_fixed_point(x_np)
    WQ_fp = to_fixed_point(WQ_np)
    WK_fp = to_fixed_point(WK_np)
    WV_fp = to_fixed_point(WV_np)
    WO_fp = to_fixed_point(WO_np)
    
    # Flatten and assign to DUT inputs
    dut.x_in.value = x_fp.flatten().tolist()
    dut.WQ_in.value = WQ_fp.flatten().tolist()
    dut.WK_in.value = WK_fp.flatten().tolist()
    dut.WV_in.value = WV_fp.flatten().tolist()
    dut.WO_in.value = WO_fp.flatten().tolist()
    
    # Drive inputs
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for computation to complete
    await wait_for_done(dut)
    
    # Read output
    y_np = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.int16)
    y_out_val = dut.out.value
    for i in range(SEQ_LEN):
        for j in range(EMB_DIM):
            idx = i * EMB_DIM + j
            y_np[i, j] = np.int16(y_out_val[idx].integer)
    
    # Compute reference output with mixed-precision
    Q = x_np @ WQ_np
    K = x_np @ WK_np
    V = x_np @ WV_np
    A = Q @ K.T / np.sqrt(EMB_DIM)
    A = softmax_approx(A)
    precisions = compute_precision(A)
    AV = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.float32)
    for i in range(SEQ_LEN):
        for j in range(EMB_DIM):
            for k in range(SEQ_LEN):
                v_quant = quantize_value(V[k, j], precisions[k])
                AV[i, j] += A[i, k] * v_quant
    y_ref = AV @ WO_np
    y_ref_fp = to_fixed_point(y_ref)
    
    # Debug: Print inputs and outputs
    print(f"Edge Case 1: Max/Min Values")
    print(f"x_np:\n{x_np}")
    print(f"Precisions: {precisions}")
    print(f"y_np (DUT output):\n{y_np}")
    print(f"y_ref_fp (Expected):\n{y_ref_fp}")
    
    # Verify results
    assert np.allclose(y_np, y_ref_fp, atol=4), f"Output mismatch!\nExpected:\n{y_ref_fp}\nGot:\n{y_np}"
    assert dut.out_valid.value == 1, "out_valid not set"
    assert dut.done.value == 1, "done not set"
    
    # Reset DUT
    await reset_dut(dut)
    
    # Edge case 2: Zero inputs
    x_np = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.float32)
    x_fp = to_fixed_point(x_np)
    dut.x_in.value = x_fp.flatten().tolist()
    dut.WQ_in.value = WQ_fp.flatten().tolist()
    dut.WK_in.value = WK_fp.flatten().tolist()
    dut.WV_in.value = WV_fp.flatten().tolist()
    dut.WO_in.value = WO_fp.flatten().tolist()
    
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await wait_for_done(dut)
    
    y_np = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.int16)
    y_out_val = dut.out.value
    for i in range(SEQ_LEN):
        for j in range(EMB_DIM):
            idx = i * EMB_DIM + j
            y_np[i, j] = np.int16(y_out_val[idx].integer)
    
    y_ref_fp = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.int16)  # Expected: all zeros
    print(f"Edge Case 2: Zero Inputs")
    print(f"y_np (DUT output):\n{y_np}")
    print(f"y_ref_fp (Expected):\n{y_ref_fp}")
    
    assert np.allclose(y_np, y_ref_fp, atol=4), f"Output mismatch!\nExpected:\n{y_ref_fp}\nGot:\n{y_np}"
    assert dut.out_valid.value == 1, "out_valid not set"
    assert dut.done.value == 1, "done not set"
    
    # Reset DUT
    await reset_dut(dut)
    
    # Edge case 3: Uniform attention scores
    x_np = np.ones((SEQ_LEN, EMB_DIM), dtype=np.float32) * 0.1
    x_fp = to_fixed_point(x_np)
    dut.x_in.value = x_fp.flatten().tolist()
    dut.WQ_in.value = WQ_fp.flatten().tolist()
    dut.WK_in.value = WK_fp.flatten().tolist()
    dut.WV_in.value = WV_fp.flatten().tolist()
    dut.WO_in.value = WO_fp.flatten().tolist()
    
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await wait_for_done(dut)
    
    y_np = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.int16)
    y_out_val = dut.out.value
    for i in range(SEQ_LEN):
        for j in range(EMB_DIM):
            idx = i * EMB_DIM + j
            y_np[i, j] = np.int16(y_out_val[idx].integer)
    
    Q = x_np @ WQ_np
    K = x_np @ WK_np
    V = x_np @ WV_np
    A = Q @ K.T / np.sqrt(EMB_DIM)
    A = softmax_approx(A)  # Should be ~1/SEQ_LEN for all elements
    precisions = compute_precision(A)
    AV = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.float32)
    for i in range(SEQ_LEN):
        for j in range(EMB_DIM):
            for k in range(SEQ_LEN):
                v_quant = quantize_value(V[k, j], precisions[k])
                AV[i, j] += A[i, k] * v_quant
    y_ref = AV @ WO_np
    y_ref_fp = to_fixed_point(y_ref)
    
    print(f"Edge Case 3: Uniform Attention")
    print(f"x_np:\n{x_np}")
    print(f"Precisions: {precisions}")
    print(f"y_np (DUT output):\n{y_np}")
    print(f"y_ref_fp (Expected):\n{y_ref_fp}")
    
    assert np.allclose(y_np, y_ref_fp, atol=4), f"Output mismatch!\nExpected:\n{y_ref_fp}\nGot:\n{y_np}"
    assert dut.out_valid.value == 1, "out_valid not set"
    assert dut.done.value == 1, "done not set"
