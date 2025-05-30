import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotb.clock import Clock
import numpy as np
import random

# Parameters matching the SystemVerilog module
DATA_WIDTH = 16
SEQ_LEN = 8
EMB_DIM = 8

@cocotb.test()
async def test_residual_random(dut):
    """Testbench for residual module with random inputs"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Initialize signals
    dut.rst_n.value = 1
    dut.start.value = 0
    dut.x_in.value = 0
    dut.sub_in.value = 0
    
    # Reset the DUT
    await reset_dut(dut)
    
    # Generate random input matrices x_in and sub_in
    x_np = np.random.randint(-2**15, 2**15-1, size=(SEQ_LEN, EMB_DIM)).astype(np.int32)
    sub_np = np.random.randint(-2**15, 2**15-1, size=(SEQ_LEN, EMB_DIM)).astype(np.int32)
    
    # Flatten and convert to binary for DUT input
    x_flat = x_np.flatten()
    sub_flat = sub_np.flatten()
    x_bin = 0
    sub_bin = 0
    for i in range(SEQ_LEN * EMB_DIM):
        x_bin = (x_bin << DATA_WIDTH) | (int(x_flat[i]) & ((1 << DATA_WIDTH) - 1))
        sub_bin = (sub_bin << DATA_WIDTH) | (int(sub_flat[i]) & ((1 << DATA_WIDTH) - 1))
    
    # Drive inputs
    dut.x_in.value = x_bin
    dut.sub_in.value = sub_bin
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for computation to complete
    await wait_for_done(dut)
    
    # Read and verify output
    y_np = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.int32)
    
    # Extract output matrix
    y_out_val = dut.y_out.value
    for i in range(SEQ_LEN):
        for j in range(EMB_DIM):
            idx = ((i * EMB_DIM) + j) * DATA_WIDTH
            bits = (y_out_val >> idx) & ((1 << DATA_WIDTH) - 1)
            if bits & (1 << (DATA_WIDTH - 1)):  # Handle sign extension
                bits = bits - (1 << DATA_WIDTH)
            y_np[SEQ_LEN-1-i, EMB_DIM-1-j] = bits
    
    # Compute reference output
    y_ref = x_np + sub_np
    
    # Debug: Print inputs and outputs
    print(f"x_np:\n{x_np}")
    print(f"sub_np:\n{sub_np}")
    print(f"y_np (DUT output):\n{y_np}")
    print(f"y_ref (Expected):\n{y_ref}")
    
    # Verify results
    assert np.array_equal(y_np, y_ref), f"Output mismatch!\nExpected:\n{y_ref}\nGot:\n{y_np}"
    assert dut.out_valid.value == 1, "out_valid not set"
    assert dut.done.value == 1, "done not set"
    
    # Test reset during computation
    await reset_dut(dut)
    assert dut.out_valid.value == 0, "out_valid not cleared after reset"
    assert dut.done.value == 0, "done not cleared after reset"

@cocotb.test()
async def test_residual_fixed(dut):
    """Testbench for residual module with fixed inputs"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Initialize signals
    dut.rst_n.value = 1
    dut.start.value = 0
    dut.x_in.value = 0
    dut.sub_in.value = 0
    
    # Reset the DUT
    await reset_dut(dut)
    
    # Fixed input matrices x_in and sub_in (shape: SEQ_LEN x EMB_DIM = 8 x 8)
    x_np = np.array([
        [ 1, -2,  3, -4,  5, -6,  7, -8],
        [-9, 10, -11, 12, -13, 14, -15, 16],
        [17, -18, 19, -20, 21, -22, 23, -24],
        [-25, 26, -27, 28, -29, 30, -31, 32],
        [33, -34, 35, -36, 37, -38, 39, -40],
        [-41, 42, -43, 44, -45, 46, -47, 48],
        [49, -50, 51, -52, 53, -54, 55, -56],
        [-57, 58, -59, 60, -61, 62, -63, 64]
    ], dtype=np.int32)
    sub_np = np.array([
        [ 2, -3,  4, -5,  6, -7,  8, -9],
        [-10, 11, -12, 13, -14, 15, -16, 17],
        [18, -19, 20, -21, 22, -23, 24, -25],
        [-26, 27, -28, 29, -30, 31, -32, 33],
        [34, -35, 36, -37, 38, -39, 40, -41],
        [-42, 43, -44, 45, -46, 47, -48, 49],
        [50, -51, 52, -53, 54, -55, 56, -57],
        [-58, 59, -60, 61, -62, 63, -64, 65]
    ], dtype=np.int32)
    
    # Flatten and convert to binary for DUT input
    x_flat = x_np.flatten()
    sub_flat = sub_np.flatten()
    x_bin = 0
    sub_bin = 0
    for i in range(SEQ_LEN * EMB_DIM):
        x_bin = (x_bin << DATA_WIDTH) | (int(x_flat[i]) & ((1 << DATA_WIDTH) - 1))
        sub_bin = (sub_bin << DATA_WIDTH) | (int(sub_flat[i]) & ((1 << DATA_WIDTH) - 1))
    
    # Drive inputs
    dut.x_in.value = x_bin
    dut.sub_in.value = sub_bin
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for computation to complete
    await wait_for_done(dut)
    
    # Read and verify output
    y_np = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.int32)
    
    # Extract output matrix
    y_out_val = dut.y_out.value
    for i in range(SEQ_LEN):
        for j in range(EMB_DIM):
            idx = ((i * EMB_DIM) + j) * DATA_WIDTH
            bits = (y_out_val >> idx) & ((1 << DATA_WIDTH) - 1)
            if bits & (1 << (DATA_WIDTH - 1)):  # Handle sign extension
                bits = bits - (1 << DATA_WIDTH)
            y_np[SEQ_LEN-1-i, EMB_DIM-1-j] = bits
    
    # Compute reference output
    y_ref = x_np + sub_np
    
    # Debug: Print inputs and outputs
    print(f"x_np:\n{x_np}")
    print(f"sub_np:\n{sub_np}")
    print(f"y_np (DUT output):\n{y_np}")
    print(f"y_ref (Expected):\n{y_ref}")
    
    # Verify results
    assert np.array_equal(y_np, y_ref), f"Output mismatch!\nExpected:\n{y_ref}\nGot:\n{y_np}"
    assert dut.out_valid.value == 1, "out_valid not set"
    assert dut.done.value == 1, "done not set"
    
    # Test reset during computation
    await reset_dut(dut)

@cocotb.test()
async def test_residual_edge_cases(dut):
    """Testbench for residual module with edge case inputs"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Initialize signals
    dut.rst_n.value = 1
    dut.start.value = 0
    dut.x_in.value = 0
    dut.sub_in.value = 0
    
    # Reset the DUT
    await reset_dut(dut)
    
    # Edge case: Maximum and minimum values
    max_val = (1 << (DATA_WIDTH - 1)) - 1  # 32767 for 16-bit
    min_val = -(1 << (DATA_WIDTH - 1))     # -32768 for 16-bit
    x_np = np.array([
        [max_val, min_val, max_val, min_val, max_val, min_val, max_val, min_val],
        [min_val, max_val, min_val, max_val, min_val, max_val, min_val, max_val],
        [max_val, min_val, max_val, min_val, max_val, min_val, max_val, min_val],
        [min_val, max_val, min_val, max_val, min_val, max_val, min_val, max_val],
        [max_val, min_val, max_val, min_val, max_val, min_val, max_val, min_val],
        [min_val, max_val, min_val, max_val, min_val, max_val, min_val, max_val],
        [max_val, min_val, max_val, min_val, max_val, min_val, max_val, min_val],
        [min_val, max_val, min_val, max_val, min_val, max_val, min_val, max_val]
    ], dtype=np.int32)
    sub_np = np.array([
        [min_val, max_val, min_val, max_val, min_val, max_val, min_val, max_val],
        [max_val, min_val, max_val, min_val, max_val, min_val, max_val, min_val],
        [min_val, max_val, min_val, max_val, min_val, max_val, min_val, max_val],
        [max_val, min_val, max_val, min_val, max_val, min_val, max_val, min_val],
        [min_val, max_val, min_val, max_val, min_val, max_val, min_val, max_val],
        [max_val, min_val, max_val, min_val, max_val, min_val, max_val, min_val],
        [min_val, max_val, min_val, max_val, min_val, max_val, min_val, max_val],
        [max_val, min_val, max_val, min_val, max_val, min_val, max_val, min_val]
    ], dtype=np.int32)
    
    # Flatten and convert to binary for DUT input
    x_flat = x_np.flatten()
    sub_flat = sub_np.flatten()
    x_bin = 0
    sub_bin = 0
    for i in range(SEQ_LEN * EMB_DIM):
        x_bin = (x_bin << DATA_WIDTH) | (int(x_flat[i]) & ((1 << DATA_WIDTH) - 1))
        sub_bin = (sub_bin << DATA_WIDTH) | (int(sub_flat[i]) & ((1 << DATA_WIDTH) - 1))
    
    # Drive inputs
    dut.x_in.value = x_bin
    dut.sub_in.value = sub_bin
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for computation to complete
    await wait_for_done(dut)
    
    # Read and verify output
    y_np = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.int32)
    
    # Extract output matrix
    y_out_val = dut.y_out.value
    for i in range(SEQ_LEN):
        for j in range(EMB_DIM):
            idx = ((i * EMB_DIM) + j) * DATA_WIDTH
            bits = (y_out_val >> idx) & ((1 << DATA_WIDTH) - 1)
            if bits & (1 << (DATA_WIDTH - 1)):  # Handle sign extension
                bits = bits - (1 << DATA_WIDTH)
            y_np[SEQ_LEN-1-i, EMB_DIM-1-j] = bits
    
    # Compute reference output
    y_ref = x_np + sub_np
    
    # Debug: Print inputs and outputs
    print(f"x_np:\n{x_np}")
    print(f"sub_np:\n{sub_np}")
    print(f"y_np (DUT output):\n{y_np}")
    print(f"y_ref (Expected):\n{y_ref}")
    
    # Verify results
    assert np.array_equal(y_np, y_ref), f"Output mismatch!\nExpected:\n{y_ref}\nGot:\n{y_np}"
    assert dut.out_valid.value == 1, "out_valid not set"
    assert dut.done.value == 1, "done not set"
    
    # Test reset during computation
    await reset_dut(dut)

async def reset_dut(dut):
    """Reset the DUT"""
    dut.rst_n.value = 0
    await Timer(20, units="ns")
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

async def wait_for_done(dut):
    """Wait for done signal to be asserted"""
    while not dut.done.value:
        await RisingEdge(dut.clk)
