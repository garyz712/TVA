import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotb.clock import Clock
import numpy as np
import random

# Parameters matching the SystemVerilog module
DATA_WIDTH = 16
L = 8  # Sequence length
N = 1  # Number of attention heads
E = 8  # Embedding dimension per head

@cocotb.test()
async def test_attention_score(dut):
    """Testbench for attention_score module"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Initialize signals
    dut.rst_n.value = 1
    dut.start.value = 0
    dut.Q_in.value = 0
    dut.K_in.value = 0
    
    # Reset the DUT
    await reset_dut(dut)
    
    # Generate random input matrices Q and K
    Q_np = np.random.randint(-10, 10, size=(L, N, E)).astype(np.int32)
    K_np = np.random.randint(-10, 10, size=(L, N, E)).astype(np.int32)
    
    # Flatten and convert to binary for DUT input
    Q_flat = Q_np.flatten()
    K_flat = K_np.flatten()
    Q_bin = 0
    K_bin = 0
    for i in range(L * N * E):
        Q_bin = (Q_bin << DATA_WIDTH) | (int(Q_flat[i]) & ((1 << DATA_WIDTH) - 1))
        K_bin = (K_bin << DATA_WIDTH) | (int(K_flat[i]) & ((1 << DATA_WIDTH) - 1))
    
    # Drive inputs
    dut.Q_in.value = Q_bin
    dut.K_in.value = K_bin
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for computation to complete
    await wait_for_done(dut)
    
    # Read and verify output
    A_np = np.zeros((L, N, L), dtype=np.int32)
    
    # Extract output matrix
    A_out_val = dut.A_out.value  # Get the full integer value of A_out
    for l in range(L):
        for n in range(N):
            for l2 in range(L):
                idx = ((l * N * L) + (n * L) + l2) * DATA_WIDTH
                # Extract DATA_WIDTH bits using bit manipulation
                bits = (A_out_val >> idx) & ((1 << DATA_WIDTH) - 1)
                # Convert to signed integer (handle sign extension)
                if bits & (1 << (DATA_WIDTH - 1)):  # If sign bit is set
                    bits = bits - (1 << DATA_WIDTH)
                A_np[l, n, l2] = bits
    
    # Compute reference attention scores
    A_ref = np.zeros((L, N, L), dtype=np.int32)
    for l in range(L):
        for n in range(N):
            for l2 in range(L):
                dot_product = 0
                for e in range(E):
                    dot_product += Q_np[l, n, e] * K_np[l2, n, e]
                A_ref[l, n, l2] = dot_product
    
    # Debug: Print inputs and outputs to diagnose failure
    print(f"Q_np:\n{Q_np}")
    print(f"K_np:\n{K_np}")
    print(f"A_np (DUT output):\n{A_np}")
    print(f"A_ref (Expected):\n{A_ref}")
    
    # Verify results
    assert np.array_equal(A_np, A_ref), f"Output mismatch!\nExpected:\n{A_ref}\nGot:\n{A_np}"
    assert dut.out_valid.value == 1, "out_valid not set"
    assert dut.done.value == 1, "done not set"
    
    # Test reset during computation
    await reset_dut(dut)
    assert dut.out_valid.value == 0, "out_valid not cleared after reset"
    assert dut.done.value == 0, "done not cleared after reset"

@cocotb.test()
async def test_attention_score_fixed(dut):
    """Testbench for attention_score module with fixed inputs"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Initialize signals
    dut.rst_n.value = 1
    dut.start.value = 0
    dut.Q_in.value = 0
    dut.K_in.value = 0
    
    # Reset the DUT
    await reset_dut(dut)
    
    # Fixed input matrices Q and K (shape: L x N x E = 8 x 1 x 8)
    Q_np = np.array([[[ 1, -1,  2,  0, -2,  1,  0, -1]],  # Token 0
                     [[-2,  0,  1, -1,  1, -2,  2,  0]],  # Token 1
                     [[ 0,  2, -1,  1,  0, -1, -2,  1]],  # Token 2
                     [[ 1, -2,  0,  2, -1,  0,  1, -2]],  # Token 3
                     [[-1,  1, -2,  0,  2, -1,  0,  1]],  # Token 4
                     [[ 2,  0, -1,  1, -2,  2, -1,  0]],  # Token 5
                     [[ 0, -1,  2, -2,  1,  0, -2,  1]],  # Token 6
                     [[-1,  2,  0, -1,  0,  1, -1,  2]]], # Token 7
                    dtype=np.int32)
    K_np = np.array([[[ 2,  0, -1,  1, -2,  2, -1,  0]],  # Token 0
                     [[ 0, -1,  2, -2,  1,  0, -2,  1]],  # Token 1
                     [[-1,  2,  0, -1,  0,  1, -1,  2]],  # Token 2
                     [[ 1, -1,  2,  0, -2,  1,  0, -1]],  # Token 3
                     [[-2,  0,  1, -1,  1, -2,  2,  0]],  # Token 4
                     [[ 0,  2, -1,  1,  0, -1, -2,  1]],  # Token 5
                     [[ 1, -2,  0,  2, -1,  0,  1, -2]],  # Token 6
                     [[-1,  1, -2,  0,  2, -1,  0,  1]]], # Token 7
                    dtype=np.int32)
    
    # Flatten and convert to binary for DUT input
    Q_flat = Q_np.flatten()
    K_flat = K_np.flatten()
    Q_bin = 0
    K_bin = 0
    for i in range(L * N * E):
        Q_bin = (Q_bin << DATA_WIDTH) | (int(Q_flat[i]) & ((1 << DATA_WIDTH) - 1))
        K_bin = (K_bin << DATA_WIDTH) | (int(K_flat[i]) & ((1 << DATA_WIDTH) - 1))
    
    # Drive inputs
    dut.Q_in.value = Q_bin
    dut.K_in.value = K_bin
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for computation to complete
    await wait_for_done(dut)
    
    # Read and verify output
    A_np = np.zeros((L, N, L), dtype=np.int32)
    
    # Extract output matrix
    A_out_val = dut.A_out.value  # Get the full integer value of A_out
    for l in range(L):
        for n in range(N):
            for l2 in range(L):
                idx = ((l * N * L) + (n * L) + l2) * DATA_WIDTH
                # Extract DATA_WIDTH bits using bit manipulation
                bits = (A_out_val >> idx) & ((1 << DATA_WIDTH) - 1)
                # Convert to signed integer (handle sign extension)
                if bits & (1 << (DATA_WIDTH - 1)):  # If sign bit is set
                    bits = bits - (1 << DATA_WIDTH)
                A_np[l, n, l2] = bits
    
    # Compute reference attention scores
    A_ref = np.zeros((L, N, L), dtype=np.int32)
    for l in range(L):
        for n in range(N):
            for l2 in range(L):
                dot_product = 0
                for e in range(E):
                    dot_product += Q_np[l, n, e] * K_np[l2, n, e]
                A_ref[l, n, l2] = dot_product
    
    # Debug: Print inputs and outputs
    print(f"Q_np:\n{Q_np}")
    print(f"K_np:\n{K_np}")
    print(f"A_ref (Expected):\n{A_ref}")
    print(f"A_np (DUT output):\n{A_np}")
    
    # Verify results
    assert np.array_equal(A_np, A_ref), f"Output mismatch!\nExpected:\n{A_ref}\nGot:\n{A_np}"
    
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