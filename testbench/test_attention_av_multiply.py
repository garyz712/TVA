import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np
import random

# Parameters matching the module
DATA_WIDTH = 16
L = 8
N = 1
E = 8

async def reset_dut(dut):
    """Reset the DUT."""
    dut.rst_n.value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await Timer(20, units="ns")

def pack_array(data, width):
    """Pack a numpy array into a flat integer for Verilog input."""
    flat = 0
    for i, val in enumerate(data.flatten()):
        flat |= (int(val) & ((1 << width) - 1)) << (i * width)
    return flat

def unpack_array(data, width, shape):
    """Unpack a flat integer into a numpy array."""
    result = np.zeros(shape, dtype=np.int32)
    for i in range(np.prod(shape)):
        result.flat[i] = (data >> (i * width)) & ((1 << width) - 1)
    return result

def software_model(A, V, token_precision):
    """Software model for attention computation with adaptive precision."""
    Z = np.zeros((L, N, E), dtype=np.float32)
    for l in range(L):
        for n in range(N):
            for e in range(E):
                for l2 in range(L):
                    # Downcast A and V based on token_precision
                    if token_precision[l2] == 0:  # INT4 (Q1.3)
                        a_val = (A[l][n][l2] >> 12) & 0xF  # Take lower 4 bits
                        v_val = (V[l2][n][e] >> 12) & 0xF
                    elif token_precision[l2] == 1:  # INT8 (Q1.7)
                        a_val = (A[l][n][l2] >> 8) & 0xFF  # Take lower 8 bits
                        v_val = (V[l2][n][e] >> 8) & 0xFF
                    else:  # FP16 (full 16 bits)
                        a_val = A[l][n][l2]
                        v_val = V[l2][n][e]
                    # Multiply and accumulate (upcast to float for precision)
                    Z[l][n][e] += a_val * v_val
    return Z

@cocotb.test()
async def test_attention_av_multiply(dut):
    """Test the attention_av_multiply module."""
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset DUT
    await reset_dut(dut)

    # Test cases: different precision settings
    precision_cases = [
        [0] * L,  # All INT4
        [1] * L,  # All INT8
        [2] * L,  # All FP16
    ]

    for case_idx, token_precision in enumerate(precision_cases):
        dut._log.info(f"Running test case {case_idx}: token_precision = {token_precision}")

        # Generate random input data
        A = np.random.randint(0, 1 << DATA_WIDTH, (L, N, L), dtype=np.int32)
        V = np.random.randint(0, 1 << DATA_WIDTH, (L, N, E), dtype=np.int32)

        # Pack inputs
        A_in = pack_array(A, DATA_WIDTH)
        V_in = pack_array(V, DATA_WIDTH)

        # Set inputs
        dut.A_in.value = A_in
        dut.V_in.value = V_in
        for i in range(L):
            dut.token_precision[i].value = token_precision[i]
        dut.start.value = 0

        # Wait for reset to settle
        await RisingEdge(dut.clk)

        # Start computation
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0

        # Wait for done signal
        for _ in range(100):  # Timeout after 100 cycles
            await RisingEdge(dut.clk)
            if dut.done.value == 1:
                break
        else:
            raise RuntimeError("Timeout waiting for done signal")

        # Check output valid
        assert dut.out_valid.value == 1, "Output valid not set"

        # Read output
        Z_out = dut.Z_out.value
        Z_hw = unpack_array(Z_out, DATA_WIDTH, (L, N, E))

        # Compute expected output
        Z_sw = software_model(A, V, token_precision)

        # Compare results (allowing small differences due to quantization)
        tolerance = 1  # Adjust based on precision
        assert np.all(np.abs(Z_hw - Z_sw) <= tolerance), \
            f"Test case {case_idx} failed: Z_hw =\n{Z_hw}\nZ_sw =\n{Z_sw}"

        dut._log.info(f"Test case {case_idx} passed")

    dut._log.info("All test cases passed!")
