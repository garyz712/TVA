import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

# Parameters
A_ROWS = 8
V_COLS = 32
NUM_COLS = 8
TILE_SIZE = 8
WIDTH_INT4 = 4
WIDTH_INT8 = 8
WIDTH_FP16 = 16
NUM_TILES = V_COLS // TILE_SIZE  # 4
CYCLES_INT4 = 1
CYCLES_INT8 = 2
CYCLES_FP16 = 4
TOLERANCE = 0.001  # FP16 tolerance for Q1.15

# Helper function: Convert real to Q1.15 (FP16)
def real_to_q1_15(x):
    val = int(x * (2**15))
    if val > 32767:
        val = 32767  # Clamp to max positive
    if val < -32768:
        val = -32768  # Clamp to min negative
    return val & 0xFFFF  # Ensure 16-bit

# Helper function: Convert Q1.15 to real
def q1_15_to_real(x):
    return float(np.int16(x)) / (2**15)

# Helper function: Convert real to Q1.7 (INT8)
def real_to_q1_7(x):
    val = int(x * (2**7))
    if val > 127:
        val = 127
    if val < -128:
        val = -128
    return val & 0xFF

# Helper function: Convert real to Q1.3 (INT4)
def real_to_q1_3(x):
    val = int(x * (2**3))
    if val > 7:
        val = 7
    if val < -8:
        val = -8
    return val & 0xF

# Helper function: Compute expected outer product
def compute_expected(a_mem, v_mem, precision_sel):
    expected_out = np.zeros((A_ROWS, V_COLS), dtype=float)
    for k in range(NUM_COLS):
        prec = precision_sel[k]
        for i in range(A_ROWS):
            a_val = q1_15_to_real(a_mem[i][k])
            if prec == 0:  # INT4 (Q1.3)
                a_val = round(a_val * (2**3)) / (2**3)
            elif prec == 1:  # INT8 (Q1.7)
                a_val = round(a_val * (2**7)) / (2**7)
            # FP16 (Q1.15): no truncation
            for j in range(V_COLS):
                v_val = q1_15_to_real(v_mem[k][j])
                if prec == 0:  # INT4
                    v_val = round(v_val * (2**3)) / (2**3)
                elif prec == 1:  # INT8
                    v_val = round(v_val * (2**7)) / (2**7)
                prod = a_val * v_val
                expected_out[i, j] += prod
    # Convert to Q1.15
    expected_out_q = np.zeros((A_ROWS, V_COLS), dtype=np.int16)
    for i in range(A_ROWS):
        for j in range(V_COLS):
            expected_out_q[i, j] = real_to_q1_15(expected_out[i, j])
    return expected_out_q

# Helper function: Verify output
async def verify_output(dut, expected_out):
    errors = 0
    for i in range(A_ROWS):
        for j in range(V_COLS):
            actual = q1_15_to_real(dut.out_mem[i][j].value)
            expected = q1_15_to_real(expected_out[i, j])
            diff = actual - expected
            if abs(diff) > TOLERANCE:
                dut._log.error(f"Error at out_mem[{i}][{j}]: actual={actual}, expected={expected}")
                errors += 1
    if errors == 0:
        dut._log.info("Test passed: Output matches expected.")
    else:
        dut._log.error(f"Test failed: {errors} errors found.")
    return errors

# Reset DUT
async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await Timer(10, units="ns")

@cocotb.test()
async def test_attention_av_multiply(dut):
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")  # 100 MHz
    cocotb.start_soon(clock.start())

    # Initialize DUT
    await reset_dut(dut)

    # Test Case 1: All INT4
    dut._log.info("\nTest Case 1: All INT4")
    a_mem = np.zeros((A_ROWS, NUM_COLS), dtype=np.int16)
    v_mem = np.zeros((NUM_COLS, V_COLS), dtype=np.int16)
    precision_sel = [0] * NUM_COLS  # INT4 (00)
    for k in range(NUM_COLS):
        for i in range(A_ROWS):
            a_mem[i, k] = real_to_q1_15(0.125 * (i + k + 1))  # e.g., 0.125, 0.25
        for j in range(V_COLS):
            v_mem[k, j] = real_to_q1_15(0.0625 * (j + k + 1))  # e.g., 0.0625, 0.125
    # Drive inputs
    for k in range(NUM_COLS):
        dut.precision_sel[k].value = precision_sel[k]
        for i in range(A_ROWS):
            dut.a_mem[i][k].value = a_mem[i, k]
        for j in range(V_COLS):
            dut.v_mem[k][j].value = v_mem[k, j]
    expected_out = compute_expected(a_mem, v_mem, precision_sel)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.done)
    await Timer(10, units="ns")
    await verify_output(dut, expected_out)

    # Test Case 2: All INT8
    dut._log.info("\nTest Case 2: All INT8")
    precision_sel = [1] * NUM_COLS  # INT8 (01)
    for k in range(NUM_COLS):
        for i in range(A_ROWS):
            a_mem[i, k] = real_to_q1_15(0.03125 * (i + k + 1))  # e.g., 0.03125
        for j in range(V_COLS):
            v_mem[k, j] = real_to_q1_15(0.015625 * (j + k + 1))  # e.g., 0.015625
    for k in range(NUM_COLS):
        dut.precision_sel[k].value = precision_sel[k]
        for i in range(A_ROWS):
            dut.a_mem[i][k].value = a_mem[i, k]
        for j in range(V_COLS):
            dut.v_mem[k][j].value = v_mem[k, j]
    expected_out = compute_expected(a_mem, v_mem, precision_sel)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.done)
    await Timer(10, units="ns")
    await verify_output(dut, expected_out)

    # Test Case 3: All FP16
    dut._log.info("\nTest Case 3: All FP16")
    precision_sel = [2] * NUM_COLS  # FP16 (10)
    for k in range(NUM_COLS):
        for i in range(A_ROWS):
            a_mem[i, k] = real_to_q1_15(0.01 * (i + k + 1))  # e.g., 0.01
        for j in range(V_COLS):
            v_mem[k, j] = real_to_q1_15(0.005 * (j + k + 1))  # e.g., 0.005
    for k in range(NUM_COLS):
        dut.precision_sel[k].value = precision_sel[k]
        for i in range(A_ROWS):
            dut.a_mem[i][k].value = a_mem[i, k]
        for j in range(V_COLS):
            dut.v_mem[k][j].value = v_mem[k, j]
    expected_out = compute_expected(a_mem, v_mem, precision_sel)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.done)
    await Timer(10, units="ns")
    await verify_output(dut, expected_out)

    # Test Case 4: Mixed Precisions
    dut._log.info("\nTest Case 4: Mixed Precisions")
    precision_sel = [0, 1, 2, 0, 1, 2, 0, 1]  # INT4, INT8, FP16, ...
    for k in range(NUM_COLS):
        for i in range(A_ROWS):
            a_mem[i, k] = real_to_q1_15(0.05 * (i + k + 1))
        for j in range(V_COLS):
            v_mem[k, j] = real_to_q1_15(0.025 * (j + k + 1))
    for k in range(NUM_COLS):
        dut.precision_sel[k].value = precision_sel[k]
        for i in range(A_ROWS):
            dut.a_mem[i][k].value = a_mem[i, k]
        for j in range(V_COLS):
            dut.v_mem[k][j].value = v_mem[k, j]
    expected_out = compute_expected(a_mem, v_mem, precision_sel)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.done)
    await Timer(10, units="ns")
    await verify_output(dut, expected_out)

    # Test Case 5: Zero Inputs
    dut._log.info("\nTest Case 5: Zero Inputs")
    precision_sel = [2] * NUM_COLS  # FP16
    a_mem.fill(0)
    v_mem.fill(0)
    for k in range(NUM_COLS):
        dut.precision_sel[k].value = precision_sel[k]
        for i in range(A_ROWS):
            dut.a_mem[i][k].value = a_mem[i, k]
        for j in range(V_COLS):
            dut.v_mem[k][j].value = v_mem[k, j]
    expected_out = compute_expected(a_mem, v_mem, precision_sel)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.done)
    await Timer(10, units="ns")
    await verify_output(dut, expected_out)

    # Test Case 6: Random Inputs
    dut._log.info("\nTest Case 6: Random Inputs")
    rng = np.random.default_rng()
    for k in range(NUM_COLS):
        precision_sel[k] = rng.integers(0, 3)  # Random: 0, 1, or 2
        for i in range(A_ROWS):
            a_mem[i, k] = real_to_q1_15(rng.uniform(-0.5, 0.5))
        for j in range(V_COLS):
            v_mem[k, j] = real_to_q1_15(rng.uniform(-0.5, 0.5))
    for k in range(NUM_COLS):
        dut.precision_sel[k].value = precision_sel[k]
        for i in range(A_ROWS):
            dut.a_mem[i][k].value = a_mem[i, k]
        for j in range(V_COLS):
            dut.v_mem[k][j].value = v_mem[k, j]
    expected_out = compute_expected(a_mem, v_mem, precision_sel)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.done)
    await Timer(10, units="ns")
    await verify_output(dut, expected_out)

    dut._log.info("All tests completed.")
