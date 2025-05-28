import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

# Helper functions (provided by user)
def real_to_q1_15(x):
    val = int(x * (2**15))
    if val > 32767:
        val = 32767
    if val < -32768:
        val = -32768
    return np.array(val).astype(np.int16)

def q1_15_to_real(x):
    value = np.array(int(x) & 0xFFFF).astype(np.int16)
    return float(value) / (2**15)

def real_to_q1_30(x):
    val = int(x * (2**30))
    if val > (2**31 - 1):
        val = (2**31 - 1)
    if val < -(2**31):
        val = -(2**31)
    return np.array(val).astype(np.int32)

def q1_30_to_real(x):
    value = np.array(int(x) & 0xFFFFFFFF).astype(np.int32)
    return float(value) / (2**30)

def compute_expected(a, b, bias):
    """Compute expected output for MLP: y = x @ W^T + b."""
    M, K = a.shape  # a: (1, HIDDEN_DIM)
    K, N = b.shape  # b: (HIDDEN_DIM, MLP_DIM)
    expected_out = np.zeros((M, N), dtype=float)
    for k in range(K):
        for i in range(M):
            raw_a = int(a[i, k])
            a_val = q1_15_to_real(raw_a)
            for j in range(N):
                raw_b = int(b[k, j])
                b_val = q1_15_to_real(raw_b)
                expected_out[i, j] += a_val * b_val
        expected_out = np.clip(expected_out, -2.0, 1.999999999)
    expected_out = np.clip(expected_out, -1.0, 1.0-2**-15)
    for i in range(M):
        for j in range(N):
            expected_out[i, j] = expected_out[i, j] + q1_15_to_real(bias[j])
            expected_out[i, j] = np.clip(expected_out[i, j], -1.0, 1.0 - 2**-15)
    return expected_out

@cocotb.test()
async def mlp_test(dut):
    """Testbench for MLP module."""
    # Parameters
    HIDDEN_DIM = 16
    MLP_DIM = 64
    DATA_WIDTH = 16

    # Start clock (10ns period)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset the DUT
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.valid_in.value = 0
    for i in range(HIDDEN_DIM):
        dut.x[i].value = 0
    for i in range(HIDDEN_DIM * MLP_DIM):
        dut.W[i].value = 0
    for i in range(MLP_DIM):
        dut.b[i].value = 0
    await Timer(20, units="ns")
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Generate random inputs
    np.random.seed(42)  # For reproducibility
    x_real = np.random.uniform(-1, 1, (1, HIDDEN_DIM))
    W_real = np.random.uniform(-1, 1, (HIDDEN_DIM, MLP_DIM))
    b_real = np.random.uniform(-1, 1, MLP_DIM)

    # Convert to Q1.15 format
    x_q15 = np.array([real_to_q1_15(val) for val in x_real.flatten()]).reshape(1, HIDDEN_DIM)
    W_q15 = np.array([real_to_q1_15(val) for val in W_real.flatten()]).reshape(HIDDEN_DIM, MLP_DIM)
    b_q15 = np.array([real_to_q1_15(val) for val in b_real])

    # Compute expected output
    expected_out_real = compute_expected(x_q15, W_q15, b_q15)
    expected_out_q15 = np.array([real_to_q1_15(val) for val in expected_out_real.flatten()])

    # Drive inputs
    for i in range(HIDDEN_DIM):
        dut.x[i].value = int(x_q15[0, i]) & 0xFFFF
    for i in range(HIDDEN_DIM):
        for j in range(MLP_DIM):
            dut.W[i * MLP_DIM + j].value = int(W_q15[i, j]) & 0xFFFF
    for i in range(MLP_DIM):
        dut.b[i].value = int(b_q15[i]) & 0xFFFF

    # Start computation
    await RisingEdge(dut.clk)
    dut.valid_in.value = 1
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    dut.valid_in.value = 0

    # Wait for computation to complete
    timeout = 1000  # Max cycles to wait
    while dut.done.value == 0 and timeout > 0:
        await RisingEdge(dut.clk)
        timeout -= 1
    assert timeout > 0, "Test timed out waiting for done signal"

    # Check valid_out and done signals
    assert dut.valid_out.value == 1, "valid_out not set"
    assert dut.done.value == 1, "done not set"

    # Read and verify outputs
    y_out = np.zeros(MLP_DIM, dtype=np.int16)
    for i in range(MLP_DIM):
        y_out[i] = dut.y[i].value.signed_integer

    # Compare outputs (allow small tolerance for fixed-point errors)
    tolerance = 2  # Allow 2 LSBs of error due to fixed-point arithmetic
    for i in range(MLP_DIM):
        actual = y_out[i]
        expected = expected_out_q15[i]
        assert abs(actual - expected) <= tolerance, \
            f"Output y[{i}] mismatch: expected {expected}, got {actual}"

    # Wait a few more cycles to ensure stability
    for _ in range(5):
        await RisingEdge(dut.clk)