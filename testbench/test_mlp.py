import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

# Helper functions
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

def compute_expected(a, b1, bias1, b2, bias2):
    """Compute expected output for two-layer MLP: y = ReLU(x @ W1^T + b1) @ W2^T + b2."""
    M, K = a.shape  # a: (1, HIDDEN_DIM)
    K, N = b1.shape  # b1: (HIDDEN_DIM, MLP_DIM)
    P, Q = b2.shape  # b2: (MLP_DIM, OUT_DIM)
    
    # First layer: x @ W1^T + b1
    hidden_out = np.zeros((M, N), dtype=float)
    for k in range(K):
        for i in range(M):
            raw_a = int(a[i, k])
            a_val = q1_15_to_real(raw_a)
            for j in range(N):
                raw_b1 = int(b1[k, j])
                b1_val = q1_15_to_real(raw_b1)
                hidden_out[i, j] += a_val * b1_val
        hidden_out = np.clip(hidden_out, -2.0, 2.0-2**-30)
    hidden_out = np.clip(hidden_out, -1.0, 1.0-2**-15)
    for i in range(M):
        for j in range(N):
            hidden_out[i, j] = hidden_out[i, j] + q1_15_to_real(bias1[j])
            hidden_out[i, j] = max(0, hidden_out[i, j])  # ReLU
            hidden_out[i, j] = np.clip(hidden_out[i, j], -1.0, 1.0 - 2**-15)
    
    # Second layer: ReLU(hidden) @ W2^T + b2
    final_out = np.zeros((M, Q), dtype=float)
    for k in range(N):
        for i in range(M):
            hidden_val = hidden_out[i, k]
            for j in range(Q):
                raw_b2 = int(b2[k, j])
                b2_val = q1_15_to_real(raw_b2)
                final_out[i, j] += hidden_val * b2_val
        final_out = np.clip(final_out, -2.0, 2.0-2**-30)
    final_out = np.clip(final_out, -1.0, 1.0-2**-15)
    for i in range(M):
        for j in range(Q):
            final_out[i, j] = final_out[i, j] + q1_15_to_real(bias2[j])
            final_out[i, j] = np.clip(final_out[i, j], -1.0, 1.0 - 2**-15)
    
    return final_out, hidden_out

@cocotb.test()
async def mlp_test(dut):
    """Testbench for two-layer MLP module."""
    # Parameters
    HIDDEN_DIM = 16   # Matches EMB_DIM
    MLP_DIM = 64     # Matches H_MLP
    OUT_DIM = 16      # Matches EMB_DIM
    DATA_WIDTH = 16  # Q1.15 format

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
        dut.W1[i].value = 0
    for i in range(MLP_DIM * OUT_DIM):
        dut.W2[i].value = 0
    for i in range(MLP_DIM):
        dut.b1[i].value = 0
    for i in range(OUT_DIM):
        dut.b2[i].value = 0
    await Timer(20, units="ns")
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Generate random inputs
    np.random.seed(42)  # For reproducibility
    x_real = np.random.uniform(-1, 1, (1, HIDDEN_DIM))
    W1_real = np.random.uniform(-1, 1, (HIDDEN_DIM, MLP_DIM))
    b1_real = np.random.uniform(-1, 1, MLP_DIM)
    W2_real = np.random.uniform(-1, 1, (MLP_DIM, OUT_DIM))
    b2_real = np.random.uniform(-1, 1, OUT_DIM)

    # Convert to Q1.15 format
    x_q15 = np.array([real_to_q1_15(val) for val in x_real.flatten()]).reshape(1, HIDDEN_DIM)
    W1_q15 = np.array([real_to_q1_15(val) for val in W1_real.flatten()]).reshape(HIDDEN_DIM, MLP_DIM)
    b1_q15 = np.array([real_to_q1_15(val) for val in b1_real])
    W2_q15 = np.array([real_to_q1_15(val) for val in W2_real.flatten()]).reshape(MLP_DIM, OUT_DIM)
    b2_q15 = np.array([real_to_q1_15(val) for val in b2_real])

    # Compute expected output
    expected_out_real, hidden_out_real = compute_expected(x_q15, W1_q15, b1_q15, W2_q15, b2_q15)
    expected_out_q15 = np.array([real_to_q1_15(val) for val in expected_out_real.flatten()])
    hidden_out_q15 = np.array([real_to_q1_15(val) for val in hidden_out_real.flatten()])

    # Drive inputs
    for i in range(HIDDEN_DIM):
        dut.x[i].value = int(x_q15[0, i]) & 0xFFFF
    for i in range(HIDDEN_DIM):
        for j in range(MLP_DIM):
            dut.W1[i * MLP_DIM + j].value = int(W1_q15[i, j]) & 0xFFFF
    for i in range(MLP_DIM):
        for j in range(OUT_DIM):
            dut.W2[i * OUT_DIM + j].value = int(W2_q15[i, j]) & 0xFFFF
    for i in range(MLP_DIM):
        dut.b1[i].value = int(b1_q15[i]) & 0xFFFF
    for i in range(OUT_DIM):
        dut.b2[i].value = int(b2_q15[i]) & 0xFFFF

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
    h_out = np.zeros(MLP_DIM, dtype=np.int16)
    for i in range(MLP_DIM):
        h_out[i] = dut.y1[i].value.signed_integer
    
    y_out = np.zeros(OUT_DIM, dtype=np.int16)
    for i in range(OUT_DIM):
        y_out[i] = dut.y[i].value.signed_integer

    # Compare outputs (allow small tolerance for fixed-point errors)
    tolerance = 4  # Increased to 4 LSBs due to two-layer computation

    for i in range(MLP_DIM):
        actual = h_out[i]
        expected = hidden_out_q15[i]
        if abs(actual - expected) > tolerance:
            dut._log.error(f"Output h[{i}] mismatch: expected {expected}, got {actual}")
        else:
            dut._log.info(f"Output h[{i}] match: expected {expected}, got {actual}")

    for i in range(OUT_DIM):
        actual = y_out[i]
        expected = expected_out_q15[i]
        if abs(actual - expected) > tolerance:
            dut._log.error(f"Output y[{i}] mismatch: expected {expected}, got {actual}")
        else:
            dut._log.info(f"Output y[{i}] match: expected {expected}, got {actual}")

    # Wait a few more cycles to ensure stability
    for _ in range(5):
        await RisingEdge(dut.clk)