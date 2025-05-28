import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

# Helper functions (from previous context)
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

def compute_expected(x_in, WQ, WK, WV, WO, W1, b1, ln1_gamma, ln1_beta, ln2_gamma, ln2_beta):
    """Compute expected output for ViT encoder block."""
    SEQ_LEN, EMB_DIM = x_in.shape  # (SEQ_LEN, EMB_DIM)
    H_MLP = W1.shape[1]  # H_MLP from W1 (EMB_DIM, H_MLP)

    # LayerNorm 1
    ln1_out = np.zeros_like(x_in, dtype=float)
    for i in range(SEQ_LEN):
        x_mean = np.mean([q1_15_to_real(x_in[i, j]) for j in range(EMB_DIM)])
        x_var = np.var([q1_15_to_real(x_in[i, j]) for j in range(EMB_DIM)])
        for j in range(EMB_DIM):
            x_norm = (q1_15_to_real(x_in[i, j]) - x_mean) / np.sqrt(x_var + 1e-5)
            ln1_out[i, j] = x_norm * q1_15_to_real(ln1_gamma[j]) + q1_15_to_real(ln1_beta[j])
        ln1_out[i] = np.clip(ln1_out[i], -1.0, 1.0 - 2**-15)

    # Self-Attention (Q = XW_Q, K = XW_K, V = XW_V, A = softmax(QK^T/sqrt(E))V, Out = AW_O)
    Q = ln1_out @ np.array([[q1_15_to_real(WQ[i, j]) for j in range(EMB_DIM)] for i in range(EMB_DIM)])
    K = ln1_out @ np.array([[q1_15_to_real(WK[i, j]) for j in range(EMB_DIM)] for i in range(EMB_DIM)])
    V = ln1_out @ np.array([[q1_15_to_real(WV[i, j]) for j in range(EMB_DIM)] for i in range(EMB_DIM)])
    scores = Q @ K.T / np.sqrt(EMB_DIM)
    softmax_scores = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    attn_out = softmax_scores @ V
    attn_out = attn_out @ np.array([[q1_15_to_real(WO[i, j]) for j in range(EMB_DIM)] for i in range(EMB_DIM)])
    attn_out = np.clip(attn_out, -1.0, 1.0 - 2**-15)

    # Residual 1: z1 = x_in + attn_out
    res1_out = np.array([[q1_15_to_real(x_in[i, j]) for j in range(EMB_DIM)] for i in range(SEQ_LEN)]) + attn_out
    res1_out = np.clip(res1_out, -1.0, 1.0 - 2**-15)

    # LayerNorm 2
    ln2_out = np.zeros_like(res1_out)
    for i in range(SEQ_LEN):
        x_mean = np.mean(res1_out[i])
        x_var = np.var(res1_out[i])
        for j in range(EMB_DIM):
            x_norm = (res1_out[i, j] - x_mean) / np.sqrt(x_var + 1e-5)
            ln2_out[i, j] = x_norm * q1_15_to_real(ln2_gamma[j]) + q1_15_to_real(ln2_beta[j])
        ln2_out[i] = np.clip(ln2_out[i], -1.0, 1.0 - 2**-15)

    # MLP (single-layer: y = x @ W1^T + b1, truncate to EMB_DIM)
    mlp_out = np.zeros((SEQ_LEN, EMB_DIM), dtype=float)
    W1_real = np.array([[q1_15_to_real(W1[i, j]) for j in range(H_MLP)] for i in range(EMB_DIM)])
    b1_real = np.array([q1_15_to_real(b1[j]) for j in range(H_MLP)])
    for i in range(SEQ_LEN):
        mlp_hidden = ln2_out[i] @ W1_real + b1_real
        mlp_hidden = np.clip(mlp_hidden, -1.0, 1.0 - 2**-15)  # ReLU approximation
        mlp_out[i] = mlp_hidden[:EMB_DIM]  # Truncate to EMB_DIM
        mlp_out[i] = np.clip(mlp_out[i], -1.0, 1.0 - 2**-15)

    # Residual 2: z2 = z1 + mlp_out
    res2_out = res1_out + mlp_out
    res2_out = np.clip(res2_out, -1.0, 1.0 - 2**-15)
    return res2_out

@cocotb.test()
async def vit_encoder_block_test(dut):
    """Testbench for vit_encoder_block module."""
    # Parameters
    DATA_WIDTH = 16
    SEQ_LEN = 8
    EMB_DIM = 8
    H_MLP = 32

    # Start clock (10ns period)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset the DUT
    dut.rst_n.value = 0
    dut.start.value = 0
    for i in range(SEQ_LEN * EMB_DIM):
        dut.x_in[i].value = 0
        dut.WQ_in[i].value = 0
        dut.WK_in[i].value = 0
        dut.WV_in[i].value = 0
        dut.WO_in[i].value = 0
    for i in range(EMB_DIM * H_MLP):
        dut.W1_in[i].value = 0
    for i in range(H_MLP):
        dut.b1_in[i].value = 0
    for i in range(EMB_DIM):
        dut.ln1_gamma[i].value = 0
        dut.ln1_beta[i].value = 0
        dut.ln2_gamma[i].value = 0
        dut.ln2_beta[i].value = 0
    await Timer(20, units="ns")
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Generate random inputs
    np.random.seed(42)  # For reproducibility
    x_in_real = np.random.uniform(-1, 1, (SEQ_LEN, EMB_DIM))
    WQ_real = np.random.uniform(-1, 1, (EMB_DIM, EMB_DIM))
    WK_real = np.random.uniform(-1, 1, (EMB_DIM, EMB_DIM))
    WV_real = np.random.uniform(-1, 1, (EMB_DIM, EMB_DIM))
    WO_real = np.random.uniform(-1, 1, (EMB_DIM, EMB_DIM))
    W1_real = np.random.uniform(-1, 1, (EMB_DIM, H_MLP))
    b1_real = np.random.uniform(-1, 1, H_MLP)
    ln1_gamma_real = np.random.uniform(0.5, 1.5, EMB_DIM)  # Non-zero for normalization
    ln1_beta_real = np.random.uniform(-0.5, 0.5, EMB_DIM)
    ln2_gamma_real = np.random.uniform(0.5, 1.5, EMB_DIM)
    ln2_beta_real = np.random.uniform(-0.5, 0.5, EMB_DIM)

    # Convert to Q1.15 format
    x_in_q15 = np.array([real_to_q1_15(val) for val in x_in_real.flatten()]).reshape(SEQ_LEN, EMB_DIM)
    WQ_q15 = np.array([real_to_q1_15(val) for val in WQ_real.flatten()]).reshape(EMB_DIM, EMB_DIM)
    WK_q15 = np.array([real_to_q1_15(val) for val in WK_real.flatten()]).reshape(EMB_DIM, EMB_DIM)
    WV_q15 = np.array([real_to_q1_15(val) for val in WV_real.flatten()]).reshape(EMB_DIM, EMB_DIM)
    WO_q15 = np.array([real_to_q1_15(val) for val in WO_real.flatten()]).reshape(EMB_DIM, EMB_DIM)
    W1_q15 = np.array([real_to_q1_15(val) for val in W1_real.flatten()]).reshape(EMB_DIM, H_MLP)
    b1_q15 = np.array([real_to_q1_15(val) for val in b1_real])
    ln1_gamma_q15 = np.array([real_to_q1_15(val) for val in ln1_gamma_real])
    ln1_beta_q15 = np.array([real_to_q1_15(val) for val in ln1_beta_real])
    ln2_gamma_q15 = np.array([real_to_q1_15(val) for val in ln2_gamma_real])
    ln2_beta_q15 = np.array([real_to_q1_15(val) for val in ln2_beta_real])

    # Compute expected output
    expected_out_real = compute_expected(x_in_q15, WQ_q15, WK_q15, WV_q15, WO_q15, W1_q15, b1_q15, ln1_gamma_q15, ln1_beta_q15, ln2_gamma_q15, ln2_beta_q15)
    expected_out_q15 = np.array([real_to_q1_15(val) for val in expected_out_real.flatten()])

    # Drive inputs
    for i in range(SEQ_LEN * EMB_DIM):
        dut.x_in[i].value = int(x_in_q15.flatten()[i]) & 0xFFFF
    for i in range(EMB_DIM * EMB_DIM):
        dut.WQ_in[i].value = int(WQ_q15.flatten()[i]) & 0xFFFF
        dut.WK_in[i].value = int(WK_q15.flatten()[i]) & 0xFFFF
        dut.WV_in[i].value = int(WV_q15.flatten()[i]) & 0xFFFF
        dut.WO_in[i].value = int(WO_q15.flatten()[i]) & 0xFFFF
    for i in range(EMB_DIM * H_MLP):
        dut.W1_in[i].value = int(W1_q15.flatten()[i]) & 0xFFFF
    for i in range(H_MLP):
        dut.b1_in[i].value = int(b1_q15[i]) & 0xFFFF
    for i in range(EMB_DIM):
        dut.ln1_gamma[i].value = int(ln1_gamma_q15[i]) & 0xFFFF
        dut.ln1_beta[i].value = int(ln1_beta_q15[i]) & 0xFFFF
        dut.ln2_gamma[i].value = int(ln2_gamma_q15[i]) & 0xFFFF
        dut.ln2_beta[i].value = int(ln2_beta_q15[i]) & 0xFFFF

    # Start computation
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for computation to complete
    timeout = 100000  # Max cycles to wait
    while dut.done.value == 0 and timeout > 0:
        await RisingEdge(dut.clk)
        timeout -= 1
    assert timeout > 0, "Test timed out waiting for done signal"

    # Check out_valid and done signals
    assert dut.out_valid.value == 1, "out_valid not set"
    assert dut.done.value == 1, "done not set"

    # Read and verify outputs
    out_block = np.zeros(SEQ_LEN * EMB_DIM, dtype=np.int16)
    for i in range(SEQ_LEN * EMB_DIM):
        out_block[i] = dut.out_block[i].value.signed_integer

    # Compare outputs (allow small tolerance for fixed-point errors)
    tolerance = 4  # Allow 4 LSBs of error due to fixed-point arithmetic
    for i in range(SEQ_LEN * EMB_DIM):
        actual = out_block[i]
        expected = expected_out_q15[i]
        assert abs(actual - expected) <= tolerance, \
            f"Output out_block[{i}] mismatch: expected {expected}, got {actual}"

    # Wait a few more cycles to ensure stability
    for _ in range(5):
        await RisingEdge(dut.clk)