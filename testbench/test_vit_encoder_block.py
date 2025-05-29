import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

# Helper functions
def real_to_q1_15(x):
    val = int(x * np.int32(2**15))
    if val > 32767:
        val = 32767
    if val < -32768:
        val = -32768
    return np.array(val).astype(np.int16)

def q1_15_to_real(x):
    value = np.array(int(x) & 0xFFFF).astype(np.int16)
    return float(value) / (2**15)

def real_to_q1_30(x):
    """Convert float to Q1.30 format."""
    val = int(x * (2**30))
    if val > 2**31 - 1:
        val = 2**31 - 1
    if val < -(2**31):
        val = -(2**31)
    return val

def nibble_to_real_q1_3(x):
    """Convert 4-bit nibble to real value in Q1.3 format."""
    x = x & 0xF  # Ensure 4-bit
    if x & 0x8:  # Sign extend
        x = x - 0x10
    return float(x) / (2**3)

def hw_multiply_inv_sqrt8(x):
    """Placeholder for hardware inverse square root approximation."""
    # As per the reference, use a fixed value of 10 for non-zero inputs
    return 10 if x != 0 else 0

def compute_layer_norm_reference(x_np, gamma_np, beta_np, seq_len, emb_dim, data_width=16, epsilon=0x34000000):
    """Software model for LayerNorm that mirrors RTL maths (incl. fake inv-sqrt)."""
    out = np.zeros_like(x_np, dtype=np.int32)
    maxv = (1 << (data_width - 1)) - 1
    minv = -(1 << (data_width - 1))
    for i in range(seq_len):
        mean = np.sum(x_np[i]) // emb_dim
        diff = x_np[i] - mean
        var = np.sum(diff * diff) // emb_dim
        denom = var + epsilon
        inv_std = 10 if denom != 0 else 0  # Placeholder inverse-sqrt
        for j in range(emb_dim):
            tmp = (x_np[i, j] - mean) * inv_std
            val = (tmp * gamma_np[j] + beta_np[j]) # >> 15  # Q1.15 multiplication
            out[i, j] = np.clip(val, minv, maxv)
    return out

def compute_attention_reference(x_np, wq_np, wk_np, wv_np, wo_np, seq_len, emb_dim, n=1):
    """Compute expected output for attention module."""
    l, e = seq_len, emb_dim
    x_np = x_np.reshape(l * n, e)
    wq_np = wq_np.reshape(e, e)
    wk_np = wk_np.reshape(e, e)
    wv_np = wv_np.reshape(e, e)
    wo_np = wo_np.reshape(e, e)

    # Step 1: QKV generation
    def matmul_expected(a, b):
        expected_out = np.zeros((l * n, e), dtype=float)
        for k in range(e):
            for i in range(l * n):
                raw_a = int(a[i, k])
                a_val = q1_15_to_real(raw_a)
                for j in range(e):
                    raw_b = int(b[k, j])
                    b_val = q1_15_to_real(raw_b)
                    expected_out[i, j] += a_val * b_val
            expected_out = np.clip(expected_out, -2.0, 1.999999999)
        expected_q30 = np.vectorize(real_to_q1_30)(expected_out)
        expected_q15 = []
        for q30_val in expected_q30.flatten():
            sign_bit = (q30_val >> 31) & 1
            int_bit = (q30_val >> 30) & 1
            if sign_bit == int_bit:
                q15_val = (sign_bit << 15) | ((q30_val >> 15) & 0x7FFF)
            else:
                q15_val = 0x8000 if sign_bit else 0x7FFF
            if q15_val & 0x8000:
                q15_val = q15_val - 0x10000
            expected_q15.append(q15_val)
        return np.array(expected_q15, dtype=np.int16).reshape(l * n, e)

    q = matmul_expected(x_np, wq_np).reshape(l, n, e)
    k = matmul_expected(x_np, wk_np).reshape(l, n, e)
    v = matmul_expected(x_np, wv_np).reshape(l, n, e)

    def hw_multiply_inv_sqrt8(matmul_result_q30):
        matmul_signed = np.array(matmul_result_q30).astype(np.int32)
        inv_sqrt8_signed = np.int16(0x2D50)
        mult_result = np.int64(matmul_signed) * np.int64(inv_sqrt8_signed)
        q30_result = (mult_result >> 15) & 0xFFFFFFFF
        if q30_result & 0x80000000:
            q30_result = q30_result - 0x100000000
        return np.int32(q30_result)

    # Step 2: Attention scores
    q_np = q.reshape(l * n, e)
    k_np = k.reshape(l, n, e)
    matmul_result = np.zeros((l * n, l), dtype=float)
    for e_idx in range(e):
        for i in range(l * n):
            for j in range(l * n):
                q_val = q1_15_to_real(q_np[i, e_idx])
                k_val = q1_15_to_real(k_np[j, 0, e_idx])  # n=1
                matmul_result[i, j] += q_val * k_val
        matmul_result = np.clip(matmul_result, -2.0, 1.999999999)
    matmul_q30 = np.vectorize(real_to_q1_30)(matmul_result)
    a_q30 = np.array([hw_multiply_inv_sqrt8(matmul_q30[i, j]) for i in range(l * n) for j in range(l)], dtype=np.int32)

    a_q15 = []
    for q30_val in a_q30:
        sign_bit = (q30_val >> 31) & 1
        int_bit = (q30_val >> 30) & 1
        if sign_bit == int_bit:
            q15_val = (sign_bit << 15) | ((q30_val >> 15) & 0x7FFF)
        else:
            q15_val = 0x8000 if sign_bit else 0x7FFF
        if q15_val & 0x8000:
            q15_val = q15_val - 0x10000
        a_q15.append(q15_val)
    a = np.array(a_q15, dtype=np.int16).reshape(l, n, l)

    # Step 3: Softmax approximation
    a_float = np.vectorize(q1_15_to_real)(a)
    a_softmax = np.zeros((l, n, l), dtype=float)
    for n_idx in range(n):
        for i in range(l):
            relu_row = np.maximum(a_float[i, n_idx, :], 0)
            row_sum = np.sum(relu_row)
            if row_sum != 0:
                a_softmax[i, n_idx, :] = relu_row / row_sum
            else:
                a_softmax[i, n_idx, :] = 0
    a_softmax = np.clip(a_softmax, -1.0, 0.999969482421875)
    a_softmax_q15 = np.vectorize(real_to_q1_15)(a_softmax).reshape(l * n, l)

    # Step 4: Precision assignment
    a_sum = np.sum(a_softmax_q15, axis=0)
    token_precision = []
    for s in a_sum:
        if s < 16384:
            code = 0  # int4
        elif s < 32768:
            code = 1  # int8
        else:
            code = 2  # fp16
        token_precision.append(code)

    # Step 5: A*V multiplication
    av_out = np.zeros((l, e), dtype=float)
    for k_idx in range(l):
        prec = token_precision[k_idx]
        for i in range(l):
            raw_a = int(a_softmax_q15[i, k_idx])
            if prec == 0:
                a_val = nibble_to_real_q1_3(raw_a >> 12)
            elif prec == 1:
                a_val = float(np.array((raw_a & 0xFF00) >> 8).astype(np.int8)) / (2**7)
            else:
                a_val = q1_15_to_real(raw_a)
            for j in range(e):
                raw_v = int(v[k_idx, 0, j])  # n=1
                if prec == 0:
                    v_val = nibble_to_real_q1_3(raw_v >> 12)
                elif prec == 1:
                    v_val = float(np.array((raw_v & 0xFF00) >> 8).astype(np.int8)) / (2**7)
                else:
                    v_val = q1_15_to_real(raw_v)
                av_out[i, j] += a_val * v_val
    av_out = np.clip(av_out, -1.0, 0.999969482421875)
    av_q15 = np.vectorize(real_to_q1_15)(av_out)

    # Step 6: W_O multiplication
    expected_out = np.zeros((l, e), dtype=float)
    for k_idx in range(e):
        for i in range(l):
            raw_av = int(av_q15[i, k_idx])
            av_val = q1_15_to_real(raw_av)
            for j in range(e):
                raw_wo = int(wo_np[k_idx, j])
                wo_val = q1_15_to_real(raw_wo)
                expected_out[i, j] += av_val * wo_val
        expected_out = np.clip(expected_out, -2.0, 1.999999999)
    out_q30 = np.vectorize(real_to_q1_30)(expected_out)

    out_q15 = []
    for q30_val in out_q30.flatten():
        sign_bit = (q30_val >> 31) & 1
        int_bit = (q30_val >> 30) & 1
        if sign_bit == int_bit:
            q15_val = (sign_bit << 15) | ((q30_val >> 15) & 0x7FFF)
        else:
            q15_val = 0x8000 if sign_bit else 0x7FFF
        if q15_val & 0x8000:
            q15_val = q15_val - 0x10000
        out_q15.append(q15_val)
    out_q15 = np.array(out_q15, dtype=np.int16).reshape(l, e)
    print({'Q': q.reshape(l, e),
        'K': k.reshape(l, e),
        'V': v.reshape(l, e),
        'A': a.reshape(l, l),
        'A_softmax': a_softmax_q15.reshape(l, l),
        'token_precision': np.array(token_precision, dtype=np.int8),
        'AV': av_q15,
        'out': out_q15})

    return out_q15

def compute_mlp_reference(a, b1, bias1, b2, bias2, seq_len, emb_dim, h_mlp):
    """Compute expected output for two-layer MLP."""
    m, k = a.shape  # a: (SEQ_LEN, EMB_DIM)
    k, n = b1.shape  # b1: (EMB_DIM, H_MLP)
    p, q = b2.shape  # b2: (H_MLP, EMB_DIM)

    # First layer: x @ W1^T + b1
    hidden_out = np.zeros((m, n), dtype=float)
    for k_idx in range(k):
        for i in range(m):
            raw_a = int(a[i, k_idx])
            a_val = q1_15_to_real(raw_a)
            for j in range(n):
                raw_b1 = int(b1[k_idx, j])
                b1_val = q1_15_to_real(raw_b1)
                hidden_out[i, j] += a_val * b1_val
        hidden_out = np.clip(hidden_out, -2.0, 2.0 - 2**-30)
    hidden_out = np.clip(hidden_out, -1.0, 1.0 - 2**-15)
    for i in range(m):
        for j in range(n):
            hidden_out[i, j] = hidden_out[i, j] + q1_15_to_real(bias1[j])
            hidden_out[i, j] = max(0, hidden_out[i, j])  # ReLU
            hidden_out[i, j] = np.clip(hidden_out[i, j], -1.0, 1.0 - 2**-15)

    # Second layer: ReLU(hidden) @ W2^T + b2
    final_out = np.zeros((m, q), dtype=float)
    for k_idx in range(n):
        for i in range(m):
            hidden_val = hidden_out[i, k_idx]
            for j in range(q):
                raw_b2 = int(b2[k_idx, j])
                b2_val = q1_15_to_real(raw_b2)
                final_out[i, j] += hidden_val * b2_val
        final_out = np.clip(final_out, -2.0, 2.0 - 2**-30)
    final_out = np.clip(final_out, -1.0, 1.0 - 2**-15)
    for i in range(m):
        for j in range(q):
            final_out[i, j] = final_out[i, j] + q1_15_to_real(bias2[j])
            final_out[i, j] = np.clip(final_out[i, j], -1.0, 1.0 - 2**-15)

    final_out_q15 = np.vectorize(real_to_q1_15)(final_out)
    return final_out_q15

def compute_expected(x_in, WQ, WK, WV, WO, W1, b1, W2, b2, ln1_gamma, ln1_beta, ln2_gamma, ln2_beta):
    """Compute expected output for ViT encoder block with two-layer MLP."""
    SEQ_LEN, EMB_DIM = x_in.shape  # (SEQ_LEN, EMB_DIM)
    H_MLP = W1.shape[1]  # H_MLP from W1 (EMB_DIM, H_MLP)

    # LayerNorm 1
    ln1_out = compute_layer_norm_reference(x_in, ln1_gamma, ln1_beta, SEQ_LEN, EMB_DIM)

    # Self-Attention
    attn_out = compute_attention_reference(ln1_out, WQ, WK, WV, WO, SEQ_LEN, EMB_DIM)

    # Residual 1: z1 = x_in + attn_out
    res1_out = x_in + attn_out  # Direct addition in Q1.15
    res1_out = np.clip(res1_out, -(1 << 15), (1 << 15) - 1)

    # LayerNorm 2
    ln2_out = compute_layer_norm_reference(res1_out, ln2_gamma, ln2_beta, SEQ_LEN, EMB_DIM)

    # MLP
    mlp_out = compute_mlp_reference(ln2_out, W1, b1, W2, b2, SEQ_LEN, EMB_DIM, H_MLP)

    # Residual 2: z2 = z1 + mlp_out
    res2_out = res1_out + mlp_out
    res2_out = np.clip(res2_out, -(1 << 15), (1 << 15) - 1)

    return res2_out, mlp_out, ln1_out, attn_out

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
    for i in range(H_MLP * EMB_DIM):
        dut.W2_in[i].value = 0
    for i in range(H_MLP):
        dut.b1_in[i].value = 0
    for i in range(EMB_DIM):
        dut.ln1_gamma[i].value = 0
        dut.ln1_beta[i].value = 0
        dut.ln2_gamma[i].value = 0
        dut.ln2_beta[i].value = 0
        dut.b2_in[i].value = 0
    await Timer(20, units="ns")
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Generate random inputs
    np.random.seed(42)  # For reproducibility
    x_in_real = np.random.uniform(-0.9, 0.9, (SEQ_LEN, EMB_DIM))
    WQ_real = np.random.uniform(-0.9, 0.9, (EMB_DIM, EMB_DIM))
    WK_real = np.random.uniform(-0.9, 0.9, (EMB_DIM, EMB_DIM))
    WV_real = np.random.uniform(-0.9, 0.9, (EMB_DIM, EMB_DIM))
    WO_real = np.random.uniform(-0.9, 0.9, (EMB_DIM, EMB_DIM))
    W1_real = np.random.uniform(-0.9, 0.9, (EMB_DIM, H_MLP))
    b1_real = np.random.uniform(-0.9, 0.9, H_MLP)
    W2_real = np.random.uniform(-0.9, 0.9, (H_MLP, EMB_DIM))
    b2_real = np.random.uniform(-0.9, 0.9, EMB_DIM)
    ln1_gamma_real = np.random.uniform(-10*2**-15, 10*2**-15, EMB_DIM)  # Non-zero for normalization
    ln1_beta_real = np.random.uniform(-10*2**-15, 10*2**-15, EMB_DIM)
    ln2_gamma_real = np.random.uniform(-10*2**-15, 10*2**-15, EMB_DIM)
    ln2_beta_real = np.random.uniform(-10*2**-15, 10*2**-15, EMB_DIM)

    # Convert to Q1.15 format
    x_in_q15 = np.array([real_to_q1_15(val) for val in x_in_real.flatten()]).reshape(SEQ_LEN, EMB_DIM)
    WQ_q15 = np.array([real_to_q1_15(val) for val in WQ_real.flatten()]).reshape(EMB_DIM, EMB_DIM)
    WK_q15 = np.array([real_to_q1_15(val) for val in WK_real.flatten()]).reshape(EMB_DIM, EMB_DIM)
    WV_q15 = np.array([real_to_q1_15(val) for val in WV_real.flatten()]).reshape(EMB_DIM, EMB_DIM)
    WO_q15 = np.array([real_to_q1_15(val) for val in WO_real.flatten()]).reshape(EMB_DIM, EMB_DIM)
    W1_q15 = np.array([real_to_q1_15(val) for val in W1_real.flatten()]).reshape(EMB_DIM, H_MLP)
    b1_q15 = np.array([real_to_q1_15(val) for val in b1_real])
    W2_q15 = np.array([real_to_q1_15(val) for val in W2_real.flatten()]).reshape(H_MLP, EMB_DIM)
    b2_q15 = np.array([real_to_q1_15(val) for val in b2_real])
    ln1_gamma_q15 = np.array([real_to_q1_15(val) for val in ln1_gamma_real])
    ln1_beta_q15 = np.array([real_to_q1_15(val) for val in ln1_beta_real])
    ln2_gamma_q15 = np.array([real_to_q1_15(val) for val in ln2_gamma_real])
    ln2_beta_q15 = np.array([real_to_q1_15(val) for val in ln2_beta_real])

    # Compute expected output
    expected_out_q15, mlp_out_q15, ln1_out_q15, attn_out_q15 = \
        compute_expected(x_in_q15, WQ_q15, WK_q15, WV_q15, WO_q15, W1_q15, b1_q15, W2_q15, b2_q15, ln1_gamma_q15, ln1_beta_q15, ln2_gamma_q15, ln2_beta_q15)
    mlp_out_q15 = mlp_out_q15.flatten()
    expected_out_q15 = expected_out_q15.flatten()
    attn_out_q15 = attn_out_q15.flatten()

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
    for i in range(H_MLP * EMB_DIM):
        dut.W2_in[i].value = int(W2_q15.flatten()[i]) & 0xFFFF
    for i in range(H_MLP):
        dut.b1_in[i].value = int(b1_q15[i]) & 0xFFFF
    for i in range(EMB_DIM):
        dut.ln1_gamma[i].value = int(ln1_gamma_q15[i]) & 0xFFFF
        dut.ln1_beta[i].value = int(ln1_beta_q15[i]) & 0xFFFF
        dut.ln2_gamma[i].value = int(ln2_gamma_q15[i]) & 0xFFFF
        dut.ln2_beta[i].value = int(ln2_beta_q15[i]) & 0xFFFF
        dut.b2_in[i].value = int(b2_q15[i]) & 0xFFFF

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
    def unpack_row_major_reverse_rows(raw_int):
        """
        Unpack DUT x_out (rows stored in reverse order, elements LSB‑first)
        into shape (SEQ_LEN, EMB_DIM).
        """
        mask = (1 << DATA_WIDTH) - 1
        arr  = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.int32)
        for i in range(SEQ_LEN):
            for j in range(EMB_DIM):
                bit_idx = (((SEQ_LEN - 1 - i) * EMB_DIM) + j) * DATA_WIDTH
                bits    = (raw_int >> bit_idx) & mask
                if bits & (1 << (DATA_WIDTH - 1)):       # sign‑extend
                    bits -= (1 << DATA_WIDTH)
                arr[i, j] = bits
        return arr
    ln1_out_block = unpack_row_major_reverse_rows(int(dut.ln1_out.value))

    attn_out_block = np.zeros(SEQ_LEN*EMB_DIM, dtype=np.int16)
    for i in range(SEQ_LEN*EMB_DIM):
        attn_out_block[i] = dut.attn_out[i].value.signed_integer

    mlp_out_block = np.zeros(EMB_DIM, dtype=np.int16)
    for i in range(EMB_DIM):
        mlp_out_block[i] = dut.mlp_out[i].value.signed_integer

    out_block = np.zeros(SEQ_LEN * EMB_DIM, dtype=np.int16)
    for i in range(SEQ_LEN * EMB_DIM):
        out_block[i] = dut.out_block[i].value.signed_integer

    # Compare outputs (allow small tolerance for fixed-point errors)
    tolerance = 4  # Allow 4 LSBs of error due to fixed-point arithmetic
    success = True

    assert np.array_equal(ln1_out_block, ln1_out_q15), f"\nExpected:\n{ln1_out_q15}\nGot:\n{ln1_out_block}"

    for i in range(SEQ_LEN * EMB_DIM):
        actual = attn_out_block[i]
        expected = attn_out_q15[i]
        if abs(actual - expected) > tolerance:
            success = False
            dut._log.error(f"Output attn[{i}] mismatch: expected {expected}, got {actual}")
        else:
            dut._log.info(f"Output attn[{i}] match: expected {expected}, got {actual}")
    assert success

    for i in range(EMB_DIM):
        actual = mlp_out_block[i]
        expected = mlp_out_q15[i]
        if abs(actual - expected) > tolerance:
            success = False
            dut._log.error(f"Output mlp[{i}] mismatch: expected {expected}, got {actual}")
        else:
            dut._log.info(f"Output mlp[{i}] match: expected {expected}, got {actual}")

    for i in range(SEQ_LEN * EMB_DIM):
        actual = out_block[i]
        expected = expected_out_q15[i]
        if abs(actual - expected) > tolerance:
            success = False
            dut._log.error(f"Output out_block[{i}] mismatch: expected {expected}, got {actual}")
        else:
            dut._log.info(f"Output out_block[{i}] match: expected {expected}, got {actual}")
    assert success

    # Wait a few more cycles to ensure stability
    for _ in range(5):
        await RisingEdge(dut.clk)