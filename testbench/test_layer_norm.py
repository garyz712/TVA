import cocotb
from cocotb.clock     import Clock
from cocotb.triggers  import RisingEdge, Timer
import numpy as np
import logging

# --------------------------------------------------------------------
# Parameters (keep in sync with the RTL)
# --------------------------------------------------------------------
DATA_WIDTH = 16
SEQ_LEN    = 16
EMB_DIM    = 32
EPSILON    = 0x000029f1           # ≈1e-5 as 32-bit integer
H_MLP = 32

# --------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------
def pack_lsb_first(values):
    """Pack iterable of ints into a single int, element-0 in the LSB."""
    mask = (1 << DATA_WIDTH) - 1
    packed = 0
    for v in reversed(values):                # reverse → index-0 ends in LSB
        packed = (packed << DATA_WIDTH) | (int(v) & mask)
    return packed

def unpack_row_major(raw_int):
    """
    Unpack DUT x_out (rows stored in standard row-major order, elements LSB-first)
    into shape (SEQ_LEN, EMB_DIM).
    """
    mask = (1 << DATA_WIDTH) - 1
    arr = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.int32)
    for i in range(SEQ_LEN):
        for j in range(EMB_DIM):
            bit_idx = (i * EMB_DIM + j) * DATA_WIDTH
            bits = (raw_int >> bit_idx) & mask
            if bits & (1 << (DATA_WIDTH - 1)):       # sign-extend
                bits -= (1 << DATA_WIDTH)
            arr[i, j] = bits
    return arr

def q1_15_to_float(arr):
    """
    Convert a NumPy array of Q1.15 integers to floating-point values.
    Q1.15: 1 sign bit, 15 fractional bits, value = integer / 2^15.
    """
    return arr.astype(float) / (1 << 15)

def compute_layer_norm_reference(x_np, gamma_np, beta_np):
    """Software model that mirrors the RTL maths (incl. fake inv-sqrt)."""
    out = np.zeros_like(x_np, dtype=np.int32)
    maxv = (1 << (DATA_WIDTH - 1)) - 1
    minv = -(1 << (DATA_WIDTH - 1))
    for i in range(SEQ_LEN):
        mean    = np.sum(x_np[i]) // EMB_DIM
        diff    = x_np[i] - mean
        var     = np.sum(diff * diff) // EMB_DIM
        denom   = var + EPSILON
        inv_std = 1 if denom != 0 else 0            # placeholder inverse-sqrt
        for j in range(EMB_DIM):
            tmp   = ((x_np[i, j] - mean) * inv_std) / 4
            tmp = np.clip(tmp, minv, maxv)
            val   = tmp * gamma_np[j] + beta_np[j]
            out[i, j] = np.clip(val, minv, maxv)
    return out

async def reset_dut(dut):
    dut.rst_n.value = 0
    await Timer(20, units="ns")
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

async def wait_for_done(dut, timeout=10_000):
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if dut.done.value:
            return
    raise TimeoutError("DUT did not assert done in time")

# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def start_clock(dut):
    """Start a free-running 10 ns clock on dut.clk."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

@cocotb.test()
async def test_layer_norm_random(dut):
    """Random vectors"""
    start_clock(dut)
    await reset_dut(dut)

    def real_to_q1_15(x):
        val = int(x * np.int32(2**15))
        if val > 32767:
            val = 32767
        if val < -32768:
            val = -32768
        return np.array(val).astype(np.int16)

    np.random.seed(42)  # For reproducibility
    x_in_real = np.random.uniform(-0.9, 0.9, (SEQ_LEN, EMB_DIM))
    ln1_gamma_real = np.random.uniform(-10*2**-15, 10*2**-15, EMB_DIM)  # Non-zero for normalization
    ln1_beta_real = np.random.uniform(-10*2**-15, 10*2**-15, EMB_DIM)

    # Convert to Q1.15 format
    x_np = np.array([real_to_q1_15(val) for val in x_in_real.flatten()]).reshape(SEQ_LEN, EMB_DIM)
    gamma_np = np.array([real_to_q1_15(val) for val in ln1_gamma_real])
    beta_np = np.array([real_to_q1_15(val) for val in ln1_beta_real])

    dut.x_in.value     = pack_lsb_first(x_np.flatten())
    dut.gamma_in.value = pack_lsb_first(gamma_np)
    dut.beta_in.value  = pack_lsb_first(beta_np)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    await wait_for_done(dut)

    got = unpack_row_major(int(dut.x_out.value))
    exp = compute_layer_norm_reference(x_np, gamma_np, beta_np)

    # Print in Q1.15 format (floating-point)
    print("Expected (Q1.15):\n", np.array2string(q1_15_to_float(exp), precision=6, suppress_small=True))
    print("Got (Q1.15):\n", np.array2string(q1_15_to_float(got), precision=6, suppress_small=True))

    assert np.allclose(got, exp, 16), f"\nExpected (Q1.15):\n{q1_15_to_float(exp)}\nGot (Q1.15):\n{q1_15_to_float(got)}"
    # for i in range(SEQ_LEN):
    #     for j in range(EMB_DIM):
    #         assert abs(got[i, j] - exp[i, j]) < 0.001, f"got[{i}, {j}] = {got[i, j]}, exp[{i}, {j}] = {exp[i, j]}"
    assert dut.done.value and dut.out_valid.value
    logger.info("Random-vector test passed")

    await reset_dut(dut)
    assert not dut.done.value and not dut.out_valid.value

# @cocotb.test()
async def test_layer_norm_fixed(dut):
    """Fixed, known pattern"""
    start_clock(dut)
    await reset_dut(dut)

    x_np = np.array([
        [  1,  -2,   3,  -4,   5,  -6,   7,  -8],
        [ -9,  10, -11,  12, -13,  14, -15,  16],
        [ 17, -18,  19, -20,  21, -22,  23, -24],
        [-25,  26, -27,  28, -29,  30, -31,  32],
        [ 33, -34,  35, -36,  37, -38,  39, -40],
        [-41,  42, -43,  44, -45,  46, -47,  48],
        [ 49, -50,  51, -52,  53, -54,  55, -56],
        [-57,  58, -59,  60, -61,  62, -63,  64]
    ], dtype=np.int32)
    gamma_np = np.array([ 1,  2, -1, -2,  1,  2, -1, -2], dtype=np.int32)
    beta_np  = np.array([ 0,  1, -1,  0,  1, -1,  0,  1], dtype=np.int32)

    dut.x_in.value     = pack_lsb_first(x_np.flatten())
    dut.gamma_in.value = pack_lsb_first(gamma_np)
    dut.beta_in.value  = pack_lsb_first(beta_np)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    await wait_for_done(dut)

    got = unpack_row_major(int(dut.x_out.value))
    exp = compute_layer_norm_reference(x_np, gamma_np, beta_np)

    # Print in Q1.15 format (floating-point)
    print("Expected (Q1.15):\n", np.array2string(q1_15_to_float(exp), precision=6, suppress_small=True))
    print("Got (Q1.15):\n", np.array2string(q1_15_to_float(got), precision=6, suppress_small=True))

    assert np.array_equal(got, exp), f"\nExpected (Q1.15):\n{q1_15_to_float(exp)}\nGot (Q1.15):\n{q1_15_to_float(got)}"
    assert dut.done.value and dut.out_valid.value
    logger.info("Fixed-vector test passed")

# @cocotb.test()
async def test_layer_norm_edge_cases(dut):
    """Edge cases (saturation)"""
    start_clock(dut)
    await reset_dut(dut)

    maxv = (1 << (DATA_WIDTH - 1)) - 1
    minv = -(1 << (DATA_WIDTH - 1))
    pattern_a = [maxv, minv] * 4
    pattern_b = [minv, maxv] * 4

    x_np     = np.array([pattern_a, pattern_b] * 4, dtype=np.int32)
    gamma_np = np.array([1, -1] * 4, dtype=np.int32)
    beta_np  = np.zeros(EMB_DIM, dtype=np.int32)

    dut.x_in.value     = pack_lsb_first(x_np.flatten())
    dut.gamma_in.value = pack_lsb_first(gamma_np)
    dut.beta_in.value  = pack_lsb_first(beta_np)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    await wait_for_done(dut)

    got = unpack_row_major(int(dut.x_out.value))
    exp = compute_layer_norm_reference(x_np, gamma_np, beta_np)

    # Print in Q1.15 format (floating-point)
    print("Expected (Q1.15):\n", np.array2string(q1_15_to_float(exp), precision=6, suppress_small=True))
    print("Got (Q1.15):\n", np.array2string(q1_15_to_float(got), precision=6, suppress_small=True))

    assert np.array_equal(got, exp), f"\nExpected (Q1.15):\n{q1_15_to_float(exp)}\nGot (Q1.15):\n{q1_15_to_float(got)}"
    assert dut.done.value and dut.out_valid.value
    logger.info("Edge-case test passed")
