import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

# Q1.15 scaling factor
Q15_SCALE = 2**15  # 32768

# Reference model for classification head
def reference_model(patch_emb_in, W_clf_in, b_clf_in, N, E, NUM_CLASSES):
    # Reshape inputs to match software's 2D structure
    patch_emb_in = patch_emb_in.reshape(N, E)
    W_clf_in = W_clf_in.reshape(E, NUM_CLASSES)
    # Convert inputs to float for reference computation
    patch_emb_float = patch_emb_in / Q15_SCALE
    W_clf_float = W_clf_in / Q15_SCALE
    b_clf_float = b_clf_in / Q15_SCALE
    
    # Compute mean across patches (dim=0)
    mean_emb = np.mean(patch_emb_float, axis=0)
    
    # Matrix multiplication and bias addition
    logits = np.dot(mean_emb, W_clf_float) + b_clf_float
    
    # Clip to Q1.15 range [-1, 1-1/32768] and quantize back
    logits = np.clip(logits, -1.0, 1.0 - 1/Q15_SCALE)
    logits = np.round(logits * Q15_SCALE).astype(np.int16)
    
    return logits

@cocotb.test()
async def test_classification_head(dut):
    # Parameters
    N = 16              # Number of patches
    E = 64              # Embedding dimension
    NUM_CLASSES = 1     # Binary classification
    DATA_WIDTH = 16     # Q1.15 format
    
    # Start clock (100 MHz)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    for _ in range(100):
        # Reset DUT
        dut.rst_n.value = 0
        await Timer(20, units="ns")
        dut.rst_n.value = 1
        await RisingEdge(dut.clk)
    
        # Generate random Q1.15 inputs
        # np.random.seed(42)  # For reproducibility
        patch_emb_in = np.random.randint(-Q15_SCALE, Q15_SCALE, size=(N, E)).astype(np.int16)
        W_clf_in = np.random.randint(-Q15_SCALE, Q15_SCALE, size=(E, NUM_CLASSES)).astype(np.int16)
        b_clf_in = np.random.randint(-Q15_SCALE, Q15_SCALE, size=(NUM_CLASSES)).astype(np.int16)
    
        # Compute expected output using reference model
        expected_logits = reference_model(patch_emb_in.flatten(), W_clf_in.flatten(), b_clf_in, N, E, NUM_CLASSES)
    
        # Set inputs (1D arrays)
        for i in range(N * E):
            dut.patch_emb_in[i].value = int(patch_emb_in.flatten()[i])
    
        for i in range(E * NUM_CLASSES):
            dut.W_clf_in[i].value = int(W_clf_in.flatten()[i])
    
        for j in range(NUM_CLASSES):
            dut.b_clf_in[j].value = int(b_clf_in[j])
    
        # Start computation
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0
    
        # Wait for output valid (timeout after 1000 cycles)
        for _ in range(1000):
            await RisingEdge(dut.clk)
            if dut.out_valid.value == 1:
                break
    
        # Collect and print all logits for random input test
        actual_logits = []
        for c in range(NUM_CLASSES):
            actual_logits.append(dut.logits_out[c].value.signed_integer)
    
        print("Random input test logits:")
        for c in range(NUM_CLASSES):
            print(f"Logit[{c}]: actual = {actual_logits[c]}, expected = {expected_logits[c]}")
    
        # Check if out_valid is high
        assert dut.out_valid.value == 1, "Output valid signal not asserted"
    
        # Verify logits
        for c in range(NUM_CLASSES):
            assert abs(actual_logits[c] - expected_logits[c]) <= 10, \
                f"Logit[{c}] mismatch: expected {expected_logits[c]}, got {actual_logits[c]}"
    
        # Additional test case with zero inputs
        for i in range(N * E):
            dut.patch_emb_in[i].value = 0
    
        for i in range(E * NUM_CLASSES):
            dut.W_clf_in[i].value = 0
    
        for j in range(NUM_CLASSES):
            dut.b_clf_in[j].value = 0
    
    # Reset and start again
    dut.rst_n.value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for output valid
    for _ in range(1000):
        await RisingEdge(dut.clk)
        if dut.out_valid.value == 1:
            break
    
    # Collect and print all logits for zero input test
    actual_logits = []
    for c in range(NUM_CLASSES):
        actual_logits.append(dut.logits_out[c].value.signed_integer)
    
    print("Zero input test logits:")
    for c in range(NUM_CLASSES):
        print(f"Logit[{c}]: actual = {actual_logits[c]}, expected = 0")
    
    # Check zero output
    assert dut.out_valid.value == 1, "Output valid signal not asserted for zero test"
    for c in range(NUM_CLASSES):
        assert abs(actual_logits[c]) <= 1, f"Logit[{c}] non-zero for zero inputs: got {actual_logits[c]}"
    
    print("All tests passed!")