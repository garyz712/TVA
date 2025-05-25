import cocotb
from cocotb.triggers import Timer, RisingEdge, FallingEdge
from cocotb.clock import Clock
import random
import numpy as np

# Parameters matching the module
DATA_WIDTH = 16
L = 8
N = 1
TOT_ROWS = L * N
TOT_COLS = L
TOT_A_IN_BITS = DATA_WIDTH * L * N * L

@cocotb.test()
async def precision_assigner_test(dut):
    """Testbench for precision_assigner module"""

    # Start clock (10ns period)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset the module
    async def reset_dut():
        dut.rst_n.value = 0
        dut.start.value = 0
        dut.A_in.value = 0
        await Timer(20, units="ns")
        await RisingEdge(dut.clk)
        dut.rst_n.value = 1
        await RisingEdge(dut.clk)

    # Helper function to drive input and wait for done
    async def run_test_case(A_in_flat):
        dut.start.value = 0
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        
        # Apply input
        dut.A_in.value = A_in_flat
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0

        # Wait for done
        timeout = 1000  # cycles
        while dut.done.value == 0 and timeout > 0:
            await RisingEdge(dut.clk)
            timeout -= 1
        assert timeout > 0, "Timeout waiting for done signal"
        assert dut.done.value == 1, "Done signal not asserted"

        # Capture output
        token_precision = [dut.token_precision[L-1-i].value for i in range(L)]
        return token_precision

    # Helper function to compute expected precision codes
    def compute_expected_precision(A_matrix):
        # A_matrix shape: (TOT_ROWS, TOT_COLS) = (L*N, L)
        sums = np.sum(A_matrix, axis=0)  # Sum across rows for each column
        precision_codes = []
        for s in sums:
            if s < 16384:
                code = 0  # int4
            elif s < 32768:
                code = 1  # int8
            else:
                code = 2  # fp16
            precision_codes.append(code)
        return precision_codes

    # Initialize
    await reset_dut()

    # Test case 1: All zeros (expect precision code 0 for all tokens)
    A_matrix = np.zeros((TOT_ROWS, TOT_COLS), dtype=np.uint16)
    A_flat = int(''.join(['{:016b}'.format(x) for x in A_matrix.flatten()]), 2)
    token_precision = await run_test_case(A_flat)
    expected = [0] * L
    assert token_precision == expected, f"Test 1 failed: Got {token_precision}, Expected {expected}"

    # Test case 2: Values to trigger code 0 (<100)
    A_matrix = np.random.randint(1, 10, size=(TOT_ROWS, TOT_COLS), dtype=np.uint16)  # Sum < 100
    A_flat = int(''.join(['{:016b}'.format(x) for x in A_matrix.flatten()]), 2)
    token_precision = await run_test_case(A_flat)
    expected = compute_expected_precision(A_matrix)
    assert token_precision == expected, f"Test 2 failed: Got {token_precision}, Expected {expected}"

    # Test case 3: Values to trigger code 1 (100 <= sum < 200)
    A_matrix = np.random.randint(16384 // TOT_ROWS, 24576 // TOT_ROWS, size=(TOT_ROWS, TOT_COLS), dtype=np.uint16)
    A_flat = int(''.join(['{:016b}'.format(x) for x in A_matrix.flatten()]), 2)
    token_precision = await run_test_case(A_flat)
    expected = compute_expected_precision(A_matrix)
    assert token_precision == expected, f"Test 3 failed: Got {token_precision}, Expected {expected}"

    # Test case 4: Values to trigger code 2 (sum >= 200)
    A_matrix = np.random.randint(32768 // TOT_ROWS, 49152 // TOT_ROWS, size=(TOT_ROWS, TOT_COLS), dtype=np.uint16)
    A_flat = int(''.join(['{:016b}'.format(x) for x in A_matrix.flatten()]), 2)
    token_precision = await run_test_case(A_flat)
    expected = compute_expected_precision(A_matrix)
    assert token_precision == expected, f"Test 4 failed: Got {token_precision}, Expected {expected}"

    # Test case 5: Mixed values to hit all thresholds
    A_matrix = np.zeros((TOT_ROWS, TOT_COLS), dtype=np.uint16)
    A_matrix[:, 0:2] = 5  # Sum = 5 * TOT_ROWS < 100
    A_matrix[:, 2:4] = 20000 // TOT_ROWS  # Sum ~ 120
    A_matrix[:, 4:6] = 40000 // TOT_ROWS  # Sum ~ 250
    A_matrix[:, 6:8] = 10  # Sum = 10 * TOT_ROWS < 100
    A_flat = int(''.join(['{:016b}'.format(x) for x in A_matrix.flatten()]), 2)
    token_precision = await run_test_case(A_flat)
    expected = compute_expected_precision(A_matrix)
    assert token_precision == expected, f"Test 5 failed: Got {token_precision}, Expected {expected}"

    # Test case 6: Multiple runs to check reset and reuse
    for _ in range(3):
        A_matrix = np.random.randint(1, 300 // TOT_ROWS, size=(TOT_ROWS, TOT_COLS), dtype=np.uint16)
        A_flat = int(''.join(['{:016b}'.format(x) for x in A_matrix.flatten()]), 2)
        token_precision = await run_test_case(A_flat)
        expected = compute_expected_precision(A_matrix)
        assert token_precision == expected, f"Test 6 failed: Got {token_precision}, Expected {expected}"

    print("All tests passed!")
