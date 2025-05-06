import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

@cocotb.test()
async def test_qkv_generator(dut):
    """Testbench for qkv_generator module"""
    
    # Module parameters
    DATA_WIDTH = dut.DATA_WIDTH.value
    L = dut.L.value
    N = dut.N.value
    E = dut.E.value
    TOTAL_TOKENS = L * N
    
    # Initialize clock and reset
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset the DUT
    dut.rst_n.value = 0
    dut.start.value = 0
    await Timer(20, units="ns")
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    # Function to convert numpy array to flattened integer for DUT
    def array_to_int(arr, width=DATA_WIDTH):
        flat = arr.flatten()
        val = 0
        for i, v in enumerate(flat):
            val |= (int(v) & ((1 << width) - 1)) << (i * width)
        return val
    
    # Function to convert DUT output to numpy array
    def int_to_array(val, shape, width=DATA_WIDTH):
        arr = np.zeros(shape, dtype=np.int64)
        flat = arr.flatten()
        for i in range(len(flat)):
            flat[i] = (val >> (i * width)) & ((1 << width) - 1)
            if flat[i] >= (1 << (width - 1)):
                flat[i] -= (1 << width)
        return arr.reshape(shape)
    
    # Run multiple test iterations
    for test_iter in range(3):
        dut._log.info(f"Starting test iteration {test_iter + 1}")
        
        # Generate random inputs (scale to avoid overflow in fixed-point)
        x = np.random.randint(-100, 100, size=(L, N, E)).astype(np.int64)
        WQ = np.random.randint(-50, 50, size=(E, E)).astype(np.int64)
        WK = np.random.randint(-50, 50, size=(E, E)).astype(np.int64)
        WV = np.random.randint(-50, 50, size=(E, E)).astype(np.int64)
        bQ = np.random.randint(-100, 100, size=(E)).astype(np.int64)
        bK = np.random.randint(-100, 100, size=(E)).astype(np.int64)
        bV = np.random.randint(-100, 100, size=(E)).astype(np.int64)
        
        # Compute expected outputs
        Q_expected = np.zeros((L, N, E), dtype=np.int64)
        K_expected = np.zeros((L, N, E), dtype=np.int64)
        V_expected = np.zeros((L, N, E), dtype=np.int64)
        for l in range(L):
            for n in range(N):
                Q_expected[l, n, :] = np.dot(x[l, n, :], WQ) + bQ
                K_expected[l, n, :] = np.dot(x[l, n, :], WK) + bK
                V_expected[l, n, :] = np.dot(x[l, n, :], WV) + bV
        
        # Drive inputs
        dut.x_in.value = array_to_int(x)
        dut.WQ_in.value = array_to_int(WQ)
        dut.WK_in.value = array_to_int(WK)
        dut.WV_in.value = array_to_int(WV)
        dut.bQ_in.value = array_to_int(bQ)
        dut.bK_in.value = array_to_int(bK)
        dut.bV_in.value = array_to_int(bV)
        
        # Start computation
        await RisingEdge(dut.clk)
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0
        
        # Wait for completion
        timeout = 1000
        while not dut.done.value and timeout > 0:
            await RisingEdge(dut.clk)
            timeout -= 1
        assert timeout > 0, "Test timed out waiting for done signal"
        assert dut.out_valid.value, "Output valid not asserted when done"
        
        # Read and verify outputs
        Q_out = int_to_array(dut.Q_out.value, (L, N, E))
        K_out = int_to_array(dut.K_out.value, (L, N, E))
        V_out = int_to_array(dut.V_out.value, (L, N, E))

        print(f"Q out = {Q_out}")
        print(f"Q expected = {Q_expected}")
        
        print(f"K out = {K_out}")
        print(f"K expected = {K_expected}")

        print(f"V out = {V_out}")
        print(f"V expected = {V_expected}")

        # Allow small tolerance for fixed-point arithmetic
        tolerance = 1
        assert np.all(np.abs(Q_out - Q_expected) <= tolerance), \
            f"Q output mismatch in iteration {test_iter + 1}"
        assert np.all(np.abs(K_out - K_expected) <= tolerance), \
            f"K output mismatch in iteration {test_iter + 1}"
        assert np.all(np.abs(V_out - V_expected) <= tolerance), \
            f"V output mismatch in iteration {test_iter + 1}"
        
        dut._log.info(f"Test iteration {test_iter + 1} passed")
    
    dut._log.info("All tests passed successfully")