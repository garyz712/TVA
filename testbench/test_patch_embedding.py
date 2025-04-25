import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.binary import BinaryValue
import random

# parameters
DATA_WIDTH  = 4
IMG_H, IMG_W, C = 32, 32, 3
PH, PW = 16, 16
E = 8

PATCH_SIZE   = PH * PW * C
PATCH_H_CNT  = IMG_H // PH
PATCH_W_CNT  = IMG_W // PW
NUM_PATCHES  = PATCH_H_CNT * PATCH_W_CNT

@cocotb.test()
async def test_patch_embedding(dut):
    """Test patch_embedding.sv with random image and weights"""

    # Clock generation
    clock = Clock(dut.clk, 10, units="ns")  # 100 MHz
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst_n.value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Initialize inputs
    dut.start.value = 0
    dut.done.value = 0
    dut.out_valid.value = 0


    # Generate random image data
    image_data = [random.randint(0, (1 << DATA_WIDTH) - 1) for _ in range(IMG_H * IMG_W * C)]
    image_flat = BinaryValue(n_bits=DATA_WIDTH * IMG_H * IMG_W * C, bigEndian=False)
    for i in range(len(image_data)):
        image_flat[(i+1)*DATA_WIDTH-1:i*DATA_WIDTH] = image_data[i]
    dut.image_in.value = image_flat

    # Generate random weights
    w_data = [random.randint(0, (1 << DATA_WIDTH) - 1) for _ in range(PATCH_SIZE * E)]
    w_flat = BinaryValue(n_bits=DATA_WIDTH * PATCH_SIZE * E, bigEndian=False)
    for i in range(len(w_data)):
        w_flat[(i+1)*DATA_WIDTH-1:i*DATA_WIDTH] = w_data[i]
    dut.W_patch_in.value = w_flat

    # Generate random biases
    b_data = [random.randint(0, (1 << DATA_WIDTH) - 1) for _ in range(E)]
    b_flat = BinaryValue(n_bits=DATA_WIDTH * E, bigEndian=False)
    for i in range(len(b_data)):
        b_flat[(i+1)*DATA_WIDTH-1:i*DATA_WIDTH] = b_data[i]
    dut.b_patch_in.value = b_flat

    # Wait a few cycles before start
    for _ in range(5):
        await RisingEdge(dut.clk)

    # Trigger processing
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    print("Here!")
    clock = 0

    # Wait for done signal
    while not dut.done.value:
        clock += 1
        if (clock % 100 == 0):
            print(f"{clock} clocks")
        await RisingEdge(dut.clk)

    # Check validity
    assert dut.out_valid.value == 1, "Output should be valid after done."

    # inspect output
    patch_out_bin = dut.patch_out.value
    print(f"{patch_out_bin}")

