//-----------------------------------------------------------------------------
// File: weight_loader.sv
// Purpose: Burst‐read a block of model weights from AXI4 into on‐chip BRAM.
//
// Apr. 22 2025    Max Zhang    Initial version
//-----------------------------------------------------------------------------

module weight_loader #(
    parameter int ADDR_WIDTH      = 32,
    parameter int DATA_WIDTH      = 32,
    parameter int MEM_DEPTH       = 1024,                          // # of 32‐bit words
    parameter int BRAM_ADDR_WIDTH = $clog2(MEM_DEPTH),             // address bits for BRAM
    parameter logic [ADDR_WIDTH-1:0] BASE_ADDR = 'h0000_0000     // DDR base address
)(
    input  logic                         clk,
    input  logic                         rst_n,

    // Control interface
    input  logic                         start,    // pulse to begin weight load
    output logic                         done,     // high for one cycle at end

    // AXI4 Read Address Channel
    output logic [ADDR_WIDTH-1:0]        araddr,
    output logic [7:0]                   arlen,
    output logic                         arvalid,
    input  logic                         arready,

    // AXI4 Read Data  Channel
    input  logic [DATA_WIDTH-1:0]        rdata,
    input  logic                         rvalid,
    output logic                         rready,

    // Simple BRAM write interface
    output logic [BRAM_ADDR_WIDTH-1:0]   bram_addr,
    output logic [DATA_WIDTH-1:0]        bram_din,
    output logic                         bram_we,
    output logic                         bram_en
);

    //-------------------------------------------------------------------------
    // States
    //-------------------------------------------------------------------------
    typedef enum logic [1:0] {
        IDLE,
        ADDR,
        READ,
        DONE
    } state_t;

    state_t curr_state, next_state;

    //-------------------------------------------------------------------------
    // Counters
    //-------------------------------------------------------------------------
    // Counts words [0 .. MEM_DEPTH-1]
    logic [BRAM_ADDR_WIDTH-1:0] mem_index;

    //-------------------------------------------------------------------------
    // State register
    //-------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            curr_state <= IDLE;
        else
            curr_state <= next_state;
    end

    //-------------------------------------------------------------------------
    // Next‐state logic
    //-------------------------------------------------------------------------
    always_comb begin
        next_state = curr_state;
        case (curr_state)
            IDLE: if (start)         next_state = ADDR;
            ADDR: if (arvalid && arready) next_state = READ;
            READ: if (rvalid)        next_state = (mem_index == MEM_DEPTH-1) ? DONE : ADDR;
            DONE:                     next_state = IDLE;
        endcase
    end

    //-------------------------------------------------------------------------
    // Outputs & datapath
    //-------------------------------------------------------------------------
    // Defaults every cycle
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // control
            done     <= 1'b0;
            // AXI
            araddr   <= '0;
            arlen    <= 8'd0;
            arvalid  <= 1'b0;
            rready   <= 1'b0;
            // BRAM
            bram_addr<= '0;
            bram_din <= '0;
            bram_we  <= 1'b0;
            bram_en  <= 1'b0;
            // counter
            mem_index<= '0;
        end else begin
            // default deasserts
            done     <= 1'b0;
            arvalid  <= 1'b0;
            rready   <= 1'b0;
            bram_we  <= 1'b0;
            bram_en  <= 1'b0;

            case (curr_state)

                // Wait for start pulse
                IDLE: begin
                    mem_index <= '0;
                end

                // Issue one‐beat AXI read
                ADDR: begin
                    araddr  <= BASE_ADDR + (mem_index * (DATA_WIDTH/8));
                    arlen   <= 8'd0;     // single‐beat burst
                    arvalid <= 1'b1;
                end

                // Accept read data, write to BRAM
                READ: begin
                    rready    <= 1'b1;
                    if (rvalid) begin
                        // write into BRAM at address = mem_index
                        bram_addr <= mem_index;
                        bram_din  <= rdata;
                        bram_we   <= 1'b1;
                        bram_en   <= 1'b1;
                        // advance
                        mem_index <= mem_index + 1;
                    end
                end

                // All words loaded
                DONE: begin
                    done <= 1'b1;
                end

            endcase
        end
    end

endmodule
