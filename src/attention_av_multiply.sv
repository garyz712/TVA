
//------------------------------------------------------------------------------
// attention_av_multiply.sv
//
//  This module implements a matrix multiplication for the attention mechanism,
//  computing A * V with token-wise mixed precision (INT4, INT8, FP16).
//  Uses pipelined multipliers with variable latency to reduce cycle count
//  for lower precision tokens, operating at a variable clock frequency.
//
//  Apr. 10 2025    Max Zhang      Initial version
//  Apr. 11 2025    Tianwei Liu    Refactor in SV, split state machine, add comments
//  Apr. 30 2025    Max Zhang      Redesigned with pipelined variable-latency multipliers
//  May. 9 2025    Max Zhang       Redesigned with pipelined variable-latency multipliers with memory tiling
//------------------------------------------------------------------------------

module attention_av_multiply #(
    parameter int A_ROWS = 8,        // Attention matrix column size
    parameter int V_COLS = 32,       // Value matrix row size
    parameter int NUM_COLS = 8,      // Number of attention columns
    parameter int TILE_SIZE = 8,     // Submatrix size (8x8)
    parameter int WIDTH_INT4 = 4,    // INT4 width (Q1.3)
    parameter int WIDTH_INT8 = 8,    // INT8 width (Q1.7)
    parameter int WIDTH_FP16 = 16    // FP16 width (Q1.15)
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        start,
    input  logic [1:0]                  precision_sel [NUM_COLS], // Precision per column
    input  logic [WIDTH_FP16-1:0]       a_mem [A_ROWS][NUM_COLS], // Attention matrix (BRAM)
    input  logic [WIDTH_FP16-1:0]       v_mem [NUM_COLS][V_COLS], // Value matrix (BRAM)
    output logic                        done,
    output logic [WIDTH_FP16-1:0]       out_mem [A_ROWS][V_COLS] // Accumulated output matrix (BRAM)
);

    // Local parameters
    localparam int NUM_TILES = V_COLS / TILE_SIZE; // 4 tiles (32 / 8)
    localparam int WIDTH_OUT = WIDTH_FP16;         // Output in FP16 (Q1.15)
    localparam int CYCLES_INT4 = 1;
    localparam int CYCLES_INT8 = 2;
    localparam int CYCLES_FP16 = 4;

    // FSM states
    typedef enum logic [2:0] {
        IDLE,
        LOAD_COLUMN,
        LOAD_TILE,
        COMPUTE,
        UPCAST_ACCUM,
        STORE_TILE,
        DONE
    } state_t;
    state_t state, next_state;

    // Signals
    logic [WIDTH_FP16-1:0] a_vec [A_ROWS];           // Current attention column
    logic [WIDTH_FP16-1:0] v_vec [TILE_SIZE];        // Current value tile row
    logic [WIDTH_OUT-1:0]  submatrix [A_ROWS][TILE_SIZE]; // 8x8 submatrix
    logic [WIDTH_OUT-1:0]  accum [A_ROWS][V_COLS];   // Accumulator for all columns
    logic [WIDTH_INT4-1:0] a_int4 [A_ROWS];          // INT4 inputs
    logic [WIDTH_INT4-1:0] v_int4 [TILE_SIZE];
    logic [WIDTH_INT8-1:0] a_int8 [A_ROWS];          // INT8 inputs
    logic [WIDTH_INT8-1:0] v_int8 [TILE_SIZE];
    logic [WIDTH_FP16-1:0] a_fp16 [A_ROWS];          // FP16 inputs
    logic [WIDTH_FP16-1:0] v_fp16 [TILE_SIZE];
    logic [7:0]            p4 [A_ROWS][TILE_SIZE];   // INT4 partial products
    logic [15:0]           p8 [A_ROWS][TILE_SIZE];   // INT8 partial products
    logic [31:0]           p16 [A_ROWS][TILE_SIZE];  // FP16 partial products
    logic                  out4_valid [A_ROWS][TILE_SIZE];
    logic                  out8_valid [A_ROWS][TILE_SIZE];
    logic                  out16_valid [A_ROWS][TILE_SIZE];
    logic [31:0]           cycle_count;              // Cycle counter
    logic [31:0]           cycles_needed;            // Cycles for current column
    logic [31:0]           tile_idx;                 // Current tile index
    logic [31:0]           col_idx;                  // Current column index
    logic                  compute_valid;             // Compute enable
    logic                  accum_valid;              // Accumulation enable

    // Instantiate mul16_progressive for 8x8 submatrix
    genvar i, j;
    generate
        for (i = 0; i < A_ROWS; i++) begin : gen_row
            for (j = 0; j < TILE_SIZE; j++) begin : gen_col
                mul16_progressive #(
                    .WIDTH(WIDTH_FP16)
                ) mul_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .in_valid(compute_valid),
                    .a(precision_sel[col_idx] == 2'b00 ? {{12{1'b0}}, a_int4[i]} :
                       precision_sel[col_idx] == 2'b01 ? {{8{1'b0}}, a_int8[i]} :
                       a_fp16[i]),
                    .b(precision_sel[col_idx] == 2'b00 ? {{12{1'b0}}, v_int4[j]} :
                       precision_sel[col_idx] == 2'b01 ? {{8{1'b0}}, v_int8[j]} :
                       v_fp16[j]),
                    .out4_valid(out4_valid[i][j]),
                    .p4(p4[i][j]),
                    .out8_valid(out8_valid[i][j]),
                    .p8(p8[i][j]),
                    .out16_valid(out16_valid[i][j]),
                    .p16(p16[i][j])
                );
            end
        end
    endgenerate

    // Precision conversion and input selection
    always_comb begin
        for (int i = 0; i < A_ROWS; i++) begin
            a_int4[i] = a_vec[i][WIDTH_INT4-1:0];   // Truncate to Q1.3
            a_int8[i] = a_vec[i][WIDTH_INT8-1:0];   // Truncate to Q1.7
            a_fp16[i] = a_vec[i];                    // Q1.15
        end
        for (int j = 0; j < TILE_SIZE; j++) begin
            v_int4[j] = v_vec[j][WIDTH_INT4-1:0];
            v_int8[j] = v_vec[j][WIDTH_INT8-1:0];
            v_fp16[j] = v_vec[j];
        end
    end

    // Determine cycles needed for current column's precision
    always_comb begin
        case (precision_sel[col_idx])
            2'b00: cycles_needed = CYCLES_INT4; // INT4: 1 cycle
            2'b01: cycles_needed = CYCLES_INT8; // INT8: 2 cycles
            2'b10: cycles_needed = CYCLES_FP16; // FP16: 4 cycles
            default: cycles_needed = CYCLES_FP16;
        endcase
    end

    // FSM: State register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // FSM: Next state and control logic
    always_comb begin
        next_state = state;
        compute_valid = 1'b0;
        accum_valid = 1'b0;
        done = 1'b0;

        case (state)
            IDLE: begin
                if (start)
                    next_state = LOAD_COLUMN;
            end
            LOAD_COLUMN: begin
                next_state = LOAD_TILE;
            end
            LOAD_TILE: begin
                next_state = COMPUTE;
            end
            COMPUTE: begin
                compute_valid = 1'b1;
                if (cycle_count >= cycles_needed - 1)
                    next_state = UPCAST_ACCUM;
            end
            UPCAST_ACCUM: begin
                accum_valid = 1'b1;
                next_state = STORE_TILE;
            end
            STORE_TILE: begin
                if (tile_idx == NUM_TILES - 1) begin
                    if (col_idx == NUM_COLS - 1)
                        next_state = DONE;
                    else
                        next_state = LOAD_COLUMN;
                end else
                    next_state = LOAD_TILE;
            end
            DONE: begin
                done = 1'b1;
                next_state = IDLE;
            end
        endcase
    end

    // Cycle counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle_count <= 0;
        else if (state == COMPUTE) begin
            if (cycle_count < cycles_needed - 1)
                cycle_count <= cycle_count + 1;
            else
                cycle_count <= 0;
        end else
            cycle_count <= 0;
    end

    // Column index
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            col_idx <= 0;
        else if (state == STORE_TILE && tile_idx == NUM_TILES - 1 && next_state == LOAD_COLUMN)
            col_idx <= col_idx + 1;
        else if (state == DONE)
            col_idx <= 0;
    end

    // Tile index
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            tile_idx <= 0;
        else if (state == STORE_TILE && next_state == LOAD_TILE)
            tile_idx <= tile_idx + 1;
        else if (state == LOAD_COLUMN)
            tile_idx <= 0;
    end

    // Load column data
    always_ff @(posedge clk) begin
        if (state == LOAD_COLUMN) begin
            for (int i = 0; i < A_ROWS; i++)
                a_vec[i] <= a_mem[i][col_idx];
            for (int j = 0; j < TILE_SIZE; j++)
                v_vec[j] <= v_mem[col_idx][tile_idx * TILE_SIZE + j];
        end
    end

    // Load tile data
    always_ff @(posedge clk) begin
        if (state == LOAD_TILE) begin
            for (int j = 0; j < TILE_SIZE; j++)
                v_vec[j] <= v_mem[col_idx][tile_idx * TILE_SIZE + j];
        end
    end

    // Upcast and accumulate
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < A_ROWS; i++)
                for (int j = 0; j < V_COLS; j++)
                    accum[i][j] <= 0;
        end else if (state == UPCAST_ACCUM) begin
            for (int i = 0; i < A_ROWS; i++) begin
                for (int j = 0; j < TILE_SIZE; j++) begin
                    logic [WIDTH_OUT-1:0] result;
                    case (precision_sel[col_idx])
                        2'b00: result = {{8{p4[i][j][7]}}, p4[i][j], 8'h0}; // INT4 -> FP16
                        2'b01: result = {{8{p8[i][j][7]}}, p8[i][j]};       // INT8 -> FP16
                        2'b10: result = p16[i][j][WIDTH_OUT-1:0];           // FP16
                        default: result = 0;
                    endcase
                    accum[i][tile_idx * TILE_SIZE + j] <= accum[i][tile_idx * TILE_SIZE + j] + result;
                end
            end
        end
    end

    // Store output
    always_ff @(posedge clk) begin
        if (state == DONE) begin
            for (int i = 0; i < A_ROWS; i++)
                for (int j = 0; j < V_COLS; j++)
                    out_mem[i][j] <= accum[i][j];
        end
    end

endmodule
