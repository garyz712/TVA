//------------------------------------------------------------------------------
// attention_av_multiply.sv
//
// This module implements a matrix multiplication for the attention mechanism,
// computing A * V with token-wise mixed precision (INT4, INT8, FP16).
// Uses pipelined multipliers with variable latency and tiling for 16x16 A matrix
// and 16x32 V matrix, with A divided into four 8x8 blocks and V into two 8x32 blocks.
//
// Apr. 10 2025    Max Zhang      Initial version
// Apr. 11 2025    Tianwei Liu    Refactor in SV, split state machine, add comments
// Apr. 30 2025    Max Zhang      Redesigned with pipelined variable-latency multipliers
// May. 9 2025     Max Zhang      Redesigned with pipelined variable-latency multipliers with memory tiling
// May. 12 2025    Max Zhang      Worked for brute force mul16
// May. 18 2025    Tianwei Liu    Bug fix, compute & check valid in correct stage
// May. 31 2025    Max Zhang     Modified for 16x16 A matrix with row and column tiling
// Jun. 1 2025     Max Zhang     Added col_idx to track columns within attention tile
//------------------------------------------------------------------------------

module attention_av_multiply #(
    parameter int A_ROWS = 16,       // Attention matrix row size
    parameter int V_COLS = 32,       // Value matrix column size
    parameter int NUM_COLS = 16,     // Attention matrix column size
    parameter int TILE_SIZE = 8,     // Submatrix size (8x8)
    parameter int WIDTH_INT4 = 4,    // INT4 width (Q1.3)
    parameter int WIDTH_INT8 = 8,    // INT8 width (Q1.7)
    parameter int WIDTH_FP16 = 16    // FP16 width (Q1.15)
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        start,
    input  logic [1:0]                  precision_sel [NUM_COLS], // Precision per column
    input  logic [WIDTH_FP16-1:0]       a_mem [0:A_ROWS*NUM_COLS-1], // Flattened Attention matrix (16*16=256)
    input  logic [WIDTH_FP16-1:0]       v_mem [0:NUM_COLS*V_COLS-1], // Flattened Value matrix (16*32=512)
    output logic                        done,
    output logic [WIDTH_FP16-1:0]       out_mem [0:A_ROWS*V_COLS-1]  // Accumulated output matrix (16*32=512)
);

    // Local parameters
    localparam int NUM_TILES = V_COLS / TILE_SIZE; // 4 tiles (32 / 8)
    localparam int NUM_ROW_TILES = A_ROWS / TILE_SIZE; // 2 row tiles (16 / 8)
    localparam int NUM_COL_TILES = NUM_COLS / TILE_SIZE; // 2 col tiles (16 / 8)
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
    logic [WIDTH_FP16-1:0] a_vec [TILE_SIZE];        // Current attention tile row (8 rows)
    logic [WIDTH_FP16-1:0] v_vec [TILE_SIZE];        // Current value tile row
    logic [WIDTH_OUT-1:0]  submatrix [TILE_SIZE][TILE_SIZE]; // 8x8 submatrix
    logic [32-1:0]         accum [A_ROWS][V_COLS];   // Accumulator for all rows/columns
    logic [WIDTH_INT4-1:0] a_int4 [TILE_SIZE];       // INT4 inputs
    logic [WIDTH_INT4-1:0] v_int4 [TILE_SIZE];
    logic [WIDTH_INT8-1:0] a_int8 [TILE_SIZE];       // INT8 inputs
    logic [WIDTH_INT8-1:0] v_int8 [TILE_SIZE];
    logic [WIDTH_FP16-1:0] a_fp16 [TILE_SIZE];       // FP16 inputs
    logic [WIDTH_FP16-1:0] v_fp16 [TILE_SIZE];
    logic [7:0]            p4 [TILE_SIZE][TILE_SIZE];   // INT4 partial products
    logic [15:0]           p8 [TILE_SIZE][TILE_SIZE];   // INT8 partial products
    logic [31:0]           p16 [TILE_SIZE][TILE_SIZE];  // FP16 partial products
    logic                  out4_valid [TILE_SIZE][TILE_SIZE];
    logic                  out8_valid [TILE_SIZE][TILE_SIZE];
    logic                  out16_valid [TILE_SIZE][TILE_SIZE];
    logic [31:0]           tile_idx;                 // Current V-column tile index (0-3)
    logic [31:0]           col_tile_idx;             // Current A-column tile index (0-1)
    logic [31:0]           row_tile_idx;             // Current A-row tile index (0-1)
    logic [31:0]           col_idx;                  // Current column within tile (0-7)
    logic                  compute_valid;             // Compute enable

    // Instantiate mul16_progressive for 8x8 submatrix
    genvar i, j;
    generate
        for (i = 0; i < TILE_SIZE; i++) begin : gen_row
            for (j = 0; j < TILE_SIZE; j++) begin : gen_col
                mul16_progressive mul_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .valid_in(compute_valid),
                    .a(precision_sel[col_tile_idx*TILE_SIZE + col_idx] == 2'b00 ? {{12{a_int4[i][3]}}, a_int4[i]} :
                       precision_sel[col_tile_idx*TILE_SIZE + col_idx] == 2'b01 ? {{8{a_int8[i][7]}}, a_int8[i]} :
                       a_fp16[i]),
                    .b(precision_sel[col_tile_idx*TILE_SIZE + col_idx] == 2'b00 ? {{12{v_int4[j][3]}}, v_int4[j]} :
                       precision_sel[col_tile_idx*TILE_SIZE + col_idx] == 2'b01 ? {{8{v_int8[j][7]}}, v_int8[j]} :
                       v_fp16[j]),
                    .q1_6_valid(out4_valid[i][j]),
                    .q1_6_out(p4[i][j]),
                    .q1_14_valid(out8_valid[i][j]),
                    .q1_14_out(p8[i][j]),
                    .q1_30_valid(out16_valid[i][j]),
                    .q1_30_out(p16[i][j])
                );
            end
        end
    endgenerate

    // Precision conversion and input selection
    always_comb begin
        for (int i = 0; i < TILE_SIZE; i++) begin
            a_int4[i] = a_vec[i][15:12];   // Q1.3: sign bit + 3 fractional bits
            a_int8[i] = a_vec[i][15:8];    // Q1.7: sign bit + 7 fractional bits
            a_fp16[i] = a_vec[i];          // Q1.15: full precision
        end
        for (int j = 0; j < TILE_SIZE; j++) begin
            v_int4[j] = v_vec[j][15:12];
            v_int8[j] = v_vec[j][15:8];
            v_fp16[j] = v_vec[j];
        end
    end

    // FSM: State register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // Check if tile computation is ready
    logic tile_ready;
    always_comb begin
        tile_ready = 1'b0;
        for (int i = 0; i < TILE_SIZE; i++)
            for (int j = 0; j < TILE_SIZE; j++)
                tile_ready |= (precision_sel[col_tile_idx*TILE_SIZE + col_idx] == 2'b00) ? out4_valid[i][j] :
                              (precision_sel[col_tile_idx*TILE_SIZE + col_idx] == 2'b01) ? out8_valid[i][j] :
                              out16_valid[i][j];
    end

    // FSM: Next state and control logic
    always_comb begin
        next_state = state;
        compute_valid = 1'b0;
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
                next_state = UPCAST_ACCUM;
            end
            UPCAST_ACCUM: begin
                next_state = tile_ready ? STORE_TILE : UPCAST_ACCUM;
            end
            STORE_TILE: begin
                if (tile_idx == NUM_TILES - 1) begin //one column is done
                    if (col_idx == TILE_SIZE - 1) begin //one attention tile is done
                        if (col_tile_idx == NUM_COL_TILES - 1) begin //one horizontal list of attention tiles are done
                            if (row_tile_idx == NUM_ROW_TILES - 1) //every attention row and column is done
                                next_state = DONE;
                            else //jump to the next row tile
                                next_state = LOAD_COLUMN;
                        end else//compute next attention column
                            next_state = LOAD_COLUMN;
                    end else //compute next attention column
                        next_state = LOAD_COLUMN;
                end else //continue with loading values tiles
                    next_state = LOAD_TILE;
            end
            DONE: begin
                done = 1'b1;
                next_state = IDLE;
            end
            default: begin
                next_state = IDLE;
            end
        endcase
    end

    // Row tile index
    always_ff @(posedge clk or negedge rst_n) begin //likely correct
        if (!rst_n)
            row_tile_idx <= 0;
        else if (state == STORE_TILE && tile_idx == NUM_TILES - 1 && col_idx == TILE_SIZE - 1 && col_tile_idx == NUM_COL_TILES - 1 && next_state == LOAD_COLUMN)
            row_tile_idx <= row_tile_idx + 1;
        else if (state == DONE)
            row_tile_idx <= 0;
    end

    // Column tile index
    always_ff @(posedge clk or negedge rst_n) begin //likely correct
        if (!rst_n)
            col_tile_idx <= 0;
        else if (state == DONE || (state == STORE_TILE && col_tile_idx == NUM_COL_TILES - 1 && tile_idx == NUM_TILES - 1 && col_idx == TILE_SIZE - 1 && next_state == LOAD_COLUMN))
            col_tile_idx <= 0;
        else if (state == STORE_TILE && tile_idx == NUM_TILES - 1 && col_idx == TILE_SIZE - 1 && next_state == LOAD_COLUMN)
            col_tile_idx <= col_tile_idx + 1;       
    end

    // Column index within tile
    always_ff @(posedge clk or negedge rst_n) begin //likely correct
        if (!rst_n)
            col_idx <= 0;
        else if (state == DONE || state == STORE_TILE && tile_idx == NUM_TILES - 1 && next_state == LOAD_COLUMN && col_idx == TILE_SIZE - 1)
            col_idx <= 0;
        else if (state == STORE_TILE && tile_idx == NUM_TILES - 1 && next_state == LOAD_COLUMN)
            col_idx <= col_idx + 1;
    end

    // Tile index (for V columns)
    always_ff @(posedge clk or negedge rst_n) begin  //correct
        if (!rst_n)
            tile_idx <= 0;
        else if (state == STORE_TILE && next_state == LOAD_TILE)
            tile_idx <= tile_idx + 1;
        else if (next_state == LOAD_COLUMN)
            tile_idx <= 0;
    end

    // Load column data
    always_ff @(posedge clk) begin //correct
        if (state == LOAD_COLUMN) begin
            for (int i = 0; i < TILE_SIZE; i++)
                a_vec[i] <= a_mem[(row_tile_idx * TILE_SIZE + i) * NUM_COLS + col_tile_idx * TILE_SIZE + col_idx];
            
        end
    end

    // Load tile data
    always_ff @(posedge clk) begin //correct
        if (state == LOAD_TILE) begin
            for (int j = 0; j < TILE_SIZE; j++)
                v_vec[j] <= v_mem[(col_tile_idx * TILE_SIZE + col_idx) * V_COLS + tile_idx * TILE_SIZE + j];
        end
    end

    // Saturation addition
    function automatic logic signed [31:0] sat_add32
            (input logic signed [31:0] a,
             input logic signed [31:0] b);
        logic signed [31:0] sum;
        logic ovf;
        begin
            sum = a + b;
            ovf = (a[31] == b[31]) && (sum[31] != a[31]);
            if (!ovf) begin
                sat_add32 = sum;
            end else if (a[31] == 0) begin
                sat_add32 = 32'h7FFF_FFFF;
            end else begin
                sat_add32 = 32'h8000_0000;
            end
        end
    endfunction

    // Upcast and accumulate
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < A_ROWS; i++)
                for (int j = 0; j < V_COLS; j++)
                    accum[i][j] <= 0;
        end else if (state == UPCAST_ACCUM) begin
            if (tile_ready) begin
                for (int i = 0; i < TILE_SIZE; i++) begin
                    for (int j = 0; j < TILE_SIZE; j++) begin
                        logic [31:0] result;
                        case (precision_sel[col_tile_idx*TILE_SIZE + col_idx])
                            2'b00: result = $signed(p4[i][j]) <<< 24;   // Q2.6 -> Q2.30
                            2'b01: result = $signed(p8[i][j]) <<< 16;   // Q1.14 -> Q2.30
                            2'b10: result = $signed(p16[i][j]);         // Q1.30
                            default: result = 0;
                        endcase
                        
                        accum[row_tile_idx*TILE_SIZE + i][tile_idx*TILE_SIZE + j] <=
                            sat_add32(accum[row_tile_idx*TILE_SIZE + i][tile_idx*TILE_SIZE + j], result);
                    end
                end
            end
        end
    end

    // Store output: Convert Q2.30 to Q1.15
    always_ff @(posedge clk) begin
        if (next_state == DONE) begin
            for (int i = 0; i < A_ROWS; i++) begin
                for (int j = 0; j < V_COLS; j++) begin
                    logic [WIDTH_OUT-1:0] result;
                    logic sign_bit = accum[i][j][31];
                    logic int_bit = accum[i][j][30];
                    if (sign_bit == int_bit) begin
                        result = {sign_bit, accum[i][j][29:15]};
                    end else begin
                        result = sign_bit ? 16'h8000 : 16'h7FFF;
                    end
                    out_mem[i * V_COLS + j] <= result;
                end
            end
        end
    end

endmodule
