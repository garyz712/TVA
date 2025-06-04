
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
//  May. 12 2025    Max Zhang       Worked for brute force mul16
//  May. 18 2025    Tianwei Liu    Bug fix, compute & check valid in correct stage
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
    input  logic [WIDTH_FP16-1:0]       a_mem [0:A_ROWS*NUM_COLS-1], // Flattened Attention matrix
    input  logic [WIDTH_FP16-1:0]       v_mem [0:NUM_COLS*V_COLS-1], // Flattened Value matrix
    output logic                        done,
 
    output logic [WIDTH_FP16-1:0]       out_mem [0:A_ROWS*V_COLS-1] // Accumulated output matrix
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
    logic [32-1:0]         accum [A_ROWS][V_COLS];   // Accumulator for all columns
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
    logic [31:0]           tile_idx;                 // Current tile index
    logic [31:0]           col_idx;                  // Current column index
    logic                  compute_valid;             // Compute enable
    logic                  accum_valid;              // Accumulation enable

    // Instantiate mul16_progressive for 8x8 submatrix
    genvar i, j;
    generate
        for (i = 0; i < A_ROWS; i++) begin : gen_row
            for (j = 0; j < TILE_SIZE; j++) begin : gen_col
                mul16_progressive mul_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .valid_in(compute_valid),
                    .a(precision_sel[col_idx] == 2'b00 ? {{12{a_int4[i][3]}}, a_int4[i]} :
                       precision_sel[col_idx] == 2'b01 ? {{8{a_int8[i][7]}}, a_int8[i]} :
                       a_fp16[i]),
                    .b(precision_sel[col_idx] == 2'b00 ? {{12{v_int4[j][3]}}, v_int4[j]} :
                       precision_sel[col_idx] == 2'b01 ? {{8{v_int8[j][7]}}, v_int8[j]} :
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
            for (int i = 0; i < A_ROWS; i++) begin
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

    logic tile_ready;
    always_comb begin
        tile_ready = 1'b0;
        for (int i = 0; i < A_ROWS; i++)
            for (int j = 0; j < TILE_SIZE; j++)
                tile_ready |= (precision_sel[col_idx]==2'b00) ? out4_valid [i][j] :
                            (precision_sel[col_idx]==2'b01) ? out8_valid [i][j] :
                                                                out16_valid[i][j];
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
                
                next_state = UPCAST_ACCUM;
            end
            UPCAST_ACCUM: begin
                accum_valid = 1'b1;
                next_state = tile_ready ? STORE_TILE : UPCAST_ACCUM;
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
                done = 1'b1; //cocotb line await RisingEdge(dut.done) therefore fires immediately after 20 ns, well before the next clock edge.
                next_state = IDLE;
            end
            default: begin
                next_state = IDLE;
            end
        endcase
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
                a_vec[i] <= a_mem[i * NUM_COLS + col_idx]; // Adjusted for flattened array
            for (int j = 0; j < TILE_SIZE; j++)
                v_vec[j] <= v_mem[col_idx * V_COLS + tile_idx * TILE_SIZE + j]; // Adjusted for flattened array
        end
    end

    // Load tile data
    always_ff @(posedge clk) begin
        if (state == LOAD_TILE) begin
            for (int j = 0; j < TILE_SIZE; j++)
                v_vec[j] <= v_mem[col_idx * V_COLS + tile_idx * TILE_SIZE + j]; // Adjusted for flattened array
        end
    end


    function automatic logic signed [31:0] sat_add32
            (input  logic signed [31:0] a,
            input  logic signed [31:0] b);

        logic signed [31:0] sum;
        logic               ovf;
        begin
            sum = a + b;                              // 32-bit two's-complement add
            ovf = (a[31] == b[31]) && (sum[31] != a[31]);

            if (!ovf) begin
                sat_add32 = sum;                      // no overflow → pass through
            end else if (a[31] == 0) begin            // operands were positive
                sat_add32 = 32'h7FFF_FFFF;            // clamp to +0.999 999 999 (Q2.30)
            end else begin                            // operands were negative
                sat_add32 = 32'h8000_0000;            // clamp to –1.0
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
            for (int i = 0; i < A_ROWS; i++) begin
                for (int j = 0; j < TILE_SIZE; j++) begin
                    logic [31:0] result;
                    logic valid;
                    case (precision_sel[col_idx])
                    // sign-cast + arithmetic left shift does everything in one step
                        2'b00 : result = $signed(p4[i][j])  <<< 24;   // Q2.6  -> Q2.30
                        2'b01 : result = $signed(p8[i][j])  <<< 16;  //{p8[i][j], {16{1'b0}}} // Q1.14 -> Q2.30
                        2'b10 : result = $signed(p16[i][j]);          // already Q2.30

                        // 2'b00: result = {{24{p4[i][j][7]}}, p4[i][j]} << 24; // Q2.6 to Q2.30: shift left by 24
                        // 2'b01: result = {{16{p8[i][j][15]}}, p8[i][j]} << 16;     // Q1.14 (INT8) is close to Q1.15
                        // 2'b10: result = $signed(p16[i][j]); // Q1.30 truncated to Q1.15
                        default: result = 0;
                    endcase
                    if (tile_ready) begin
                        accum[i][tile_idx*TILE_SIZE+j] <= sat_add32(accum[i][tile_idx*TILE_SIZE+j], result);
                        //accum[i][tile_idx*TILE_SIZE+j] <= accum[i][tile_idx*TILE_SIZE+j] + result;
                    end
                end
            end
        end
    end

    // Store output: Convert Q2.30 to Q1.15 by checking integer bits
    always_ff @(posedge clk) begin
        if (next_state == DONE) begin
            for (int i = 0; i < A_ROWS; i++) begin
                for (int j = 0; j <V_COLS; j++) begin
                    logic [WIDTH_OUT-1:0] result;
                    logic sign_bit = accum[i][j][31];
                    logic int_bit = accum[i][j][30];
                    if (sign_bit == int_bit) begin
                        // In range: take sign bit and top 15 fractional bits
                        result = {sign_bit, accum[i][j][29:15]};
                    end else begin
                        // Out of range: saturate based on sign
                        result = sign_bit ? 16'h8000 : 16'h7FFF;
                    end
                    out_mem[i * V_COLS + j] <= result;
                end
            end
        end
    end

endmodule
