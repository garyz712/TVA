//------------------------------------------------------------------------------
// attention_av_multiply.sv
//
//  Module implements a matrix multiplication for the attention mechanism,
//  computing A * V with token-wise mixed precision (INT4, INT8, FP16).
//  Uses pipelined multipliers with variable latency to reduce cycle count
//  for lower precision tokens, operating at a single clock frequency.
//
//  Apr 10 2025    Max Zhang      Initial version
//  Apr 11 2025    Tianwei Liu    Refactor in SV, split state machine, add comments
//  [Date]         [Your Name]    Redesigned with pipelined variable-latency multipliers
//------------------------------------------------------------------------------

module attention_av_multiply #(
    parameter int DATA_WIDTH = 16,
    parameter int L = 8,
    parameter int N = 1,
    parameter int E = 8,
    parameter int P = 16  // Number of parallel multipliers
)(
    input  logic                        clk,
    input  logic                        rst_n,
    // Control
    input  logic                        start,
    output logic                        done,
    // Input A: shape (L, N, L)
    input  logic [DATA_WIDTH*L*N*L-1:0] A_in,
    // Input V: shape (L, N, E)
    input  logic [DATA_WIDTH*L*N*E-1:0] V_in,
    // Per-token precision codes: length L (0: INT4, 1: INT8, 2: FP16)
    input  logic [3:0]                  token_precision [L-1:0],
    // Output Z: shape (L, N, E)
    output logic [DATA_WIDTH*L*N*E-1:0] Z_out,
    output logic                        out_valid
);

    // Packed arrays for synthesis clarity
    logic [DATA_WIDTH-1:0] A_arr [L-1:0][N-1:0][L-1:0];
    logic [DATA_WIDTH-1:0] V_arr [L-1:0][N-1:0][E-1:0];
    logic [31:0] Z_arr [L-1:0][N-1:0][E-1:0]; // 32-bit accumulators

    // State machine encoding
    typedef enum logic [1:0] {
        S_IDLE = 2'd0,
        S_LOAD = 2'd1,
        S_MUL  = 2'd2,
        S_DONE = 2'd3
    } state_t;
    state_t state, next_state;

    // Counters
    logic [$clog2(L)-1:0] l2_cnt;              // Token index
    logic [$clog2(L*N*E/P + 4)-1:0] cycle_cnt; // Cycles per token, max latency 4
    logic [$clog2(4)-1:0] latency;             // Pipeline latency (K)

    // Multiplier pipeline signals
    logic [DATA_WIDTH-1:0] mult_in_a [P-1:0];
    logic [DATA_WIDTH-1:0] mult_in_v [P-1:0];
    logic [3:0] mult_prec [P-1:0];
    logic [31:0] mult_out [P-1:0];
    logic mult_valid [P-1:0];

    // Compute cycles needed per token
    localparam int TOTAL_MULTS = L * N * E;
    localparam int CYCLES_PER_BATCH = (TOTAL_MULTS + P - 1) / P; // ceil(TOTAL_MULTS / P)

    // Instantiate P pipelined multipliers
    genvar i;
    generate
        for (i = 0; i < P; i++) begin : gen_mult
            adaptive_multiplier mult (
                .clk(clk),
                .rst_n(rst_n),
                .a_in(mult_in_a[i]),
                .v_in(mult_in_v[i]),
                .precision(mult_prec[i]),
                .product(mult_out[i]),
                .valid(mult_valid[i])
            );
        end
    endgenerate

    // Sequential logic for state and counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            out_valid <= 1'b0;
            l2_cnt <= '0;
            cycle_cnt <= '0;
            latency <= '0;
        end else begin
            state <= next_state;
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    out_valid <= 1'b0;
                    l2_cnt <= '0;
                    cycle_cnt <= '0;
                    latency <= '0;
                end
                S_LOAD: begin
                    l2_cnt <= '0;
                    cycle_cnt <= '0;
                    latency <= get_latency(token_precision[0]);
                end
                S_MUL: begin
                    if (cycle_cnt == CYCLES_PER_BATCH + latency - 1) begin
                        if (l2_cnt == L - 1) begin
                            l2_cnt <= '0;
                            cycle_cnt <= '0;
                            latency <= '0;
                        end else begin
                            l2_cnt <= l2_cnt + 1;
                            cycle_cnt <= '0;
                            latency <= get_latency(token_precision[l2_cnt + 1]);
                        end
                    end else begin
                        cycle_cnt <= cycle_cnt + 1;
                    end
                end
                S_DONE: begin
                    done <= 1'b1;
                    out_valid <= 1'b1;
                end
            endcase
        end
    end

    // Next state logic
    always_comb begin
        next_state = state;
        case (state)
            S_IDLE: if (start) next_state = S_LOAD;
            S_LOAD: next_state = S_MUL;
            S_MUL: if (cycle_cnt == CYCLES_PER_BATCH + latency - 1 && l2_cnt == L - 1) next_state = S_DONE;
            S_DONE: next_state = S_IDLE;
            default: next_state = S_IDLE;
        endcase
    end

    // Load inputs into arrays
    always_ff @(posedge clk) begin
        if (state == S_LOAD) begin
            for (int l = 0; l < L; l++) begin
                for (int n_ = 0; n_ < N; n_++) begin
                    for (int l2 = 0; l2 < L; l2++) begin
                        A_arr[l][n_][l2] <= A_in[((l*N*L)+(n_*L)+l2)*DATA_WIDTH +: DATA_WIDTH];
                    end
                end
            end
            for (int l2 = 0; l2 < L; l2++) begin
                for (int n_ = 0; n_ < N; n_++) begin
                    for (int e_ = 0; e_ < E; e_++) begin
                        V_arr[l2][n_][e_] <= V_in[((l2*N*E)+(n_*E)+e_)*DATA_WIDTH +: DATA_WIDTH];
                    end
                end
            end
            for (int l = 0; l < L; l++) begin
                for (int n_ = 0; n_ < N; n_++) begin
                    for (int e_ = 0; e_ < E; e_++) begin
                        Z_arr[l][n_][e_] <= '0;
                    end
                end
            end
        end
    end

    // Feed multipliers and accumulate results
    integer mult_idx, l, n, e;
    always_ff @(posedge clk) begin
        if (state == S_MUL) begin
            // Feed multipliers
            if (cycle_cnt < CYCLES_PER_BATCH) begin
                for (int p = 0; p < P; p++) begin
                    mult_idx = cycle_cnt * P + p;
                    if (mult_idx < TOTAL_MULTS) begin
                        l = mult_idx / (N * E);
                        n = (mult_idx / E) % N;
                        e = mult_idx % E;
                        mult_in_a[p] <= A_arr[l][n][l2_cnt];
                        mult_in_v[p] <= V_arr[l2_cnt][n][e];
                        mult_prec[p] <= token_precision[l2_cnt];
                    end else begin
                        mult_in_a[p] <= '0;
                        mult_in_v[p] <= '0;
                        mult_prec[p] <= '0;
                    end
                end
            end

            // Accumulate results after latency
            if (cycle_cnt >= latency) begin
                for (int p = 0; p < P; p++) begin
                    if (mult_valid[p]) begin
                        mult_idx = (cycle_cnt - latency) * P + p;
                        if (mult_idx < TOTAL_MULTS) begin
                            l = mult_idx / (N * E);
                            n = (mult_idx / E) % N;
                            e = mult_idx % E;
                            Z_arr[l][n][e] <= Z_arr[l][n][e] + mult_out[p];
                        end
                    end
                end
            end
        end
    end

    // Pack output
    always_ff @(posedge clk) begin
        if (state == S_DONE) begin
            for (int l = 0; l < L; l++) begin
                for (int n_ = 0; n_ < N; n_++) begin
                    for (int e_ = 0; e_ < E; e_++) begin
                        Z_out[((l*N*E)+(n_*E)+e_)*DATA_WIDTH +: DATA_WIDTH] <= Z_arr[l][n_][e_][15:0];
                    end
                end
            end
        end
    end

    // Function to determine multiplier latency based on precision
    function logic [1:0] get_latency(input logic [3:0] prec);
        case (prec)
            4'd0: get_latency = 2'd1; // INT4: 1 cycle
            4'd1: get_latency = 2'd2; // INT8: 2 cycles
            4'd2: get_latency = 2'd4; // FP16: 4 cycles
            default: get_latency = 2'd4; // Default to FP16
        endcase
    endfunction

endmodule

// Adaptive multiplier module with variable pipeline latency
module adaptive_multiplier (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic [15:0]           a_in,
    input  logic [15:0]           v_in,
    input  logic [3:0]            precision,
    output logic [31:0]           product,
    output logic                  valid
);
    logic [3:0]  val_a_int4, val_v_int4;
    logic [7:0]  val_a_int8, val_v_int8;
    logic [15:0] val_a_fp16, val_v_fp16;
    logic [7:0]  prod_int4;
    logic [15:0] prod_int8;
    logic [31:0] prod_fp16;
    logic [31:0] result;
    logic [1:0]  latency;

    // Pipeline registers (max depth 4)
    logic [31:0] pipe [3:0];
    logic [3:0]  valid_pipe;

    always_comb begin
        case (precision)
            4'd0: begin // INT4
                val_a_int4 = a_in[3:0];
                val_v_int4 = v_in[3:0];
                prod_int4 = val_a_int4 * val_v_int4;
                result = {{24{1'b0}}, prod_int4} >> 6;
                latency = 2'd1;
            end
            4'd1: begin // INT8
                val_a_int8 = a_in[7:0];
                val_v_int8 = v_in[7:0];
                prod_int8 = val_a_int8 * val_v_int8;
                result = {{16{1'b0}}, prod_int8} >> 14;
                latency = 2'd2;
            end
            4'd2, default: begin // FP16
                val_a_fp16 = a_in;
                val_v_fp16 = v_in;
                prod_fp16 = val_a_fp16 * val_v_fp16; // Simplified, assumes FP16 hardware
                result = {{16{1'b0}}, prod_fp16[15:0]};
                latency = 2'd4;
            end
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 4; i++) begin
                pipe[i] <= '0;
                valid_pipe[i] <= 1'b0;
            end
        end else begin
            pipe[0] <= result;
            valid_pipe[0] <= 1'b1;
            for (int i = 1; i < 4; i++) begin
                pipe[i] <= pipe[i-1];
                valid_pipe[i] <= valid_pipe[i-1];
            end
        end
    end

    always_comb begin
        product = pipe[latency - 1];
        valid = valid_pipe[latency - 1];
    end
endmodule
