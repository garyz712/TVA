//------------------------------------------------------------------------------
// attention_av_multiply.sv
//
//  attention_av_multiply
//  module implements a matrix multiplication operation tailored for the
//  attention mechanism in neural networks, specifically computing the product
//  of an attention weight matrix AA and a value matrix VV. It operates on
//  fixed-point or integer data with per-token precision control, producing an
//  output matrix ZZ.
//
//  Apr 10 2025    Max Zhang      initial version
//  Apr 11 2025    Tianwei Liu    refactor in SV
//  Apr 11 2025    Tianwei Liu    split state machine logic
//  Apr 11 2025    Tianwei Liu    add comments
//------------------------------------------------------------------------------

// IO Description
//
// Parameters:
//    DATA_WIDTH: Bit width of each element (default 16 bits).
//    L: Number of tokens (sequence length, default 8).
//    N: Number of attention heads (default 1).
//    E: Embedding dimension per head (default 8).
//
// Inputs:
//
//    clk: Clock signal for synchronous operation.
//    rst_n: Active-low asynchronous reset.
//    start: Control signal to initiate computation.
//    A_in: Flattened input for matrix AA
//    V_in: Flattened input for matrix VV
//    token_precision [L-1:0]: Array of 4-bit precision codes,
//        one per token, controlling downcasting of VV.
//
// Outputs:
//
//    done: Indicates computation is complete.
//    Z_out: Flattened output matrix ZZ
//    out_valid: Signals valid output data.

module attention_av_multiply #(
    parameter int DATA_WIDTH = 16,
    parameter int L = 8,
    parameter int N = 1,
    parameter int E = 8
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
    // Per-token precision codes: length L
    input  logic [3:0]                  token_precision [L-1:0],
    // Output Z: shape (L, N, E)
    output logic [DATA_WIDTH*L*N*E-1:0] Z_out,
    output logic                        out_valid
);
    // Use SystemVerilog packed arrays for synthesis clarity
    logic [DATA_WIDTH-1:0] A_arr [L-1:0][N-1:0][L-1:0];
    logic [DATA_WIDTH-1:0] V_arr [L-1:0][N-1:0][E-1:0];
    logic [DATA_WIDTH-1:0] Z_arr [L-1:0][N-1:0][E-1:0];

    // State machine encoding
    typedef enum logic [1:0] {
        S_IDLE = 2'd0,
        S_LOAD = 2'd1,
        S_MUL  = 2'd2,
        S_DONE = 2'd3
    } state_t;
    state_t state, next_state;

    // Internal signals for accumulation
    logic [DATA_WIDTH-1:0] Z_temp [L-1:0][N-1:0][E-1:0];
    logic [DATA_WIDTH-1:0] product;
    logic [DATA_WIDTH-1:0] valA_down;
    logic [DATA_WIDTH-1:0] valV_down;

    // Counter for MUL state to iterate over l2
    logic [$clog2(L)-1:0] l2_cnt;

    // Sequential logic for state and control signals
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            out_valid <= 1'b0;
            l2_cnt <= '0;
        end else begin
            state <= next_state;
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    out_valid <= 1'b0;
                    l2_cnt <= '0;
                end
                S_LOAD: begin
                    l2_cnt <= '0;
                end
                S_MUL: begin
                    if (l2_cnt == L-1) begin
                        l2_cnt <= '0;
                    end else begin
                        l2_cnt <= l2_cnt + 1;
                    end
                end
                S_DONE: begin
                    done <= 1'b1;
                    out_valid <= 1'b1;
                    l2_cnt <= '0;
                end
            endcase
        end
    end

    // Next state logic
    always_comb begin
        next_state = state;
        case (state)
            S_IDLE: begin
                if (start) begin
                    next_state = S_LOAD;
                end
            end
            S_LOAD: begin
                next_state = S_MUL;
            end
            S_MUL: begin
                if (l2_cnt == L-1) begin
                    next_state = S_DONE;
                end
            end
            S_DONE: begin
                next_state = S_IDLE;
            end
            default: next_state = S_IDLE;
        endcase
    end

    // Load inputs into arrays
    always_ff @(posedge clk) begin
        if (state == S_LOAD) begin
            // Unpack A_in
            for (int l = 0; l < L; l++) begin
                for (int n_ = 0; n_ < N; n_++) begin
                    for (int l2 = 0; l2 < L; l2++) begin
                        A_arr[l][n_][l2] <= A_in[((l*N*L)+(n_*L)+l2)*DATA_WIDTH +: DATA_WIDTH];
                    end
                end
            end
            // Unpack V_in
            for (int l2 = 0; l2 < L; l2++) begin
                for (int n_ = 0; n_ < N; n_++) begin
                    for (int e_ = 0; e_ < E; e_++) begin
                        V_arr[l2][n_][e_] <= V_in[((l2*N*E)+(n_*E)+e_)*DATA_WIDTH +: DATA_WIDTH];
                    end
                end
            end
            // Initialize Z_arr
            for (int l = 0; l < L; l++) begin
                for (int n_ = 0; n_ < N; n_++) begin
                    for (int e_ = 0; e_ < E; e_++) begin
                        Z_arr[l][n_][e_] <= '0;
                    end
                end
            end
        end
    end

    // Multiplication and accumulation
    always_ff @(posedge clk) begin
        if (state == S_MUL) begin
            for (int l = 0; l < L; l++) begin
                for (int n_ = 0; n_ < N; n_++) begin
                    for (int e_ = 0; e_ < E; e_++) begin
                        // Downcast A, V based on precision code
                        case (token_precision[l2_cnt])
                            4'd0: begin // INT4 (Q1.3 format)
                                valA_down = {{(DATA_WIDTH-4){1'b0}}, A_arr[l][n_][l2_cnt][3:0]};
                                valV_down = {{(DATA_WIDTH-4){1'b0}}, V_arr[l2_cnt][n_][e_][3:0]};
                            end
                            4'd1: begin // INT8 (Q1.7 format)
                                valA_down = {{(DATA_WIDTH-8){1'b0}}, A_arr[l][n_][l2_cnt][7:0]};
                                valV_down = {{(DATA_WIDTH-8){1'b0}}, V_arr[l2_cnt][n_][e_][7:0]};
                            end
                            4'd2: begin // FP16
                                valA_down = A_arr[l][n_][l2_cnt];
                                valV_down = V_arr[l2_cnt][n_][e_];
                            end
                            default: begin // FP16
                                valA_down = A_arr[l][n_][l2_cnt];
                                valV_down = V_arr[l2_cnt][n_][e_];
                            end
                        endcase

                        // Multiply in lower precision, upcast product to FP16 automatically
                        product = valA_down * valV_down;
                        // Accumulation in FP16
                        Z_arr[l][n_][e_] <= Z_arr[l][n_][e_] + product;
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
                        Z_out[((l*N*E)+(n_*E)+e_)*DATA_WIDTH +: DATA_WIDTH] <= Z_arr[l][n_][e_];
                    end
                end
            end
        end
    end

endmodule
