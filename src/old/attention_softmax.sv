//------------------------------------------------------------------------------
// softmax_approx.sv
//
// softmax_approx
// This module computes a row-wise softmax approximation for attention scores.
// It takes an input matrix A_in of shape (L, N, L) and produces normalized
// attention weights A_out of the same shape. The implementation:
// 1. Computes row sums (exponentiation omitted for approximation)
// 2. Normalizes each element by its row sum
// 3. Outputs valid results when computation is complete
//
// Note: This is an approximation that skips the exponential function for
// hardware efficiency. For better accuracy, consider adding a lookup table
// or piecewise linear approximation of exp().
//
// Apr 10 2025    Max Zhang      Initial version
// Apr 13 2025    Tianwei Liu    Refactor to SystemVerilog
// Apr 13 2025    Tianwei Liu    split state machine for clarity
// Apr 13 2025    Tianwei Liu    comments
//------------------------------------------------------------------------------


// Parameters:
//     DATA_WIDTH: Bit width of each element (default 16)
//     L: Sequence length (default 8)
//     N: Number of attention heads (default 1)

module softmax_approx #(
    parameter int DATA_WIDTH = 16,
    parameter int L = 8,
    parameter int N = 1
)(
    input  logic                        clk,
    input  logic                        rst_n,
    // Control signals
    input  logic                        start,
    output logic                        done,
    // Input: A_in of shape (L, N, L)
    input  logic [DATA_WIDTH*L*N*L-1:0] A_in,
    // Output: A_out of shape (L, N, L)
    output logic [DATA_WIDTH*L*N*L-1:0] A_out,
    output logic                        out_valid //FIXME: WHY???
);

    // Internal array storage
    logic [DATA_WIDTH-1:0] A_arr [L][N][L];
    logic [DATA_WIDTH-1:0] A_out_arr [L][N][L];

    // State machine
    typedef enum logic [1:0] {
        S_IDLE  = 2'd0,
        S_LOAD  = 2'd1,
        S_SOFT  = 2'd2,
        S_DONE  = 2'd3
    } state_t;
    state_t state, next_state;

    // Counters
    logic [$clog2(L)-1:0] l_cnt, l2_cnt;
    logic [$clog2(N)-1:0] n_cnt;

    // Temporary registers
    logic [DATA_WIDTH-1:0] row_sum [L][N];

    // Sequential state transition
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            out_valid <= 1'b0;
            l_cnt <= '0;
            n_cnt <= '0;
            l2_cnt <= '0;
        end else begin
            state <= next_state;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    out_valid <= 1'b0;
                    l_cnt <= '0;
                    n_cnt <= '0;
                    l2_cnt <= '0;
                end

                S_LOAD: begin
                    // Reset counters for computation
                    l_cnt <= '0;
                    n_cnt <= '0;
                    l2_cnt <= '0;
                end

                S_SOFT: begin
                    if (l2_cnt == L-1) begin
                        l2_cnt <= '0;
                        if (n_cnt == N-1) begin
                            n_cnt <= '0;
                            if (l_cnt == L-1) begin
                                l_cnt <= '0;
                            end else begin
                                l_cnt <= l_cnt + 1;
                            end
                        end else begin
                            n_cnt <= n_cnt + 1;
                        end
                    end else begin
                        l2_cnt <= l2_cnt + 1;
                    end
                end

                S_DONE: begin
                    out_valid <= 1'b1;
                    done <= 1'b1;
                end
            endcase
        end
    end

    // Next state logic
    always_comb begin
        next_state = state;
        case (state)
            S_IDLE: if (start) next_state = S_LOAD;
            S_LOAD: next_state = S_SOFT;
            S_SOFT: if (l_cnt == L-1 && n_cnt == N-1 && l2_cnt == L-1) 
                       next_state = S_DONE;
            S_DONE: next_state = S_IDLE;
            default: next_state = S_IDLE;
        endcase
    end

    // Load input data
    always_ff @(posedge clk) begin
        if (state == S_LOAD) begin
            for (int l = 0; l < L; l++) begin
                for (int n = 0; n < N; n++) begin
                    for (int l2 = 0; l2 < L; l2++) begin
                        A_arr[l][n][l2] <= A_in[((l*N*L)+(n*L)+l2)*DATA_WIDTH +: DATA_WIDTH];
                    end
                end
            end
        end
    end

    // Softmax computation
    always_ff @(posedge clk) begin
        if (state == S_SOFT) begin
            // Row sum accumulation
            if (l2_cnt == 0) begin
                row_sum[l_cnt][n_cnt] <= A_arr[l_cnt][n_cnt][0];
            end else begin
                row_sum[l_cnt][n_cnt] <= row_sum[l_cnt][n_cnt] + A_arr[l_cnt][n_cnt][l2_cnt];
            end

            // Division (placeholder - replace with actual approximation)
            // Note: In real implementation, this would need proper fixed-point division
            // or approximation (like reciprocal multiplication)
            A_out_arr[l_cnt][n_cnt][l2_cnt] <= A_arr[l_cnt][n_cnt][l2_cnt]; // Placeholder
        end
    end

    // Output packing
    always_ff @(posedge clk) begin
        if (state == S_DONE) begin
            for (int l = 0; l < L; l++) begin
                for (int n = 0; n < N; n++) begin
                    for (int l2 = 0; l2 < L; l2++) begin
                        A_out[((l*N*L)+(n*L)+l2)*DATA_WIDTH +: DATA_WIDTH] <= A_out_arr[l][n][l2];
                    end
                end
            end
        end
    end

endmodule
