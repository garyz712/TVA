//------------------------------------------------------------------------------
// attention_score.sv
//
// attention_score
// This module computes the attention scores in a transformer model's self-attention
// mechanism using signed arithmetic, correct input/output indexing, and fixed timing.
//
// Apr 10 2025    Max Zhang      initial version
// Apr 12 2025    Tianwei Liu    refactor into SV style
// Apr 12 2025    Tianwei Liu    split state machine logic
// Apr 13 2025    Tianwei Liu    add comments
// Apr 28 2025    Tianwei Liu    fix dot product computation bug
//------------------------------------------------------------------------------

module attention_score #(
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
    // Inputs Q, K: each (L, N, E)
    input  logic [DATA_WIDTH*L*N*E-1:0] Q_in,
    input  logic [DATA_WIDTH*L*N*E-1:0] K_in,
    // Output: A of shape (L, N, L)
    output logic [DATA_WIDTH*L*N*L-1:0] A_out,
    output logic                        out_valid
);
    // Internal arrays using packed dimensions for synthesis
    logic signed [DATA_WIDTH-1:0] Q [L-1:0][N-1:0][E-1:0];
    logic signed [DATA_WIDTH-1:0] K [L-1:0][N-1:0][E-1:0];
    logic signed [DATA_WIDTH-1:0] A [L-1:0][N-1:0][L-1:0];

    // Temporary sum for dot product (wider to avoid overflow)
    logic signed [2*DATA_WIDTH-1:0] sum_temp [L-1:0][N-1:0][L-1:0];

    // State machine encoding
    typedef enum logic [1:0] {
        S_IDLE    = 2'd0,
        S_LOAD    = 2'd1,
        S_COMPUTE = 2'd2,
        S_DONE    = 2'd3
    } state_t;
    state_t state, next_state;

    // Counters for computation
    logic [$clog2(L)-1:0] l_cnt, l2_cnt;
    logic [$clog2(E)-1:0] e_cnt;

    // Sequential logic for state and control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            out_valid <= 1'b0;
            l_cnt <= '0;
            l2_cnt <= '0;
            e_cnt <= '0;
        end else begin
            state <= next_state;
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    out_valid <= 1'b0;
                    l_cnt <= '0;
                    l2_cnt <= '0;
                    e_cnt <= '0;
                end
                S_LOAD: begin
                    l_cnt <= '0;
                    l2_cnt <= '0;
                    e_cnt <= '0;
                end
                S_COMPUTE: begin
                    e_cnt <= e_cnt + 1;
                    if (e_cnt == E-1) begin
                        e_cnt <= '0;
                        if (l2_cnt == L-1) begin
                            l2_cnt <= '0;
                            if (l_cnt == L-1) begin
                                l_cnt <= '0;
                            end else begin
                                l_cnt <= l_cnt + 1;
                            end
                        end else begin
                            l2_cnt <= l2_cnt + 1;
                        end
                    end
                end
                S_DONE: begin
                    l_cnt <= '0;
                    l2_cnt <= '0;
                    e_cnt <= '0;
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
            S_IDLE: begin
                if (start) begin
                    next_state = S_LOAD;
                end
            end
            S_LOAD: begin
                next_state = S_COMPUTE;
            end
            S_COMPUTE: begin
                if (l_cnt == L-1 && l2_cnt == L-1 && e_cnt == E-1) begin
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
            // Unpack Q_in and K_in
            for (int l = 0; l < L; l++) begin
                for (int n_ = 0; n_ < N; n_++) begin
                    for (int e_ = 0; e_ < E; e_++) begin
                        Q[l][n_][e_] <= Q_in[((l*N*E)+(n_*E)+(e_))*DATA_WIDTH +: DATA_WIDTH];
                        K[l][n_][e_] <= K_in[((l*N*E)+(n_*E)+(e_))*DATA_WIDTH +: DATA_WIDTH];
                    end
                end
            end
            // Initialize sum_temp
            for (int l = 0; l < L; l++) begin
                for (int n_ = 0; n_ < N; n_++) begin
                    for (int l2 = 0; l2 < L; l2++) begin
                        sum_temp[l][n_][l2] <= '0;
                    end
                end
            end
        end
    end

    // Compute dot product iteratively
    always_ff @(posedge clk) begin
        if (state == S_COMPUTE) begin
            for (int n_ = 0; n_ < N; n_++) begin
                // Compute one term of the dot product
                logic signed [2*DATA_WIDTH-1:0] next_sum;
                next_sum = sum_temp[l_cnt][n_][l2_cnt] + 
                           (Q[l_cnt][n_][e_cnt] * K[l2_cnt][n_][e_cnt]);
                sum_temp[l_cnt][n_][l2_cnt] <= next_sum;
                // Store result and reset sum_temp when dot product is complete
                if (e_cnt == E-1) begin
                    A[l_cnt][n_][l2_cnt] <= next_sum[DATA_WIDTH-1:0]; // Use updated sum
                    sum_temp[l_cnt][n_][l2_cnt] <= '0;
                end
            end
        end
    end

    // Pack output
    always_ff @(posedge clk) begin
        if (state == S_DONE) begin
            for (int l = 0; l < L; l++) begin
                for (int n_ = 0; n_ < N; n_++) begin
                    for (int l2 = 0; l2 < L; l2++) begin
                        A_out[(((l)*N*L)+(n_*L)+(l2))*DATA_WIDTH +: DATA_WIDTH] <= A[l][n_][l2];
                    end
                end
            end
        end
    end

endmodule