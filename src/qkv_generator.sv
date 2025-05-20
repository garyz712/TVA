// Top-level of a QKV Matrix Generator using our matmul_array.
// May 9 2025    Max Zhang    Initial version
// May 13 2025   Max Zhang    update multiply logic
//--------------------------------------------------------------------------------
module qkv_generator #(
    parameter int DATA_WIDTH = 16,  // Data width for inputs (Q1.15)
    parameter int L = 8,           // Sequence length
    parameter int N = 1,           // Batch size
    parameter int E = 8            // Embedding dimension
)(
    input  logic                           clk,
    input  logic                           rst_n,
    input  logic                           start,
    output logic                           done,

    // Input x: shape (L*N, E), flattened
    input  logic [DATA_WIDTH-1:0]          x_in [L*N*E],
    // Weights WQ/WK/WV: shape (E, E), flattened
    input  logic [DATA_WIDTH-1:0]          WQ_in [E*E],
    input  logic [DATA_WIDTH-1:0]          WK_in [E*E],
    input  logic [DATA_WIDTH-1:0]          WV_in [E*E],

    // Outputs Q, K, V: each (L*N, E), flattened
    output logic [DATA_WIDTH-1:0]          Q_out [L*N*E],
    output logic [DATA_WIDTH-1:0]          K_out [L*N*E],
    output logic [DATA_WIDTH-1:0]          V_out [L*N*E],
    output logic                           out_valid
);

    // Local parameters
    localparam int TOTAL_TOKENS = L * N;
    localparam int OUT_DATA_WIDTH = 32;  // matmul_array output is 32-bit

    // State definitions
    typedef enum logic [2:0] {
        S_IDLE,
        S_LOAD,
        S_COMPUTE_Q,
        S_COMPUTE_K,
        S_COMPUTE_V,
        S_DONE
    } state_t;

    state_t curr_state, next_state;

    // Internal storage
    logic [DATA_WIDTH-1:0] Q_mem [TOTAL_TOKENS*E];  // Output storage
    logic [DATA_WIDTH-1:0] K_mem [TOTAL_TOKENS*E];
    logic [DATA_WIDTH-1:0] V_mem [TOTAL_TOKENS*E];

    // matmul_array control signals
    logic matmul_start;
    logic matmul_done;
    logic [DATA_WIDTH-1:0] matmul_a_in [TOTAL_TOKENS*E];  // Input matrix X
    logic [DATA_WIDTH-1:0] matmul_b_in [E*E];            // Weight matrix (WQ, WK, or WV)
    logic [OUT_DATA_WIDTH-1:0] matmul_c_out [TOTAL_TOKENS*E];  // Output matrix

    // Register to control matmul_start pulse
    logic start_pulse;

    // Instantiate matmul_array
    matmul_array #(
        .M(TOTAL_TOKENS),
        .K(E),
        .N(E)
    ) matmul_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(matmul_start),
        .a_in(matmul_a_in),
        .b_in(matmul_b_in),
        .c_out(matmul_c_out),
        .done(matmul_done)
    );

    // State register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            curr_state <= S_IDLE;
        else
            curr_state <= next_state;
    end

    // Start pulse logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            start_pulse <= 1'b0;
        end else begin
            // Assert start_pulse for one cycle after inputs are loaded
            if ((curr_state == S_LOAD ) ||
                (curr_state == S_COMPUTE_Q && matmul_done) ||
                (curr_state == S_COMPUTE_K && matmul_done)) begin
                start_pulse <= 1'b1;
            end else begin
                start_pulse <= 1'b0;
            end
        end
    end

    // Assign matmul_start
    assign matmul_start = start_pulse;

    // Next-state logic
    always_comb begin
        next_state = curr_state;

        case (curr_state)
            S_IDLE: begin
                if (start)
                    next_state = S_LOAD;
            end
            S_LOAD: begin
                next_state = S_COMPUTE_Q;
            end
            S_COMPUTE_Q: begin
                if (matmul_done)
                    next_state = S_COMPUTE_K;
            end
            S_COMPUTE_K: begin
                if (matmul_done)
                    next_state = S_COMPUTE_V;
            end
            S_COMPUTE_V: begin
                if (matmul_done)
                    next_state = S_DONE;
            end
            S_DONE: begin
                next_state = S_IDLE;
            end
            default: next_state = S_IDLE;
        endcase
    end

    // Data path and matrix multiplication control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (integer i = 0; i < TOTAL_TOKENS*E; i++) begin
                matmul_a_in[i] <= '0;
                Q_mem[i] <= '0;
                K_mem[i] <= '0;
                V_mem[i] <= '0;
            end
            for (integer i = 0; i < E*E; i++) begin
                matmul_b_in[i] <= '0;
            end
        end else begin
            case (curr_state)
                S_LOAD: begin
                    // Load x_in into matmul_a_in and WQ_in into matmul_b_in
                    for (integer i = 0; i < TOTAL_TOKENS*E; i++)
                        matmul_a_in[i] <= x_in[i];
                    for (integer i = 0; i < E*E; i++)
                        matmul_b_in[i] <= WQ_in[i];
                end
                S_COMPUTE_Q: begin
                    // Store matmul output to Q_mem (convert Q2.30 to Q1.15 with saturation)
                    if (matmul_done)
                        for (integer i = 0; i < TOTAL_TOKENS*E; i++) begin
                            logic sign_bit = matmul_c_out[i][31];
                            logic int_bit = matmul_c_out[i][30];
                            if (sign_bit == int_bit) begin
                                // In range: take sign bit and top 15 fractional bits
                                Q_mem[i] <= {sign_bit, matmul_c_out[i][29:15]};
                            end else begin
                                // Out of range: saturate based on sign
                                Q_mem[i] <= sign_bit ? 16'h8000 : 16'h7FFF;
                            end
                        end
                    // Prepare WK_in for next computation
                    if (matmul_done)
                        for (integer i = 0; i < E*E; i++)
                            matmul_b_in[i] <= WK_in[i];
                end
                S_COMPUTE_K: begin
                    // Store matmul output to K_mem (convert Q2.30 to Q1.15 with saturation)
                    if (matmul_done)
                        for (integer i = 0; i < TOTAL_TOKENS*E; i++) begin
                            logic sign_bit = matmul_c_out[i][31];
                            logic int_bit = matmul_c_out[i][30];
                            if (sign_bit == int_bit) begin
                                // In range: take sign bit and top 15 fractional bits
                                K_mem[i] <= {sign_bit, matmul_c_out[i][29:15]};
                            end else begin
                                // Out of range: saturate based on sign
                                K_mem[i] <= sign_bit ? 16'h8000 : 16'h7FFF;
                            end
                        end
                    // Prepare WV_in for next computation
                    if (matmul_done)
                        for (integer i = 0; i < E*E; i++)
                            matmul_b_in[i] <= WV_in[i];
                end
                S_COMPUTE_V: begin
                    // Store matmul output to V_mem (convert Q2.30 to Q1.15 with saturation)
                    if (matmul_done)
                        for (integer i = 0; i < TOTAL_TOKENS*E; i++) begin
                            logic sign_bit = matmul_c_out[i][31];
                            logic int_bit = matmul_c_out[i][30];
                            if (sign_bit == int_bit) begin
                                // In range: take sign bit and top 15 fractional bits
                                V_mem[i] <= {sign_bit, matmul_c_out[i][29:15]};
                            end else begin
                                // Out of range: saturate based on sign
                                V_mem[i] <= sign_bit ? 16'h8000 : 16'h7FFF;
                            end
                        end
                end
                S_DONE: begin
                    // Assign outputs
                    for (integer i = 0; i < TOTAL_TOKENS*E; i++) begin
                        Q_out[i] <= Q_mem[i];
                        K_out[i] <= K_mem[i];
                        V_out[i] <= V_mem[i];
                    end
                end
                default: ;
            endcase
        end
    end

    // Output control signals
    always_comb begin
        done = (curr_state == S_DONE);
        out_valid = (curr_state == S_DONE);
    end

endmodule
