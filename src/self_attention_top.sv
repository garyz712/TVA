//------------------------------------------------------------------------------
// self_attention.sv
//
// Top-level module for self-attention mechanism with token-wise mixed precision.
// Integrates qkv_generator, attention_score, precision_assigner, softmax_approx,
// attention_av_multiply, and matmul_array for output projection with W_O.
// Assumes a single attention head (N=1) and uses matmul_array for all matrix
// multiplications.
//
// May 24 2025    Generated       Initial version
//------------------------------------------------------------------------------

module self_attention_top #(
    parameter int DATA_WIDTH = 16,  // Data width (Q1.15 for FP16)
    parameter int L = 8,           // Sequence length
    parameter int E = 8,           // Embedding dimension
    parameter int N = 1            // Number of attention heads (fixed to 1)
)(
    input  logic                           clk,
    input  logic                           rst_n,
    input  logic                           start,
    output logic                           done,
    output logic                           out_valid,

    // Input: shape (L, E), flattened
    input  logic [DATA_WIDTH-1:0]          x_in [L*E],

    // Weight matrices: WQ, WK, WV, WO, each shape (E, E), flattened
    input  logic [DATA_WIDTH-1:0]          WQ_in [E*E],
    input  logic [DATA_WIDTH-1:0]          WK_in [E*E],
    input  logic [DATA_WIDTH-1:0]          WV_in [E*E],
    input  logic [DATA_WIDTH-1:0]          WO_in [E*E],

    // Output: shape (L, E), flattened
    output logic [DATA_WIDTH-1:0]          out [L*E]
);

    // Local parameters
    localparam int TOTAL_TOKENS = L * N;  // L * 1 = L
    localparam int OUT_DATA_WIDTH = 32;   // matmul_array output width (Q2.30)

    // State machine states
    typedef enum logic [3:0] {
        S_IDLE,
        S_QKV,
        S_ATTENTION_SCORE,
        S_PRECISION_ASSIGN,
        S_SOFTMAX,
        S_AV_MULTIPLY,
        S_WO_MULTIPLY,
        S_DONE
    } state_t;

    state_t curr_state, next_state;

    // Internal signals and storage
    logic [DATA_WIDTH-1:0] Q [L*E];       // Query matrix: (L, E)
    logic [DATA_WIDTH-1:0] K [L*E];       // Key matrix: (L, E)
    logic [DATA_WIDTH-1:0] V [L*E];       // Value matrix: (L, E)
    logic [DATA_WIDTH-1:0] A [L*L];       // Attention scores: (L, L)
    //logic [DATA_WIDTH-1:0] A_softmax [L*L];
    logic [1:0]            token_precision [L]; // Precision codes per token
    logic [DATA_WIDTH-1:0] AV_out [L*E];  // A * V output: (L, E)

    // Control signals for submodules
    logic qkv_start, qkv_done, qkv_out_valid;
    logic attn_score_start, attn_score_done, attn_score_out_valid;
    logic prec_assign_start, prec_assign_done;
    logic softmax_start, softmax_done, softmax_out_valid;
    logic av_multiply_start, av_multiply_done;
    logic wo_multiply_start, wo_multiply_done;

    // matmul_array signals for W_O multiplication
    logic [DATA_WIDTH-1:0]  wo_matmul_a_in [L*E];   // Input: AV_out
    logic [DATA_WIDTH-1:0]  wo_matmul_b_in [E*E];   // Input: W_O
    logic [OUT_DATA_WIDTH-1:0] wo_matmul_c_out [L*E]; // Output: final result

    // Start pulse logic for submodules
    logic qkv_start_pulse, attn_score_start_pulse, prec_assign_start_pulse;
    logic softmax_start_pulse, av_multiply_start_pulse, wo_multiply_start_pulse;

    // Instantiate qkv_generator
    qkv_generator #(
        .DATA_WIDTH(DATA_WIDTH),
        .L(L),
        .N(N),
        .E(E)
    ) qkv_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(qkv_start),
        .done(qkv_done),
        .x_in(x_in),
        .WQ_in(WQ_in),
        .WK_in(WK_in),
        .WV_in(WV_in),
        .Q_out(Q),
        .K_out(K),
        .V_out(V),
        .out_valid(qkv_out_valid)
    );

    // Instantiate attention_score
    attention_score #(
        .DATA_WIDTH(DATA_WIDTH),
        .L(L),
        .N(N),
        .E(E)
    ) attn_score_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(attn_score_start),
        .done(attn_score_done),
        .Q_in(Q),
        .K_in(K),
        .A_out(A),
        .out_valid(attn_score_out_valid)
    );


    // self_attention_top.sv

    // Add this before the softmax_approx instantiation
    logic [DATA_WIDTH-1:0] A_flat [L*N*L-1:0];
    logic [DATA_WIDTH*L*N*L-1:0] A_packed;

    // Assign A to A_flat (flatten the 2D array)
    always_comb begin
        for (integer i = 0; i < L; i++) begin
            for (integer j = 0; j < L; j++) begin
                A_flat[i*L + j] = A[i*L + j];
            end
        end
    end

    // Assign A to A_packed (flatten unpacked to packed)
    always_comb begin
        for (integer i = 0; i < L*L; i++) begin
            A_packed[i*DATA_WIDTH +: DATA_WIDTH] = A[i];
        end
    end

    // Modify the softmax_approx instantiation around line 129
        softmax_approx #(
            .DATA_WIDTH(DATA_WIDTH),
            .L(L),
            .N(N)
        ) softmax_inst (
            .clk(clk),
            .rst_n(rst_n),
            .start(softmax_start),
            .done(softmax_done),
            .A_in(A_flat),      // Use flattened array
            .A_out(A_flat),     // Output back to flattened array
            .out_valid(softmax_out_valid)
        );

    // Reassign A_flat back to A if needed (since A_out overwrites A)
    always_comb begin
        for (integer i = 0; i < L; i++) begin
            for (integer j = 0; j < L; j++) begin
                A[i*L + j] = A_flat[i*L + j];
            end
        end
    end


    // Instantiate precision_assigner
    precision_assigner #(
        .DATA_WIDTH(DATA_WIDTH),
        .L(L),
        .N(N)
    ) prec_assign_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(prec_assign_start),
        .done(prec_assign_done),
        .A_in(A_packed),
        .token_precision(token_precision)
    );

    // Instantiate softmax_approx
    // softmax_approx #(
    //     .DATA_WIDTH(DATA_WIDTH),
    //     .L(L),
    //     .N(N)
    // ) softmax_inst (
    //     .clk(clk),
    //     .rst_n(rst_n),
    //     .start(softmax_start),
    //     .done(softmax_done),
    //     .A_in(A),
    //     .A_out(A_softmax),
    //     .out_valid(softmax_out_valid)
    // );

    // Instantiate attention_av_multiply
    attention_av_multiply #(
        .A_ROWS(L),
        .V_COLS(E),
        .NUM_COLS(L),
        .TILE_SIZE(8),
        .WIDTH_INT4(4),
        .WIDTH_INT8(8),
        .WIDTH_FP16(16)
    ) av_multiply_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(av_multiply_start),
        .precision_sel(token_precision),
        .a_mem(A),
        .v_mem(V),
        .done(av_multiply_done),
        .out_mem(AV_out)
    );

    // Instantiate matmul_array for W_O multiplication
    matmul_array #(
        .M(L),
        .K(E),
        .N(E)
    ) wo_matmul_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(wo_multiply_start),
        .a_in(wo_matmul_a_in),
        .b_in(wo_matmul_b_in),
        .c_out(wo_matmul_c_out),
        .done(wo_multiply_done)
    );

    // Start pulse generation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            qkv_start_pulse <= 1'b0;
            attn_score_start_pulse <= 1'b0;
            prec_assign_start_pulse <= 1'b0;
            softmax_start_pulse <= 1'b0;
            av_multiply_start_pulse <= 1'b0;
            wo_multiply_start_pulse <= 1'b0;
        end else begin
            // Assert pulses for one cycle at state transitions
            qkv_start_pulse <= (curr_state == S_IDLE && start);
            attn_score_start_pulse <= (curr_state == S_QKV && qkv_done);
            prec_assign_start_pulse <= (curr_state == S_ATTENTION_SCORE && attn_score_done);
            softmax_start_pulse <= (curr_state == S_PRECISION_ASSIGN && prec_assign_done);
            av_multiply_start_pulse <= (curr_state == S_SOFTMAX && softmax_done);
            wo_multiply_start_pulse <= (curr_state == S_AV_MULTIPLY && av_multiply_done);
        end
    end

    // Assign start signals
    assign qkv_start = qkv_start_pulse;
    assign attn_score_start = attn_score_start_pulse;
    assign prec_assign_start = prec_assign_start_pulse;
    assign softmax_start = softmax_start_pulse;
    assign av_multiply_start = av_multiply_start_pulse;
    assign wo_multiply_start = wo_multiply_start_pulse;

    // State machine: Sequential logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            curr_state <= S_IDLE;
        end else begin
            curr_state <= next_state;
        end
    end

    // State machine: Next state logic
    always_comb begin
        next_state = curr_state;
        done = 1'b0;
        out_valid = 1'b0;

        case (curr_state)
            S_IDLE: begin
                if (start)
                    next_state = S_QKV;
            end
            S_QKV: begin
                if (qkv_done)
                    next_state = S_ATTENTION_SCORE;
            end
            S_ATTENTION_SCORE: begin
                if (attn_score_done)
                    next_state = S_SOFTMAX;
            end
            S_SOFTMAX: begin
                if (softmax_done)
                    next_state = S_PRECISION_ASSIGN;
            end
            S_PRECISION_ASSIGN: begin
                if (prec_assign_done)
                    next_state = S_AV_MULTIPLY;
            end           
            S_AV_MULTIPLY: begin
                if (av_multiply_done)
                    next_state = S_WO_MULTIPLY;
            end
            S_WO_MULTIPLY: begin
                if (wo_multiply_done)
                    next_state = S_DONE;
            end
            S_DONE: begin
                done = 1'b1;
                out_valid = 1'b1;
                next_state = S_IDLE;
            end
            default: next_state = S_IDLE;
        endcase
    end

    // Data path for W_O multiplication
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (integer i = 0; i < L*E; i++) begin
                wo_matmul_a_in[i] <= '0;
                out[i] <= '0;
            end
            for (integer i = 0; i < E*E; i++) begin
                wo_matmul_b_in[i] <= '0;
            end
        end else begin
            case (curr_state)
                S_AV_MULTIPLY: begin
                    if (av_multiply_done) begin
                        // Load AV_out and W_O for matmul
                        for (integer i = 0; i < L*E; i++)
                            wo_matmul_a_in[i] <= AV_out[i];
                        for (integer i = 0; i < E*E; i++)
                            wo_matmul_b_in[i] <= WO_in[i];
                    end
                end
                S_WO_MULTIPLY: begin
                    if (wo_multiply_done) begin
                        // Convert Q2.30 to Q1.15 with saturation
                        for (integer i = 0; i < L*E; i++) begin
                            logic sign_bit = wo_matmul_c_out[i][31];
                            logic int_bit = wo_matmul_c_out[i][30];
                            if (sign_bit == int_bit) begin
                                // In range: take sign bit and top 15 fractional bits
                                out[i] <= {sign_bit, wo_matmul_c_out[i][29:15]};
                            end else begin
                                // Out of range: saturate based on sign
                                out[i] <= sign_bit ? 16'h8000 : 16'h7FFF;
                            end
                        end
                    end
                end
                S_DONE: begin
                    // Outputs already assigned
                end
                default: ;
            endcase
        end
    end

endmodule

