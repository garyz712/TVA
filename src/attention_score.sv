//------------------------------------------------------------------------------
// attention_score.sv
//
// This module computes the attention scores in a transformer model's self-attention
// mechanism using the matmul_array module for matrix multiplication.
// Computes A = Q * K^T where Q, K are (L, N, E), and A is (L, N, L).
//
// Apr 10 2025    Max Zhang      initial version
// Apr 12 2025    Tianwei Liu    refactor into SV style
// Apr 12 2025    Tianwei Liu    split state machine logic
// Apr 13 2025    Tianwei Liu    add comments
// Apr 28 2025    Tianwei Liu    fix dot product computation bug
// May 20 2025    Tianwei Liu    refactor to use matmul_array
//------------------------------------------------------------------------------

module attention_score #(
    parameter int DATA_WIDTH = 16,
    parameter int L = 8,
    parameter int N = 1,
    parameter int E = 8
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        start,
    output logic                        done,
    input  logic [DATA_WIDTH-1:0] Q_in [0:L*N*E-1],
    input  logic [DATA_WIDTH-1:0] K_in [0:L*N*E-1],
    output logic [DATA_WIDTH-1:0] A_out [0:L*N*L-1],
    output logic                        out_valid
);

    // Local parameters
    localparam int MATMUL_OUT_WIDTH = 32;  // Output width of matmul (2*DATA_WIDTH)
    
    // State machine states
    typedef enum logic [1:0] {
        IDLE = 2'd0,
        MATMUL = 2'd1,
        DONE = 2'd2
    } state_t;
    
    state_t state, next_state;
    
    // Internal signals
    logic [MATMUL_OUT_WIDTH-1:0] matmul_out [0:L*N*L-1];
    logic [DATA_WIDTH-1:0] K_transpose [0:L*N*E-1];
    logic matmul_start, matmul_done;
    
    // Transpose K matrix (L,N,E) -> (E,N,L)
    always_comb begin
        for (int i = 0; i < L; i++) begin
            for (int j = 0; j < N; j++) begin
                for (int k = 0; k < E; k++) begin
                    K_transpose[k*N*L + j*L + i] = K_in[i*N*E + j*E + k];
                end
            end
        end
    end
    
    // Matrix multiplication instance
    matmul_array #(
        .M(L*N),
        .K(E),
        .N(L)
    ) matmul_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(matmul_start),
        .a_in(Q_in),
        .b_in(K_transpose),
        .c_out(matmul_out),
        .done(matmul_done)
    );
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = state;
        matmul_start = 1'b0;
        done = 1'b0;
        out_valid = 1'b0;
        
        case (state)
            IDLE: begin
                if (start) begin
                    next_state = MATMUL;
                    matmul_start = 1'b1;
                end
            end
            MATMUL: begin
                if (matmul_done) begin
                    next_state = DONE;
                end
            end
            DONE: begin
                done = 1'b1;
                out_valid = 1'b1;
                next_state = IDLE;
            end
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Output assignment (truncate to DATA_WIDTH)
    always_ff @(posedge clk) begin
        if (state == MATMUL && matmul_done) begin
            for (int i = 0; i < L*N*L; i++) begin
                logic sign_bit = matmul_out[i][31];
                logic int_bit = matmul_out[i][30];
                if (sign_bit == int_bit) begin
                    // In range: take sign bit and top 15 fractional bits
                    A_out[i] <= {sign_bit, matmul_out[i][29:15]};
                end else begin
                    // Out of range: saturate based on sign
                    A_out[i] <= sign_bit ? 16'h8000 : 16'h7FFF;
                end
            end
        end
    end

endmodule