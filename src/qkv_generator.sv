module qkv_projection #(
    parameter int DATA_WIDTH = 16,  // Data width for inputs (FP16 for now)
    parameter int L = 8,           // Sequence length
    parameter int N = 1,           // Batch size
    parameter int E = 8            // Embedding dimension
)(
    input  logic                           clk,
    input  logic                           rst_n,
    // Control
    input  logic                           start,
    output logic                           done,

    // Input x: shape (L*N, E), flattened
    input  logic [DATA_WIDTH*L*N*E-1:0]    x_in,

    // Weights WQ/WK/WV: shape (E, E)
    input  logic [DATA_WIDTH*E*E-1:0]      WQ_in,
    input  logic [DATA_WIDTH*E*E-1:0]      WK_in,
    input  logic [DATA_WIDTH*E*E-1:0]      WV_in,

    // Outputs Q, K, V: each (L*N, E), flattened
    output logic [DATA_WIDTH*L*N*E-1:0]    Q_out,
    output logic [DATA_WIDTH*L*N*E-1:0]    K_out,
    output logic [DATA_WIDTH*L*N*E-1:0]    V_out,
    output logic                           out_valid
);

    // Local parameters
    localparam int TOTAL_TOKENS = L*N;
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
    logic [DATA_WIDTH-1:0] x_mem [0:TOTAL_TOKENS-1][0:E-1];  // Input matrix
    logic [DATA_WIDTH-1:0] WQ [0:E-1][0:E-1];                // Weight matrices
    logic [DATA_WIDTH-1:0] WK [0:E-1][0:E-1];
    logic [DATA_WIDTH-1:0] WV [0:E-1][0:E-1];
    logic [DATA_WIDTH-1:0] Q_mem [0:TOTAL_TOKENS-1][0:E-1];  // Output storage
    logic [DATA_WIDTH-1:0] K_mem [0:TOTAL_TOKENS-1][0:E-1];
    logic [DATA_WIDTH-1:0] V_mem [0:TOTAL_TOKENS-1][0:E-1];

    // matmul_array control signals
    logic matmul_start;
    logic matmul_done;
    logic [DATA_WIDTH*TOTAL_TOKENS*E-1:0] matmul_a_in;  // Input matrix X
    logic [DATA_WIDTH*E*E-1:0] matmul_b_in;            // Weight matrix (WQ, WK, or WV)
    logic [OUT_DATA_WIDTH*TOTAL_TOKENS*E-1:0] matmul_c_out;  // Output matrix

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

    // Unpack inputs combinationally
    always_comb begin
        integer i, j;
        for (i = 0; i < E; i++) begin
            for (j = 0; j < E; j++) begin
                WQ[i][j] = WQ_in[((i*E)+j+1)*DATA_WIDTH-1 -: DATA_WIDTH];
                WK[i][j] = WK_in[((i*E)+j+1)*DATA_WIDTH-1 -: DATA_WIDTH];
                WV[i][j] = WV_in[((i*E)+j+1)*DATA_WIDTH-1 -: DATA_WIDTH];
            end
        end
    end

    // State register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            curr_state <= S_IDLE;
        else
            curr_state <= next_state;
    end

    // Next-state logic and control signals
    always_comb begin
        next_state = curr_state;
        matmul_start = 1'b0;

        case (curr_state)
            S_IDLE: begin
                if (start)
                    next_state = S_LOAD;
            end
            S_LOAD: begin
                next_state = S_COMPUTE_Q;
            end
            S_COMPUTE_Q: begin
                matmul_start = (curr_state != next_state);  // Start on entering state
                if (matmul_done)
                    next_state = S_COMPUTE_K;
            end
            S_COMPUTE_K: begin
                matmul_start = (curr_state != next_state);
                if (matmul_done)
                    next_state = S_COMPUTE_V;
            end
            S_COMPUTE_V: begin
                matmul_start = (curr_state != next_state);
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
            for (integer i = 0; i < TOTAL_TOKENS; i++)
                for (integer j = 0; j < E; j++) begin
                    x_mem[i][j] <= '0;
                    Q_mem[i][j] <= '0;
                    K_mem[i][j] <= '0;
                    V_mem[i][j] <= '0;
                end
            matmul_a_in <= '0;
            matmul_b_in <= '0;
        end else begin
            case (curr_state)
                S_LOAD: begin
                    // Unpack x_in into x_mem
                    for (integer tok = 0; tok < TOTAL_TOKENS; tok++)
                        for (integer d = 0; d < E; d++)
                            x_mem[tok][d] <= x_in[((tok*E)+d+1)*DATA_WIDTH-1 -: DATA_WIDTH];
                    // Prepare matmul_a_in (X) for all computations
                    for (integer tok = 0; tok < TOTAL_TOKENS; tok++)
                        for (integer d = 0; d < E; d++)
                            matmul_a_in[((tok*E)+d+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= x_in[((tok*E)+d+1)*DATA_WIDTH-1 -: DATA_WIDTH];
                end
                S_COMPUTE_Q: begin
                    // Set matmul_b_in to WQ
                    for (integer i = 0; i < E; i++)
                        for (integer j = 0; j < E; j++)
                            matmul_b_in[((i*E)+j+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= WQ[i][j];
                    // Store matmul output directly to Q_mem (truncate to 16-bit)
                    if (matmul_done)
                        for (integer tok = 0; tok < TOTAL_TOKENS; tok++)
                            for (integer d = 0; d < E; d++)
                                Q_mem[tok][d] <= matmul_c_out[((tok*E)+d+1)*OUT_DATA_WIDTH-1 -: OUT_DATA_WIDTH][DATA_WIDTH-1:0];
                end
                S_COMPUTE_K: begin
                    // Set matmul_b_in to WK
                    for (integer i = 0; i < E; i++)
                        for (integer j = 0; j < E; j++)
                            matmul_b_in[((i*E)+j+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= WK[i][j];
                    if (matmul_done)
                        for (integer tok = 0; tok < TOTAL_TOKENS; tok++)
                            for (integer d = 0; d < E; d++)
                                K_mem[tok][d] <= matmul_c_out[((tok*E)+d+1)*OUT_DATA_WIDTH-1 -: OUT_DATA_WIDTH][DATA_WIDTH-1:0];
                end
                S_COMPUTE_V: begin
                    // Set matmul_b_in to WV
                    for (integer i = 0; i < E; i++)
                        for (integer j = 0; j < E; j++)
                            matmul_b_in[((i*E)+j+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= WV[i][j];
                    if (matmul_done)
                        for (integer tok = 0; tok < TOTAL_TOKENS; tok++)
                            for (integer d = 0; d < E; d++)
                                V_mem[tok][d] <= matmul_c_out[((tok*E)+d+1)*OUT_DATA_WIDTH-1 -: OUT_DATA_WIDTH][DATA_WIDTH-1:0];
                end
                S_DONE: begin
                    // Pack outputs
                    for (integer tok = 0; tok < TOTAL_TOKENS; tok++)
                        for (integer d = 0; d < E; d++) begin
                            Q_out[((tok*E)+d+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= Q_mem[tok][d];
                            K_out[((tok*E)+d+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= K_mem[tok][d];
                            V_out[((tok*E)+d+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= V_mem[tok][d];
                        end
                end
            endcase
        end
    end

    // Output control signals
    always_comb begin
        done = (curr_state == S_DONE);
        out_valid = (curr_state == S_DONE);
    end

endmodule
