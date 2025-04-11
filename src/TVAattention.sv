module self_attention_with_internal_qkv #(
    parameter DATA_WIDTH = 16,  // e.g. 16-bit for FP16 or fixed
    parameter L = 8,            // Sequence length
    parameter N = 1,            // Batch size
    parameter E = 8             // Embedding dimension
)(
    input  wire                                  clk,
    input  wire                                  rst_n,
    input  wire                                  start,
    output reg                                   done,

    // Input: x of shape (L, N, E), flattened
    input  wire [DATA_WIDTH*L*N*E-1:0]           x_in,

    // Output: final attention output, same shape (L, N, E)
    output reg  [DATA_WIDTH*L*N*E-1:0]           out_attention,
    output reg                                   out_valid
);

    //--------------------------------------------------------------------------
    // 1) Weights and Biases for Q, K, V
    //--------------------------------------------------------------------------
    // In a PyTorch-like definition for a single head:
    //   W_Q, W_K, W_V: each is E x E
    //   b_Q, b_K, b_V: each is E
    // For demonstration, we store these in reg arrays. You might actually
    // load them from a file or keep them in BRAM/ROM. 

    reg [DATA_WIDTH-1:0] WQ [0:E-1][0:E-1];
    reg [DATA_WIDTH-1:0] WK [0:E-1][0:E-1];
    reg [DATA_WIDTH-1:0] WV [0:E-1][0:E-1];

    // Optional biases
    reg [DATA_WIDTH-1:0] bQ [0:E-1];
    reg [DATA_WIDTH-1:0] bK [0:E-1];
    reg [DATA_WIDTH-1:0] bV [0:E-1];

    // Example initialization (or load from file). 
    // This is just an illustrative block, *not* a real design.
    integer i, j;
    initial begin
        for (i = 0; i < E; i = i + 1) begin
            for (j = 0; j < E; j = j + 1) begin
                WQ[i][j] = {DATA_WIDTH{1'b0}};
                WK[i][j] = {DATA_WIDTH{1'b0}};
                WV[i][j] = {DATA_WIDTH{1'b0}};
            end
            bQ[i] = {DATA_WIDTH{1'b0}};
            bK[i] = {DATA_WIDTH{1'b0}};
            bV[i] = {DATA_WIDTH{1'b0}};
        end
    end

    //--------------------------------------------------------------------------
    // 2) Store input x in a 3D array: x[L, N, E]
    //--------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] x_arr [0:L-1][0:N-1][0:E-1];
    integer l, n_, e_;
    // For the final output
    reg [DATA_WIDTH-1:0] Z [0:L-1][0:N-1][0:E-1];

    //--------------------------------------------------------------------------
    // 3) Q, K, V embeddings: same shape (L, N, E)
    //--------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] Q [0:L-1][0:N-1][0:E-1];
    reg [DATA_WIDTH-1:0] K [0:L-1][0:N-1][0:E-1];
    reg [DATA_WIDTH-1:0] V [0:L-1][0:N-1][0:E-1];

    //--------------------------------------------------------------------------
    // 4) Attention scores A: shape (L, N, L)
    //    For self-attention, we typically do (QK^T) across the same sequence length L.
    //    A[l, n, l2] = dot(Q[l, n], K[l2, n]) / sqrt(E)
    //--------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] A [0:L-1][0:N-1][0:L-1];

    //--------------------------------------------------------------------------
    // 5) Token-wise precision code for each "key token" l2
    //    We'll treat each position l2 as a token for precision assignment
    //--------------------------------------------------------------------------
    reg [3:0] token_precision [0:L-1];

    //--------------------------------------------------------------------------
    // FSM
    //--------------------------------------------------------------------------
    localparam ST_IDLE      = 4'd0;
    localparam ST_LOAD      = 4'd1;
    localparam ST_GENQ      = 4'd2;
    localparam ST_GENK      = 4'd3;
    localparam ST_GENV      = 4'd4;
    localparam ST_QK        = 4'd5;
    localparam ST_SOFTMAX   = 4'd6;
    localparam ST_PRECISION = 4'd7;
    localparam ST_MULV      = 4'd8;
    localparam ST_DONE      = 4'd9;

    reg [3:0] state;

    // Indices used within the state machine
    integer l2; 

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= ST_IDLE;
            done        <= 1'b0;
            out_valid   <= 1'b0;
        end else begin
            case(state)
                // Wait for 'start'
                ST_IDLE: begin
                    done      <= 1'b0;
                    out_valid <= 1'b0;
                    if (start) state <= ST_LOAD;
                end

                //------------------------------------------------------------
                // ST_LOAD: parse input x_in => x_arr[L,N,E]
                //------------------------------------------------------------
                ST_LOAD: begin
                    for (l = 0; l < L; l = l + 1) begin
                        for (n_ = 0; n_ < N; n_ = n_ + 1) begin
                            for (e_ = 0; e_ < E; e_ = e_ + 1) begin
                                x_arr[l][n_][e_] 
                                    = x_in[ ((l*N*E)+(n_*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH ];
                            end
                        end
                    end
                    state <= ST_GENQ;
                end

                //------------------------------------------------------------
                // ST_GENQ: Q = x_arr * WQ + bQ
                // For each token (l, n) do a dot-product with each row i of WQ
                //------------------------------------------------------------
                ST_GENQ: begin
                    for (l = 0; l < L; l = l + 1) begin
                        for (n_ = 0; n_ < N; n_ = n_ + 1) begin
                            for (i = 0; i < E; i = i + 1) begin
                                reg [DATA_WIDTH-1:0] sum_temp;
                                sum_temp = bQ[i]; // start with bias
                                for (e_ = 0; e_ < E; e_ = e_ + 1) begin
                                    // sum_temp += x_arr[l][n_][e_] * WQ[i][e_]
                                    sum_temp = sum_temp + (x_arr[l][n_][e_] * WQ[i][e_]);
                                end
                                Q[l][n_][i] = sum_temp;
                            end
                        end
                    end
                    state <= ST_GENK;
                end

                //------------------------------------------------------------
                // ST_GENK: K = x_arr * WK + bK
                //------------------------------------------------------------
                ST_GENK: begin
                    for (l = 0; l < L; l = l + 1) begin
                        for (n_ = 0; n_ < N; n_ = n_ + 1) begin
                            for (i = 0; i < E; i = i + 1) begin
                                reg [DATA_WIDTH-1:0] sum_temp;
                                sum_temp = bK[i];
                                for (e_ = 0; e_ < E; e_ = e_ + 1) begin
                                    sum_temp = sum_temp + (x_arr[l][n_][e_] * WK[i][e_]);
                                end
                                K[l][n_][i] = sum_temp;
                            end
                        end
                    end
                    state <= ST_GENV;
                end

                //------------------------------------------------------------
                // ST_GENV: V = x_arr * WV + bV
                //------------------------------------------------------------
                ST_GENV: begin
                    for (l = 0; l < L; l = l + 1) begin
                        for (n_ = 0; n_ < N; n_ = n_ + 1) begin
                            for (i = 0; i < E; i = i + 1) begin
                                reg [DATA_WIDTH-1:0] sum_temp;
                                sum_temp = bV[i];
                                for (e_ = 0; e_ < E; e_ = e_ + 1) begin
                                    sum_temp = sum_temp + (x_arr[l][n_][e_] * WV[i][e_]);
                                end
                                V[l][n_][i] = sum_temp;
                            end
                        end
                    end
                    state <= ST_QK;
                end

                //------------------------------------------------------------
                // ST_QK: A[l, n, l2] = dot(Q[l,n], K[l2,n]) / sqrt(E)
                // This is NxN if we consider L tokens -> L x L attention
                //------------------------------------------------------------
                ST_QK: begin
                    for (l = 0; l < L; l = l + 1) begin
                        for (n_ = 0; n_ < N; n_ = n_ + 1) begin
                            for (l2 = 0; l2 < L; l2 = l2 + 1) begin
                                reg [DATA_WIDTH-1:0] sum_temp;
                                sum_temp = {DATA_WIDTH{1'b0}};
                                for (e_ = 0; e_ < E; e_ = e_ + 1) begin
                                    sum_temp = sum_temp + (Q[l][n_][e_] * K[l2][n_][e_]);
                                end
                                // For demonstration, not applying real float scale
                                A[l][n_][l2] = sum_temp; 
                            end
                        end
                    end
                    state <= ST_SOFTMAX;
                end

                //------------------------------------------------------------
                // ST_SOFTMAX: row-wise softmax over l2 for each (l, n)
                //------------------------------------------------------------
                ST_SOFTMAX: begin
                    for (l = 0; l < L; l = l + 1) begin
                        for (n_ = 0; n_ < N; n_ = n_ + 1) begin
                            // sum row
                            reg [DATA_WIDTH-1:0] row_sum;
                            row_sum = {DATA_WIDTH{1'b0}};
                            for (l2 = 0; l2 < L; l2 = l2 + 1) begin
                                row_sum = row_sum + A[l][n_][l2];
                            end
                            // divide
                            for (l2 = 0; l2 < L; l2 = l2 + 1) begin
                                // A[l][n_][l2] = A[l][n_][l2] / row_sum
                                A[l][n_][l2] = A[l][n_][l2]; // placeholder
                            end
                        end
                    end
                    state <= ST_PRECISION;
                end

                //------------------------------------------------------------
                // ST_PRECISION: Decide per-token precision (one code per l2)
                // We'll sum usage across (l, n) for each l2. 
                //------------------------------------------------------------
                ST_PRECISION: begin
                    for (l2 = 0; l2 < L; l2 = l2 + 1) begin
                        reg [DATA_WIDTH-1:0] col_sum;
                        col_sum = {DATA_WIDTH{1'b0}};
                        for (l = 0; l < L; l = l + 1) begin
                            for (n_ = 0; n_ < N; n_ = n_ + 1) begin
                                col_sum = col_sum + A[l][n_][l2];
                            end
                        end
                        // Simple threshold logic
                        if (col_sum < 100)
                            token_precision[l2] = 4'd0; // INT4
                        else if (col_sum < 200)
                            token_precision[l2] = 4'd1; // INT8
                        else
                            token_precision[l2] = 4'd2; // FP16
                    end
                    state <= ST_MULV;
                end

                //------------------------------------------------------------
                // ST_MULV: Z[l, n, e] = sum_{l2} A[l,n,l2] * downcast(V[l2,n,e])
                //------------------------------------------------------------
                ST_MULV: begin
                    // Initialize Z
                    for (l = 0; l < L; l = l + 1) begin
                        for (n_ = 0; n_ < N; n_ = n_ + 1) begin
                            for (e_ = 0; e_ < E; e_ = e_ + 1) begin
                                Z[l][n_][e_] = {DATA_WIDTH{1'b0}};
                            end
                        end
                    end

                    for (l = 0; l < L; l = l + 1) begin
                        for (n_ = 0; n_ < N; n_ = n_ + 1) begin
                            for (l2 = 0; l2 < L; l2 = l2 + 1) begin
                                // For each e in embedding
                                for (e_ = 0; e_ < E; e_ = e_ + 1) begin
                                    reg [DATA_WIDTH-1:0] valV_downcast;
                                    reg [DATA_WIDTH-1:0] product;

                                    // Downcast V[l2][n_][e_] by token_precision[l2]
                                    case(token_precision[l2])
                                        4'd0: begin
                                            valV_downcast = {
                                                {(DATA_WIDTH-4){1'b0}},
                                                V[l2][n_][e_][3:0]
                                            };
                                        end
                                        4'd1: begin
                                            valV_downcast = {
                                                {(DATA_WIDTH-8){1'b0}},
                                                V[l2][n_][e_][7:0]
                                            };
                                        end
                                        4'd2: begin
                                            valV_downcast = V[l2][n_][e_];
                                        end
                                        default: begin
                                            valV_downcast = V[l2][n_][e_];
                                        end
                                    endcase

                                    // Multiply A[l][n_][l2] * valV_downcast
                                    product = A[l][n_][l2] * valV_downcast;
                                    // Accumulate
                                    Z[l][n_][e_] = Z[l][n_][e_] + product;
                                end
                            end
                        end
                    end

                    state <= ST_DONE;
                end

                //------------------------------------------------------------
                // ST_DONE: pack Z => out_attention (L, N, E)
                //------------------------------------------------------------
                ST_DONE: begin
                    for (l = 0; l < L; l = l + 1) begin
                        for (n_ = 0; n_ < N; n_ = n_ + 1) begin
                            for (e_ = 0; e_ < E; e_ = e_ + 1) begin
                                out_attention[
                                    ((l*N*E)+(n_*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH
                                ] = Z[l][n_][e_];
                            end
                        end
                    end
                    out_valid <= 1'b1;
                    done      <= 1'b1;
                    state     <= ST_IDLE; // or remain ST_DONE
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
