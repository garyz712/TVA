module qkv_projection #(
    parameter int DATA_WIDTH = 16,
    parameter int L = 8,   // sequence length
    parameter int N = 1,   // batch size
    parameter int E = 8    // embedding dimension
)(
    input  logic                           clk,
    input  logic                           rst_n,
    // Control
    input  logic                           start,
    output logic                           done,

    // Input x: shape (L, N, E), flattened
    input  logic [DATA_WIDTH*L*N*E-1:0]    x_in,

    // Weights WQ/WK/WV => (E x E), biases => (E)
    input  logic [DATA_WIDTH*E*E-1:0]      WQ_in,
    input  logic [DATA_WIDTH*E*E-1:0]      WK_in,
    input  logic [DATA_WIDTH*E*E-1:0]      WV_in,
    input  logic [DATA_WIDTH*E-1:0]        bQ_in,
    input  logic [DATA_WIDTH*E-1:0]        bK_in,
    input  logic [DATA_WIDTH*E-1:0]        bV_in,

    // Outputs Q, K, V => each (L, N, E), flattened
    output logic [DATA_WIDTH*L*N*E-1:0]    Q_out,
    output logic [DATA_WIDTH*L*N*E-1:0]    K_out,
    output logic [DATA_WIDTH*L*N*E-1:0]    V_out,
    output logic                           out_valid
);

    // ----------------------------------------------------------------
    // 0) Local parameters
    // ----------------------------------------------------------------
    localparam int TOTAL_TOKENS = L*N;
    // We'll compute Q,K,V for each of the (L*N) tokens, each dimension E

    // ----------------------------------------------------------------
    // 1) State definitions
    // ----------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_LOAD,
        S_COMPUTE_Q,
        S_COMPUTE_K,
        S_COMPUTE_V,
        S_DONE
    } state_t;

    state_t curr_state, next_state;

    // ----------------------------------------------------------------
    // 2) Internal storage
    // ----------------------------------------------------------------
    // Flattened input x => local array x_mem[i][e]
    // i = token index from 0..(L*N-1)
    logic [DATA_WIDTH-1:0] x_mem [0:TOTAL_TOKENS-1][0:E-1];

    // We'll store partial results for Q, K, V likewise
    logic [DATA_WIDTH-1:0] Q_mem [0:TOTAL_TOKENS-1][0:E-1];
    logic [DATA_WIDTH-1:0] K_mem [0:TOTAL_TOKENS-1][0:E-1];
    logic [DATA_WIDTH-1:0] V_mem [0:TOTAL_TOKENS-1][0:E-1];

    // We also store weight arrays in [row][col] format
    logic [DATA_WIDTH-1:0] WQ [0:E-1][0:E-1];
    logic [DATA_WIDTH-1:0] WK [0:E-1][0:E-1];
    logic [DATA_WIDTH-1:0] WV [0:E-1][0:E-1];
    logic [DATA_WIDTH-1:0] bQ [0:E-1];
    logic [DATA_WIDTH-1:0] bK [0:E-1];
    logic [DATA_WIDTH-1:0] bV [0:E-1];

    // We'll have a counter to iterate over each token i
    logic [$clog2(TOTAL_TOKENS):0] idx;
    // Another counter for dimension e
    logic [$clog2(E):0] dim;

    // done for each stage
    logic stage_done;

    // ----------------------------------------------------------------
    // 3) State register
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            curr_state <= S_IDLE;
        else
            curr_state <= next_state;
    end

    // ----------------------------------------------------------------
    // 4) Next-State Logic + Control Signals
    // ----------------------------------------------------------------
    always_comb begin
        // defaults
        next_state   = curr_state;
        stage_done   = 1'b0;

        case (curr_state)

            S_IDLE: begin
                if (start)
                    next_state = S_LOAD;
            end

            S_LOAD: begin
                // once we finish loading, go compute Q
                stage_done  = 1'b1;
                next_state  = S_COMPUTE_Q;
            end

            S_COMPUTE_Q: begin
                // We'll time-multiplex the Q computation over tokens
                if (idx == (TOTAL_TOKENS - 1) && dim == (E - 1)) begin
                    stage_done  = 1'b1;
                    next_state  = S_COMPUTE_K;
                end
            end

            S_COMPUTE_K: begin
                if (idx == (TOTAL_TOKENS - 1) && dim == (E - 1)) begin
                    stage_done  = 1'b1;
                    next_state  = S_COMPUTE_V;
                end
            end

            S_COMPUTE_V: begin
                if (idx == (TOTAL_TOKENS - 1) && dim == (E - 1)) begin
                    stage_done  = 1'b1;
                    next_state  = S_DONE;
                end
            end

            S_DONE: begin
                // stay or go back to idle
                next_state = S_IDLE;
            end

            default: next_state = S_IDLE;

        endcase
    end

    // ----------------------------------------------------------------
    // 5) Output Logic + Internal Datapath (always_ff or always_comb?)
    //    We'll do a common always_ff with non-blocking for regs
    //    and a small always_comb for immediate signals.
    // ----------------------------------------------------------------
    // We'll do a simple approach: use an always_ff for sequential regs,
    // and compute partial sums with blocking (in a small local logic).
    // For real design, you'd pipeline or use a dedicated MAC sub-block.
    //

    // 5a) Weights are unpacked combinationally
    integer i, j;
    always_comb begin
        for (i=0; i<E; i++) begin
            bQ[i] = bQ_in[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH];
            bK[i] = bK_in[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH];
            bV[i] = bV_in[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH];
            for (j=0; j<E; j++) begin
                WQ[i][j] = WQ_in[((i*E)+j+1)*DATA_WIDTH-1 -: DATA_WIDTH];
                WK[i][j] = WK_in[((i*E)+j+1)*DATA_WIDTH-1 -: DATA_WIDTH];
                WV[i][j] = WV_in[((i*E)+j+1)*DATA_WIDTH-1 -: DATA_WIDTH];
            end
        end
    end

    // 5b) Provide out_valid, done
    always_comb begin
        // defaults
        done      = 1'b0;
        out_valid = 1'b0;

        if (curr_state == S_DONE) begin
            done      = 1'b1;
            out_valid = 1'b1;
        end
    end

    // 5c) Actually do the data moves in always_ff
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            idx <= '0;
            dim <= '0;
        end
        else begin
            case(curr_state)

                // Load x_in into x_mem
                S_LOAD: begin
                    // We do it in one shot for simplicity
                    // (Alternatively, you could time-multiplex here too.)
                    integer tok, d;
                    for(tok=0; tok < TOTAL_TOKENS; tok++) begin
                        for(d=0; d<E; d++) begin
                            // Flattened index
                            x_mem[tok][d] = x_in[ ((tok*E) + d +1)*DATA_WIDTH -1 -: DATA_WIDTH ];
                        end
                    end
                    idx <= 0;
                    dim <= 0;
                end

                // Compute Q => For each token idx, dimension dim
                S_COMPUTE_Q: begin
                    // We'll do partial sums across e
                    // sum_temp = bQ[dim] + sum(x_mem[idx][k] * WQ[dim][k], k=0..E-1)
                    // For demonstration, do a single multiply each cycle:
                    // We'll accumulate in an internal register:
                    static logic [DATA_WIDTH-1:0] sum_temp;
                    static logic [DATA_WIDTH-1:0] partial_index;
                    // partial_index is the "k" dimension in the dot product

                    if (dim == 0 && idx == 0) begin
                        // reset partial accumulators
                        sum_temp     = bQ[0];
                        partial_index= 0;
                    end

                    // multiply accum
                    sum_temp = sum_temp + (x_mem[idx][partial_index] * WQ[dim][partial_index]);

                    // increment partial_index
                    if (partial_index == (E-1)) begin
                        // store sum_temp to Q_mem
                        Q_mem[idx][dim] = sum_temp;

                        // Move to next dim or next idx
                        sum_temp = bQ[dim]; // reinit for next dimension
                        partial_index = 0;

                        if (dim < E-1)
                            dim <= dim + 1;
                        else begin
                            dim <= 0;
                            if (idx < (TOTAL_TOKENS-1))
                                idx <= idx + 1;
                        end

                    end else begin
                        partial_index = partial_index + 1;
                    end
                end

                // Compute K similarly
                S_COMPUTE_K: begin
                    static logic [DATA_WIDTH-1:0] sum_temp;
                    static logic [DATA_WIDTH-1:0] partial_index;
                    if (dim == 0 && idx == 0) begin
                        sum_temp     = bK[0];
                        partial_index= 0;
                    end

                    sum_temp = sum_temp + (x_mem[idx][partial_index] * WK[dim][partial_index]);

                    if (partial_index == (E-1)) begin
                        K_mem[idx][dim] = sum_temp;
                        sum_temp = bK[dim];
                        partial_index = 0;

                        if (dim < E-1)
                            dim <= dim + 1;
                        else begin
                            dim <= 0;
                            if (idx < (TOTAL_TOKENS-1))
                                idx <= idx + 1;
                        end
                    end else begin
                        partial_index = partial_index + 1;
                    end
                end

                // Compute V
                S_COMPUTE_V: begin
                    static logic [DATA_WIDTH-1:0] sum_temp;
                    static logic [DATA_WIDTH-1:0] partial_index;
                    if (dim == 0 && idx == 0) begin
                        sum_temp     = bV[0];
                        partial_index= 0;
                    end

                    sum_temp = sum_temp + (x_mem[idx][partial_index] * WV[dim][partial_index]);

                    if (partial_index == (E-1)) begin
                        V_mem[idx][dim] = sum_temp;
                        sum_temp = bV[dim];
                        partial_index = 0;

                        if (dim < E-1)
                            dim <= dim + 1;
                        else begin
                            dim <= 0;
                            if (idx < (TOTAL_TOKENS-1))
                                idx <= idx + 1;
                        end
                    end else begin
                        partial_index = partial_index + 1;
                    end
                end

                // S_DONE: pack Q_mem/K_mem/V_mem into Q_out/K_out/V_out
                default: begin
                    if (curr_state == S_DONE) begin
                        integer tok, d;
                        for(tok=0; tok< TOTAL_TOKENS; tok++) begin
                            for(d=0; d<E; d++) begin
                                Q_out[ ((tok*E) + d +1)*DATA_WIDTH -1 -: DATA_WIDTH ] = Q_mem[tok][d];
                                K_out[ ((tok*E) + d +1)*DATA_WIDTH -1 -: DATA_WIDTH ] = K_mem[tok][d];
                                V_out[ ((tok*E) + d +1)*DATA_WIDTH -1 -: DATA_WIDTH ] = V_mem[tok][d];
                            end
                        end
                    end
                end
            endcase
        end
    end

endmodule


