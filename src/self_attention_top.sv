module self_attention #(
    parameter int DATA_WIDTH = 16,
    parameter int SEQ_LEN    = 8,   // number of tokens
    parameter int EMB_DIM    = 8    // embedding dimension
)(
    input  logic                                      clk,
    input  logic                                      rst_n,

    // Control
    input  logic                                      start,
    output logic                                      done,

    // Input: x_in, shape (SEQ_LEN, EMB_DIM), flattened
    input  logic [DATA_WIDTH*SEQ_LEN*EMB_DIM -1:0]    x_in,

    // Q, K, V weights => each is (EMB_DIM x EMB_DIM), plus bias => (EMB_DIM)
    input  logic [DATA_WIDTH*EMB_DIM*EMB_DIM -1:0]    WQ_in,
    input  logic [DATA_WIDTH*EMB_DIM -1:0]            bQ_in,
    input  logic [DATA_WIDTH*EMB_DIM*EMB_DIM -1:0]    WK_in,
    input  logic [DATA_WIDTH*EMB_DIM -1:0]            bK_in,
    input  logic [DATA_WIDTH*EMB_DIM*EMB_DIM -1:0]    WV_in,
    input  logic [DATA_WIDTH*EMB_DIM -1:0]            bV_in,

    // Output: final attention result, shape (SEQ_LEN, EMB_DIM)
    output logic [DATA_WIDTH*SEQ_LEN*EMB_DIM -1:0]    attn_out,
    output logic                                      out_valid
);

    //***************************************************************
    // 1) States of the top-level FSM
    //***************************************************************
    typedef enum logic [2:0] {
        S_IDLE,
        S_QKV,      // compute Q, K, V
        S_QK,       // compute A = QK^T
        S_SOFTMAX,  // apply row-wise softmax
        S_AXV,      // compute A x V
        S_DONE
    } state_t;

    state_t curr_state, next_state;

    //***************************************************************
    // 2) Internal Memories
    //***************************************************************
    // We'll keep x_in in x_mem[seq][dim].
    // Then we compute Q, K, V => each [seq][dim].
    // We compute attention matrix A => shape (SEQ_LEN, SEQ_LEN).
    // Finally, we produce output => shape (SEQ_LEN, EMB_DIM).

    logic [DATA_WIDTH-1:0] x_mem [0:SEQ_LEN-1][0:EMB_DIM-1];
    logic [DATA_WIDTH-1:0] Q_mem [0:SEQ_LEN-1][0:EMB_DIM-1];
    logic [DATA_WIDTH-1:0] K_mem [0:SEQ_LEN-1][0:EMB_DIM-1];
    logic [DATA_WIDTH-1:0] V_mem [0:SEQ_LEN-1][0:EMB_DIM-1];

    // A => shape (SEQ_LEN, SEQ_LEN), each element is DATA_WIDTH for demonstration.
    logic [DATA_WIDTH-1:0] A_mem [0:SEQ_LEN-1][0:SEQ_LEN-1];

    // Final output => attn_mem[seq][dim]
    logic [DATA_WIDTH-1:0] attn_mem [0:SEQ_LEN-1][0:EMB_DIM-1];

    // We'll store weights in 2D arrays: WQ[row][col], etc.
    // row => EMB_DIM, col => EMB_DIM
    logic [DATA_WIDTH-1:0] WQ [0:EMB_DIM-1][0:EMB_DIM-1];
    logic [DATA_WIDTH-1:0] WK [0:EMB_DIM-1][0:EMB_DIM-1];
    logic [DATA_WIDTH-1:0] WV [0:EMB_DIM-1][0:EMB_DIM-1];

    // biases
    logic [DATA_WIDTH-1:0] bQ [0:EMB_DIM-1];
    logic [DATA_WIDTH-1:0] bK [0:EMB_DIM-1];
    logic [DATA_WIDTH-1:0] bV [0:EMB_DIM-1];

    // partial sums for dot-products, counters, etc.
    logic [$clog2(SEQ_LEN):0]  seq_i, seq_j;
    logic [$clog2(EMB_DIM):0]  dim_i;
    logic [31:0]               sum_temp; // up to 32 bits for partial accumulation

    // For row-wise softmax, we might store row sums in row_sum[] and do naive “divide each element by row_sum”.
    // Real design: exponent-based or approximate. We'll do a naive approach.

    // row_sums for each row => row_sums[seq_i]
    logic [31:0] row_sums [0:SEQ_LEN-1];

    //***************************************************************
    // 3) State Register
    //***************************************************************
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n)
            curr_state <= S_IDLE;
        else
            curr_state <= next_state;
    end

    //***************************************************************
    // 4) Next-State Logic (always_comb)
    //***************************************************************
    always_comb begin
        next_state = curr_state;
        case(curr_state)
            S_IDLE:    if(start) next_state = S_QKV;

            // S_QKV => compute Q, K, V
            // once done => S_QK
            S_QKV:     if(/* we detect QKV done */ 1'b0) next_state = S_QK;

            // S_QK => compute A=QK^T
            // once done => S_SOFTMAX
            S_QK:      if(/* qk done */ 1'b0) next_state = S_SOFTMAX;

            // S_SOFTMAX => row-wise
            // once done => S_AXV
            S_SOFTMAX: if(/* softmax done */ 1'b0) next_state = S_AXV;

            // S_AXV => compute A x V => final attn
            // once done => S_DONE
            S_AXV:     if(/* done */ 1'b0) next_state = S_DONE;

            S_DONE:    next_state = S_IDLE;

            default:   next_state = S_IDLE;
        endcase
    end

    //***************************************************************
    // 5) Unpack Weights in always_comb
    //***************************************************************
    integer w_r, w_c;
    always_comb begin
        // WQ
        for(w_r=0; w_r<EMB_DIM; w_r++) begin
            bQ[w_r] = bQ_in[(w_r+1)*DATA_WIDTH -1 -: DATA_WIDTH];
            for(w_c=0; w_c<EMB_DIM; w_c++) begin
                WQ[w_r][w_c] = WQ_in[((w_r*EMB_DIM)+w_c+1)*DATA_WIDTH -1 -: DATA_WIDTH];
            end
        end
        // WK
        for(w_r=0; w_r<EMB_DIM; w_r++) begin
            bK[w_r] = bK_in[(w_r+1)*DATA_WIDTH -1 -: DATA_WIDTH];
            for(w_c=0; w_c<EMB_DIM; w_c++) begin
                WK[w_r][w_c] = WK_in[((w_r*EMB_DIM)+w_c+1)*DATA_WIDTH -1 -: DATA_WIDTH];
            end
        end
        // WV
        for(w_r=0; w_r<EMB_DIM; w_r++) begin
            bV[w_r] = bV_in[(w_r+1)*DATA_WIDTH -1 -: DATA_WIDTH];
            for(w_c=0; w_c<EMB_DIM; w_c++) begin
                WV[w_r][w_c] = WV_in[((w_r*EMB_DIM)+w_c+1)*DATA_WIDTH -1 -: DATA_WIDTH];
            end
        end
    end

    //***************************************************************
    // 6) The Main Datapath in always_ff
    //***************************************************************
    integer i, j;
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            done      <= 1'b0;
            out_valid <= 1'b0;

            // init arrays
            for(i=0; i<SEQ_LEN; i++) begin
                for(j=0; j<EMB_DIM; j++) begin
                    x_mem[i][j]      <= '0;
                    Q_mem[i][j]      <= '0;
                    K_mem[i][j]      <= '0;
                    V_mem[i][j]      <= '0;
                    attn_mem[i][j]   <= '0;
                end
                for(j=0; j<SEQ_LEN; j++) begin
                    A_mem[i][j] <= '0;
                end
                row_sums[i] <= 32'd0;
            end
            seq_i   <= 0;
            seq_j   <= 0;
            dim_i   <= 0;
            sum_temp<= 32'd0;

        end else begin
            // default each cycle
            done      <= 1'b0;
            out_valid <= 1'b0;

            case(curr_state)
                //---------------------------------------------------------
                // S_IDLE: unpack x_in => x_mem
                //---------------------------------------------------------
                S_IDLE: begin
                    for(i=0; i<SEQ_LEN; i++) begin
                        for(j=0; j<EMB_DIM; j++) begin
                            int flat_idx = (i*EMB_DIM)+j;
                            x_mem[i][j] <= x_in[(flat_idx+1)*DATA_WIDTH -1 -: DATA_WIDTH];
                        end
                    end
                    seq_i   <= 0;
                    seq_j   <= 0;
                    dim_i   <= 0;
                    sum_temp<= 0;
                end

                //---------------------------------------------------------
                // S_QKV: compute Q, K, V => time multiplex
                //---------------------------------------------------------
                // e.g. one dimension at a time, partial dot product
                // We'll do:
                //   Q[i][dim_i] = bQ[dim_i] + sum_{k}( x_mem[i][k]*WQ[dim_i][k] )
                // similarly for K, V
                S_QKV: begin
                    // Example partial approach:
                    static logic [DATA_WIDTH-1:0] sumQ, sumK, sumV;
                    static logic [$clog2(EMB_DIM):0] dot_idx;

                    if(seq_i == 0 && dim_i == 0 && seq_j == 0) begin
                        sumQ <= bQ[0];
                        sumK <= bK[0];
                        sumV <= bV[0];
                        dot_idx <= 0;
                    end

                    // accumulate one partial multiply for Q, K, V
                    sumQ <= sumQ + ( x_mem[seq_i][dot_idx] * WQ[dim_i][dot_idx] );
                    sumK <= sumK + ( x_mem[seq_i][dot_idx] * WK[dim_i][dot_idx] );
                    sumV <= sumV + ( x_mem[seq_i][dot_idx] * WV[dim_i][dot_idx] );

                    if(dot_idx < (EMB_DIM-1)) begin
                        dot_idx <= dot_idx + 1;
                    end else begin
                        // store results
                        Q_mem[seq_i][dim_i] <= sumQ[DATA_WIDTH-1:0];
                        K_mem[seq_i][dim_i] <= sumK[DATA_WIDTH-1:0];
                        V_mem[seq_i][dim_i] <= sumV[DATA_WIDTH-1:0];

                        // next dimension or next seq
                        dot_idx <= 0;
                        sumQ <= bQ[dim_i+1]; // next dim’s bias
                        sumK <= bK[dim_i+1];
                        sumV <= bV[dim_i+1];

                        if(dim_i < (EMB_DIM-1)) begin
                            dim_i <= dim_i + 1;
                        end else begin
                            dim_i <= 0;
                            if(seq_i < (SEQ_LEN-1)) begin
                                seq_i <= seq_i + 1;
                            end else begin
                                // done all QKV
                                // you’d set a “qkv_done” or so
                            end
                        end
                    end
                end

                //---------------------------------------------------------
                // S_QK: compute A= QK^T => shape (SEQ_LEN, SEQ_LEN)
                //---------------------------------------------------------
                // partial dot: A[i][j] = dot(Q[i], K[j])
                // We'll do a time-multiplex approach again
                S_QK: begin
                    // Similar approach: partial dot of dimension EMB_DIM
                    // e.g. sum_temp <= sum_temp + Q_mem[i][dot_idx]*K_mem[j][dot_idx]
                end

                //---------------------------------------------------------
                // S_SOFTMAX: naive row-wise softmax across dimension j
                //---------------------------------------------------------
                // We'll do: for each row i:
                //   row_sum = sum(A[i][j]) over j
                //   then A[i][j] = A[i][j]/row_sum   (placeholder, integer divide)
                S_SOFTMAX: begin
                    // time multiplex each row i. 
                    // e.g. partial approach:
                    // 1) accumulate row_sum
                    // 2) divide each A[i][j] by row_sum
                end

                //---------------------------------------------------------
                // S_AXV: compute attn_mem = A x V => shape (SEQ_LEN, EMB_DIM)
                //---------------------------------------------------------
                // attn_mem[i][dim] = sum_{j} A[i][j]* V_mem[j][dim]
                S_AXV: begin
                    // partial dot approach again. 
                    // e.g. sum_temp <= sum_temp + (A[i][j]*V_mem[j][dim])
                end

                //---------------------------------------------------------
                // S_DONE: flatten attn_mem => attn_out, done=1
                //---------------------------------------------------------
                S_DONE: begin
                    done      <= 1'b1;
                    out_valid <= 1'b1;
                    // flatten
                    for(i=0; i<SEQ_LEN; i++) begin
                        for(j=0; j<EMB_DIM; j++) begin
                            attn_out[ ((i*EMB_DIM)+j+1)*DATA_WIDTH -1 -: DATA_WIDTH ]
                                <= attn_mem[i][j];
                        end
                    end
                end

                default: /* no-op */;
            endcase
        end
    end

endmodule

