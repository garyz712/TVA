/**************************************************************
 * mlp_block.sv
 *
 * A time-multiplexed 2-layer MLP skeleton with:
 *   - Input shape (L, N, E)
 *   - Hidden dimension H
 *   - Output shape (L, N, E)
 *   - Avoids large unrolled for-loops by using counters
 **************************************************************/
module mlp_block #(
    parameter int DATA_WIDTH = 16,
    parameter int L = 8,   // Sequence length
    parameter int N = 1,   // Batch size
    parameter int E = 8,   // Input/Output embedding dimension
    parameter int H = 32   // Hidden dimension
)(
    input  logic                               clk,
    input  logic                               rst_n,

    // Control
    input  logic                               start,     // Begin operation
    output logic                               done,      // Done with entire MLP
    output logic                               out_valid, // Final output is valid

    // Input: Flattened (L, N, E)
    input  logic [DATA_WIDTH*L*N*E-1:0]        x_in,

    // MLP weights/biases:
    //   W1: (E x H)
    //   b1: (H)
    //   W2: (H x E)
    //   b2: (E)
    input  logic [DATA_WIDTH*E*H-1:0]          W1_in,
    input  logic [DATA_WIDTH*H-1:0]            b1_in,
    input  logic [DATA_WIDTH*H*E-1:0]          W2_in,
    input  logic [DATA_WIDTH*E-1:0]            b2_in,

    // Output: Flattened (L, N, E)
    output logic [DATA_WIDTH*L*N*E-1:0]        out_mlp
);

    // --------------------------------------------------------------------------
    // 0) Local parameters
    // --------------------------------------------------------------------------
    localparam int TOTAL_TOKENS = L*N;  
    // We'll treat each (l, n) as one "token", so we have TOTAL_TOKENS tokens.

    // --------------------------------------------------------------------------
    // 1) FSM Definition
    // --------------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_LOAD,  // load x_in into x_mem
        S_FC1,   // compute first FC => hidden_mem
        S_ACT,   // ReLU across hidden_mem
        S_FC2,   // compute second FC => out_mem
        S_DONE
    } state_t;

    state_t curr_state, next_state;

    // --------------------------------------------------------------------------
    // 2) State Register
    // --------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            curr_state <= S_IDLE;
        else
            curr_state <= next_state;
    end

    // --------------------------------------------------------------------------
    // 3) Next-State Logic (always_comb)
    // --------------------------------------------------------------------------
    // We'll track when each step is finished using flags: load_done, fc1_done,
    // act_done, fc2_done.
    logic load_done, fc1_done, act_done, fc2_done;

    always_comb begin
        // default
        next_state = curr_state;

        case (curr_state)
            S_IDLE:
                if (start)
                    next_state = S_LOAD;

            S_LOAD:
                if (load_done)
                    next_state = S_FC1;

            S_FC1:
                if (fc1_done)
                    next_state = S_ACT;

            S_ACT:
                if (act_done)
                    next_state = S_FC2;

            S_FC2:
                if (fc2_done)
                    next_state = S_DONE;

            S_DONE:
                next_state = S_IDLE;

            default:
                next_state = S_IDLE;
        endcase
    end

    // --------------------------------------------------------------------------
    // 4) Output & Control Signals
    // --------------------------------------------------------------------------
    always_comb begin
        // Defaults
        done      = 1'b0;
        out_valid = 1'b0;

        // For each step, we also default them to zero
        // We'll set them true in our always_ff if we finish each step
    end

    // We'll assert `done` and `out_valid` only in S_DONE
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done      <= 1'b0;
            out_valid <= 1'b0;
        end
        else begin
            if (curr_state == S_DONE) begin
                done      <= 1'b1;
                out_valid <= 1'b1;
            end
            else begin
                done      <= 1'b0;
                out_valid <= 1'b0;
            end
        end
    end

    // --------------------------------------------------------------------------
    // 5) Internal Memory Arrays
    // --------------------------------------------------------------------------
    // x_mem[i][e], hidden_mem[i][h], out_mem[i][e]
    logic [DATA_WIDTH-1:0] x_mem      [0:TOTAL_TOKENS-1][0:E-1];
    logic [DATA_WIDTH-1:0] hidden_mem [0:TOTAL_TOKENS-1][0:H-1];
    logic [DATA_WIDTH-1:0] out_mem    [0:TOTAL_TOKENS-1][0:E-1];

    // We'll store final output in out_mem, and then pack it into out_mlp.

    // --------------------------------------------------------------------------
    // 6) Weights & Biases in 2D Arrays
    // --------------------------------------------------------------------------
    logic [DATA_WIDTH-1:0] W1 [0:E-1][0:H-1];  // W1[e_in][h_out]
    logic [DATA_WIDTH-1:0] b1 [0:H-1];
    logic [DATA_WIDTH-1:0] W2 [0:H-1][0:E-1];  // W2[h_in][e_out]
    logic [DATA_WIDTH-1:0] b2 [0:E-1];

    // Unpack them in combinational logic
    integer i, j;
    always_comb begin
        // b1, b2
        for (j = 0; j < H; j++) begin
            b1[j] = b1_in[(j+1)*DATA_WIDTH-1 -: DATA_WIDTH];
        end
        for (i = 0; i < E; i++) begin
            b2[i] = b2_in[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH];
        end

        // W1: E x H
        for (i = 0; i < E; i++) begin
            for (j = 0; j < H; j++) begin
                W1[i][j] = W1_in[ ((i*H) + j + 1)*DATA_WIDTH -1 -: DATA_WIDTH];
            end
        end

        // W2: H x E
        for (i = 0; i < H; i++) begin
            for (j = 0; j < E; j++) begin
                W2[i][j] = W2_in[ ((i*E) + j + 1)*DATA_WIDTH -1 -: DATA_WIDTH];
            end
        end
    end

    // --------------------------------------------------------------------------
    // 7) Sequential Logic: time-multiplexed load, FC1, ACT, FC2, output
    // --------------------------------------------------------------------------

    // -- S_LOAD counters
    logic [$clog2(TOTAL_TOKENS):0] load_tok;
    logic [$clog2(E):0]            load_e;

    // -- FC1 counters
    // We'll do hidden_mem[token][outH] = b1[outH] + sum_{e=0..E-1} x_mem[token][e]*W1[e][outH]
    logic [$clog2(TOTAL_TOKENS):0] fc1_token;
    logic [$clog2(H):0]            fc1_outH;
    logic [$clog2(E):0]            fc1_inE;  
    // partial accumulators
    static logic [DATA_WIDTH-1:0] sum_temp_fc1;
    static logic [DATA_WIDTH-1:0] partial_e_fc1;

    // -- S_ACT counters
    logic [$clog2(TOTAL_TOKENS):0] act_tok;
    logic [$clog2(H):0]            act_h;

    // -- FC2 counters
    // out_mem[token][outE] = b2[outE] + sum_{h=0..H-1} hidden_mem[token][h]*W2[h][outE]
    logic [$clog2(TOTAL_TOKENS):0] fc2_token;
    logic [$clog2(E):0]            fc2_outE;
    logic [$clog2(H):0]            fc2_inH;
    static logic [DATA_WIDTH-1:0] sum_temp_fc2;
    static logic [DATA_WIDTH-1:0] partial_h_fc2;

    // We'll define them in an always_ff, resetting as we move states

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // reset all counters & flags
            load_tok     <= '0;
            load_e       <= '0;
            load_done    <= 1'b0;

            fc1_token    <= '0;
            fc1_outH     <= '0;
            fc1_inE      <= '0;
            sum_temp_fc1 <= '0;
            partial_e_fc1<= '0;
            fc1_done     <= 1'b0;

            act_tok      <= '0;
            act_h        <= '0;
            act_done     <= 1'b0;

            fc2_token    <= '0;
            fc2_outE     <= '0;
            fc2_inH      <= '0;
            sum_temp_fc2 <= '0;
            partial_h_fc2<= '0;
            fc2_done     <= 1'b0;

        end else begin

            case (curr_state)

                //--------------------------------------------------------
                // S_LOAD: load x_in => x_mem, one element per cycle
                //--------------------------------------------------------
                S_LOAD: begin
                    // store one element from x_in into x_mem
                    x_mem[load_tok][load_e] <= x_in[
                        ((load_tok*E) + load_e +1)*DATA_WIDTH -1 -: DATA_WIDTH
                    ];

                    // increment e
                    if (load_e == (E-1)) begin
                        load_e <= 0;
                        if (load_tok == (TOTAL_TOKENS-1)) begin
                            // done loading
                            load_done <= 1'b1;
                        end
                        else begin
                            load_tok <= load_tok + 1;
                        end
                    end
                    else begin
                        load_e <= load_e + 1;
                    end
                end

                //--------------------------------------------------------
                // S_FC1: hidden_mem[tok][outH] = b1[outH] + sum(x_mem[tok][e]*W1[e][outH])
                // Time-multiplex: one multiply per cycle
                //--------------------------------------------------------
                S_FC1: begin
                    if (!fc1_done) begin
                        // If this is the first cycle in FC1
                        if (fc1_token == 0 && fc1_outH == 0 && fc1_inE == 0) begin
                            sum_temp_fc1 <= b1[0];  // start partial sum
                            partial_e_fc1<= 0;
                        end

                        // accumulate
                        sum_temp_fc1 <= sum_temp_fc1 +
                                                        (x_mem[fc1_token][partial_e_fc1] * W1[partial_e_fc1][fc1_outH]);

                        // increment partial_e_fc1
                        if (partial_e_fc1 == (E-1)) begin
                            // store result
                            hidden_mem[fc1_token][fc1_outH] <= sum_temp_fc1;

                            // re-init sum_temp for next output dimension or next token
                            sum_temp_fc1 <= b1[fc1_outH]; // might be overwritten below
                            partial_e_fc1<= 0;

                            if (fc1_outH < (H-1)) begin
                                // next outH
                                fc1_outH <= fc1_outH + 1;
                                // new sum
                                sum_temp_fc1 <= b1[fc1_outH+1];
                            end
                            else begin
                                fc1_outH <= 0;
                                if (fc1_token < (TOTAL_TOKENS-1)) begin
                                    fc1_token <= fc1_token + 1;
                                    sum_temp_fc1 <= b1[0];
                                end
                                else begin
                                    // done with FC1 for all tokens
                                    fc1_done <= 1'b1;
                                end
                            end
                        end
                        else begin
                            partial_e_fc1 <= partial_e_fc1 + 1;
                        end
                    end
                end

                //--------------------------------------------------------
                // S_ACT: ReLU across hidden_mem, 1 element per cycle
                //--------------------------------------------------------
                S_ACT: begin
                    if (!act_done) begin
                        // read hidden_mem, clamp if negative
                        if (hidden_mem[act_tok][act_h][DATA_WIDTH-1] == 1'b1) begin
                            hidden_mem[act_tok][act_h] <= '0;
                        end

                        // increment act_h
                        if (act_h == (H-1)) begin
                            act_h <= 0;
                            if (act_tok == (TOTAL_TOKENS-1)) begin
                                act_done <= 1'b1;
                            end
                            else begin
                                act_tok <= act_tok + 1;
                            end
                        end
                        else begin
                            act_h <= act_h + 1;
                        end
                    end
                end

                //--------------------------------------------------------
                // S_FC2: out_mem[tok][outE] = b2[outE] + sum_{h=0..H-1}(hidden_mem[tok][h]*W2[h][outE])
                //--------------------------------------------------------
                S_FC2: begin
                    if (!fc2_done) begin
                        // if first cycle in FC2
                        if (fc2_token == 0 && fc2_outE == 0 && fc2_inH == 0) begin
                            sum_temp_fc2 <= b2[0];
                            partial_h_fc2<= 0;
                        end

                        // accumulate
                        sum_temp_fc2 <= sum_temp_fc2 +
                                                        (hidden_mem[fc2_token][partial_h_fc2] * W2[partial_h_fc2][fc2_outE]);

                        // increment partial_h_fc2
                        if (partial_h_fc2 == (H-1)) begin
                            // store in out_mem
                            out_mem[fc2_token][fc2_outE] <= sum_temp_fc2;

                            // re-init for next outE or next token
                            sum_temp_fc2 <= b2[fc2_outE];
                            partial_h_fc2<= 0;

                            if (fc2_outE < (E-1)) begin
                                fc2_outE <= fc2_outE + 1;
                                sum_temp_fc2 <= b2[fc2_outE+1];
                            end
                            else begin
                                fc2_outE <= 0;
                                if (fc2_token < (TOTAL_TOKENS-1)) begin
                                    fc2_token <= fc2_token + 1;
                                    sum_temp_fc2 <= b2[0];
                                end
                                else begin
                                    fc2_done <= 1'b1;
                                end
                            end
                        end
                        else begin
                            partial_h_fc2 <= partial_h_fc2 + 1;
                        end
                    end
                end

                //--------------------------------------------------------
                // S_DONE: pack out_mem => out_mlp
                //--------------------------------------------------------
                default: begin
                    if (curr_state == S_DONE) begin
                        integer t_, e_;
                        for (t_ = 0; t_ < TOTAL_TOKENS; t_++) begin
                            for (e_ = 0; e_ < E; e_++) begin
                                out_mlp[ ((t_*E) + e_ +1)*DATA_WIDTH -1 -: DATA_WIDTH ]
                                    <= out_mem[t_][e_];
                            end
                        end
                    end
                end

            endcase
        end
    end

endmodule
