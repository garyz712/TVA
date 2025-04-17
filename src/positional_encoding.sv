//-----------------------------------------------------------------------------
// positional_encoding.sv
//
// Adds learned positional embeddings to input embeddings.  Given a flattened
// array A_in of shape (NUM_TOKENS, E) and a learned positional table
// pos_table_in of the same shape, this module performs a time-multiplexed
// addition to produce out_embed = A_in + pos_table_in.  Control is provided
// via start/done signals; the result is made valid when done.
//-----------------------------------------------------------------------------
module positional_encoding #(
    parameter int DATA_WIDTH  = 16,
    parameter int NUM_TOKENS  = 196,  // e.g., 14x14 if that's how many patches or tokens
    parameter int E           = 128   // embedding dimension
)(
    input  logic                                      clk,
    input  logic                                      rst_n,

    // Control
    input  logic                                      start,
    output logic                                      done,

    // Input embeddings shape (NUM_TOKENS, E), flattened
    input  logic [DATA_WIDTH*NUM_TOKENS*E -1:0]        A_in,

    // Learned positional table shape (NUM_TOKENS, E)
    input  logic [DATA_WIDTH*NUM_TOKENS*E -1:0]        pos_table_in,

    // Output = A_in + pos_table_in, shape (NUM_TOKENS, E)
    output logic [DATA_WIDTH*NUM_TOKENS*E -1:0]        out_embed,
    output logic                                       out_valid //FIXME: WHY?
);

    //-------------------------------------------------------------------------
    // 1) State Definitions
    //-------------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_LOAD,      // (optional) load pos_table into local arrays if needed
        S_ADD,       // time-multiplex add
        S_DONE
    } state_t;

    state_t curr_state, next_state;

    //-------------------------------------------------------------------------
    // 2) Internal Memories / Arrays
    //-------------------------------------------------------------------------
    // If you want, you can store pos_table and input in local 2D arrays:
    //   A_mem[token][dim]
    //   pos_mem[token][dim]
    //   out_mem[token][dim]
    // Or read them directly from the flattened bus with a helper function.

    logic [DATA_WIDTH-1:0] out_mem [0:NUM_TOKENS-1][0:E-1];

    // We keep counters to cycle over (token_idx, dim_e)
    logic [$clog2(NUM_TOKENS):0] token_idx;
    logic [$clog2(E):0]          dim_e;

    //-------------------------------------------------------------------------
    // 3) State Register
    //-------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            curr_state <= S_IDLE;
        else
            curr_state <= next_state;
    end

    //-------------------------------------------------------------------------
    // 4) Next-State Logic (always_comb)
    //-------------------------------------------------------------------------
    always_comb begin
        // default
        next_state = curr_state;

        case (curr_state)
            S_IDLE: begin
                if (start)
                    next_state = S_LOAD;
            end

            // S_LOAD => If needed, parse pos_table_in. Then go to S_ADD
            S_LOAD: begin
                next_state = S_ADD;
            end

            // S_ADD => do one addition (A_in + pos_table_in) each cycle
            // if we've done all (NUM_TOKENS * E) then S_DONE
            S_ADD: begin
                if ((token_idx == (NUM_TOKENS-1)) && (dim_e == (E-1)))
                    next_state = S_DONE;
            end

            S_DONE: begin
                next_state = S_IDLE;
            end

            default: next_state = S_IDLE;
        endcase
    end

    //-------------------------------------------------------------------------
    // 5) Helper Functions to Access Flattened Buses
    //-------------------------------------------------------------------------

    // get_embed() => returns A_in[token_idx, dim_e]
    function logic [DATA_WIDTH-1:0] get_embed(
        input logic [$clog2(NUM_TOKENS):0] t_idx,
        input logic [$clog2(E):0]         d_idx
    );
        int flat_idx;
        flat_idx = (t_idx * E) + d_idx;
        get_embed = A_in[((flat_idx + 1) * DATA_WIDTH) -1 -: DATA_WIDTH];
    endfunction

    // get_pos() => returns pos_table_in[token_idx, dim_e]
    function logic [DATA_WIDTH-1:0] get_pos(
        input logic [$clog2(NUM_TOKENS):0] t_idx,
        input logic [$clog2(E):0]         d_idx
    );
        int flat_idx;
        flat_idx = (t_idx * E) + d_idx;
        get_pos = pos_table_in[((flat_idx + 1) * DATA_WIDTH) -1 -: DATA_WIDTH];
    endfunction

    //-------------------------------------------------------------------------
    // 6) Datapath + Output Logic in always_ff
    //-------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done       <= 1'b0;
            out_valid  <= 1'b0;

            token_idx  <= 0;
            dim_e      <= 0;

            // init out_mem
            for (int i = 0; i < NUM_TOKENS; i = i + 1) begin
                for (int j = 0; j < E; j = j + 1) begin
                    out_mem[i][j] <= '0;
                end
            end
        end else begin
            // defaults each cycle
            done      <= 1'b0;
            out_valid <= 1'b0;

            case (curr_state)

                S_IDLE: begin
                    // No action, waiting for start
                end

                S_LOAD: begin
                    // Reset counters at start of operation
                    token_idx <= 0;
                    dim_e     <= 0;
                end

                S_ADD: begin
                    // add one element
                    logic [DATA_WIDTH-1:0] valA, valP;
                    valA = get_embed(token_idx, dim_e);
                    valP = get_pos(token_idx, dim_e);

                    // For real design, watch out for sign bits or saturations
                    out_mem[token_idx][dim_e] <= valA + valP;

                    // increment dim_e
                    if (dim_e < (E-1)) begin
                        dim_e <= dim_e + 1;
                    end else begin
                        // next row
                        dim_e <= 0;
                        if (token_idx < (NUM_TOKENS-1))
                            token_idx <= token_idx + 1;
                    end
                end

                S_DONE: begin
                    done      <= 1'b1;
                    out_valid <= 1'b1;

                    // Flatten out_mem => out_embed
                    for (int t = 0; t < NUM_TOKENS; t = t + 1) begin
                        for (int d = 0; d < E; d = d + 1) begin
                            out_embed[(((t * E) + d + 1) * DATA_WIDTH) -1 -: DATA_WIDTH] <= out_mem[t][d];
                        end
                    end
                end

                default: /* no-op */ ;

            endcase
        end
    end

endmodule
