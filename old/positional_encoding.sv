//-----------------------------------------------------------------------------
// positional_encoding.sv
// Time-multiplexed positional embedding addition using a pre-computed
// FP16 sin/cos ROM. out_embed = A_in + PosROM[token_idx,dim_e].
// ROM is synchronous (updates on clock edge).
//
// Apr. 15 2025    Max Zhang      Initial version
// Apr. 20 2025    Tianwei Liu    Synchronous ROM read
//-----------------------------------------------------------------------------
module positional_encoding #(
    parameter int DATA_WIDTH  = 16,
    parameter int NUM_TOKENS  = 196,      // e.g. 14×14 patches (or +1 for [CLS])
    parameter int E           = 128       // embedding dimension
)(
    input  logic                                      clk,
    input  logic                                      rst_n,

    // Control
    input  logic                                      start,
    output logic                                      done,

    // Input embeddings shape (NUM_TOKENS, E), flattened
    input  logic [DATA_WIDTH*NUM_TOKENS*E -1:0]        A_in,

    // Output = A_in + pos_val_ROM, shape (NUM_TOKENS, E)
    output logic [DATA_WIDTH*NUM_TOKENS*E -1:0]        out_embed,
    output logic                                       out_valid
);

    //-------------------------------------------------------------------------
    // 1) FSM States
    //-------------------------------------------------------------------------
    typedef enum logic [1:0] {
        S_IDLE,
        S_ADD,
        S_DONE
    } state_t;
    state_t curr_state, next_state;

    //-------------------------------------------------------------------------
    // 2) Counters & Local RAM for Result
    //-------------------------------------------------------------------------
    logic [$clog2(NUM_TOKENS)-1:0] token_idx, token_idx_dly;
    logic [$clog2(E)-1:0]          dim_e, dim_e_dly;
    logic [DATA_WIDTH-1:0]         out_mem [0:NUM_TOKENS-1][0:E-1];

    //-------------------------------------------------------------------------
    // 3) Instantiate the ROM
    //-------------------------------------------------------------------------
    // ROM is synchronous (pos_val updates on clock edge).
    logic [DATA_WIDTH-1:0] pos_val;
    positional_encoding_rom_fp16 #(
        .DATA_WIDTH (DATA_WIDTH),
        .NUM_TOKENS (NUM_TOKENS),
        .E          (E),
        .MEMFILE    ("pos_embed_fp16.mem")
    ) u_posrom (
        .clk        (clk),
        .rst_n      (rst_n),
        .token_idx  (token_idx),
        .dim        (dim_e),
        .pos_val    (pos_val)
    );

    //-------------------------------------------------------------------------
    // 4) Helper to read A_in
    //-------------------------------------------------------------------------
    function logic [DATA_WIDTH-1:0] get_embed(
        input logic [$clog2(NUM_TOKENS)-1:0] t,
        input logic [$clog2(E)-1:0]          d
    );
        get_embed = A_in[(t*E + d+1)*DATA_WIDTH -1 -: DATA_WIDTH];
    endfunction

    //-------------------------------------------------------------------------
    // 5) FSM: state register
    //-------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) curr_state <= S_IDLE;
        else         curr_state <= next_state;
    end

    //-------------------------------------------------------------------------
    // 6) FSM: next-state logic
    //-------------------------------------------------------------------------
    always_comb begin
        next_state = curr_state;
        case (curr_state)
            S_IDLE: begin
                if (start)
                    next_state = S_ADD;
                else
                    next_state = S_IDLE;
            end
            S_ADD: begin
                if (token_idx_dly == NUM_TOKENS-1 && dim_e_dly == E-1)
                    next_state = S_DONE;
                else
                    next_state = S_ADD;
            end
            S_DONE: next_state = S_IDLE;
        endcase
    end

    //-------------------------------------------------------------------------
    // 7) FSM: outputs & datapath
    //-------------------------------------------------------------------------
    // Flatten out_mem → out_embed in DONE state
    integer i, j;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done          <= 1'b0;
            out_valid     <= 1'b0;
            token_idx     <= '0;
            dim_e         <= '0;
            token_idx_dly <= '0;
            dim_e_dly     <= '0;
            for (i=0; i<NUM_TOKENS; i++)
                for (j=0; j<E; j++)
                    out_mem[i][j] <= '0;
        end else begin
            // Default deassert
            done      <= 1'b0;
            out_valid <= 1'b0;

            case (curr_state)
                S_IDLE: begin
                    token_idx     <= 0;
                    dim_e         <= 0;
                    token_idx_dly <= 0;
                    dim_e_dly     <= 0;
                end

                S_ADD: begin
                    // Delay indices to align with ROM output
                    token_idx_dly <= token_idx;
                    dim_e_dly     <= dim_e;

                    // Perform addition using delayed indices (when pos_val is valid)
                    if (token_idx_dly < NUM_TOKENS && dim_e_dly < E) begin
                        out_mem[token_idx_dly][dim_e_dly] <= get_embed(token_idx_dly, dim_e_dly) + pos_val;
                    end

                    // Advance counters
                    if (dim_e < E-1)
                        dim_e <= dim_e + 1;
                    else begin
                        dim_e <= 0;
                        if (token_idx < NUM_TOKENS-1)
                            token_idx <= token_idx + 1;
                    end
                end

                S_DONE: begin
                    done      <= 1'b1;
                    out_valid <= 1'b1;
                    // Flatten
                    for (i=0; i<NUM_TOKENS; i++)
                        for (j=0; j<E; j++)
                            out_embed[((i*E + j)+1)*DATA_WIDTH -1 -: DATA_WIDTH]
                                <= out_mem[i][j];
                end
            endcase
        end
    end

endmodule
