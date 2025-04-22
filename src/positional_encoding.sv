//-----------------------------------------------------------------------------
// positional_encoding.sv
//    Time‑multiplexed positional embedding addition using a pre‑computed
//    FP16 sin/cos ROM.  out_embed = A_in + PosROM[token_idx,dim_e].
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
    output logic                                      out_valid
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
    logic [$clog2(NUM_TOKENS)-1:0] token_idx;
    logic [$clog2(E)-1:0]          dim_e;
    logic [DATA_WIDTH-1:0]         out_mem [0:NUM_TOKENS-1][0:E-1];

    //-------------------------------------------------------------------------
    // 3) Instantiate the ROM
    //-------------------------------------------------------------------------
    // This ROM is assumed asynchronous (pos_val updates combinationally).
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
        int idx = t*E + d;
        get_embed = A_in[(idx+1)*DATA_WIDTH -1 -: DATA_WIDTH];
    endfunction

    //-------------------------------------------------------------------------
    // 5) FSM: state register
    //-------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) curr_state <= S_IDLE;
        else         curr_state <= next_state;
    end

    //-------------------------------------------------------------------------
    // 6) FSM: next‑state logic
    //-------------------------------------------------------------------------
    always_comb begin
        next_state = curr_state;
        case (curr_state)
            S_IDLE: next_state = start ? S_ADD  : S_IDLE;
            S_ADD:  next_state = (token_idx == NUM_TOKENS-1 && dim_e == E-1)
                                 ? S_DONE : S_ADD;
            S_DONE: next_state = S_IDLE;
        endcase
    end

    //-------------------------------------------------------------------------
    // 7) FSM: outputs & datapath
    //-------------------------------------------------------------------------
    // flatten out_mem → out_embed in DONE state
    integer i,j;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done      <= 1'b0;
            out_valid <= 1'b0;
            token_idx <= '0;
            dim_e     <= '0;
            for (i=0; i<NUM_TOKENS; i++)
                for (j=0; j<E; j++)
                  out_mem[i][j] <= '0;
        end else begin
            // default deassert
            done      <= 1'b0;
            out_valid <= 1'b0;

            case (curr_state)
                S_IDLE: begin
                    token_idx <= 0;
                    dim_e     <= 0;
                end

                S_ADD: begin
                    // read embed + rom
                    logic [DATA_WIDTH-1:0] a = get_embed(token_idx, dim_e);
                    out_mem[token_idx][dim_e] <= a + pos_val;

                    // advance counters
                    if (dim_e < E-1)      dim_e     <= dim_e + 1;
                    else begin
                        dim_e     <= 0;
                        if (token_idx < NUM_TOKENS-1)
                            token_idx <= token_idx + 1;
                    end
                end

                S_DONE: begin
                    done      <= 1'b1;
                    out_valid <= 1'b1;
                    // flatten
                    for (i=0; i<NUM_TOKENS; i++)
                        for (j=0; j<E; j++)
                            out_embed[((i*E + j)+1)*DATA_WIDTH -1 -: DATA_WIDTH]
                                <= out_mem[i][j];
                end
            endcase
        end
    end

endmodule
