//------------------------------------------------------------------------------
// vit_encoder_block.sv
//
// This module implements a single encoder block from a Vision Transformer 
// (ViT). It consists of the following components in sequence:
// - LayerNorm 1 (LN1)
// - Multi-head Self-Attention
// - Residual Connection (Input + Attention)
// - LayerNorm 2 (LN2)
// - MLP Block (2-layer Feedforward Network)
// - Residual Connection (Previous Residual + MLP Output)
//
//------------------------------------------------------------------------------
module vit_encoder_block #(
    parameter int DATA_WIDTH = 16,
    parameter int SEQ_LEN    = 8,
    parameter int EMB_DIM    = 8,
    parameter int H_MLP      = 32 // hidden dimension in MLP
)(
    input  logic                                      clk,
    input  logic                                      rst_n,

    // Control
    input  logic                                      start,
    output logic                                      done,

    // Input x_in: shape (SEQ_LEN, EMB_DIM)
    // Flattened => DATA_WIDTH*(SEQ_LEN*EMB_DIM)
    input  logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0]     x_in,

    // LN1 gamma/beta
    input  logic [DATA_WIDTH*EMB_DIM-1:0]            ln1_gamma,
    input  logic [DATA_WIDTH*EMB_DIM-1:0]            ln1_beta,

    // LN2 gamma/beta
    input  logic [DATA_WIDTH*EMB_DIM-1:0]            ln2_gamma,
    input  logic [DATA_WIDTH*EMB_DIM-1:0]            ln2_beta,

    // Self-Attention Weights
    input  logic [DATA_WIDTH*EMB_DIM*EMB_DIM -1:0]   WQ_in,
    input  logic [DATA_WIDTH*EMB_DIM*EMB_DIM -1:0]   WK_in,
    input  logic [DATA_WIDTH*EMB_DIM*EMB_DIM -1:0]   WV_in,
    input  logic [DATA_WIDTH*EMB_DIM -1:0]           bQ_in,
    input  logic [DATA_WIDTH*EMB_DIM -1:0]           bK_in,
    input  logic [DATA_WIDTH*EMB_DIM -1:0]           bV_in,

    // MLP Weights => W1, b1, W2, b2
    // Suppose W1: (EMB_DIM x H_MLP), b1: (H_MLP)
    //         W2: (H_MLP x EMB_DIM), b2: (EMB_DIM)
    input  logic [DATA_WIDTH*EMB_DIM*H_MLP -1:0]      W1_in,
    input  logic [DATA_WIDTH*H_MLP -1:0]             b1_in,
    input  logic [DATA_WIDTH*H_MLP*EMB_DIM -1:0]     W2_in,
    input  logic [DATA_WIDTH*EMB_DIM -1:0]           b2_in,

    // Output: shape (SEQ_LEN, EMB_DIM)
    output logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0]    out_block,
    output logic                                      out_valid
);

    //***************************************************************
    // 1) Internal Wires for Submodules
    //***************************************************************
    // LN1
    logic ln1_start, ln1_done, ln1_out_valid;
    logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0] ln1_out;

    // Self-Attn
    logic attn_start, attn_done, attn_out_valid;
    logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0] attn_out;

    // Residual1
    logic res1_start, res1_done, res1_out_valid;
    logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0] res1_out; // x + attn_out

    // LN2
    logic ln2_start, ln2_done, ln2_out_valid;
    logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0] ln2_out;

    // MLP
    logic mlp_start, mlp_done, mlp_out_valid;
    logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0] mlp_out;

    // Residual2
    logic res2_start, res2_done, res2_out_valid;
    logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0] res2_out; // res1_out + mlp_out

    //***************************************************************
    // 2) Submodule Instantiations
    //***************************************************************

    // (A) LN1: LN(x_in)
    layer_norm #(
        .DATA_WIDTH(DATA_WIDTH),
        .SEQ_LEN   (SEQ_LEN),
        .EMB_DIM   (EMB_DIM)
    ) u_ln1 (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (ln1_start),
        .done     (ln1_done),
        .x_in     (x_in),
        .gamma_in (ln1_gamma),
        .beta_in  (ln1_beta),
        .x_out    (ln1_out),
        .out_valid(ln1_out_valid)
    );

    // (B) Self-Attention
    self_attention #(
        .DATA_WIDTH(DATA_WIDTH),
        .SEQ_LEN   (SEQ_LEN),
        .EMB_DIM   (EMB_DIM)
    ) u_attn (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (attn_start),
        .done     (attn_done),
        .x_in     (ln1_out),
        .WQ_in    (WQ_in),
        .WK_in    (WK_in),
        .WV_in    (WV_in),
        .bQ_in    (bQ_in),
        .bK_in    (bK_in),
        .bV_in    (bV_in),
        .attn_out (attn_out),
        .out_valid(attn_out_valid)
    );

    // (C) Residual #1: z1 = x_in + attn_out
    residual #(
        .DATA_WIDTH(DATA_WIDTH),
        .SEQ_LEN   (SEQ_LEN),
        .EMB_DIM   (EMB_DIM)
    ) u_res1 (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (res1_start),
        .done     (res1_done),
        .x_in     (x_in),
        .sub_in   (attn_out),
        .y_out    (res1_out),
        .out_valid(res1_out_valid)
    );

    // (D) LN2: LN(z1)
    layer_norm #(
        .DATA_WIDTH(DATA_WIDTH),
        .SEQ_LEN   (SEQ_LEN),
        .EMB_DIM   (EMB_DIM)
    ) u_ln2 (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (ln2_start),
        .done     (ln2_done),
        .x_in     (res1_out),
        .gamma_in (ln2_gamma),
        .beta_in  (ln2_beta),
        .x_out    (ln2_out),
        .out_valid(ln2_out_valid)
    );

    // (E) MLP
    mlp_block #(
        .DATA_WIDTH(DATA_WIDTH),
        .L        (SEQ_LEN), // "L" is SEQ_LEN in your earlier code
        .N        (1),       // batch=1
        .E        (EMB_DIM),
        .H        (H_MLP)
    ) u_mlp (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (mlp_start),
        .done     (mlp_done),
        .x_in     (ln2_out),
        .W1_in    (W1_in),
        .b1_in    (b1_in),
        .W2_in    (W2_in),
        .b2_in    (b2_in),
        .out_mlp  (mlp_out),
        .out_valid(mlp_out_valid)
    );

    // (F) Residual #2: z2 = z1 + mlp_out
    residual #(
        .DATA_WIDTH(DATA_WIDTH),
        .SEQ_LEN   (SEQ_LEN),
        .EMB_DIM   (EMB_DIM)
    ) u_res2 (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (res2_start),
        .done     (res2_done),
        .x_in     (res1_out),
        .sub_in   (mlp_out),
        .y_out    (res2_out),
        .out_valid(res2_out_valid)
    );

    //***************************************************************
    // 3) FSM for Orchestrating LN1 -> ATT -> RES1 -> LN2 -> MLP -> RES2
    //***************************************************************

    typedef enum logic [3:0] {
        S_IDLE,
        S_LN1,
        S_WAIT_LN1,
        S_ATT,
        S_WAIT_ATT,
        S_RES1,
        S_WAIT_RES1,
        S_LN2,
        S_WAIT_LN2,
        S_MLP,
        S_WAIT_MLP,
        S_RES2,
        S_WAIT_RES2,
        S_DONE
    } state_t;

    state_t curr_state, next_state;

    // State register
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n)
            curr_state <= S_IDLE;
        else
            curr_state <= next_state;
    end

    // Next-state logic
    always_comb begin
        next_state = curr_state;
        case(curr_state)
            S_IDLE:       if(start)       next_state = S_LN1;

            // LN1
            S_LN1:        next_state = S_WAIT_LN1;
            S_WAIT_LN1:   if(ln1_done)    next_state = S_ATT;

            // Self-Attention
            S_ATT:        next_state = S_WAIT_ATT;
            S_WAIT_ATT:   if(attn_done)   next_state = S_RES1;

            // Residual #1
            S_RES1:       next_state = S_WAIT_RES1;
            S_WAIT_RES1:  if(res1_done)   next_state = S_LN2;

            // LN2
            S_LN2:        next_state = S_WAIT_LN2;
            S_WAIT_LN2:   if(ln2_done)    next_state = S_MLP;

            // MLP
            S_MLP:        next_state = S_WAIT_MLP;
            S_WAIT_MLP:   if(mlp_done)    next_state = S_RES2;

            // Residual #2
            S_RES2:       next_state = S_WAIT_RES2;
            S_WAIT_RES2:  if(res2_done)   next_state = S_DONE;

            S_DONE:       next_state = S_IDLE;
            default:      next_state = S_IDLE;
        endcase
    end

    //***************************************************************
    // 4) Output & Control Logic
    //***************************************************************
    always_comb begin
        // default
        done       = 1'b0;
        out_valid  = 1'b0;

        // submodule starts default
        ln1_start  = 1'b0;
        attn_start = 1'b0;
        res1_start = 1'b0;
        ln2_start  = 1'b0;
        mlp_start  = 1'b0;
        res2_start = 1'b0;

        case(curr_state)
            S_LN1:   ln1_start  = 1'b1;
            S_ATT:   attn_start = 1'b1;
            S_RES1:  res1_start = 1'b1;
            S_LN2:   ln2_start  = 1'b1;
            S_MLP:   mlp_start  = 1'b1;
            S_RES2:  res2_start = 1'b1;

            S_DONE: begin
                done      = 1'b1;
                out_valid = 1'b1;
            end
            default: /* no-op */
        endcase
    end

    //***************************************************************
    // 5) Final Output
    //***************************************************************
    // The final block output is `res2_out`.
    // We'll register or pipeline if needed.
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            out_block <= '0;
        end else if(curr_state == S_DONE) begin
            out_block <= res2_out;
        end
    end

endmodule
