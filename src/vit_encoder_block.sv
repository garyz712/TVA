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
// May 06 2025    Max Zhang      Initial version
// May 27 2025    Tianwei Liu    Use updated module declaration
//------------------------------------------------------------------------------
module vit_encoder_block #(
    parameter int DATA_WIDTH = 16,
    parameter int SEQ_LEN    = 8,
    parameter int EMB_DIM    = 8,
    parameter int H_MLP      = 32,  // Hidden dimension in MLP
    parameter logic [31:0] EPS = 32'h34000000  // ~1e-5 for LayerNorm
)(
    input  logic                                      clk,
    input  logic                                      rst_n, // Active-low reset
    // Control
    input  logic                                      start,
    output logic                                      done,
    // Input x_in: shape (SEQ_LEN, EMB_DIM), flattened
    input  logic [DATA_WIDTH-1:0]     x_in [SEQ_LEN*EMB_DIM],
    // LN1 gamma/beta
    input  logic [DATA_WIDTH-1:0]            ln1_gamma [EMB_DIM],
    input  logic [DATA_WIDTH-1:0]            ln1_beta [EMB_DIM],
    // LN2 gamma/beta
    input  logic [DATA_WIDTH-1:0]            ln2_gamma [EMB_DIM],
    input  logic [DATA_WIDTH-1:0]            ln2_beta [EMB_DIM],
    // Self-Attention Weights: WQ, WK, WV, WO, each (EMB_DIM, EMB_DIM)
    input  logic [DATA_WIDTH-1:0]    WQ_in [EMB_DIM * EMB_DIM],
    input  logic [DATA_WIDTH-1:0]    WK_in [EMB_DIM * EMB_DIM],
    input  logic [DATA_WIDTH-1:0]    WV_in [EMB_DIM * EMB_DIM],
    input  logic [DATA_WIDTH-1:0]    WO_in [EMB_DIM * EMB_DIM],
    // MLP Weights: W1 (EMB_DIM x H_MLP), b1 (H_MLP), W2 (H_MLP x EMB_DIM), b2 (EMB_DIM)
    input  logic [DATA_WIDTH-1:0]      W1_in [EMB_DIM * H_MLP],
    input  logic [DATA_WIDTH-1:0]             b1_in [H_MLP],
    input  logic [DATA_WIDTH-1:0]     W2_in [H_MLP*EMB_DIM],
    input  logic [DATA_WIDTH-1:0]           b2_in [EMB_DIM],
    // Output: shape (SEQ_LEN, EMB_DIM), flattened
    output logic [DATA_WIDTH-1:0]    out_block [SEQ_LEN*EMB_DIM],
    output logic                                      out_valid
);

    //***************************************************************
    // 1) Internal Wires for Submodules
    //***************************************************************
    // LN1
    logic ln1_start, ln1_done, ln1_out_valid;
    logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0] ln1_out;
    // Self-Attention
    logic attn_start, attn_done, attn_out_valid;
    logic [DATA_WIDTH-1:0] attn_out [SEQ_LEN*EMB_DIM];
    // Residual1
    logic res1_start, res1_done, res1_out_valid;
    logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0] res1_out; // x_in + attn_out
    // LN2
    logic ln2_start, ln2_done, ln2_out_valid;
    logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0] ln2_out;
    // MLP
    logic mlp_start, mlp_done, mlp_out_valid, mlp_valid_in;
    logic [DATA_WIDTH-1:0] mlp_in [EMB_DIM];
    logic [DATA_WIDTH-1:0] mlp_out [EMB_DIM]; // Changed to EMB_DIM
    logic [DATA_WIDTH-1:0] mlp_out_array [SEQ_LEN*EMB_DIM];
    // Residual2
    logic res2_start, res2_done, res2_out_valid;
    logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0] res2_out; // res1_out + mlp_out

    //***************************************************************
    // 2) Array-to-Flat and Flat-to-Array Conversions
    //***************************************************************
    // Convert arrays to flattened for layer_norm and residual
    logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0] x_in_flat;
    logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0] attn_out_flat;
    logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0] mlp_out_flat;
    // Convert flattened to array for self_attention_top and mlp
    logic [DATA_WIDTH-1:0] ln1_out_array [SEQ_LEN*EMB_DIM];
    logic [DATA_WIDTH-1:0] ln2_out_array [SEQ_LEN*EMB_DIM];
    // Convert gamma/beta arrays to flattened
    logic [DATA_WIDTH*EMB_DIM-1:0] ln1_gamma_flat;
    logic [DATA_WIDTH*EMB_DIM-1:0] ln1_beta_flat;
    logic [DATA_WIDTH*EMB_DIM-1:0] ln2_gamma_flat;
    logic [DATA_WIDTH*EMB_DIM-1:0] ln2_beta_flat;

    always_comb begin
        // x_in to flattened
        for (int i = 0; i < SEQ_LEN*EMB_DIM; i++) begin
            x_in_flat[i*DATA_WIDTH +: DATA_WIDTH] = x_in[i];
        end
        // ln1_out to array
        for (int i = 0; i < SEQ_LEN; i++) begin
            for (int j = 0; j < EMB_DIM; j++) begin
                ln1_out_array[i*SEQ_LEN+j] = ln1_out[(i*SEQ_LEN+j)*DATA_WIDTH +: DATA_WIDTH];
            end
        end
        // ln2_out to array
        // for (int i = 0; i < SEQ_LEN*EMB_DIM; i++) begin
        //     ln2_out_array[i] = ln2_out[i*DATA_WIDTH +: DATA_WIDTH];
        // end
        // attn_out to flattened
        for (int i = 0; i < SEQ_LEN; i++) begin
            for (int j = 0; j < EMB_DIM; j++) begin
                ln2_out_array[i*SEQ_LEN+j] = ln2_out[(i*SEQ_LEN+j)*DATA_WIDTH +: DATA_WIDTH];
            end
        end

        // for (int i = 0; i < SEQ_LEN*EMB_DIM; i++) begin
        //     attn_out_flat[i*DATA_WIDTH +: DATA_WIDTH] = attn_out[i];
        // end
        // // mlp_out_array to flattened
        // for (int i = 0; i < SEQ_LEN*EMB_DIM; i++) begin
        //     mlp_out_flat[i*DATA_WIDTH +: DATA_WIDTH] = mlp_out_array[i];
        // end
        for (int i = 0; i < SEQ_LEN*EMB_DIM; i++) begin
            attn_out_flat[i*DATA_WIDTH +: DATA_WIDTH] = attn_out[i];
        end
        // mlp_out_array to flattened
        for (int i = 0; i < SEQ_LEN*EMB_DIM; i++) begin
            mlp_out_flat[i*DATA_WIDTH +: DATA_WIDTH] = mlp_out_array[i];
        end

        // gamma/beta to flattened
        for (int i = 0; i < EMB_DIM; i++) begin
            ln1_gamma_flat[i*DATA_WIDTH +: DATA_WIDTH] = ln1_gamma[i];
            ln1_beta_flat[i*DATA_WIDTH +: DATA_WIDTH] = ln1_beta[i];
            ln2_gamma_flat[i*DATA_WIDTH +: DATA_WIDTH] = ln2_gamma[i];
            ln2_beta_flat[i*DATA_WIDTH +: DATA_WIDTH] = ln2_beta[i];
        end
    end

    //***************************************************************
    // 3) Submodule Instantiations
    //***************************************************************

    // (A) LN1: LN(x_in)
    layer_norm #(
        .DATA_WIDTH(DATA_WIDTH),
        .SEQ_LEN   (SEQ_LEN),
        .EMB_DIM   (EMB_DIM),
        .EPS       (EPS)
    ) u_ln1 (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (ln1_start),
        .done     (ln1_done),
        .x_in     (x_in_flat),
        .gamma_in (ln1_gamma_flat),
        .beta_in  (ln1_beta_flat),
        .x_out    (ln1_out),
        .out_valid(ln1_out_valid)
    );

    // (B) Self-Attention
    self_attention_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .L         (SEQ_LEN),
        .E         (EMB_DIM),
        .N         (1)
    ) u_attn (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (attn_start),
        .done     (attn_done),
        .out_valid(attn_out_valid),
        .x_in     (ln1_out_array),
        .WQ_in    (WQ_in),
        .WK_in    (WK_in),
        .WV_in    (WV_in),
        .WO_in    (WO_in),
        .out      (attn_out)
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
        .x_in     (x_in_flat),
        .sub_in   (attn_out_flat),
        .y_out    (res1_out),
        .out_valid(res1_out_valid)
    );

    // (D) LN2: LN(z1)
    layer_norm #(
        .DATA_WIDTH(DATA_WIDTH),
        .SEQ_LEN   (SEQ_LEN),
        .EMB_DIM   (EMB_DIM),
        .EPS       (EPS)
    ) u_ln2 (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (ln2_start),
        .done     (ln2_done),
        .x_in     (res1_out),
        .gamma_in (ln2_gamma_flat),
        .beta_in  (ln2_beta_flat),
        .x_out    (ln2_out),
        .out_valid(ln2_out_valid)
    );

    // (E) MLP (processes one token at a time)
    mlp #(
        .HIDDEN_DIM(EMB_DIM),
        .MLP_DIM   (H_MLP),
        .OUT_DIM   (EMB_DIM),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_mlp (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (mlp_start),
        .valid_in (mlp_valid_in),
        .x        (mlp_in),
        .W1       (W1_in),
        .b1       (b1_in),
        .W2       (W2_in),
        .b2       (b2_in),
        .valid_out(mlp_out_valid),
        .done     (mlp_done),
        .y        (mlp_out)
    );

    // MLP control: process each token in sequence
    logic [31:0] token_idx;
    logic mlp_start_pulse;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            token_idx <= 0;
            mlp_valid_in <= 0;
            mlp_out_array <= '{default:0};
            mlp_start_pulse <= 0;
        end else begin
            if (curr_state == S_MLP || curr_state == S_WAIT_MLP) begin
                if (token_idx < SEQ_LEN) begin
                    mlp_valid_in <= 1;
                    // Assign current token's embedding
                    for (int i = 0; i < EMB_DIM; i++) begin
                        mlp_in[i] = ln2_out_array[token_idx*EMB_DIM + i];
                    end
                    // Pulse mlp_start for one cycle when mlp_out_valid is received or for first token
                    if ((token_idx == 0 && curr_state == S_MLP) || (mlp_out_valid && token_idx < SEQ_LEN)) begin
                        mlp_start_pulse <= 1;
                    end else begin
                        mlp_start_pulse <= 0;
                    end
                    // Store MLP output and increment token_idx
                    if (mlp_out_valid) begin
                        for (int i = 0; i < EMB_DIM; i++) begin
                            mlp_out_array[token_idx*EMB_DIM + i] = mlp_out[i];
                        end
                        token_idx <= token_idx + 1;
                    end
                end else begin
                    mlp_valid_in <= 0;
                    mlp_start_pulse <= 0;
                end
            end else begin
                token_idx <= 0;
                mlp_valid_in <= 0;
                mlp_start_pulse <= 0;
            end
        end
    end

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
        .sub_in   (mlp_out_flat),
        .y_out    (res2_out),
        .out_valid(res2_out_valid)
    );

    //***************************************************************
    // 4) FSM for Orchestrating LN1 -> ATT -> RES1 -> LN2 -> MLP -> RES2
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
        if (!rst_n)
            curr_state <= S_IDLE;
        else
            curr_state <= next_state;
    end

    // Next-state logic
    always_comb begin
        next_state = curr_state;
        case (curr_state)
            S_IDLE: begin
                if (start)       next_state = S_LN1;
            end
            // LN1
            S_LN1: begin
                next_state = S_WAIT_LN1;
            end
            S_WAIT_LN1: begin
                if (ln1_done)    next_state = S_ATT;
            end
            // Self-Attention
            S_ATT: begin
                next_state = S_WAIT_ATT;
            end
            S_WAIT_ATT: begin
                if (attn_done)   next_state = S_RES1;
            end
            // Residual #1
            S_RES1: begin
                next_state = S_WAIT_RES1;
            end
            S_WAIT_RES1: begin
                if (res1_done)   next_state = S_LN2;
            end
            // LN2
            S_LN2: begin
                next_state = S_WAIT_LN2;
            end
            S_WAIT_LN2: begin
                if (ln2_done)    next_state = S_MLP;
            end
            // MLP
            S_MLP: begin
                next_state = S_WAIT_MLP;
            end
            S_WAIT_MLP: begin
                if (token_idx == SEQ_LEN && mlp_done) next_state = S_RES2;
            end
            // Residual #2
            S_RES2: begin
                next_state = S_WAIT_RES2;
            end
            S_WAIT_RES2: begin
                if (res2_done)   next_state = S_DONE;
            end
            S_DONE: begin
                next_state = S_IDLE;
            end
            default: begin
                next_state = S_IDLE;
            end
        endcase
    end

    //***************************************************************
    // 5) Output & Control Logic
    //***************************************************************
    always_comb begin
        // Default
        done       = 1'b0;
        out_valid  = 1'b0;
        // Submodule starts default
        ln1_start  = 1'b0;
        attn_start = 1'b0;
        res1_start = 1'b0;
        ln2_start  = 1'b0;
        mlp_start  = 1'b0;
        res2_start = 1'b0;

        case (curr_state)
            S_LN1:   ln1_start  = 1'b1;
            S_ATT:   attn_start = 1'b1;
            S_RES1:  res1_start = 1'b1;
            S_LN2:   ln2_start  = 1'b1;
            S_MLP:   mlp_start  = mlp_start_pulse;
            S_WAIT_MLP: mlp_start = mlp_start_pulse;
            S_RES2:  res2_start = 1'b1;
            S_DONE: begin
                done      = 1'b1;
                out_valid = 1'b1;
            end
            default: begin
                // do nothing
            end
        endcase
    end

    //***************************************************************
    // 6) Final Output
    //***************************************************************
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < SEQ_LEN*EMB_DIM; i++)
                out_block[i] <= '0;
        end else if (curr_state == S_WAIT_RES2 && res2_done) begin
            for (int i = 0; i < SEQ_LEN*EMB_DIM; i++) begin
                out_block[i] <= res2_out[i*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    end

endmodule