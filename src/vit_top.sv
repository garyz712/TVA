//------------------------------------------------------------------------------
// vit_top.sv
//
// Top-level SystemVerilog module for Vision Transformer inference.
// Implements a minimal ViT architecture using the following pipeline:
//     - Patch Embedding
//     - Positional Encoding
//     - Stacked Transformer Encoder Blocks (time-multiplexed)
//     - Classification Head
// Supports external loading of all trained weights and positional tables.
//     - Time-multiplexes a single encoder block across NUM_LAYERS passes.
//     - Designed for simple pipeline control via internal FSM.
//     - Requires weight input or pre-storage mechanism for full model.
//
// Apr. 16 2025    Max Zhang      Initial version
// Apr. 21 2025    Tianwei Liu    Syntax fix and comments
//------------------------------------------------------------------------------
module vit_top #(
    parameter int DATA_WIDTH   = 16,
    parameter int IMG_H        = 224,
    parameter int IMG_W        = 224,
    parameter int C            = 3,
    parameter int PH           = 16,
    parameter int PW           = 16,
    parameter int E            = 128,   // embedding dimension
    parameter int NUM_LAYERS   = 12,    // number of encoder blocks
    parameter int NUM_CLASSES  = 1000
)(
    input  logic                                             clk,
    input  logic                                             rst_n,

    // Control
    input  logic                                             start,
    output logic                                             done,

    // Input image: shape (IMG_H, IMG_W, C)
    // Flattened => DATA_WIDTH*(IMG_H*IMG_W*C)
    input  logic [DATA_WIDTH*IMG_H*IMG_W*C -1:0]             image_in,

    // Learned positional table:
    // shape (L, E) => L = number_of_patches, E=embedding dim
    // Flatten => DATA_WIDTH*(L*E)
    input  logic [DATA_WIDTH*((IMG_H/PH)*(IMG_W/PW))*E -1:0] pos_table_in,

    // Weights for each submodule:
    // (A) patch_embedding => W_patch_in, b_patch_in
    input  logic [DATA_WIDTH*(PH*PW*C*E) -1:0]               W_patch_in,
    input  logic [DATA_WIDTH*E -1:0]                         b_patch_in,

    // (B) For each of the NUM_LAYERS encoder blocks:
    //     LN1: gamma, beta
    //     LN2: gamma, beta
    //     Q,K,V weights, MLP weights, etc.
    // For simplicity, we might store them in arrays or pass them in a big bus.
    // We'll just define placeholders to show structure – in a real design, you need
    // a large bus or external memory to feed these. Or you’d store them in a memory.

    // We'll demonstrate for ONE layer's weights. If you truly have 12 layers,
    // you'd replicate or store them in an array. 
    // ***Skeleton:***
    // LN1 gamma, LN1 beta, LN2 gamma, LN2 beta
    // Q,K,V => each (E*E), bQ,bK,bV => each (E)
    // MLP => W1 (E*4E?), b1 (4E?), W2 (4E?*E), b2(E?), etc.

    // (C) Classification Head
    // W_clf: shape (E x NUM_CLASSES), b_clf: shape (NUM_CLASSES)
    input  logic [DATA_WIDTH*E*NUM_CLASSES -1:0]             W_clf_in,
    input  logic [DATA_WIDTH*NUM_CLASSES -1:0]               b_clf_in,

    // Final output => shape (NUM_CLASSES)
    output logic [DATA_WIDTH*NUM_CLASSES -1:0]               logits_out,
    output logic                                             logits_valid
);

    //****************************************************************
    // 1) Derived Parameter: L = number_of_patches
    //****************************************************************
    localparam int L = (IMG_H/PH)*(IMG_W/PW);

    //****************************************************************
    // 2) Wires to Connect Each Submodule
    //****************************************************************

    // Patch embedding
    logic patch_start, patch_done, patch_out_valid;
    logic [DATA_WIDTH*L*E -1:0] patch_out;  // shape (L, E)

    // Positional encoding
    logic pos_start, pos_done, pos_out_valid;
    logic [DATA_WIDTH*L*E -1:0] embed_out; // shape (L, E) = patch_out + pos_table

    // We'll do an array of "encoder_in" and "encoder_out" if we time-multiplex the same block
    // OR we'll do a single submodule repeated in a loop if we physically replicate it.
    // For a minimal skeleton, we do a single instance and feed it multiple times in an FSM loop
    // for each layer. That's typical if we can't replicate 12 copies in hardware.

    // We'll store the output of each layer in some reg.

    // For classification
    logic class_start, class_done;
    logic [DATA_WIDTH*NUM_CLASSES -1:0] class_logits;
    logic                               class_valid;

    //****************************************************************
    // 3) Submodule Instantiations
    //****************************************************************

    // (A) Patch Embedding
    patch_embedding #(
        .DATA_WIDTH(DATA_WIDTH),
        .IMG_H     (IMG_H),
        .IMG_W     (IMG_W),
        .C         (C),
        .PH        (PH),
        .PW        (PW),
        .E         (E)
    ) u_patch_embed (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (patch_start),
        .done       (patch_done),
        .image_in   (image_in),
        .W_patch_in (W_patch_in),
        .b_patch_in (b_patch_in),
        // Output => shape (L,E)
        .patch_out  (patch_out),
        .out_valid  (patch_out_valid)
    );

    // (B) Positional Encoding
    positional_encoding #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_TOKENS(L),
        .E         (E)
    ) u_pos_enc (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (pos_start),
        .done      (pos_done),
        // Input => patch_out, pos_table_in
        .A_in      (patch_out),
        .pos_table_in(pos_table_in),
        .out_embed (embed_out),
        .out_valid (pos_out_valid)
    );

    // (C) We'll have ONE `vit_encoder_block` for time-multiplexing multiple layers
    // inside an FSM loop. So we'll define:
    logic enc_start, enc_done, enc_out_valid;
    logic [DATA_WIDTH*L*E -1:0] enc_in;
    logic [DATA_WIDTH*L*E -1:0] enc_out;

    vit_encoder_block #(
        .DATA_WIDTH(DATA_WIDTH),
        .SEQ_LEN   (L),
        .EMB_DIM   (E),
        .H_MLP     (4*E) // typically 4*E hidden dimension
    ) u_enc_block (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (enc_start),
        .done      (enc_done),
        .x_in      (enc_in),

        // LN1 gamma/beta, LN2 gamma/beta, QKV, MLP weights...
        // For real design, you'd pass the correct layer's weights here.
        // We'll omit them or show placeholders:
        .ln1_gamma (/* ln1_gamma_of_current_layer */),
        .ln1_beta  (/* ln1_beta_of_current_layer  */),
        .ln2_gamma (/* ln2_gamma_of_current_layer */),
        .ln2_beta  (/* ln2_beta_of_current_layer  */),

        .WQ_in     (/* WQ_of_current_layer */),
        .WK_in     (/* WK_of_current_layer */),
        .WV_in     (/* WV_of_current_layer */),
        .bQ_in     (/* bQ_of_current_layer */),
        .bK_in     (/* bK_of_current_layer */),
        .bV_in     (/* bV_of_current_layer */),

        // MLP
        .W1_in     (/* W1_of_current_layer */),
        .b1_in     (/* b1_of_current_layer */),
        .W2_in     (/* W2_of_current_layer */),
        .b2_in     (/* b2_of_current_layer */),

        .out_block (enc_out),
        .out_valid (enc_out_valid)
    );

    // (D) Classification Head
    classification_head #(
        .DATA_WIDTH (DATA_WIDTH),
        .E          (E),
        .NUM_CLASSES(NUM_CLASSES)
    ) u_class_head (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (class_start),
        .done      (class_done),
        // Typically we feed the "CLS token" from enc_out. 
        // But if we didn't explicitly add a CLS token, let's assume we take e.g. enc_out[0].
        // We'll pass the entire enc_out if the submodule is designed that way, or just a slice.
        .cls_in    (/* something like enc_out_of_first_token */),
        .W_clf_in  (W_clf_in),
        .b_clf_in  (b_clf_in),
        .logits_out(class_logits),
        .out_valid (class_valid)
    );

    // We'll store final "logits_out" from class_head => "logits_out"

    //****************************************************************
    // 4) Top-Level FSM for Orchestrating the Pipeline
    //****************************************************************

    typedef enum logic [3:0] {
        S_IDLE,
        S_PATCH,
        S_WAIT_PATCH,
        S_POSENC,
        S_WAIT_POSENC,
        S_LAYER_START,
        S_WAIT_LAYER,
        S_LAYER_CHECK,
        S_CLASS,
        S_WAIT_CLASS,
        S_DONE
    } state_t;

    state_t curr_state, next_state;
    // We'll keep a layer counter to do multiple repeated passes through `u_enc_block`.
    logic [$clog2(NUM_LAYERS):0] layer_idx;

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
            S_IDLE: begin
                if(start)
                    next_state = S_PATCH;
            end

            // 1) Patch embedding
            S_PATCH: begin
                next_state = S_WAIT_PATCH;
            end
            S_WAIT_PATCH: begin
                if(patch_done)
                    next_state = S_POSENC;
            end

            // 2) Positional Encoding
            S_POSENC: begin
                next_state = S_WAIT_POSENC;
            end
            S_WAIT_POSENC: begin
                if(pos_done)
                    next_state = S_LAYER_START;
            end

            // 3) Repeated encoder blocks
            S_LAYER_START: begin
                next_state = S_WAIT_LAYER;
            end
            S_WAIT_LAYER: begin
                if(enc_done)
                    next_state = S_LAYER_CHECK;
            end
            S_LAYER_CHECK: begin
                // if we have more layers to do => next layer
                // else => classification
                if(layer_idx == (NUM_LAYERS-1))
                    next_state = S_CLASS;
                else
                    next_state = S_LAYER_START;
            end

            // 4) Classification
            S_CLASS: begin
                next_state = S_WAIT_CLASS;
            end
            S_WAIT_CLASS: begin
                if(class_done)
                    next_state = S_DONE;
            end

            S_DONE: next_state = S_IDLE;

            default: next_state = S_IDLE;
        endcase
    end

    //****************************************************************
    // 5) Output & Control Logic
    //****************************************************************
    always_comb begin
        // defaults
        done          = 1'b0;
        logits_valid  = 1'b0;

        patch_start   = 1'b0;
        pos_start     = 1'b0;
        enc_start     = 1'b0;
        class_start   = 1'b0;

        // submodule inputs
        enc_in        = embed_out;  // or the previous layer’s output, if layering
        // In reality, after the first layer finishes, we feed its output back into enc_in.

        case(curr_state)
            S_PATCH:    patch_start = 1'b1;
            S_POSENC:   pos_start   = 1'b1;
            S_LAYER_START: enc_start= 1'b1;
            S_CLASS:    class_start = 1'b1;

            S_DONE: begin
                done         = 1'b1;
                logits_valid = 1'b1;
            end
            default: /* no-op */
            begin end
        endcase
    end

    //****************************************************************
    // 6) Sequencing Data from Layer to Layer
    //****************************************************************
    // We need to hold the output from the current layer
    // so we can feed it to the next layer. We'll do this in always_ff
    // with a "layer_idx" counter.

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            layer_idx <= 0;
            // We also store the block output in a reg or memory
        end else begin
            case(curr_state)
                S_IDLE: begin
                    layer_idx <= 0;
                end

                // Once an encoder block finishes:
                S_WAIT_LAYER: if(enc_done) begin
                    // store enc_out if needed
                end

                S_LAYER_CHECK: begin
                    if(layer_idx < (NUM_LAYERS-1))
                        layer_idx <= layer_idx + 1;
                end

                // once we’re done, go to classification
            endcase
        end
    end

    // For a minimal skeleton, let’s show how we pass the data:
    //  - After pos_enc, we store that in a reg "block_input_reg"
    //  - Then each time we finish an encoder block, we store "enc_out" back
    //    into "block_input_reg" if we have more layers.
    //  - At the end, we start classification with "cls_in" from "block_input_reg".


    //****************************************************************
    // 7) Final Classification Output
    //****************************************************************
    // "u_class_head" yields "class_logits", "class_valid" when done
    // We connect those to "logits_out" and "logits_valid".
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            logits_out <= '0;
        end else if(curr_state == S_WAIT_CLASS && class_done) begin
            logits_out <= class_logits;
        end
    end

endmodule
