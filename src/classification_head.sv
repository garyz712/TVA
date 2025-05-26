//------------------------------------------------------------------------------
// classification_head.sv
//
// Implements a linear classification layer for neural networks in Q1.15 format.
// Performs: global_avg_pool -> linear(embedding * weights + bias)
// 
// May 02 2025    Max Zhang      Initial version
// May 26 2025    Tianwei Liu    bug fix
//------------------------------------------------------------------------------
module classification_head #(
    parameter int DATA_WIDTH   = 16,
    parameter int E            = 64,
    parameter int N            = 16,
    parameter int NUM_CLASSES  = 1
)(
    input  logic                                   clk,
    input  logic                                   rst_n,
    input  logic                                   start,
    input  logic signed [DATA_WIDTH-1:0]           patch_emb_in [N*E-1:0],
    input  logic signed [DATA_WIDTH-1:0]           W_clf_in [E*NUM_CLASSES-1:0],
    input  logic signed [DATA_WIDTH-1:0]           b_clf_in [NUM_CLASSES-1:0],
    output logic signed [DATA_WIDTH-1:0]           logits_out [NUM_CLASSES-1:0],
    output logic                                   out_valid
);

    localparam int ACC_WIDTH = DATA_WIDTH + $clog2(N);  // To accumulate N values safely

    typedef logic signed [DATA_WIDTH-1:0] data_t;
    typedef logic signed [ACC_WIDTH-1:0] acc_t;
    typedef logic signed [2*DATA_WIDTH-1:0] mult_t;

    acc_t mean_emb [E-1:0];
    logic [$clog2(N):0] count;
    logic processing, done;

    // Stage 1: Compute mean embedding (sum over N patches and divide by N)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= 0;
            processing <= 0;
            for (int i = 0; i < E; i++) mean_emb[i] <= 0;
        end else if (start) begin
            count <= 0;
            processing <= 1;
            for (int i = 0; i < E; i++) mean_emb[i] <= 0;
        end else if (processing) begin
            if (count < N) begin
                for (int i = 0; i < E; i++) begin
                    mean_emb[i] <= mean_emb[i] + patch_emb_in[count * E + i];
                end
                count <= count + 1;
            end else begin
                processing <= 0;
                done <= 1;
            end
        end else begin
            done <= 0;
        end
    end

    // Stage 2: Compute logits = W^T * mean + b
    logic signed [2*DATA_WIDTH-1:0] dot_products [NUM_CLASSES-1:0];
    logic stage2_done;

    mult_t product;
    logic signed [DATA_WIDTH-1:0] mean_scaled;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int c = 0; c < NUM_CLASSES; c++) dot_products[c] <= 0;
            stage2_done <= 0;
        end else if (done) begin
            for (int c = 0; c < NUM_CLASSES; c++) begin
                dot_products[c] = 0;
                for (int i = 0; i < E; i++) begin
                    // Truncate accumulator to fixed-point format
                    mean_scaled = mean_emb[i] / N;
                    product = mean_scaled * W_clf_in[c * E + i];
                    dot_products[c] += product;
                end
                dot_products[c] += b_clf_in[c] <<< 15;
            end
            stage2_done <= 1;
        end else begin
            stage2_done <= 0;
        end
    end

    logic [15:0] result;

    // Stage 3: Output result
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int c = 0; c < NUM_CLASSES; c++) logits_out[c] <= 0;
            out_valid <= 0;
        end else if (stage2_done) begin
            for (int c = 0; c < NUM_CLASSES; c++) begin
                if (dot_products[c] > 32'sh3FFFFFFF)
                    result = 16'sh7FFF;
                else if (dot_products[c] < 32'shC0000000)
                    result = 16'sh8000;
                else
                    result = {dot_products[c][31], dot_products[c][29:15]};
                logits_out[c] <= result;  // truncate
            end
            out_valid <= 1;
        end else begin
            out_valid <= 0;
        end
    end

endmodule
