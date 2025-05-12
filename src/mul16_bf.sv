//------------------------------------------------------------------------------
// mul16_bf.sv
//
// Signed Q0.15 (Q0.3, Q0.8) multiplier brute force version
// Not optimized for efficiency AT ALL. Just use for test.
//
// May 10 2025    Tianwei Liu    Initial version
//------------------------------------------------------------------------------
module mul16_progressive #(parameter int WIDTH = 16)(
    input  logic         clk,
    input  logic         rst_n,
    input  logic [15:0]  a,            // Signed Q0.15 input
    input  logic [15:0]  b,            // Signed Q0.15 input
    input  logic         in_valid,

    output logic [7:0]   q1_6_out,     // Q1.6 output (1 integer, 6 fractional)
    output logic         q1_6_valid,
    output logic [15:0]  q1_14_out,    // Q1.14 output (1 integer, 14 fractional)
    output logic         q1_14_valid,
    output logic [31:0]  q1_30_out,    // Q1.30 output (1 integer, 30 fractional)
    output logic         q1_30_valid
);

    // Internal signals
    logic [31:0]  product_full;        // Full 32-bit product (Q1.30 format)
    logic [7:0]   product_comb4;        // Full 32-bit product (Q1.30 format)
    logic [15:0]  product_comb8;        // Full 32-bit product (Q1.30 format)
    logic [31:0]  product_comb16;        // Full 32-bit product (Q1.30 format)
    logic [31:0]  product_reg1, product_reg2, product_reg3; // Pipeline registers
    logic         valid_reg1, valid_reg2, valid_reg3, valid_reg4; // Valid signal pipeline

    logic [7:0] expand_a4;
    logic [7:0] expand_a_tc4;
    logic [15:0] expand_a8;
    logic [15:0] expand_a_tc8;
    logic [31:0] expand_a16;
    logic [31:0] expand_a_tc16;

    always_comb begin
        product_comb4 = '0;
        expand_a4     = a[7:0];
        expand_a_tc4  = ~expand_a4 + 1;
        for (int i = 0; i < 4; ++i) begin
            if (i != 3 && b[i] == 1) begin
                product_comb4 += (expand_a4 << i);
            end
            if (i == 3 && b[i] == 1) begin
                product_comb4 += (expand_a_tc4 << i);
            end
        end
    end

    always_comb begin
        product_comb8 = '0;
        expand_a8     = a;
        expand_a_tc8  = ~expand_a8 +1;
        for (int i = 0; i < 8; ++i) begin
            if (i != 7 && b[i] == 1) begin
                product_comb8 += (expand_a8 << i);
            end
            if (i == 7 && b[i] == 1) begin
                product_comb8 += (expand_a_tc8 << i);
            end
        end
    end

    always_comb begin
        product_comb16 = '0;
        expand_a16     = {{16{a[15]}}, a};
        expand_a_tc16  = ~expand_a16 + 1;
        for (int i = 0; i < 16; ++i) begin
            if (i != 15 && b[i] == 1) begin
                product_comb16 += (expand_a16 << i);
            end
            if (i == 15 && b[i] == 1) begin
                product_comb16 += (expand_a_tc16 << i);
            end
        end
    end


    // Stage 1: Compute full product (Q1.30) and Q1.6 output
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            product_full   <= 32'b0;
            product_reg1   <= 32'b0;
            q1_6_out       <= 8'b0;
            q1_6_valid     <= 1'b0;
            valid_reg1     <= 1'b0;
        end else begin
            // Compute 16-bit x 16-bit signed multiplication
            product_full   <= {{24{product_comb4[7]}}, product_comb4};
            product_reg1   <= product_full;
            valid_reg1     <= in_valid;

            // Q1.6 output: Take lower 8 bits (1 integer, 6 fractional, 1 sign)
            q1_6_out       <= product_comb4;
            q1_6_valid     <= in_valid;
        end
    end

    // Stage 2: Q1.14 output
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            product_reg2   <= 32'b0;
            q1_14_out      <= 16'b0;
            q1_14_valid    <= 1'b0;
            valid_reg2     <= 1'b0;
        end else begin
            product_reg2   <= product_reg1;
            valid_reg2     <= valid_reg1;

            // Q1.14 output: Take lower 16 bits (1 integer, 14 fractional, 1 sign)
            q1_14_out      <= product_comb8;
            q1_14_valid    <= valid_reg1;
        end
    end

    // Stage 3: Pipeline for Q1.30
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            product_reg3   <= 32'b0;
            valid_reg3     <= 1'b0;
        end else begin
            product_reg3   <= product_reg2;
            valid_reg3     <= valid_reg2;
        end
    end

    // Stage 4: Q1.30 output
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q1_30_out      <= 32'b0;
            q1_30_valid    <= 1'b0;
            valid_reg4     <= 1'b0;
        end else begin
            // Q1.30 output: Full product
            q1_30_out      <= product_comb16;
            q1_30_valid    <= valid_reg3;
            valid_reg4     <= valid_reg3;
        end
    end

endmodule
