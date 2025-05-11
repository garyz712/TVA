//------------------------------------------------------------------------------
// mul16_bf.sv
//
// Signed Q0.15 (Q0.3, Q0.8) multiplier brute force version
// Not optimized for efficiency AT ALL. Just use for test.
//
// May 10 2025    Tianwei Liu    Initial version
//------------------------------------------------------------------------------
module mul16 (
    input  logic         clk,
    input  logic         rst_n,
    input  logic [15:0]  a,            // Signed Q0.15 input
    input  logic [15:0]  b,            // Signed Q0.15 input
    input  logic         valid_in,

    output logic [7:0]   q1_6_out,     // Q1.6 output (1 integer, 6 fractional)
    output logic         q1_6_valid,
    output logic [15:0]  q1_14_out,    // Q1.14 output (1 integer, 14 fractional)
    output logic         q1_14_valid,
    output logic [31:0]  q1_30_out,    // Q1.30 output (1 integer, 30 fractional)
    output logic         q1_30_valid
);

    // Internal signals
    logic [31:0]  product_full;        // Full 32-bit product (Q1.30 format)
    logic [31:0]  product_reg1, product_reg2, product_reg3; // Pipeline registers
    logic         valid_reg1, valid_reg2, valid_reg3, valid_reg4; // Valid signal pipeline

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
            product_full   <= $signed(a) * $signed(b); // Q0.15 * Q0.15 = Q1.30
            product_reg1   <= product_full;
            valid_reg1     <= valid_in;

            // Q1.6 output: Take upper 8 bits (1 integer, 6 fractional, 1 sign)
            // From Q1.30, select bits [30:23] for Q1.6 (shifted to get 6 fractional bits)
            q1_6_out       <= product_full[30:23];
            q1_6_valid     <= valid_in;
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

            // Q1.14 output: Take upper 16 bits (1 integer, 14 fractional, 1 sign)
            // From Q1.30, select bits [30:15] for Q1.14
            q1_14_out      <= product_reg1[30:15];
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
            q1_30_out      <= product_reg3;
            q1_30_valid    <= valid_reg3;
            valid_reg4     <= valid_reg3;
        end
    end

endmodule