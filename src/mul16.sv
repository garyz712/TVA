//------------------------------------------------------------------------------
// mul16.sv
//
// Signed Q0.15 (Q0.3, Q0.8) multiplier brute force version
// Not optimized for efficiency AT ALL. Just use for test.
//
// May 10 2025    Tianwei Liu    Initial version
// May 11 2025    Tianwei Liu    Fix fixed-point multiplication
// May 11 2025    Tianwei Liu    Add clock division
// May 12 2025    Tianwei Liu    Fix clock division alighment
// May 18 2025    Tianwei Liu    Make pipelined
//------------------------------------------------------------------------------
module mul16_progressive (
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

    // Function for 16x4 multiplication (b is unsigned)
    function automatic logic signed [19:0] fp_mul_unsigned
            (input logic signed [15:0] a,
             input logic [3:0] b); // Treat b as unsigned

        logic signed [19:0] result;
        logic signed [19:0] expand_a;
        begin
            result = '0;
            expand_a = {{4{a[15]}}, a}; // Sign-extend a to 20 bits
            for (int i = 0; i < 4; ++i) begin
                if (b[i] == 1) begin
                    result += (expand_a << i);
                end
            end
            fp_mul_unsigned = result;
        end
    endfunction

    function automatic logic signed [19:0] fp_mul
            (input logic signed [15:0] a,
             input logic signed [3:0] b); // Treat b as signed

        logic signed [19:0] result;
        logic signed [19:0] expand_a;
        logic signed [19:0] expand_a_tc;
        begin
            result = '0;
            expand_a = {{4{a[15]}}, a}; // Sign-extend a to 20 bits
            expand_a_tc = ~expand_a + 1;
            for (int i = 0; i < 4; ++i) begin
                if (b[i] == 1) begin
                    result += i == 3 ? (expand_a_tc << i) : (expand_a << i);
                end
            end
            fp_mul = result;
        end
    endfunction

    // Internal registers for pipelining
    logic [15:0] a_reg1, a_reg2, a_reg3, a_reg4;
    logic [15:0] b_reg1, b_reg2, b_reg3, b_reg4;
    logic [4:0]  valid_reg;

    // Partial product registers
    logic signed [19:0] pp1, pp1s, pp2, pp2s, pp3, pp4; // 16x4 multiplication results
    logic signed [31:0] accum1, accum2, accum3, accum4;
    logic signed [31:0] result1, result2;


    // Stage 1: First 16x4 multiplication (b[3:0])
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_reg1 <= 0;
            b_reg1 <= 0;
            valid_reg[0] <= 0;
            pp1 <= 0;
            accum1 <= 0;
        end else begin
            a_reg1 <= a;
            b_reg1 <= b;
            valid_reg[0] <= valid_in;
            
            // Compute partial product: a * b[3:0] (b unsigned)
            pp1 <= fp_mul_unsigned(a, b[3:0]);
            pp1s <= fp_mul(a, b[3:0]);
            accum1 <= $signed(pp1); // Store first partial product
            result1 <= $signed(pp1s);
        end
    end

    // Q1.6 output (truncate to 1 integer, 6 fractional bits)
    // For stage 1, we approximate without sign correction (partial result)
    assign q1_6_out = result1[7:0];
    assign q1_6_valid = valid_reg[1];

    // Stage 2: Second 16x4 multiplication (b[7:4]) and accumulation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_reg2 <= 0;
            b_reg2 <= 0;
            valid_reg[1] <= 0;
            pp2 <= 0;
            accum2 <= 0;
        end else begin
            a_reg2 <= a_reg1;
            b_reg2 <= b_reg1;
            valid_reg[1] <= valid_reg[0];
            
            // Compute partial product: a * b[7:4], shift left by 4
            pp2 <= fp_mul_unsigned(a_reg1, b_reg1[7:4]);
            pp2s <= fp_mul(a_reg1, b_reg1[7:4]);
            accum2 <= $signed(accum1) + ($signed(pp2) << 4);
            result2 <= $signed(accum1) + ($signed(pp2s) << 4);
        end
    end

    // Q1.14 output (truncate to 1 integer, 14 fractional bits)
    // Still approximate, as sign correction is applied later
    assign q1_14_out = result2[15:0]; // Bit 18 to 3 (14 fractional)
    assign q1_14_valid = valid_reg[2];

    // Stage 3: Third 16x4 multiplication (b[11:8]) and accumulation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_reg3 <= 0;
            b_reg3 <= 0;
            valid_reg[2] <= 0;
            pp3 <= 0;
            accum3 <= 0;
        end else begin
            a_reg3 <= a_reg2;
            b_reg3 <= b_reg2;
            valid_reg[2] <= valid_reg[1];
            
            // Compute partial product: a * b[11:8], shift left by 8
            pp3 <= fp_mul_unsigned(a_reg2, b_reg2[11:8]);
            accum3 <= $signed(accum2) + ($signed(pp3) << 8);
        end
    end

    // Stage 4: Fourth 16x4 multiplication (b[15:12]) and final accumulation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_reg4 <= 0;
            b_reg4 <= 0;
            valid_reg[3] <= 0;
            valid_reg[4] <= 0;
            pp4 <= 0;
            accum4 <= 0;
        end else begin
            a_reg4 <= a_reg3;
            b_reg4 <= b_reg3;
            valid_reg[3] <= valid_reg[2];
            valid_reg[4] <= valid_reg[3];
            
            // Compute partial product: a * b[15:12], shift left by 12
            pp4 <= fp_mul(a_reg3, b_reg3[15:12]);
            accum4 <= $signed(accum3) + ($signed(pp4) << 12);
        end

    end


    // Q1.30 output (1 integer, 30 fractional bits)
    assign q1_30_out = accum4; // Full 32-bit result
    assign q1_30_valid = valid_reg[4];

endmodule