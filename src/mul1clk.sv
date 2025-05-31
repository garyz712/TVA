//------------------------------------------------------------------------------
// mul1clk.sv
//
// Signed Q0.15 multiplier in 1 clock
//
// May 31 2025    Tianwei Liu    Initial version
//------------------------------------------------------------------------------
module mul1clk (
    input  logic         clk,
    input  logic         rst_n,
    input  logic [15:0]  a,            // Signed Q0.15 input
    input  logic [15:0]  b,            // Signed Q0.15 input
    input  logic         valid_in,

    output logic [31:0]  q1_30_out,    // Q1.30 output (1 integer, 30 fractional)
    output logic         q1_30_valid
);

    // Function for 16x16 multiplication
    function automatic logic signed [31:0] fp_mul
            (input logic signed [15:0] a,
             input logic [15:0] b); // Treat b as unsigned

        logic [31:0] expand_a;
        logic [31:0] expand_a_tc;
        begin
            fp_mul = '0;
            expand_a     = {{16{a[15]}}, a};
            expand_a_tc  = ~expand_a + 1;
            for (int i = 0; i < 16; ++i) begin
                if (i != 15 && b[i] == 1) begin
                    fp_mul += (expand_a << i);
                end
                if (i == 15 && b[i] == 1) begin
                    fp_mul += (expand_a_tc << i);
                end
            end
        end
    endfunction

    logic [1:0] valid_reg;
    logic [31:0] pp1s, result1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_reg[0] <= 0;
            result1 <= 0;
            pp1s <= 0;
        end else begin
            valid_reg[0] <= valid_in;
            valid_reg[1] <= valid_reg[0];
            
            // Compute partial product: a * b[3:0] (b unsigned)
            pp1s <= fp_mul(a, b);
            result1 <= $signed(pp1s);
        end
    end

    // Q1.30 output
    assign q1_30_out = result1;
    assign q1_30_valid = valid_reg[1];

endmodule