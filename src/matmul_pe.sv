//------------------------------------------------------------------------------
// matmul_pe.sv
//
// One processing element of the matmul systolic array. Uses the mul16 component
// to compute the 16x16 bit multiplication in 4 cycles.
//
// May 9 2025    Tianwei Liu    Initial version
//------------------------------------------------------------------------------
module matmul_pe (
    input  logic        clk,          // System clock
    input  logic        rst_n,        // Active-low reset
    input  logic        reset_acc,    // Signal to reset accumulator
    input  logic        a_valid_in,   // Valid signal for a_in from left
    input  logic [15:0] a_in,         // 16-bit input from left
    input  logic        b_valid_in,   // Valid signal for b_in from top
    input  logic [15:0] b_in,         // 16-bit input from top
    output logic        a_valid_out,  // Valid signal to right
    output logic [15:0] a_out,        // 16-bit output to right
    output logic        b_valid_out,  // Valid signal to bottom
    output logic [15:0] b_out,        // 16-bit output to bottom
    output logic [31:0] c_out         // Accumulated result
);

    // Register inputs and pass to outputs
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_valid_out <= 0;
            a_out       <= 0;
            b_valid_out <= 0;
            b_out       <= 0;
        end else begin
            a_valid_out <= a_valid_in;
            a_out       <= a_in;
            b_valid_out <= b_valid_in;
            b_out       <= b_in;
        end
    end

    // Multiplier instantiation
    logic        out16_valid;
    logic [31:0] p16;
    mul16_progressive mul (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(a_valid_in & b_valid_in),
        .a(a_in),
        .b(b_in),
        .out4_valid(),  // Not used
        .p4(),          // Not used
        .out8_valid(),  // Not used
        .p8(),          // Not used
        .out16_valid(out16_valid),
        .p16(p16)
    );

    // Accumulator
    logic [31:0] acc;
    always_ff @(posedge clk) begin
        if (!rst_n || reset_acc) begin
            acc <= 0;
        end else if (out16_valid) begin
            acc <= acc + p16;
        end
    end

    assign c_out = acc;

endmodule