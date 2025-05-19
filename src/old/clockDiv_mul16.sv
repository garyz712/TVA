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

    // Internal signals
    logic [31:0]  product_comb;        // Full 32-bit product (Q1.30 format)
    logic [15:0]  product16_comb;      // 16-bit product
    logic [7:0]   product8_comb;       // 8-bit product
    // logic [31:0]  product_reg1, product_reg2, product_reg3; // Pipeline registers
    // logic         valid_reg1, valid_reg2, valid_reg3, valid_reg4; // Valid signal pipeline
    logic valid_1;
    logic valid_2;
    logic valid_4;

    logic [31:0] expand_a;
    logic [31:0] expand_a_tc;
    logic [15:0] expand_a16;
    logic [15:0] expand_a_tc16;
    logic [7:0]  expand_a8;
    logic [7:0]  expand_a_tc8;

    // clock division
    logic clk2, clk4;
    logic [1:0] div_counter;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            clk2        <= 0;
            clk4        <= 0;
            div_counter <= '0;
        end else begin
            div_counter <= div_counter + 1;
            clk2        <= div_counter[0];
            clk4        <= ~div_counter[1];
        end
    end

    // comb logic to compute 8 bit output (1 cycle)
    always_comb begin
        product8_comb = '0;
        expand_a8     = {{4{a[3]}}, a[3:0]};
        expand_a_tc8  = ~expand_a8 + 1;
        for (int i = 0; i < 4; ++i) begin
            if (i != 3 && b[i] == 1) begin
                product8_comb += (expand_a8 << i);
            end
            if (i == 3 && b[i] == 1) begin
                product8_comb += (expand_a_tc8 << i);
            end
        end
    end

    // comb logic to compute 16 bit output (2 cycle)
    always_comb begin
        product16_comb = '0;
        expand_a16     = {{8{a[7]}}, a[7:0]};
        expand_a_tc16  = ~expand_a16 + 1;
        for (int i = 0; i < 8; ++i) begin
            if (i != 7 && b[i] == 1) begin
                product16_comb += (expand_a16 << i);
            end
            if (i == 7 && b[i] == 1) begin
                product16_comb += (expand_a_tc16 << i);
            end
        end
    end

    // comb logic to compute 32 bit output (4 cycle)
    always_comb begin
        product_comb = '0;
        expand_a     = {{16{a[15]}}, a};
        expand_a_tc  = ~expand_a + 1;
        for (int i = 0; i < 16; ++i) begin
            if (i != 15 && b[i] == 1) begin
                product_comb += (expand_a << i);
            end
            if (i == 15 && b[i] == 1) begin
                product_comb += (expand_a_tc << i);
            end
        end
    end

    // Clock 1: Compute Q1.6 output
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q1_6_out       <= 8'b0;
            q1_6_valid     <= 1'b0;
        end else begin
            // Compute 16-bit x 16-bit signed multiplication
            // product_full   <= $signed(a) * $signed(b); // Q0.15 * Q0.15 = Q1.30

            // Q1.6 output: Take upper 8 bits (1 integer, 6 fractional, 1 sign)
            // From Q1.30, select bits [30:23] for Q1.6 (shifted to get 6 fractional bits)
            q1_6_out       <= product8_comb;
            q1_6_valid     <= valid_in;
        end
    end

    // Clock 2: Q1.14 output
    always_ff @(posedge clk2 or negedge rst_n) begin
        if (!rst_n) begin
            q1_14_out      <= 16'b0;
            q1_14_valid    <= 1'b0;
        end else begin

            // Q1.14 output: Take upper 16 bits (1 integer, 14 fractional, 1 sign)
            // From Q1.30, select bits [30:15] for Q1.14
            q1_14_out      <= product16_comb;
            q1_14_valid    <= valid_in;
        end
    end

    // Clock 4: Q1.30 output
    always_ff @(posedge clk4 or negedge rst_n) begin
        if (!rst_n) begin
            q1_30_out      <= 32'b0;
            q1_30_valid    <= 1'b0;
        end else begin
            // Q1.30 output: Full product
            q1_30_out      <= product_comb;
            q1_30_valid    <= valid_in;
        end
    end

endmodule
