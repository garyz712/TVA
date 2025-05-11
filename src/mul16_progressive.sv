//------------------------------------------------------------------------------
// mul16_progressive.sv
//
// Signed 16-bit multiplier that treats inputs as Q1.15 (signed) and outputs
// Q0.3 (4 bits), Q0.7 (8 bits), and Q0.15 (16 bits) with the first bit as the sign.
//
// May 10 2025    Tianwei Liu    Modified for signed multiplication
//------------------------------------------------------------------------------
module mul16_progressive (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [15:0] a,          // Signed Q1.15 input
    input  logic [15:0] b,          // Signed Q1.15 input
    input  logic        valid_in,
    output logic [3:0]  q0_3_out,   // Q0.3: 1 sign bit + 3 fractional bits
    output logic        q0_3_valid,
    output logic [7:0]  q0_7_out,   // Q0.7: 1 sign bit + 7 fractional bits
    output logic        q0_7_valid,
    output logic [15:0] q0_15_out,  // Q0.15: 1 sign bit + 15 fractional bits
    output logic        q0_15_valid
);

    // Pipeline registers
    logic signed [19:0] partial_sum_stage1; // 20 bits for 16x4-bit partial product
    logic signed [23:0] partial_sum_stage2; // 24 bits for accumulation
    logic signed [27:0] partial_sum_stage3; // 28 bits for accumulation
    logic signed [31:0] partial_sum_stage4; // 32 bits for final product
    logic signed [15:0] a_reg_stage1, a_reg_stage2, a_reg_stage3;
    logic signed [11:0] b_reg_stage1;      // Store b[15:4]
    logic signed [7:0]  b_reg_stage2;      // Store b[15:8]
    logic signed [3:0]  b_reg_stage3;      // Store b[15:12]
    logic               valid_stage1, valid_stage2, valid_stage3, valid_stage4;

    // Stage 1: Compute partial product for b[3:0], output Q0.3
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            partial_sum_stage1 <= 20'sb0;
            a_reg_stage1       <= 16'sb0;
            b_reg_stage1       <= 12'sb0;
            valid_stage1       <= 1'b0;
            q0_3_out           <= 4'sb0;
            q0_3_valid         <= 1'b0;
        end else begin
            if (valid_in) begin
                partial_sum_stage1 <= $signed(a) * $signed(b[3:0]); // Signed 16x4-bit multiply
                a_reg_stage1       <= $signed(a);
                b_reg_stage1       <= $signed(b[15:4]); // Store upper bits
            end
            valid_stage1 <= valid_in;
            // Q0.3: sign bit + bits [6:4] for 3 fractional bits
            q0_3_out     <= {partial_sum_stage1[19], partial_sum_stage1[6:4]};
            q0_3_valid   <= valid_stage1;
        end
    end

    // Stage 2: Compute partial product for b[7:4], accumulate, output Q0.7
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            partial_sum_stage2 <= 24'sb0;
            a_reg_stage2       <= 16'sb0;
            b_reg_stage2       <= 8'sb0;
            valid_stage2       <= 1'b0;
            q0_7_out           <= 8'sb0;
            q0_7_valid         <= 1'b0;
        end else begin
            partial_sum_stage2 <= (partial_sum_stage1 <<< 4) + ($signed(a_reg_stage1) * $signed(b_reg_stage1[3:0]));
            a_reg_stage2       <= a_reg_stage1;
            b_reg_stage2       <= $signed(b_reg_stage1[11:4]);
            valid_stage2       <= valid_stage1;
            // Q0.7: sign bit + bits [10:4] for 7 fractional bits
            q0_7_out           <= {partial_sum_stage2[23], partial_sum_stage2[10:4]};
            q0_7_valid         <= valid_stage2;
        end
    end

    // Stage 3: Compute partial product for b[11:8], accumulate
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            partial_sum_stage3 <= 28'sb0;
            a_reg_stage3       <= 16'sb0;
            b_reg_stage3       <= 4'sb0;
            valid_stage3       <= 1'b0;
        end else begin
            partial_sum_stage3 <= (partial_sum_stage2 <<< 4) + ($signed(a_reg_stage2) * $signed(b_reg_stage2[3:0]));
            a_reg_stage3       <= a_reg_stage2;
            b_reg_stage3       <= $signed(b_reg_stage2[7:4]);
            valid_stage3       <= valid_stage2;
        end
    end

    // Stage 4: Compute partial product for b[15:12], accumulate, output Q0.15
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            partial_sum_stage4 <= 32'sb0;
            valid_stage4       <= 1'b0;
            q0_15_out          <= 16'sb0;
            q0_15_valid        <= 1'b0;
        end else begin
            partial_sum_stage4 <= (partial_sum_stage3 <<< 4) + ($signed(a_reg_stage3) * $signed(b_reg_stage3[3:0]));
            valid_stage4       <= valid_stage3;
            // Q0.15: sign bit + bits [18:4] for 15 fractional bits
            q0_15_out          <= {partial_sum_stage4[31], partial_sum_stage4[18:4]};
            q0_15_valid        <= valid_stage4;
        end
    end

endmodule
