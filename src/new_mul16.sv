//------------------------------------------------------------------------------
// new_mul16.sv
//
//
// new mul 16 that treats input as Q1.2, Q1.6, and Q1.14, and outputs
// the corresponding representation of numbers.
//
// May 10 2025    Tianwei Liu    Initial version
//------------------------------------------------------------------------------
module four_stage_multiplier (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [15:0] a,
    input  logic [15:0] b,
    input  logic        valid_in,
    output logic [3:0]  q1_2_out,
    output logic        q1_2_valid,
    output logic [7:0]  q1_6_out,
    output logic        q1_6_valid,
    output logic [15:0] q1_14_out,
    output logic        q1_14_valid
);

    // Pipeline registers
    logic [19:0] partial_sum_stage1; // 20 bits for 16x4-bit partial product
    logic [23:0] partial_sum_stage2; // 24 bits to accommodate shift and add
    logic [27:0] partial_sum_stage3; // 28 bits for further accumulation
    logic [31:0] partial_sum_stage4; // 32 bits for final product
    logic [15:0] a_reg_stage1, a_reg_stage2, a_reg_stage3;
    logic [11:0] b_reg_stage1;      // Store b[15:4] for next stages
    logic [7:0]  b_reg_stage2;      // Store b[15:8]
    logic [3:0]  b_reg_stage3;      // Store b[15:12]
    logic        valid_stage1, valid_stage2, valid_stage3, valid_stage4;

    // Stage 1: Compute partial product for b[3:0], output Q1.2
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            partial_sum_stage1 <= 20'b0;
            a_reg_stage1       <= 16'b0;
            b_reg_stage1       <= 12'b0;
            valid_stage1       <= 1'b0;
            q1_2_out           <= 4'b0;
            q1_2_valid         <= 1'b0;
        end else begin
            if (valid_in) begin
                partial_sum_stage1 <= a * b[3:0]; // 16x4-bit multiply
                a_reg_stage1       <= a;
                b_reg_stage1       <= b[15:4]; // Store upper bits
            end
            valid_stage1 <= valid_in;
            q1_2_out     <= partial_sum_stage1[7:4]; // Q1.2: bits [7:4]
            q1_2_valid   <= valid_stage1;
        end
    end

    // Stage 2: Compute partial product for b[7:4], accumulate, output Q1.6
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            partial_sum_stage2 <= 24'b0;
            a_reg_stage2       <= 16'b0;
            b_reg_stage2       <= 8'b0;
            valid_stage2       <= 1'b0;
            q1_6_out           <= 8'b0;
            q1_6_valid         <= 1'b0;
        end else begin
            partial_sum_stage2 <= (partial_sum_stage1 << 4) + (a_reg_stage1 * b_reg_stage1[3:0]);
            a_reg_stage2       <= a_reg_stage1;
            b_reg_stage2       <= b_reg_stage1[11:4];
            valid_stage2       <= valid_stage1;
            q1_6_out           <= partial_sum_stage2[11:4]; // Q1.6: bits [11:4]
            q1_6_valid         <= valid_stage2;
        end
    end

    // Stage 3: Compute partial product for b[11:8], accumulate
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            partial_sum_stage3 <= 28'b0;
            a_reg_stage3       <= 16'b0;
            b_reg_stage3       <= 4'b0;
            valid_stage3       <= 1'b0;
        end else begin
            partial_sum_stage3 <= (partial_sum_stage2 << 4) + (a_reg_stage2 * b_reg_stage2[3:0]);
            a_reg_stage3       <= a_reg_stage2;
            b_reg_stage3       <= b_reg_stage2[7:4];
            valid_stage3       <= valid_stage2;
        end
    end

    // Stage 4: Compute partial product for b[15:12], accumulate, output Q1.14
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            partial_sum_stage4 <= 32'b0;
            valid_stage4       <= 1'b0;
            q1_14_out          <= 16'b0;
            q1_14_valid        <= 1'b0;
        end else begin
            partial_sum_stage4 <= (partial_sum_stage3 << 4) + (a_reg_stage3 * b_reg_stage3[3:0]);
            valid_stage4       <= valid_stage3;
            q1_14_out          <= partial_sum_stage4[19:4]; // Q1.14: bits [19:4]
            q1_14_valid        <= valid_stage4;
        end
    end

endmodule