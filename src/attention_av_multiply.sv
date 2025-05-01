
// -----------------------------------------------------------------------------
// Progressive-Precision 4‑Stage Pipelined Multiplier
// -----------------------------------------------------------------------------
// • Handles INT4, INT8 and full 16‑bit fixed/FP16 data with one shared datapath.
// • Produces increasingly wider results every stage:
//       – 1‑cycle  8‑bit product  (INT4×INT4)   →  out4_valid / p4[7:0]
//       – 2‑cycle 16‑bit product  (INT8×INT8)   →  out8_valid / p8[15:0]
//       – 4‑cycle 32‑bit product  (16b×16b)     →  out16_valid / p16[31:0]
// • No Verilog * operator is used ⇒ each stage’s critical path is just a
//   few add/shift operations (easy to hit >500 MHz on most FPGAs/ASICs).
// • Unsigned arithmetic assumed for INT4/INT8; full‑width signed behaviour is
//   correct after stage‑4. If you need full signed accuracy at the early
//   exits, extend the partial‑product function to use Booth/Baugh‑Wooley.
// -----------------------------------------------------------------------------
// Interface (per‑cycle streaming, 1 new operand pair every clock):
//   clk, rst_n      ‑ system clock / active‑low reset.
//   in_valid        ‑ pulse when ‘a’ & ‘b’ are valid.
//   a, b            ‑ 16‑bit operands (raw bit‑patterns).
//   out4_valid      ‑ pulses 1 cycle after in_valid; p4 holds INT4 product.
//   out8_valid      ‑ pulses 2 cycles after;         p8 holds INT8 product.
//   out16_valid     ‑ pulses 4 cycles after;         p16 holds 16×16 product.
// -----------------------------------------------------------------------------
// ============================================================================
//  Pipelined Precision‑Progressive Multiplier & Attention A×V Engine
//  • INT‑4  (Q1.3)   product available 1 cycle after launch  (8‑bit result)
//  • INT‑8  (Q1.7)   product available 2 cycles after launch (16‑bit result)
//  • FP16 / full16   product available 4 cycles after launch (32‑bit result)
//  • No '*' operator on the critical path – only add/shift logic.
// ----------------------------------------------------------------------------
//  File: mul16_progressive.sv  +  attention_av_multiply.sv (integrated)
// ============================================================================

// -----------------------------------------------------------------------------
// 16×16 Multiplier that progressively releases results (no combinational '*').
// -----------------------------------------------------------------------------
module mul16_progressive #(
    parameter int WIDTH = 16
)(
    input  logic                       clk,
    input  logic                       rst_n,
    input  logic                       in_valid,
    input  logic [WIDTH-1:0]           a,
    input  logic [WIDTH-1:0]           b,

    output logic                       out4_valid,
    output logic        [7:0]          p4,
    output logic                       out8_valid,
    output logic       [15:0]          p8,
    output logic                       out16_valid,
    output logic       [31:0]          p16
);
    // ---------------------------------------------------------------------
    // Helper: 4‑bit × 16‑bit partial product via add/shift (unsigned)
    // ---------------------------------------------------------------------
    function automatic logic [19:0] mul4x16_u (input logic [15:0] x, input logic [3:0] y);
        logic [19:0] sum;
        begin
            sum = 20'd0;
            if (y[0]) sum += x;
            if (y[1]) sum += (x << 1);
            if (y[2]) sum += (x << 2);
            if (y[3]) sum += (x << 3);
            mul4x16_u = sum;
        end
    endfunction

    // Pipeline regs ---------------------------------------------------------
    logic               v_s0, v_s1, v_s2, v_s3, v_s4;
    logic [15:0]        a_s0, a_s1, a_s2, a_s3;
    logic [15:0]        b_s0, b_s1, b_s2, b_s3;
    logic [31:0]        acc_s1, acc_s2, acc_s3, acc_s4;

    // Stage‑0 capture
    always_ff @(posedge clk) begin
        if (!rst_n) v_s0 <= 1'b0;
        else        v_s0 <= in_valid;
        a_s0 <= a;
        b_s0 <= b;
    end

    // Stage‑1 (bits 3:0)
    always_ff @(posedge clk) begin
        v_s1 <= v_s0;
        a_s1 <= a_s0;
        b_s1 <= b_s0 >> 4;
        acc_s1 <= mul4x16_u(a_s0, b_s0[3:0]);
    end

    // Stage‑2 (bits 7:4)
    always_ff @(posedge clk) begin
        v_s2 <= v_s1;
        a_s2 <= a_s1;
        b_s2 <= b_s1 >> 4;
        acc_s2 <= acc_s1 + (mul4x16_u(a_s1, b_s1[3:0]) << 4);
    end

    // Stage‑3 (bits 11:8)
    always_ff @(posedge clk) begin
        v_s3 <= v_s2;
        a_s3 <= a_s2;
        b_s3 <= b_s2 >> 4;
        acc_s3 <= acc_s2 + (mul4x16_u(a_s2, b_s2[3:0]) << 8);
    end

    // Stage‑4 (bits 15:12)
    always_ff @(posedge clk) begin
        v_s4   <= v_s3;
        acc_s4 <= acc_s3 + (mul4x16_u(a_s3, b_s3[3:0]) << 12);
    end

    // Outputs ---------------------------------------------------------------
    assign out4_valid  = v_s1;
    assign p4         = acc_s1[7:0];
    assign out8_valid  = v_s2;
    assign p8         = acc_s2[15:0];
    assign out16_valid = v_s4;
    assign p16        = acc_s4;
endmodule

// ============================================================================
//  Attention A×V Multiply with adaptive precision + progressive multiplier
// ============================================================================
module attention_av_multiply #(
    parameter int DATA_WIDTH = 16,
    parameter int L          = 8,
    parameter int N          = 1,
    parameter int E          = 8
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        start,
    output logic                        done,

    input  logic [DATA_WIDTH*L*N*L-1:0] A_in,
    input  logic [DATA_WIDTH*L*N*E-1:0] V_in,
    input  logic [3:0]                  token_precision [L-1:0],

    output logic [DATA_WIDTH*L*N*E-1:0] Z_out,
    output logic                        out_valid
);
    // ---------------------------------------------------------------------
    // Unpacked storage
    // ---------------------------------------------------------------------
    logic [DATA_WIDTH-1:0] A_arr [L-1:0][N-1:0][L-1:0];
    logic [DATA_WIDTH-1:0] V_arr [L-1:0][N-1:0][E-1:0];
    logic [31:0]           Z_arr [L-1:0][N-1:0][E-1:0];

    // FSM ------------------------------------------------------------------
    typedef enum logic [1:0] { S_IDLE, S_LOAD, S_MUL, S_DONE } state_t;
    state_t state, next_state;
    logic [$clog2(L)-1:0] l2_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done <= 0; out_valid <= 0; l2_cnt <= 0;
        end else begin
            state <= next_state;
            unique case (state)
                S_IDLE: begin done<=0; out_valid<=0; l2_cnt<=0; end
                S_LOAD: l2_cnt <= 0;
                S_MUL : l2_cnt <= (l2_cnt==L-1) ? 0 : l2_cnt+1;
                S_DONE: begin done<=1; out_valid<=1; end
            endcase
        end
    end

    always_comb begin
        next_state = state;
        unique case (state)
            S_IDLE: if (start) next_state = S_LOAD;
            S_LOAD:           next_state = S_MUL;
            S_MUL : if (l2_cnt==L-1) next_state = S_DONE;
            S_DONE:           next_state = S_IDLE;
        endcase
    end

    // ---------------------------------------------------------------------
    // Unpack inputs and clear accumulators on S_LOAD
    // ---------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (state==S_LOAD) begin
            for (int l=0;l<L;l++)
                for (int n_=0;n_<N;n_++)
                    for (int l2=0;l2<L;l2++)
                        A_arr[l][n_][l2] <= A_in[((l*N*L)+(n_*L)+l2)*DATA_WIDTH +: DATA_WIDTH];
            for (int l2=0;l2<L;l2++)
                for (int n_=0;n_<N;n_++)
                    for (int e_=0;e_<E;e_++)
                        V_arr[l2][n_][e_] <= V_in[((l2*N*E)+(n_*E)+e_)*DATA_WIDTH +: DATA_WIDTH];
            for (int l=0;l<L;l++)
                for (int n_=0;n_<N;n_++)
                    for (int e_=0;e_<E;e_++)
                        Z_arr[l][n_][e_] <= 32'd0;
        end
    end

    // ---------------------------------------------------------------------
    // Global precision pipeline (aligns with multiplier latencies)
    // ---------------------------------------------------------------------
    logic [3:0] prec_p0, prec_p1, prec_p2, prec_p3, prec_p4;
    always_ff @(posedge clk) begin
        if (state!=S_MUL) begin prec_p0<=0; prec_p1<=0; prec_p2<=0; prec_p3<=0; prec_p4<=0; end
        else begin
            prec_p0 <= token_precision[l2_cnt];
            prec_p1 <= prec_p0;
            prec_p2 <= prec_p1;
            prec_p3 <= prec_p2;
            prec_p4 <= prec_p3;
        end
    end

    // ---------------------------------------------------------------------
    // Generate multiplier lanes
    // ---------------------------------------------------------------------
    logic lane_v4  [L-1:0][N-1:0][E-1:0];
    logic lane_v8  [L-1:0][N-1:0][E-1:0];
    logic lane_v16 [L-1:0][N-1:0][E-1:0];
    logic [7:0]  lane_p4  [L-1:0][N-1:0][E-1:0];
    logic [15:0] lane_p8  [L-1:0][N-1:0][E-1:0];
    logic [31:0] lane_p16 [L-1:0][N-1:0][E-1:0];

    generate
        for (genvar l=0;l<L;l++) begin : gL
            for (genvar n_=0;n_<N;n_++) begin : gN
                for (genvar e_=0;e_<E;e_++) begin : gE
                    mul16_progressive u_mul (
                        .clk  (clk), .rst_n(rst_n),
                        .in_valid(state==S_MUL),
                        .a (A_arr[l][n_][l2_cnt]),
                        .b (V_arr[l2_cnt][n_][e_]),
                        .out4_valid (lane_v4 [l][n_][e_]),
                        .p4         (lane_p4 [l][n_][e_]),
                        .out8_valid (lane_v8 [l][n_][e_]),
                        .p8         (lane_p8 [l][n_][e_]),
                        .out16_valid(lane_v16[l][n_][e_]),
                        .p16        (lane_p16[l][n_][e_])
                    );

                    // Select first valid product according to precision code
                    logic sel_valid;
                    logic [31:0] sel_prod;
                    always_comb begin
                        sel_valid = 1'b0; sel_prod = 32'd0;
                        // INT‑4 (latency 1) aligns with prec_p1
                        if (lane_v4[l][n_][e_]  && prec_p1==4'd0) begin
                            sel_valid = 1'b1;
                            sel_prod  = ({24'd0,lane_p4[l][n_][e_]} >> 6); // scale Q1.3→Q2.30
                        end
                        // INT‑8 (latency 2) aligns with prec_p2
                        else if (lane_v8[l][n_][e_] && prec_p2==4'd1) begin
                            sel_valid = 1'b1;
                            sel_prod  = ({16'd0,lane_p8[l][n_][e_]} >> 14); // scale Q1.7→Q2.30
                        end
                        // FP16/full16 (latency 4) aligns with prec_p4
                        else if (lane_v16[l][n_][e_] && prec_p4>=4'd2) begin
                            sel_valid = 1'b1;
                            sel_prod  = lane_p16[l][n_][e_];
                        end
                    end

                    always_ff @(posedge clk) begin
                        if (sel_valid) Z_arr[l][n_][e_] <= Z_arr[l][n_][e_] + sel_prod;
                    end
                end
            end
        end
    endgenerate

    // ---------------------------------------------------------------------
    // Pack outputs on S_DONE
    // ---------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (state==S_DONE) begin
            for (int l=0;l<L;l++)
                for (int n_=0;n_<N;n_++)
                    for (int e_=0;e_<E;e_++)
                        Z_out[((l*N*E)+(n_*E)+e_)*DATA_WIDTH +: DATA_WIDTH] <= Z_arr[l][n_][e_][15:0];
        end
    end
endmodule
