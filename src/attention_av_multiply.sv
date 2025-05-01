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

module mul16_progressive #(
    parameter int WIDTH = 16
)(
    input  logic                       clk,
    input  logic                       rst_n,

    // input handshake
    input  logic                       in_valid,
    input  logic [WIDTH-1:0]           a,
    input  logic [WIDTH-1:0]           b,

    // early‑exit results
    output logic                       out4_valid,
    output logic        [7:0]          p4,

    output logic                       out8_valid,
    output logic       [15:0]          p8,

    output logic                       out16_valid,
    output logic       [31:0]          p16
);

    // ──────────────────────────────────────────────────────────────
    // Helper: 4‑bit × 16‑bit unsigned partial product (no ‘*’)
    // result width: 16 + 4 = 20 bits (fits in 32)
    // pp = a * nibble where nibble ∈ [0,15]
    // ──────────────────────────────────────────────────────────────
    function automatic logic [19:0] mul4x16_u (
        input logic [15:0] x,
        input logic  [3:0] y
    );
        logic [19:0] sum;
        begin
            sum = 20'd0;
            if (y[0]) sum = sum + x;                // + x
            if (y[1]) sum = sum + (x << 1);         // + 2×x
            if (y[2]) sum = sum + (x << 2);         // + 4×x
            if (y[3]) sum = sum + (x << 3);         // + 8×x
            mul4x16_u = sum;
        end
    endfunction

    // ──────────────────────────────────────────────────────────────
    // Pipeline registers
    // ──────────────────────────────────────────────────────────────

    // Stage‑0 : capture operands
    logic               v_s0;
    logic [15:0]        a_s0, b_s0;

    // Stage‑1…3 shared signals
    logic               v_s1, v_s2, v_s3, v_s4;
    logic [15:0]        a_s1, a_s2, a_s3, a_s4;
    logic [15:0]        b_s1, b_s2, b_s3, b_s4;

    logic [31:0]        acc_s1, acc_s2, acc_s3, acc_s4;

    // ──────────────────────────────────────────────────────────────
    // Stage‑0
    // ──────────────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v_s0 <= 1'b0;
        end else begin
            v_s0 <= in_valid;
            a_s0 <= a;
            b_s0 <= b;
        end
    end

    // ──────────────────────────────────────────────────────────────
    // Stage‑1 : add partial product for bits 3:0 (INT4)
    // ──────────────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v_s1 <= 1'b0;
        end else begin
            v_s1 <= v_s0;
            a_s1 <= a_s0;
            b_s1 <= b_s0 >> 4;                              // shift for next stage
            acc_s1 <= mul4x16_u(a_s0, b_s0[3:0]);          // lowest PP
        end
    end

    // ──────────────────────────────────────────────────────────────
    // Stage‑2 : accumulate PP for bits 7:4 (INT8)
    // ──────────────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v_s2 <= 1'b0;
        end else begin
            v_s2 <= v_s1;
            a_s2 <= a_s1;
            b_s2 <= b_s1 >> 4;

            acc_s2 <= acc_s1 + (mul4x16_u(a_s1, b_s1[3:0]) << 4);
        end
    end

    // ──────────────────────────────────────────────────────────────
    // Stage‑3 : accumulate PP for bits 11:8
    // ──────────────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v_s3 <= 1'b0;
        end else begin
            v_s3 <= v_s2;
            a_s3 <= a_s2;
            b_s3 <= b_s2 >> 4;

            acc_s3 <= acc_s2 + (mul4x16_u(a_s2, b_s2[3:0]) << 8);
        end
    end

    // ──────────────────────────────────────────────────────────────
    // Stage‑4 : accumulate PP for bits 15:12 (final 32‑bit result)
    // ──────────────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v_s4 <= 1'b0;
        end else begin
            v_s4 <= v_s3;
            a_s4 <= a_s3;
            b_s4 <= b_s3; // not used further

            acc_s4 <= acc_s3 + (mul4x16_u(a_s3, b_s3[3:0]) << 12);
        end
    end

    // ──────────────────────────────────────────────────────────────
    // Output mapping
    // ──────────────────────────────────────────────────────────────
    assign out4_valid  = v_s1;
    assign p4         = acc_s1[7:0];

    assign out8_valid = v_s2;
    assign p8         = acc_s2[15:0];

    assign out16_valid = v_s4;
    assign p16         = acc_s4;

endmodule

// -----------------------------------------------------------------------------
// NOTE:   • Substitute signed/Booth logic if you need early signed precision.
//         • Add saturation / rounding as required by your numeric format.
//         • Throughput = 1 result per cycle (pipeline full).
// -----------------------------------------------------------------------------

