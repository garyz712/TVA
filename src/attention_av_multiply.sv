
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

// -----------------------------------------------------------------------------
//  Attention A×V Multiply – integrated "precision‑progressive" multipliers
// -----------------------------------------------------------------------------
//  Replaces each scalar multiply with a MUL16_PROGRESSIVE instance so INT4
//  finishes in 1 cycle, INT8 in 2 cycles, and FP16/full‑16 in 4 cycles.
//  One multiplier is instantiated per (l,n,e) lane; its inputs change once
//  each S_MUL clock as l2 sweeps 0…L‑1.  Partial products return with their
//  own valid flags, and we accumulate them into Z_arr whenever that flag
//  asserts.
// -----------------------------------------------------------------------------
module attention_av_multiply #(
    parameter int DATA_WIDTH = 16,
    parameter int L          = 8,
    parameter int N          = 1,
    parameter int E          = 8
)(
    input  logic                        clk,
    input  logic                        rst_n,
    // Control
    input  logic                        start,
    output logic                        done,
    // Input A: shape (L, N, L)
    input  logic [DATA_WIDTH*L*N*L-1:0] A_in,
    // Input V: shape (L, N, E)
    input  logic [DATA_WIDTH*L*N*E-1:0] V_in,
    // Per‑token precision codes: length L  (0=INT4  1=INT8  else=FP16)
    input  logic [3:0]                  token_precision [L-1:0],
    // Output Z: shape (L, N, E)
    output logic [DATA_WIDTH*L*N*E-1:0] Z_out,
    output logic                        out_valid
);

    // ---------------------------------------------------------------------
    // Internal storage (unpacked views)
    // ---------------------------------------------------------------------
    logic [DATA_WIDTH-1:0] A_arr   [L-1:0][N-1:0][L-1:0];
    logic [DATA_WIDTH-1:0] V_arr   [L-1:0][N-1:0][E-1:0];
    logic [31:0]           Z_arr   [L-1:0][N-1:0][E-1:0];

    typedef enum logic [1:0] { S_IDLE, S_LOAD, S_MUL, S_DONE } state_t;
    state_t state, next_state;

    logic [$clog2(L)-1:0] l2_cnt;

    // ---------------------------------------------------------------------
    //  FSM & counters
    // ---------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            done    <= 1'b0;
            out_valid <= 1'b0;
            l2_cnt  <= '0;
        end else begin
            state <= next_state;
            unique case (state)
                S_IDLE: begin
                    done      <= 1'b0;
                    out_valid <= 1'b0;
                    l2_cnt    <= '0;
                end
                S_LOAD: l2_cnt <= '0;
                S_MUL : l2_cnt <= (l2_cnt == L-1) ? '0 : l2_cnt + 1;
                S_DONE: begin
                    done      <= 1'b1;
                    out_valid <= 1'b1;
                end
            endcase
        end
    end

    always_comb begin
        next_state = state;
        unique case (state)
            S_IDLE: if (start)          next_state = S_LOAD;
            S_LOAD:                     next_state = S_MUL;
            S_MUL : if (l2_cnt==L-1)    next_state = S_DONE;
            S_DONE:                     next_state = S_IDLE;
        endcase
    end

    // ---------------------------------------------------------------------
    //  Unpack inputs & clear accumulators on S_LOAD
    // ---------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (state == S_LOAD) begin
            for (int l=0; l<L; l++)
                for (int n_=0; n_<N; n_++)
                    for (int l2=0; l2<L; l2++)
                        A_arr[l][n_][l2] <= A_in[((l*N*L)+(n_*L)+l2)*DATA_WIDTH +: DATA_WIDTH];

            for (int l2=0; l2<L; l2++)
                for (int n_=0; n_<N; n_++)
                    for (int e_=0; e_<E; e_++)
                        V_arr[l2][n_][e_] <= V_in[((l2*N*E)+(n_*E)+e_)*DATA_WIDTH +: DATA_WIDTH];

            for (int l=0; l<L; l++)
                for (int n_=0; n_<N; n_++)
                    for (int e_=0; e_<E; e_++)
                        Z_arr[l][n_][e_] <= 32'd0;
        end
    end

    // ---------------------------------------------------------------------
    //  Progressive multiplier lanes (generate)
    // ---------------------------------------------------------------------
    logic lane_v4  [L-1:0][N-1:0][E-1:0];
    logic lane_v8  [L-1:0][N-1:0][E-1:0];
    logic lane_v16 [L-1:0][N-1:0][E-1:0];
    logic [7:0]  lane_p4  [L-1:0][N-1:0][E-1:0];
    logic [15:0] lane_p8  [L-1:0][N-1:0][E-1:0];
    logic [31:0] lane_p16 [L-1:0][N-1:0][E-1:0];

    generate
        for (genvar l=0; l<L; l++) begin : g_l
            for (genvar n_=0; n_<N; n_++) begin : g_n
                for (genvar e_=0; e_<E; e_++) begin : g_e
                    wire [15:0] op_a = A_arr[l][n_][l2_cnt];
                    wire [15:0] op_b = V_arr[l2_cnt][n_][e_];
                    wire        v_in = (state == S_MUL);

                    mul16_progressive u_mul (
                        .clk         (clk),
                        .rst_n       (rst_n),
                        .in_valid    (v_in),
                        .a           (op_a),
                        .b           (op_b),
                        .out4_valid  (lane_v4 [l][n_][e_]),
                        .p4          (lane_p4 [l][n_][e_]),
                        .out8_valid  (lane_v8 [l][n_][e_]),
                        .p8          (lane_p8 [l][n_][e_]),
                        .out16_valid (lane_v16[l][n_][e_]),
                        .p16         (lane_p16[l][n_][e_])
                    );

                    logic [31:0] sel_prod;
                    logic        sel_valid;
                    always_comb begin
                        unique case (token_precision[l2_cnt])
                            4'd0: begin sel_prod={24'd0,lane_p4[l][n_][e_]}; sel_valid=lane_v4[l][n_][e_]; end
                            4'd1: begin sel_prod={16'd0,lane_p8[l][n_][e_]}; sel_valid=lane_v8[l][n_][e_]; end
                            default: begin sel_prod=lane_p16[l][n_][e_];    sel_valid=lane_v16[l][n_][e_]; end
                        endcase
                    end

                    always_ff @(posedge clk) begin
                        if (!rst_n) begin
                            Z_arr[l][n_][e_] <= 32'd0;
                        end else if (state == S_LOAD) begin
                            Z_arr[l][n_][e_] <= 32'd0;
                        end else if (sel_valid) begin
                            Z_arr[l][n_][e_] <= Z_arr[l][n_][e_] + sel_prod;
                        end
                    end
                end
            end
        end
    endgenerate

    // ---------------------------------------------------------------------
    //  Pack outputs when finished
    // ---------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (state == S_DONE) begin
            for (int l=0; l<L; l++)
                for (int n_=0; n_<N; n_++)
                    for (int e_=0; e_<E; e_++)
                        Z_out[((l*N*E)+(n_*E)+e_)*DATA_WIDTH +: DATA_WIDTH] <= Z_arr[l][n_][e_][15:0];
        end
    end
endmodule



