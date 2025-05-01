
//------------------------------------------------------------------------------
// attention_av_multiply.sv
//
//  This module implements a matrix multiplication for the attention mechanism,
//  computing A * V with token-wise mixed precision (INT4, INT8, FP16).
//  Uses pipelined multipliers with variable latency to reduce cycle count
//  for lower precision tokens, operating at a variable clock frequency.
//
//  Apr. 10 2025    Max Zhang      Initial version
//  Apr. 11 2025    Tianwei Liu    Refactor in SV, split state machine, add comments
//  Apr. 30 2025    Max Zhang      Redesigned with pipelined variable-latency multipliers
//------------------------------------------------------------------------------
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
