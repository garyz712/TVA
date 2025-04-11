module self_attention_top #(
    parameter DATA_WIDTH = 16,
    parameter L = 8,
    parameter N = 1,
    parameter E = 8
)(
    input  wire                                       clk,
    input  wire                                       rst_n,
    input  wire                                       start,
    output reg                                        done,

    // Input x (L, N, E)
    input  wire [DATA_WIDTH*L*N*E-1:0]                x_in,
    
    // Weight matrices (E x E), biases (E)
    input  wire [DATA_WIDTH*E*E-1:0]                  WQ_in,
    input  wire [DATA_WIDTH*E*E-1:0]                  WK_in,
    input  wire [DATA_WIDTH*E*E-1:0]                  WV_in,
    input  wire [DATA_WIDTH*E-1:0]                    bQ_in,
    input  wire [DATA_WIDTH*E-1:0]                    bK_in,
    input  wire [DATA_WIDTH*E-1:0]                    bV_in,

    // Final Output: (L, N, E)
    output wire [DATA_WIDTH*L*N*E-1:0]                out_attention,
    output wire                                       out_valid
);

    // FSM to orchestrate submodules
    localparam ST_IDLE=0, ST_QKV=1, ST_QKV_WAIT=2,
               ST_QK=3,  ST_QK_WAIT=4,
               ST_SOFT=5,ST_SOFT_WAIT=6,
               ST_PREC=7,ST_PREC_WAIT=8,
               ST_MUL=9, ST_MUL_WAIT=10,
               ST_DONE=11;

    reg [3:0] state;
    reg start_qkv, start_qk, start_soft, start_prec, start_mul;
    wire done_qkv, done_qk, done_soft, done_prec, done_mul;
    reg [DATA_WIDTH*L*N*E-1:0] Q_wire, K_wire, V_wire;
    reg [DATA_WIDTH*L*N*L-1:0] A_wire;
    reg [DATA_WIDTH*L*N*L-1:0] A_softmax_wire;
    reg [3:0] token_prec [0:L-1];
    reg [DATA_WIDTH*L*N*E-1:0] Z_wire;

    wire qkv_valid, qk_valid, soft_valid, prec_valid, mul_valid;

    //---------------------------------------------
    // 1) QKV Generator
    //---------------------------------------------
    qkv_generator #(
        .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N), .E(E)
    ) u_qkv_gen (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (start_qkv),
        .done           (done_qkv),

        .x_in           (x_in),
        .WQ_in          (WQ_in),
        .WK_in          (WK_in),
        .WV_in          (WV_in),
        .bQ_in          (bQ_in),
        .bK_in          (bK_in),
        .bV_in          (bV_in),

        .Q_out          (Q_wire),
        .K_out          (K_wire),
        .V_out          (V_wire),
        .out_valid      (qkv_valid)
    );

    //---------------------------------------------
    // 2) Attention Score: A = QK^T
    //---------------------------------------------
    attention_score #(
        .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N), .E(E)
    ) u_attn_score (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (start_qk),
        .done           (done_qk),

        .Q_in           (Q_wire),
        .K_in           (K_wire),
        .A_out          (A_wire),
        .out_valid      (qk_valid)
    );

    //---------------------------------------------
    // 3) Softmax Approx
    //---------------------------------------------
    softmax_approx #(
        .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N)
    ) u_softmax (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (start_soft),
        .done           (done_soft),

        .A_in           (A_wire),
        .A_out          (A_softmax_wire),
        .out_valid      (soft_valid)
    );

    //---------------------------------------------
    // 4) Token Precision
    //---------------------------------------------
    token_precision_analyzer #(
        .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N)
    ) u_token_prec (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (start_prec),
        .done           (done_prec),

        .A_in           (A_softmax_wire),
        .token_precision(token_prec),
        .out_valid      (prec_valid)
    );

    //---------------------------------------------
    // 5) A x V (Adaptive Precision)
    //---------------------------------------------
    attention_av_multiply #(
        .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N), .E(E)
    ) u_attn_av (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (start_mul),
        .done           (done_mul),

        .A_in           (A_softmax_wire), // after softmax
        .V_in           (V_wire),
        .token_precision(token_prec),
        .Z_out          (Z_wire),
        .out_valid      (mul_valid)
    );

    //---------------------------------------------
    // Top-Level FSM
    //---------------------------------------------
    reg out_valid_reg;
    assign out_attention = Z_wire;
    assign out_valid     = out_valid_reg;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state          <= ST_IDLE;
            out_valid_reg  <= 1'b0;
            done           <= 1'b0;
            start_qkv      <= 1'b0;
            start_qk       <= 1'b0;
            start_soft     <= 1'b0;
            start_prec     <= 1'b0;
            start_mul      <= 1'b0;
        end else begin
            case(state)
                ST_IDLE: begin
                    out_valid_reg <= 1'b0;
                    done          <= 1'b0;
                    if(start) begin
                        start_qkv  <= 1'b1;
                        state      <= ST_QKV;
                    end
                end

                // 1) QKV
                ST_QKV: begin
                    start_qkv <= 1'b0; // only assert for 1 cycle
                    if(done_qkv) begin
                        start_qk <= 1'b1;
                        state    <= ST_QK;
                    end
                end

                ST_QK: begin
                    start_qk <= 1'b0;
                    if(done_qk) begin
                        start_soft <= 1'b1;
                        state      <= ST_SOFT;
                    end
                end

                ST_SOFT: begin
                    start_soft <= 1'b0;
                    if(done_soft) begin
                        start_prec <= 1'b1;
                        state      <= ST_PREC;
                    end
                end

                ST_PREC: begin
                    start_prec <= 1'b0;
                    if(done_prec) begin
                        start_mul <= 1'b1;
                        state     <= ST_MUL;
                    end
                end

                ST_MUL: begin
                    start_mul <= 1'b0;
                    if(done_mul) begin
                        state          <= ST_DONE;
                    end
                end

                ST_DONE: begin
                    out_valid_reg <= 1'b1;
                    done          <= 1'b1;
                    state         <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
