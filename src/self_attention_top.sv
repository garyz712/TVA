// File: attention_top.sv
module attention_top #(
  parameter int DATA_WIDTH = 16,
  parameter int L = 8,
  parameter int N = 1,
  parameter int E = 8,
  parameter int H = 32 // for MLP hidden dimension
)(
  input  logic                                     clk,
  input  logic                                     rst_n,
  input  logic                                     start,
  output logic                                     done,

  // Input x (L, N, E)
  input  logic [DATA_WIDTH*L*N*E-1:0]              x_in,

  // Weights for QKV
  input  logic [DATA_WIDTH*E*E-1:0]                WQ_in,
  input  logic [DATA_WIDTH*E*E-1:0]                WK_in,
  input  logic [DATA_WIDTH*E*E-1:0]                WV_in,
  input  logic [DATA_WIDTH*E-1:0]                  bQ_in,
  input  logic [DATA_WIDTH*E-1:0]                  bK_in,
  input  logic [DATA_WIDTH*E-1:0]                  bV_in,

  // Weights for MLP
  //   W1 (E x H), b1 (H)
  //   W2 (H x E), b2 (E)
  input  logic [DATA_WIDTH*E*H-1:0]                W1_in,
  input  logic [DATA_WIDTH*H-1:0]                  b1_in,
  input  logic [DATA_WIDTH*H*E-1:0]                W2_in,
  input  logic [DATA_WIDTH*E-1:0]                  b2_in,

  // Final output from MLP (L, N, E)
  output logic [DATA_WIDTH*L*N*E-1:0]              out_final,
  output logic                                     out_valid
);

  //============================================================
  // Internal wires
  //============================================================
  logic                                    qkv_done, qkv_start;
  logic [DATA_WIDTH*L*N*E-1:0]             Q_wire, K_wire, V_wire;
  logic                                    qkv_out_valid;

  logic                                    qk_done, qk_start;
  logic [DATA_WIDTH*L*N*L-1:0]             A_wire;
  logic                                    qk_out_valid;

  logic [DATA_WIDTH*L*N*L-1:0]             A_softmax_wire;
  logic [3:0]                              token_prec_wire [0:L-1];

  // Output of A×V multiply => attention result
  logic [DATA_WIDTH*L*N*E-1:0]             attention_out;
  
  // MLP signals
  logic                                    mlp_start, mlp_done;
  logic [DATA_WIDTH*L*N*E-1:0]             mlp_out;
  logic                                    mlp_out_valid;

  // FSM
  typedef enum logic [3:0] {
    ST_IDLE, ST_QKV, ST_QK, ST_SOFTMAX, ST_PREC, ST_AXV,
    ST_MLP, ST_DONE
  } top_state_t;
  top_state_t state;

  //============================================================
  // 1) QKV Projection
  //============================================================
  qkv_projection #(
    .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N), .E(E)
  ) u_qkv (
    .clk         (clk),
    .rst_n       (rst_n),
    .start       (qkv_start),
    .done        (qkv_done),
    .x_in        (x_in),
    .WQ_in       (WQ_in),
    .WK_in       (WK_in),
    .WV_in       (WV_in),
    .bQ_in       (bQ_in),
    .bK_in       (bK_in),
    .bV_in       (bV_in),
    .Q_out       (Q_wire),
    .K_out       (K_wire),
    .V_out       (V_wire),
    .out_valid   (qkv_out_valid)
  );

  //============================================================
  // 2) QK Engine
  //============================================================
  qk_engine #(
    .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N), .E(E)
  ) u_qk (
    .clk         (clk),
    .rst_n       (rst_n),
    .start       (qk_start),
    .done        (qk_done),
    .Q_in        (Q_wire),
    .K_in        (K_wire),
    .A_out       (A_wire),
    .out_valid   (qk_out_valid)
  );

  //============================================================
  // 3) Softmax
  //============================================================
  softmax_unit #(
    .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N)
  ) u_softmax (
    .A_in        (A_wire),
    .A_out       (A_softmax_wire)
  );

  //============================================================
  // 4) Precision Assigner
  //============================================================
  precision_assigner #(
    .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N)
  ) u_prec_assigner (
    .A_in           (A_softmax_wire),
    .token_precision(token_prec_wire)
  );

  //============================================================
  // 5) A×V Multiply => attention_out
  //============================================================
  av_multiply #(
    .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N), .E(E)
  ) u_av_mul (
    .A_in            (A_softmax_wire),
    .V_in            (V_wire),
    .token_precision (token_prec_wire),
    .Z_out           (attention_out)
  );

  //============================================================
  // 6) MLP (2-layer feed-forward)
  //============================================================
  mlp_block #(
    .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N), .E(E), .H(H)
  ) u_mlp (
    .clk       (clk),
    .rst_n     (rst_n),
    .start     (mlp_start),
    .done      (mlp_done),
    .x_in      (attention_out), // MLP input is the result of attention
    .W1_in     (W1_in),
    .b1_in     (b1_in),
    .W2_in     (W2_in),
    .b2_in     (b2_in),
    .out_mlp   (mlp_out),
    .out_valid (mlp_out_valid)
  );

  //============================================================
  // FSM: orchestrates the flow
  //============================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      state       <= ST_IDLE;
      done        <= 1'b0;
      out_valid   <= 1'b0;
      qkv_start   <= 1'b0;
      qk_start    <= 1'b0;
      mlp_start   <= 1'b0;
    end else begin
      case(state)
        // Wait for 'start'
        ST_IDLE: begin
          done        <= 1'b0;
          out_valid   <= 1'b0;
          qkv_start   <= 1'b0;
          qk_start    <= 1'b0;
          mlp_start   <= 1'b0;
          if(start) state <= ST_QKV;
        end

        // 1) QKV
        ST_QKV: begin
          qkv_start <= 1'b1;
          if(qkv_done) begin
            qkv_start <= 1'b0;
            state     <= ST_QK;
          end
        end

        // 2) QK
        ST_QK: begin
          qk_start <= 1'b1;
          if(qk_done) begin
            qk_start <= 1'b0;
            state    <= ST_SOFTMAX;
          end
        end

        // 3) Softmax (combinational placeholder)
        ST_SOFTMAX: begin
          // We could add pipeline/wait cycles if needed
          state <= ST_PREC;
        end

        // 4) Precision assignment (also combinational)
        ST_PREC: begin
          state <= ST_AXV;
        end

        // 5) A×V multiply (combinational in skeleton)
        ST_AXV: begin
          // Next, we feed that result to MLP
          state <= ST_MLP;
        end

        // 6) MLP
        ST_MLP: begin
          mlp_start <= 1'b1;
          if(mlp_done) begin
            mlp_start <= 1'b0;
            state     <= ST_DONE;
          end
        end

        // 7) Done
        ST_DONE: begin
          // The MLP output is in mlp_out
          // Drive final signals
          done        <= 1'b1;
          out_valid   <= 1'b1;
          // Present final result
          // out_final = mlp_out
          out_final   <= mlp_out;
          state       <= ST_IDLE;
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

endmodule

