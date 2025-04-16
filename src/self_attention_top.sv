//
// File: attention_top.sv
//
// Purpose: Top-level orchestrator for single-head Self-Attention + MLP
// using time-multiplexed submodules.
//
// - QKV Projection
// - QK Engine
// - Softmax (combinational placeholder in this skeleton)
// - Precision Assigner (time-multiplexed)
// - A×V Multiply (time-multiplexed)
// - MLP (time-multiplexed)
//
// Follows best practice:
//  - Non-blocking (<=) in always_ff
//  - FSM with separate next-state logic (always_comb)
//  - Minimizes large nested loops
//

module attention_top #(
  parameter int DATA_WIDTH = 16,
  parameter int L = 8,        // sequence length
  parameter int N = 1,        // batch size
  parameter int E = 8,        // embedding dimension
  parameter int H = 32        // MLP hidden dimension (typical ~4*E)
)(
  input  logic                                      clk,
  input  logic                                      rst_n,
  input  logic                                      start,
  output logic                                      done,

  //----------------------------------------------------------------
  //  Input: x_in shape (L, N, E), flattened
  //----------------------------------------------------------------
  input  logic [DATA_WIDTH*L*N*E-1:0]               x_in,

  //----------------------------------------------------------------
  //  Weights for QKV:
  //    WQ, WK, WV => (E x E)
  //    bQ, bK, bV => (E)
  //----------------------------------------------------------------
  input  logic [DATA_WIDTH*E*E-1:0]                 WQ_in,
  input  logic [DATA_WIDTH*E*E-1:0]                 WK_in,
  input  logic [DATA_WIDTH*E*E-1:0]                 WV_in,
  input  logic [DATA_WIDTH*E-1:0]                   bQ_in,
  input  logic [DATA_WIDTH*E-1:0]                   bK_in,
  input  logic [DATA_WIDTH*E-1:0]                   bV_in,

  //----------------------------------------------------------------
  //  Weights for MLP:
  //    W1 => (E x H), b1 => (H)
  //    W2 => (H x E), b2 => (E)
  //----------------------------------------------------------------
  input  logic [DATA_WIDTH*E*H-1:0]                 W1_in,
  input  logic [DATA_WIDTH*H-1:0]                   b1_in,
  input  logic [DATA_WIDTH*H*E-1:0]                 W2_in,
  input  logic [DATA_WIDTH*E-1:0]                   b2_in,

  //----------------------------------------------------------------
  //  Final output shape: (L, N, E)
  //----------------------------------------------------------------
  output logic [DATA_WIDTH*L*N*E-1:0]               out_final,
  output logic                                      out_valid
);

  //****************************************************************
  //
  //  1) FSM States
  //
  //****************************************************************
  typedef enum logic [3:0] {
    S_IDLE       = 4'd0,
    S_QKV        = 4'd1,   // Start QKV projection
    S_WAIT_QKV   = 4'd2,   // Wait QKV done
    S_QK         = 4'd3,   // Start QK engine
    S_WAIT_QK    = 4'd4,   // Wait QK done
    S_SOFTMAX    = 4'd5,   // Combinational softmax
    S_PREC       = 4'd6,   // Start precision assigner
    S_WAIT_PREC  = 4'd7,   // Wait precision done
    S_AXV        = 4'd8,   // Start A×V multiply
    S_WAIT_AXV   = 4'd9,   // Wait A×V done
    S_MLP        = 4'd10,  // Start MLP
    S_WAIT_MLP   = 4'd11,  // Wait MLP done
    S_DONE       = 4'd12
  } state_t;

  state_t curr_state, next_state;

  //****************************************************************
  //
  //  2) Wires/Regs to Connect Submodules
  //
  //****************************************************************

  // QKV submodule
  logic qkv_start, qkv_done, qkv_out_valid;
  logic [DATA_WIDTH*L*N*E-1:0] Q_wire;
  logic [DATA_WIDTH*L*N*E-1:0] K_wire;
  logic [DATA_WIDTH*L*N*E-1:0] V_wire;

  // QK engine
  logic qk_start, qk_done, qk_out_valid;
  logic [DATA_WIDTH*L*N*L-1:0] A_wire;  // shape (L, N, L) => flattened

  // Softmax (combinational placeholder)
  logic [DATA_WIDTH*L*N*L-1:0] A_softmax_wire;

  // Precision assigner
  logic prec_start, prec_done;
  logic [3:0] token_prec [0:L-1];  // 4-bit code per token

  // A×V multiply
  logic axv_start, axv_done;
  logic [DATA_WIDTH*L*N*E-1:0] attention_out;  // shape (L, N, E)

  // MLP block
  logic mlp_start, mlp_done, mlp_out_valid;
  logic [DATA_WIDTH*L*N*E-1:0] mlp_out;

  // Final outputs
  // out_final, out_valid, done


  //****************************************************************
  //
  //  3) Instantiate Submodules
  //
  //****************************************************************

  // (A) QKV Projection
  qkv_projection #(
    .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N), .E(E)
  ) u_qkv (
    .clk        (clk),
    .rst_n      (rst_n),
    .start      (qkv_start),
    .done       (qkv_done),

    .x_in       (x_in),
    .WQ_in      (WQ_in),
    .WK_in      (WK_in),
    .WV_in      (WV_in),
    .bQ_in      (bQ_in),
    .bK_in      (bK_in),
    .bV_in      (bV_in),

    .Q_out      (Q_wire),
    .K_out      (K_wire),
    .V_out      (V_wire),
    .out_valid  (qkv_out_valid)
  );

  // (B) QK Engine
  qk_engine #(
    .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N), .E(E)
  ) u_qk (
    .clk       (clk),
    .rst_n     (rst_n),
    .start     (qk_start),
    .done      (qk_done),

    .Q_in      (Q_wire),
    .K_in      (K_wire),
    .A_out     (A_wire),
    .out_valid (qk_out_valid)
  );

  // (C) Softmax (Combinational or time-multiplexed)
  softmax_unit #(
    .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N)
  ) u_softmax (
    .A_in  (A_wire),
    .A_out (A_softmax_wire)
  );

  // (D) Precision Assigner
  precision_assigner #(
    .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N)
  ) u_prec_assigner (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (prec_start),
    .done           (prec_done),
    .A_in           (A_softmax_wire),
    .token_precision(token_prec)
  );

  // (E) A×V multiply
  av_multiply #(
    .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N), .E(E)
  ) u_av_mul (
    .clk             (clk),
    .rst_n           (rst_n),
    .start           (axv_start),
    .done            (axv_done),

    .A_in            (A_softmax_wire),
    .V_in            (V_wire),
    .token_precision (token_prec),
    .Z_out           (attention_out)
  );

  // (F) MLP Block
  mlp_block #(
    .DATA_WIDTH(DATA_WIDTH), .L(L), .N(N), .E(E), .H(H)
  ) u_mlp (
    .clk       (clk),
    .rst_n     (rst_n),
    .start     (mlp_start),
    .done      (mlp_done),
    .x_in      (attention_out),

    .W1_in     (W1_in),
    .b1_in     (b1_in),
    .W2_in     (W2_in),
    .b2_in     (b2_in),

    .out_mlp   (mlp_out),
    .out_valid (mlp_out_valid)
  );


  //****************************************************************
  //
  //  4) Top-Level FSM
  //
  //****************************************************************

  // 4a) State register
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n)
      curr_state <= S_IDLE;
    else
      curr_state <= next_state;
  end

  // 4b) Next-state logic (always_comb)
  always_comb begin
    // Default
    next_state = curr_state;

    case(curr_state)

      S_IDLE: begin
        if(start)
          next_state = S_QKV;
      end

      //-----------------------------------------------------------
      // QKV
      //-----------------------------------------------------------
      S_QKV: begin
        // we assert qkv_start in output logic
        next_state = S_WAIT_QKV;
      end

      S_WAIT_QKV: begin
        if(qkv_done)  // submodule signals it's done
          next_state = S_QK;
      end

      //-----------------------------------------------------------
      // QK
      //-----------------------------------------------------------
      S_QK: begin
        next_state = S_WAIT_QK;
      end

      S_WAIT_QK: begin
        if(qk_done)
          next_state = S_SOFTMAX;
      end

      //-----------------------------------------------------------
      // Softmax
      //-----------------------------------------------------------
      // We'll assume it's purely combinational. 
      // We might wait 1 cycle or so. We'll do a single cycle wait:
      S_SOFTMAX: begin
        next_state = S_PREC;
      end

      //-----------------------------------------------------------
      // Precision
      //-----------------------------------------------------------
      S_PREC: begin
        next_state = S_WAIT_PREC;
      end

      S_WAIT_PREC: begin
        if(prec_done)
          next_state = S_AXV;
      end

      //-----------------------------------------------------------
      // A×V Multiply
      //-----------------------------------------------------------
      S_AXV: begin
        next_state = S_WAIT_AXV;
      end

      S_WAIT_AXV: begin
        if(axv_done)
          next_state = S_MLP;
      end

      //-----------------------------------------------------------
      // MLP
      //-----------------------------------------------------------
      S_MLP: begin
        next_state = S_WAIT_MLP;
      end

      S_WAIT_MLP: begin
        if(mlp_done)
          next_state = S_DONE;
      end

      //-----------------------------------------------------------
      // DONE
      //-----------------------------------------------------------
      S_DONE: begin
        // Return to idle or remain done
        next_state = S_IDLE;
      end

      default: next_state = S_IDLE;

    endcase
  end

  // 4c) Output/Control logic (always_comb)
  always_comb begin
    // Default all signals
    done        = 1'b0;
    out_valid   = 1'b0;

    // Submodule control signals, default off
    qkv_start   = 1'b0;
    qk_start    = 1'b0;
    prec_start  = 1'b0;
    axv_start   = 1'b0;
    mlp_start   = 1'b0;

    case(curr_state)

      S_QKV:     qkv_start  = 1'b1;
      S_QK:      qk_start   = 1'b1;
      S_PREC:    prec_start = 1'b1;
      S_AXV:     axv_start  = 1'b1;
      S_MLP:     mlp_start  = 1'b1;

      S_DONE: begin
        done      = 1'b1;
        out_valid = 1'b1;
      end
    endcase
  end

  //****************************************************************
  //
  //  5) Output Final
  //
  //****************************************************************

  // The final MLP output is `mlp_out` (shape L,N,E).
  // We present it as `out_final`.
  // Typically, you'd want to register or pipeline it. 
  // For simplicity, we'll do direct assignment:

  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      out_final <= '0;
    end
    else if (curr_state == S_DONE) begin
      out_final <= mlp_out;  
      // Alternatively, you might do this assignment in S_WAIT_MLP when mlp_done==1,
      // or in an always_comb if mlp_out is stable. 
    end
  end

endmodule
