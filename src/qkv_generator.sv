// File: qkv_projection.sv
module qkv_projection #(
  parameter int DATA_WIDTH = 16, // e.g. 16-bit
  parameter int L = 8,          // sequence length
  parameter int N = 1,          // batch size
  parameter int E = 8           // embedding dimension
)(
  input  logic                          clk,
  input  logic                          rst_n,

  // Control
  input  logic                          start,
  output logic                          done,

  // Input x: shape (L, N, E) flattened
  input  logic [DATA_WIDTH*L*N*E-1:0]   x_in,

  // Weights WQ, WK, WV each (E x E), plus biases bQ, bK, bV (E)
  // For simplicity, we pass them in as large flattened arrays
  input  logic [DATA_WIDTH*E*E-1:0]     WQ_in,
  input  logic [DATA_WIDTH*E*E-1:0]     WK_in,
  input  logic [DATA_WIDTH*E*E-1:0]     WV_in,
  input  logic [DATA_WIDTH*E-1:0]       bQ_in,
  input  logic [DATA_WIDTH*E-1:0]       bK_in,
  input  logic [DATA_WIDTH*E-1:0]       bV_in,

  // Output Q, K, V => each (L, N, E), flattened
  output logic [DATA_WIDTH*L*N*E-1:0]   Q_out,
  output logic [DATA_WIDTH*L*N*E-1:0]   K_out,
  output logic [DATA_WIDTH*L*N*E-1:0]   V_out,

  output logic                          out_valid
);

  // Internal arrays (for demonstration)
  logic [DATA_WIDTH-1:0] x_arr [0:L-1][0:N-1][0:E-1];
  logic [DATA_WIDTH-1:0] Q_arr [0:L-1][0:N-1][0:E-1];
  logic [DATA_WIDTH-1:0] K_arr [0:L-1][0:N-1][0:E-1];
  logic [DATA_WIDTH-1:0] V_arr [0:L-1][0:N-1][0:E-1];

  // Weight/bias 2D arrays for convenience
  logic [DATA_WIDTH-1:0] WQ [0:E-1][0:E-1];
  logic [DATA_WIDTH-1:0] WK [0:E-1][0:E-1];
  logic [DATA_WIDTH-1:0] WV [0:E-1][0:E-1];
  logic [DATA_WIDTH-1:0] bQ [0:E-1];
  logic [DATA_WIDTH-1:0] bK [0:E-1];
  logic [DATA_WIDTH-1:0] bV [0:E-1];

  // Simple FSM
  typedef enum logic [2:0] {
    S_IDLE, S_LOAD, S_GENQ, S_GENK, S_GENV, S_DONE
  } state_t;
  state_t state;

  // Unpack weights/biases into 2D arrays
  integer i, j;
  always_comb begin
    for (i=0; i<E; i++) begin
      bQ[i] = bQ_in[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH];
      bK[i] = bK_in[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH];
      bV[i] = bV_in[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH];
      for (j=0; j<E; j++) begin
        WQ[i][j] = WQ_in[((i*E)+j+1)*DATA_WIDTH-1 -: DATA_WIDTH];
        WK[i][j] = WK_in[((i*E)+j+1)*DATA_WIDTH-1 -: DATA_WIDTH];
        WV[i][j] = WV_in[((i*E)+j+1)*DATA_WIDTH-1 -: DATA_WIDTH];
      end
    end
  end

  // FSM
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state     <= S_IDLE;
      done      <= 1'b0;
      out_valid <= 1'b0;
    end else begin
      case (state)
        S_IDLE: begin
          done      <= 1'b0;
          out_valid <= 1'b0;
          if (start) state <= S_LOAD;
        end

        // 1) Load x into x_arr
        S_LOAD: begin
          integer l_, n_, e_;
          for (l_ = 0; l_<L; l_++) begin
            for (n_ = 0; n_<N; n_++) begin
              for (e_ = 0; e_<E; e_++) begin
                x_arr[l_][n_][e_] = x_in[
                  ((l_*N*E)+(n_*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH
                ];
              end
            end
          end
          state <= S_GENQ;
        end

        // 2) Generate Q = x_arr * WQ + bQ
        S_GENQ: begin
          integer l_, n_, row, idx;
          for (l_ = 0; l_<L; l_++) begin
            for (n_ = 0; n_<N; n_++) begin
              for (row=0; row<E; row++) begin
                logic [DATA_WIDTH-1:0] sum_temp;
                sum_temp = bQ[row];
                for (idx=0; idx<E; idx++) begin
                  // Real design: floating/fixed multiply+acc
                  sum_temp = sum_temp + (x_arr[l_][n_][idx] * WQ[row][idx]);
                end
                Q_arr[l_][n_][row] = sum_temp;
              end
            end
          end
          state <= S_GENK;
        end

        // 3) Generate K = x_arr * WK + bK
        S_GENK: begin
          integer l_, n_, row, idx;
          for (l_ = 0; l_<L; l_++) begin
            for (n_ = 0; n_<N; n_++) begin
              for (row=0; row<E; row++) begin
                logic [DATA_WIDTH-1:0] sum_temp;
                sum_temp = bK[row];
                for (idx=0; idx<E; idx++) begin
                  sum_temp = sum_temp + (x_arr[l_][n_][idx] * WK[row][idx]);
                end
                K_arr[l_][n_][row] = sum_temp;
              end
            end
          end
          state <= S_GENV;
        end

        // 4) Generate V = x_arr * WV + bV
        S_GENV: begin
          integer l_, n_, row, idx;
          for (l_ = 0; l_<L; l_++) begin
            for (n_ = 0; n_<N; n_++) begin
              for (row=0; row<E; row++) begin
                logic [DATA_WIDTH-1:0] sum_temp;
                sum_temp = bV[row];
                for (idx=0; idx<E; idx++) begin
                  sum_temp = sum_temp + (x_arr[l_][n_][idx] * WV[row][idx]);
                end
                V_arr[l_][n_][row] = sum_temp;
              end
            end
          end
          state <= S_DONE;
        end

        // 5) Done => Pack Q, K, V back to Q_out, K_out, V_out
        S_DONE: begin
          integer l_, n_, e_;
          for (l_ = 0; l_<L; l_++) begin
            for (n_ = 0; n_<N; n_++) begin
              for (e_ = 0; e_<E; e_++) begin
                Q_out[((l_*N*E)+(n_*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH]
                  = Q_arr[l_][n_][e_];
                K_out[((l_*N*E)+(n_*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH]
                  = K_arr[l_][n_][e_];
                V_out[((l_*N*E)+(n_*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH]
                  = V_arr[l_][n_][e_];
              end
            end
          end
          out_valid <= 1'b1;
          done      <= 1'b1;
          state     <= S_IDLE; // or remain in DONE
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule

