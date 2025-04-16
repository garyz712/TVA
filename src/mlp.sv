// File: mlp.sv
module mlp_block #(
  parameter int DATA_WIDTH = 16,
  parameter int L = 8,
  parameter int N = 1,
  parameter int E = 8,
  parameter int H = 32 // hidden dimension (commonly 4*E in Transformers)
)(
  input  logic                             clk,
  input  logic                             rst_n,
  input  logic                             start,
  output logic                             done,

  // Input: (L, N, E)
  input  logic [DATA_WIDTH*L*N*E-1:0]      x_in,

  // Weights for MLP:
  //   W1: (E x H)
  //   b1: (H)
  //   W2: (H x E)
  //   b2: (E)
  input  logic [DATA_WIDTH*E*H-1:0]        W1_in,
  input  logic [DATA_WIDTH*H-1:0]          b1_in,
  input  logic [DATA_WIDTH*H*E-1:0]        W2_in,
  input  logic [DATA_WIDTH*E-1:0]          b2_in,

  // Output: (L, N, E)
  output logic [DATA_WIDTH*L*N*E-1:0]      out_mlp,
  output logic                             out_valid
);

  // Internal states, arrays
  typedef enum logic [2:0] {S_IDLE, S_LOAD, S_FC1, S_ACT, S_FC2, S_DONE} state_t;
  state_t state;

  // Input array
  logic [DATA_WIDTH-1:0] x_arr [0:L-1][0:N-1][0:E-1];

  // Weights
  logic [DATA_WIDTH-1:0] W1 [0:E-1][0:H-1]; // row: E, col: H
  logic [DATA_WIDTH-1:0] b1 [0:H-1];
  logic [DATA_WIDTH-1:0] W2 [0:H-1][0:E-1]; // row: H, col: E
  logic [DATA_WIDTH-1:0] b2 [0:E-1];

  // Hidden array: shape (L, N, H)
  logic [DATA_WIDTH-1:0] hidden [0:L-1][0:N-1][0:H-1];

  // Output array: shape (L, N, E)
  logic [DATA_WIDTH-1:0] out_arr [0:L-1][0:N-1][0:E-1];

  // Unpack weight inputs into arrays
  integer i, j;
  always_comb begin
    for (i=0; i<E; i++) begin
      for (j=0; j<H; j++) begin
        W1[i][j] = W1_in[( (i*H) + j +1 )*DATA_WIDTH -1 -: DATA_WIDTH];
      end
    end
    for (j=0; j<H; j++) begin
      b1[j] = b1_in[(j+1)*DATA_WIDTH-1 -: DATA_WIDTH];
    end

    for (i=0; i<H; i++) begin
      for (j=0; j<E; j++) begin
        W2[i][j] = W2_in[( (i*E)+ j +1 )*DATA_WIDTH -1 -: DATA_WIDTH];
      end
    end
    for (i=0; i<E; i++) begin
      b2[i] = b2_in[(i+1)*DATA_WIDTH -1 -: DATA_WIDTH];
    end
  end

  // FSM
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      state     <= S_IDLE;
      done      <= 1'b0;
      out_valid <= 1'b0;
    end else begin
      case(state)
        S_IDLE: begin
          done      <= 1'b0;
          out_valid <= 1'b0;
          if(start) state <= S_LOAD;
        end

        // 1) Load x_in into x_arr
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
          state <= S_FC1;
        end

        // 2) FC1: hidden = x_arr * W1 + b1, shape => (L,N,H)
        S_FC1: begin
          integer l_, n_, h_, e_;
          for (l_ = 0; l_<L; l_++) begin
            for (n_ = 0; n_<N; n_++) begin
              for (h_ = 0; h_<H; h_++) begin
                logic [DATA_WIDTH-1:0] sum_temp;
                sum_temp = b1[h_];
                for (e_ = 0; e_<E; e_++) begin
                  // sum_temp += x_arr[l_][n_][e_] * W1[e_][h_]
                  sum_temp = sum_temp + (x_arr[l_][n_][e_] * W1[e_][h_]);
                end
                hidden[l_][n_][h_] = sum_temp;
              end
            end
          end
          state <= S_ACT;
        end

        // 3) Activation: ReLU on hidden
        S_ACT: begin
          integer l_, n_, h_;
          for (l_ = 0; l_<L; l_++) begin
            for (n_ = 0; n_<N; n_++) begin
              for (h_ = 0; h_<H; h_++) begin
                // ReLU
                if (hidden[l_][n_][h_][DATA_WIDTH-1] == 1'b1) begin
                  // if sign bit is 1 => negative => clamp to 0
                  hidden[l_][n_][h_] = '0; 
                end
              end
            end
          end
          state <= S_FC2;
        end

        // 4) FC2: out_arr = hidden * W2 + b2, shape => (L,N,E)
        S_FC2: begin
          integer l_, n_, e_, h_;
          for (l_ = 0; l_<L; l_++) begin
            for (n_ = 0; n_<N; n_++) begin
              for (e_ = 0; e_<E; e_++) begin
                logic [DATA_WIDTH-1:0] sum_temp;
                sum_temp = b2[e_];
                for (h_ = 0; h_<H; h_++) begin
                  sum_temp = sum_temp + (hidden[l_][n_][h_] * W2[h_][e_]);
                end
                out_arr[l_][n_][e_] = sum_temp;
              end
            end
          end
          state <= S_DONE;
        end

        S_DONE: begin
          // Pack out_arr into out_mlp
          integer l_, n_, e_;
          for (l_ = 0; l_<L; l_++) begin
            for (n_ = 0; n_<N; n_++) begin
              for (e_ = 0; e_<E; e_++) begin
                out_mlp[
                  ((l_*N*E)+(n_*E)+ e_ +1)*DATA_WIDTH -1 -: DATA_WIDTH
                ] = out_arr[l_][n_][e_];
              end
            end
          end
          out_valid <= 1'b1;
          done      <= 1'b1;
          state     <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
