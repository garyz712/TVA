// File: ws_array.sv
module ws_array #(
  parameter int DATA_WIDTH = 16,
  parameter int M = 4,    // # inputs (rows)
  parameter int N = 4,    // # outputs (cols)
  parameter int K = 128   // vector length
)(
  input  logic                              clk,
  input  logic                              rst_n,
  input  logic                              start_load, // begin weight load
  input  logic                              start_compute, // begin activation stream
  // Weight memory [M][N]
  input  logic [DATA_WIDTH-1:0]             W_mem [0:M-1][0:N-1],
  // Activation streams: K cycles of M values
  input  logic [DATA_WIDTH-1:0]             A_stream [0:K-1][0:M-1],
  output logic [2*DATA_WIDTH-1:0]           C_out  [0:M-1][0:N-1], // result
  output logic                              done
);

  // internal control FSM
  typedef enum logic [1:0] { IDLE, LOAD, COMPUTE, DONE } state_t;
  state_t state, next_state;
  integer i,j, cycle;

  // instantiate PEs
  logic load_w_sig;
  logic valid_in [0:M-1][0:N-1];
  logic a_in     [0:M-1][0:N-1];
  logic acc_out  [0:M-1][0:N-1];
  logic v_out    [0:M-1][0:N-1];

  genvar r,c;
  generate
    for(r=0; r<M; r++) begin
      for(c=0; c<N; c++) begin
        ws_pe #(.DATA_WIDTH(DATA_WIDTH)) pe (
          .clk      (clk),
          .rst_n    (rst_n),
          .load_w   (state==LOAD),
          .valid_in (state==COMPUTE ? valid_in[r][c] : 1'b0),
          .w_in     (W_mem[r][c]),
          .a_in     (a_in[r][c]),
          .accum_out(acc_out[r][c]),
          .valid_out(v_out[r][c])
        );
      end
    end
  endgenerate

  // FSM
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin state <= IDLE; cycle <= 0; end
    else         state <= next_state;
  end

  always_comb begin
    next_state = state;
    case(state)
      IDLE:      if (start_load)    next_state = LOAD;
      LOAD:      if (cycle==M*N-1)  next_state = COMPUTE;
      COMPUTE:  if (cycle==K-1)    next_state = DONE;
      DONE:      next_state = IDLE;
    endcase
  end

  // cycle counter & done
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) cycle <= 0;
    else if (state==LOAD || state==COMPUTE) cycle <= cycle + 1;
    if (state==DONE) done <= 1; else if(state==IDLE) done <= 0;
  end

  // drive activations in COMPUTE phase
  always_comb begin
    for (i=0; i<M; i++) begin
      for (j=0; j<N; j++) begin
        // broadcast A_stream[cycle][i] to every column in that row
        a_in[i][j]     = (state==COMPUTE) ? A_stream[cycle][i] : '0;
        valid_in[i][j] = (state==COMPUTE);
      end
    end
  end

  // capture output when DONE
  always_ff @(posedge clk) begin
    if (state==DONE) begin
      for (i=0; i<M; i++)
        for (j=0; j<N; j++)
          C_out[i][j] <= acc_out[i][j];
    end
  end

endmodule
