//-----------------------------------------------------------------------------
// systolic_array.sv
//-----------------------------------------------------------------------------

module systolic_array #(
  parameter int DATA_W = 16,   // activation/weight width
  parameter int ACC_W  = 32,   // accumulator width
  parameter int M      = 16,   // # of rows in the tile
  parameter int N      = 16    // # of columns in the tile
)(
  input  logic                         clk,
  input  logic                         rst_n,

  // Inputs: stream M activations in parallel each cycle
  input  logic signed [DATA_W-1:0]     act_in [0:M-1],
  // Inputs: stream N weights in parallel each cycle
  input  logic signed [DATA_W-1:0]     wgt_in [0:N-1],

  // Pulse high for one cycle at the start of a new tile
  input  logic                         tile_start,

  // Outputs: M×N partial sums after reduction is complete
  output logic signed [ACC_W-1:0]      psum_out [0:M-1][0:N-1]
);

  // Internal wires carrying activations across columns (0..N)
  logic signed [DATA_W-1:0] a_wire [0:M-1][0:N];
  // Internal wires carrying weights down rows (0..M)
  logic signed [DATA_W-1:0] w_wire [0:M][0:N-1];
  // Internal wires for partial sums (0..N)
  logic signed [ACC_W-1:0]  psum_wire [0:M-1][0:N];

  // Generate a grid of PEs
  genvar i, j;
  generate
    for (i = 0; i < M; i++) begin: gen_rows
      for (j = 0; j < N; j++) begin: gen_cols

        // 1) On the top edge of each column j, feed in wgt_in[j]:
        assign w_wire[0][j] = wgt_in[j];
        // 2) On the left edge of each row i, feed in act_in[i]:
        assign a_wire[i][0] = act_in[i];
        // 3) Initialize partial sums at the start of each tile:
        //    if tile_start==1, we zero psum_wire at [i][j]
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n)
            psum_wire[i][j] <= '0;
          else if (tile_start)
            psum_wire[i][j] <= '0;
        end

        // Instantiate one PE at (i,j)
        systolic_PE #(.DATA_W(DATA_W), .ACC_W(ACC_W)) pe_u (
          .clk     (clk),
          .rst_n   (rst_n),
          .a_in    (a_wire[i][j]),     // input activation
          .w_in    (w_wire[i][j]),     // input weight
          .psum_in (psum_wire[i][j]),  // incoming partial sum

          .a_out    (a_wire[i][j+1]),  // to next column
          .w_out    (w_wire[i+1][j]),  // to next row
          .psum_out (psum_wire[i][j+1])// updated sum for next column
        );
      end

      // After the last column N, capture the final partial sums:
      assign psum_out[i] = psum_wire[i][N];
    end
  endgenerate

endmodule
