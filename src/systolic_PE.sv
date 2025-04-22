//-----------------------------------------------------------------------------
// systolic_PE.sv
//-----------------------------------------------------------------------------

module systolic_PE #(
  parameter int DATA_W = 16,    // bit‑width of inputs a_in, w_in
  parameter int ACC_W  = 32     // bit‑width of the partial sum
)(
  input  logic                  clk,      // clock
  input  logic                  rst_n,    // active‑low reset

  // Inputs
  input  logic signed [DATA_W-1:0] a_in,   // activation coming from left
  input  logic signed [DATA_W-1:0] w_in,   // weight coming from above

  // Feedback of accumulated sum
  input  logic signed [ACC_W-1:0]  psum_in,

  // Outputs
  output logic signed [ACC_W-1:0]  psum_out, // updated partial sum
  output logic signed [DATA_W-1:0] a_out,    // pass a_in to right
  output logic signed [DATA_W-1:0] w_out     // pass w_in downward
);

  // On each rising clock edge (or reset), update outputs:
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // If reset is asserted, clear all registers:
      psum_out <= '0;  // zero the accumulator
      a_out    <= '0;  // clear the passed‑through activation
      w_out    <= '0;  // clear the passed‑through weight
    end else begin
      // Normal operation:

      // 1) Multiply a_in * w_in, then add previous partial sum:
      psum_out <= psum_in + a_in * w_in;

      // 2) Forward the input activation one column to the right:
      a_out    <= a_in;

      // 3) Forward the input weight one row down:
      w_out    <= w_in;
    end
  end

endmodule
