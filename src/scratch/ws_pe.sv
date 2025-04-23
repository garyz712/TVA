// File: ws_pe.sv
module ws_pe #(
  parameter int DATA_WIDTH = 16
)(
  input  logic                     clk,
  input  logic                     rst_n,

  // control
  input  logic                     load_w,    // high one cycle to load new weight
  input  logic                     valid_in,  // indicates data_in is valid
  input  logic [DATA_WIDTH-1:0]    w_in,       // new weight when load_w=1
  input  logic [DATA_WIDTH-1:0]    a_in,       // activation stream

  // outputs
  output logic [2*DATA_WIDTH-1:0]  accum_out,  // final dot-product result
  output logic                     valid_out   // goes high when multiplication happened
);

  // local weight register
  logic [DATA_WIDTH-1:0] w_local;
  // local accumulator
  logic [2*DATA_WIDTH-1:0] accum;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      w_local <= '0;
      accum   <= '0;
    end else begin
      if (load_w) begin
        w_local <= w_in;      // latch the weight once
      end
      if (valid_in) begin
        accum <= accum + (w_local * a_in);
      end
    end
  end

  assign accum_out = accum;
  assign valid_out = valid_in;

endmodule
