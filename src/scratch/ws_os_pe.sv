// ws_os_pe.sv  – one PE that supports WS (mode=0) and OS (mode=1)
module ws_os_pe #(
  parameter int DW = 16          // data width
)(
  input  logic                     clk,
  input  logic                     rst_n,

  // mode control
  input  logic                     mode,     // 0 = WS, 1 = OS

  // weight‑stationary interface (mode=0)
  input  logic                     load_w,   // 1 exactly one cycle when its weight is on w_in
  input  logic [DW-1:0]            w_in,

  // streaming operands (both modes)
  input  logic                     valid_in,
  input  logic [DW-1:0]            a_in,
  input  logic [DW-1:0]            b_in,

  // accumulate result
  output logic [2*DW-1:0]          accum_out
);

  // PE registers
  logic [DW-1:0]   w_reg;          // stationary weight (WS)  OR  streamed b (OS)
  logic [2*DW-1:0] psum;

  // ------------------------------------------------------------------
  // register update
  // ------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      w_reg <= '0;
      psum  <= '0;
    end
    else begin
      //----------------------------------------------------------------
      // weight register
      //----------------------------------------------------------------
      if (mode==1'b0) begin                        // WS: preload weight
        if (load_w)  w_reg <= w_in;
      end
      else begin                                   // OS: b value changes every cycle
        if (valid_in) w_reg <= b_in;               // treat b_in as w_reg
      end

      //----------------------------------------------------------------
      // MAC
      //----------------------------------------------------------------
      if (valid_in) begin
        // multiply a_in by w_reg whatever interpretation
        psum <= psum + (a_in * w_reg);
      end
    end
  end

  // ------------------------------------------------------------------
  // clear psum at start of every C‑tile (external signal)
  // (simpler: expose clr_accum port; here we use load_w==1 & mode==0)
  // ------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n)
    if(!rst_n)
      psum <= '0;
    else if(load_w && mode==1'b0)   // first load of a new output row‑tile
      psum <= '0;

  assign accum_out = psum;

endmodule
