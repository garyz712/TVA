// ===============================================================
// ws_os_array.sv  –  one systolic grid, two modes (WS / OS)
// ===============================================================
module ws_os_array #(
  parameter int DW      = 16,       // data width
  parameter int M_TILE  = 8,        // PE rows
  parameter int N_TILE  = 8,        // PE cols
  parameter int K_TILE  = 8         // inner dimension per tile
)(
  input  logic                         clk,
  input  logic                         rst_n,

  // ---------------- control ----------------
  input  logic                         start,   // pulse to begin a tile
  input  logic                         mode,    // 0 = weight‑stationary, 1 = output‑stationary
  output logic                         done,    // high for 1 cycle when tile is finished

  // ---------------- weight tile (WS mode) --
  input  logic [DW-1:0]                W_tile   [0:M_TILE-1][0:N_TILE-1],

  // ---------------- streamed operands ------
  // a_bus : M_TILE values per cycle (row broadcast)
  input  logic [DW-1:0]                A_stream [0:M_TILE-1],
  // b_bus : N_TILE values per cycle (col broadcast) –– only used in OS mode
  input  logic [DW-1:0]                B_stream [0:N_TILE-1],

  // ---------------- result ------------------
  output logic [2*DW-1:0]              C_tile   [0:M_TILE-1][0:N_TILE-1]
);

  // ------------------------------------------------------------
  //  local FSM  :  IDLE  →  LOAD_WS  (if mode=0)  OR  CLEAR_OS
  //                → COMPUTE  (K_TILE cycles)  → DONE
  // ------------------------------------------------------------
  typedef enum logic [1:0] {S_IDLE, S_LOAD, S_COMP, S_DONE} st_t;
  st_t st, nst;

  always_ff @(posedge clk or negedge rst_n)
    if(!rst_n) st<=S_IDLE; else st<=nst;

  // counters
  logic [$clog2(M_TILE*N_TILE):0]  w_cnt;   // weight load counter
  logic [$clog2(K_TILE):0]         k_cnt;   // inner‑dim counter

  always_comb begin
    nst = st;
    case(st)
      S_IDLE:  if(start)                         nst = (mode==1'b0) ? S_LOAD : S_COMP;
      S_LOAD:  if(w_cnt == M_TILE*N_TILE-1)      nst = S_COMP;
      S_COMP:  if(k_cnt == K_TILE-1)             nst = S_DONE;
      S_DONE:                                     nst = S_IDLE;
    endcase
  end

  // counter logic & done flag
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      w_cnt <= 0;  k_cnt <= 0;  done <= 0;
    end
    else begin
      done <= (nst==S_DONE);
      w_cnt <= (st==S_LOAD) ? w_cnt+1 : 0;
      k_cnt <= (st==S_COMP) ? k_cnt+1 : 0;
    end
  end

  // ------------------------------------------------------------
  //  PE grid instantiation
  // ------------------------------------------------------------
  genvar r,c;
  logic load_en [0:M_TILE-1][0:N_TILE-1];   // one‑cycle weight write
  logic valid   [0:M_TILE-1][0:N_TILE-1];
  logic [DW-1:0] a_bus [0:M_TILE-1];
  logic [DW-1:0] b_bus [0:N_TILE-1];

  // load enable: row‑major weight loading when mode==0 (WS)
  generate
    for(r=0; r<M_TILE; r++)
      for(c=0; c<N_TILE; c++)
        assign load_en[r][c] = (mode==1'b0) && (st==S_LOAD) &&
                               (w_cnt == r*N_TILE + c);
  endgenerate

  // activation & validity
  generate
    for(r=0; r<M_TILE; r++) begin
      assign a_bus[r] = (st==S_COMP) ? A_stream[r] : '0;
      for(c=0; c<N_TILE; c++) begin
        assign b_bus[c]      = (mode==1'b1 && st==S_COMP) ? B_stream[c] : '0;
        assign valid[r][c]   = (st==S_COMP);   // always assert during COMP
      end
    end
  endgenerate

  // PEs
  generate
    for(r=0; r<M_TILE; r++) begin : ROW
      for(c=0; c<N_TILE; c++) begin : COL
        ws_os_pe #(.DW(DW)) u_pe (
          .clk      (clk),
          .rst_n    (rst_n),
          .mode     (mode),
          .load_w   (load_en[r][c]),
          .w_in     (W_tile[r][c]),
          .valid_in (valid[r][c]),
          .a_in     (a_bus[r]),
          .b_in     (b_bus[c]),
          .accum_out(C_tile[r][c])
        );
      end
    end
  endgenerate

endmodule
