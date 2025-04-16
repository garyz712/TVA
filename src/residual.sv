module residual #(
  parameter int DATA_WIDTH = 16,
  parameter int SEQ_LEN    = 8,
  parameter int EMB_DIM    = 8
)(
  input  logic                                      clk,
  input  logic                                      rst_n,

  // Control
  input  logic                                      start,
  output logic                                      done,

  // x_in, sub_in => shape (SEQ_LEN, EMB_DIM)
  // Flattened
  input  logic [DATA_WIDTH*SEQ_LEN*EMB_DIM -1:0]    x_in,
  input  logic [DATA_WIDTH*SEQ_LEN*EMB_DIM -1:0]    sub_in,

  // Output => x_in + sub_in, shape (SEQ_LEN, EMB_DIM)
  output logic [DATA_WIDTH*SEQ_LEN*EMB_DIM -1:0]    y_out,
  output logic                                      out_valid
);

  //--------------------------------------------------------------------------
  // 1) State definitions
  //--------------------------------------------------------------------------
  typedef enum logic [1:0] {
    S_IDLE,
    S_ADD,
    S_DONE
  } state_t;

  state_t curr_state, next_state;

  // We'll index row_i in [0..SEQ_LEN-1], col_i in [0..EMB_DIM-1].
  logic [$clog2(SEQ_LEN):0] row_i;
  logic [$clog2(EMB_DIM):0] col_i;

  // local 2D arrays to store output
  logic [DATA_WIDTH-1:0] y_mem [0:SEQ_LEN-1][0:EMB_DIM-1];

  //--------------------------------------------------------------------------
  // 2) Helper functions to read x_in, sub_in
  //--------------------------------------------------------------------------
  function logic [DATA_WIDTH-1:0] get_x(
    input logic [$clog2(SEQ_LEN):0] r,
    input logic [$clog2(EMB_DIM):0] c
  );
    int flat_idx = (r*EMB_DIM)+c;
    get_x = x_in[(flat_idx+1)*DATA_WIDTH -1 -: DATA_WIDTH];
  endfunction

  function logic [DATA_WIDTH-1:0] get_sub(
    input logic [$clog2(SEQ_LEN):0] r,
    input logic [$clog2(EMB_DIM):0] c
  );
    int flat_idx = (r*EMB_DIM)+c;
    get_sub = sub_in[(flat_idx+1)*DATA_WIDTH -1 -: DATA_WIDTH];
  endfunction

  //--------------------------------------------------------------------------
  // 3) State register
  //--------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n)
      curr_state <= S_IDLE;
    else
      curr_state <= next_state;
  end

  //--------------------------------------------------------------------------
  // 4) Next-state logic (always_comb)
  //--------------------------------------------------------------------------
  always_comb begin
    next_state = curr_state;

    case(curr_state)
      S_IDLE: if(start) next_state = S_ADD;

      S_ADD: begin
        // once we've processed all row_i, col_i => S_DONE
        if(row_i == (SEQ_LEN-1) && col_i == (EMB_DIM-1))
          next_state = S_DONE;
      end

      S_DONE: next_state = S_IDLE;

      default: next_state = S_IDLE;
    endcase
  end

  //--------------------------------------------------------------------------
  // 5) Datapath + output logic
  //--------------------------------------------------------------------------
  integer i, j;
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      done      <= 1'b0;
      out_valid <= 1'b0;
      row_i     <= 0;
      col_i     <= 0;

      for(i=0; i<SEQ_LEN; i++) begin
        for(j=0; j<EMB_DIM; j++)
          y_mem[i][j] <= '0;
      end

    end else begin
      // defaults
      done      <= 1'b0;
      out_valid <= 1'b0;

      case(curr_state)
        S_IDLE: begin
          row_i <= 0;
          col_i <= 0;
        end

        S_ADD: begin
          // do one element => y_mem[row_i][col_i] = x_in + sub_in
          logic [DATA_WIDTH-1:0] xv;
          logic [DATA_WIDTH-1:0] sv;
          xv = get_x(row_i, col_i);
          sv = get_sub(row_i, col_i);
          y_mem[row_i][col_i] <= xv + sv; // watch out for saturations

          // increment col_i, row_i
          if(col_i < (EMB_DIM-1)) begin
            col_i <= col_i + 1;
          end else begin
            col_i <= 0;
            if(row_i < (SEQ_LEN-1))
              row_i <= row_i + 1;
          end
        end

        S_DONE: begin
          done      <= 1'b1;
          out_valid <= 1'b1;

          // flatten y_mem => y_out
          for(i=0; i<SEQ_LEN; i++) begin
            for(j=0; j<EMB_DIM; j++) begin
              y_out[ ((i*EMB_DIM)+ j +1)*DATA_WIDTH -1 -: DATA_WIDTH ]
                <= y_mem[i][j];
            end
          end
        end

        default: /* no-op */;
      endcase
    end
  end

endmodule
