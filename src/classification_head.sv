module classification_head #(
  parameter int DATA_WIDTH   = 16,    // bits for each element
  parameter int E            = 128,   // embedding dimension
  parameter int NUM_CLASSES  = 1000   // number of classes
)(
  input  logic                                   clk,
  input  logic                                   rst_n,

  // Control
  input  logic                                   start,
  output logic                                   done,

  // Input: final embedding vector shape (E) 
  // Flattened => DATA_WIDTH*E
  input  logic [DATA_WIDTH*E -1:0]               cls_in,

  // Weights W_clf => shape (E x NUM_CLASSES), flattened
  // b_clf => shape (NUM_CLASSES)
  input  logic [DATA_WIDTH*E*NUM_CLASSES -1:0]   W_clf_in,
  input  logic [DATA_WIDTH*NUM_CLASSES -1:0]     b_clf_in,

  // Output => shape (NUM_CLASSES), flattened
  // => DATA_WIDTH*NUM_CLASSES
  output logic [DATA_WIDTH*NUM_CLASSES -1:0]     logits_out,
  output logic                                   out_valid
);

  //****************************************************************
  // 1) Local Parameters and FSM States
  //****************************************************************
  typedef enum logic [1:0] {
    S_IDLE,
    S_DOT,   // accumulate partial sums for each class
    S_DONE
  } state_t;

  state_t curr_state, next_state;

  // We'll treat the input embedding as a vector of length E:
  //   cls_in_vec[e] for e in [0..E-1]
  // We'll store the weight matrix in a local 2D array: W_clf[e][class]
  // We'll store the bias b_clf[class]
  // We'll produce an array logits_mem[class].

  //****************************************************************
  // 2) Internal Storage
  //****************************************************************
  logic [DATA_WIDTH-1:0] cls_vec [0:E-1];
  logic [DATA_WIDTH-1:0] b_clf_arr [0:NUM_CLASSES-1];
  logic [DATA_WIDTH-1:0] W_clf_arr [0:E-1][0:NUM_CLASSES-1];

  // For the final output, we keep partial sums for each class
  // up to 32 bits if we do integer accumulations:
  logic [31:0] logits_mem [0:NUM_CLASSES-1];

  // We'll have counters
  logic [$clog2(NUM_CLASSES):0] class_idx;
  logic [$clog2(E):0]           emb_idx;

  // A partial sum register for the current class:
  logic [31:0] sum_temp;

  //****************************************************************
  // 3) State Register
  //****************************************************************
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n)
      curr_state <= S_IDLE;
    else
      curr_state <= next_state;
  end

  //****************************************************************
  // 4) Next-State Logic (always_comb)
  //****************************************************************
  always_comb begin
    next_state = curr_state;
    case(curr_state)
      S_IDLE: begin
        if(start)
          next_state = S_DOT;
      end

      // S_DOT => we do partial dot product for each class dimension
      // once we do all E elements => store sum => next class
      // once we do all classes => S_DONE
      S_DOT: begin
        if((class_idx == (NUM_CLASSES-1)) && (emb_idx == E)) 
          next_state = S_DONE;
      end

      S_DONE: begin
        // go idle or remain done
        next_state = S_IDLE;
      end

      default: next_state = S_IDLE;
    endcase
  end

  //****************************************************************
  // 5) Unpack Weights and Input in always_comb
  //****************************************************************
  // We can do a loop:
  integer i, j;
  always_comb begin
    // cls_in => cls_vec
    for(i=0; i<E; i++) begin
      cls_vec[i] = cls_in[(i+1)*DATA_WIDTH -1 -: DATA_WIDTH];
    end

    // b_clf_in => b_clf_arr
    for(i=0; i<NUM_CLASSES; i++) begin
      b_clf_arr[i] = b_clf_in[(i+1)*DATA_WIDTH -1 -: DATA_WIDTH];
    end

    // W_clf_in => W_clf_arr[e][class]
    // Flattening: index = (e*NUM_CLASSES + class)
    for(i=0; i<E; i++) begin
      for(j=0; j<NUM_CLASSES; j++) begin
        W_clf_arr[i][j] 
          = W_clf_in[ ((i*NUM_CLASSES)+ j+1)*DATA_WIDTH -1 -: DATA_WIDTH ];
      end
    end
  end

  //****************************************************************
  // 6) Main Datapath in always_ff
  //****************************************************************
  integer c;
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      done      <= 1'b0;
      out_valid <= 1'b0;

      for(c=0; c<NUM_CLASSES; c++) begin
        logits_mem[c] <= 32'd0;
      end

      class_idx <= 0;
      emb_idx   <= 0;
      sum_temp  <= 32'd0;

    end else begin
      // defaults
      done      <= 1'b0;
      out_valid <= 1'b0;

      case(curr_state)
        S_IDLE: begin
          // no action, waiting for start
          // reset counters
          class_idx <= 0;
          emb_idx   <= 0;
          sum_temp  <= '0;
          for(c=0; c<NUM_CLASSES; c++) begin
            logits_mem[c] <= 32'd0;
          end
        end

        S_DOT: begin
          // partial dot product for the current class => class_idx
          // sum_temp accumulates: sum_temp += cls_vec[emb_idx] * W_clf_arr[emb_idx][class_idx]
          // if emb_idx==0 => init sum_temp = b_clf_arr[class_idx]
          if(emb_idx == 0) begin
            // start new class => reset partial sum with bias
            sum_temp <= {{(32-DATA_WIDTH){1'b0}}, b_clf_arr[class_idx]}; 
          end else begin
            // multiply
            sum_temp <= sum_temp + ( cls_vec[emb_idx-1] * W_clf_arr[emb_idx-1][class_idx] );
          end

          if(emb_idx < E) begin
            emb_idx <= emb_idx + 1;
          end else begin
            // store final sum for this class => after finishing E multiples
            // in real design, watch sign extension, Q format, etc.
            // we store in logits_mem[class_idx]
            logits_mem[class_idx] <= sum_temp;

            // move to next class
            emb_idx <= 0;
            if(class_idx < (NUM_CLASSES-1)) begin
              class_idx <= class_idx + 1;
            end
          end
        end

        S_DONE: begin
          // flatten logits_mem => logits_out
          integer cc;
          for(cc=0; cc<NUM_CLASSES; cc++) begin
            // truncate to DATA_WIDTH if needed
            logits_out[ ((cc+1)*DATA_WIDTH)-1 -: DATA_WIDTH ]
              <= logits_mem[cc][DATA_WIDTH-1:0];
          end
          done      <= 1'b1;
          out_valid <= 1'b1;
        end

        default: /* no-op */;
      endcase
    end
  end

endmodule
