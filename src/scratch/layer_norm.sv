module layer_norm #(
    parameter int DATA_WIDTH = 16,
    parameter int SEQ_LEN    = 8,   // number of tokens or rows
    parameter int EMB_DIM    = 8,   // embedding dimension
    parameter logic [31:0] EPS = 32'h34000000 // ~1e-5 in float, or small in fixed
)(
    input  logic                                      clk,
    input  logic                                      rst_n,

    // Control
    input  logic                                      start,
    output logic                                      done,

    // Input shape (SEQ_LEN, EMB_DIM), flattened
    input  logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0]     x_in,

    // LN parameters: gamma, beta => each (EMB_DIM)
    // Flattened => DATA_WIDTH*EMB_DIM for each
    input  logic [DATA_WIDTH*EMB_DIM-1:0]             gamma_in,
    input  logic [DATA_WIDTH*EMB_DIM-1:0]             beta_in,

    // Output shape (SEQ_LEN, EMB_DIM), LN(x), flattened
    output logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0]     x_out,
    output logic                                      out_valid
);

    // --------------------------------------------------------------------------
    // 1) Internal parameters and definitions
    // --------------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_ROW_MEAN,    // compute mean for current row
        S_ROW_VAR,     // compute var for current row
        S_ROW_NORM,    // normalize each element => out_mem[row_i][col_i]
        S_NEXT_ROW,    // increment row
        S_DONE
    } state_t;

    state_t curr_state, next_state;

    // We'll process row by row, so row_i in [0..SEQ_LEN-1], col_i in [0..EMB_DIM-1].
    logic [$clog2(SEQ_LEN):0] row_i;
    logic [$clog2(EMB_DIM):0] col_i;

    // We'll store the input in a 2D array x_mem[row_i][col_i]
    // We'll store the output in out_mem[row_i][col_i].
    // We'll store gamma, beta in 1D arrays: gamma_arr[col_i], beta_arr[col_i].
    // We'll keep partial sums in 32 bits for mean/var. 
    // Real LN uses float. This skeleton uses integer or naive approach.

    logic [DATA_WIDTH-1:0] x_mem [0:SEQ_LEN-1][0:EMB_DIM-1];
    logic [DATA_WIDTH-1:0] out_mem[0:SEQ_LEN-1][0:EMB_DIM-1];

    // LN parameters
    logic [DATA_WIDTH-1:0] gamma_arr [0:EMB_DIM-1];
    logic [DATA_WIDTH-1:0] beta_arr  [0:EMB_DIM-1];

    // We'll keep track of row_mean, row_var in 32 bits.
    logic [31:0] row_mean;
    logic [31:0] row_var;
    // We'll do approximate sqrt. 
    // We'll define a small function or placeholder for sqrt.

    // Epsilon
    localparam logic [31:0] EPSILON = EPS;

    // --------------------------------------------------------------------------
    // 2) Functions to access flattened x_in
    // --------------------------------------------------------------------------
    function logic [DATA_WIDTH-1:0] get_x(
        input logic [$clog2(SEQ_LEN):0] r,
        input logic [$clog2(EMB_DIM):0] c
    );
        int flat_idx = (r*EMB_DIM) + c;
        get_x = x_in[(flat_idx+1)*DATA_WIDTH -1 -: DATA_WIDTH];
    endfunction

    // --------------------------------------------------------------------------
    // 3) State register
    // --------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n)
            curr_state <= S_IDLE;
        else
            curr_state <= next_state;
    end

    // --------------------------------------------------------------------------
    // 4) Next-state logic (always_comb)
    // --------------------------------------------------------------------------
    always_comb begin
        next_state = curr_state;

        case(curr_state)
            S_IDLE: if(start) next_state = S_ROW_MEAN;

            // S_ROW_MEAN => sum up the row => row_mean
            // once col_i == EMB_DIM-1 => go S_ROW_VAR
            S_ROW_MEAN: if(col_i == (EMB_DIM-1)) next_state = S_ROW_VAR;

            // S_ROW_VAR => sum up (x - mean)^2 => row_var
            // once col_i == EMB_DIM-1 => go S_ROW_NORM
            S_ROW_VAR:  if(col_i == (EMB_DIM-1)) next_state = S_ROW_NORM;

            // S_ROW_NORM => for each col_i => out_mem[row_i][col_i] = ...
            // once col_i == EMB_DIM-1 => go S_NEXT_ROW
            S_ROW_NORM: if(col_i == (EMB_DIM-1)) next_state = S_NEXT_ROW;

            // S_NEXT_ROW => if row_i < SEQ_LEN-1 => row_i++
            // else => S_DONE
            S_NEXT_ROW: next_state = (row_i == (SEQ_LEN-1)) ? S_DONE : S_ROW_MEAN;

            S_DONE: next_state = S_IDLE;

            default: next_state = S_IDLE;
        endcase
    end

    // --------------------------------------------------------------------------
    // 5) Datapath and output logic
    // --------------------------------------------------------------------------
    integer i, j;
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            row_i     <= 0;
            col_i     <= 0;
            done      <= 1'b0;
            out_valid <= 1'b0;

            row_mean  <= 32'd0;
            row_var   <= 32'd0;

            // unpack x_in, gamma_in, beta_in in S_IDLE for convenience:
            for(i=0; i<SEQ_LEN; i++) begin
                for(j=0; j<EMB_DIM; j++) begin
                    x_mem[i][j] <= '0;
                    out_mem[i][j] <= '0;
                end
            end
            for(j=0; j<EMB_DIM; j++) begin
                gamma_arr[j] <= '0;
                beta_arr[j]  <= '0;
            end

        end else begin
            // Default
            done      <= 1'b0;
            out_valid <= 1'b0;

            case(curr_state)

                S_IDLE: begin
                    // Unpack x_in
                    for(i=0; i<SEQ_LEN; i++) begin
                        for(j=0; j<EMB_DIM; j++) begin
                            x_mem[i][j] <= get_x(i, j);
                        end
                    end
                    // Unpack gamma, beta
                    for(j=0; j<EMB_DIM; j++) begin
                        gamma_arr[j] <= gamma_in[(j+1)*DATA_WIDTH -1 -: DATA_WIDTH];
                        beta_arr[j]  <= beta_in[(j+1)*DATA_WIDTH -1 -: DATA_WIDTH];
                    end
                    row_i <= 0;
                    col_i <= 0;
                    row_mean <= 32'd0;
                    row_var  <= 32'd0;
                end

                // S_ROW_MEAN => partial sum across the row => row_mean
                // sum all x_mem[row_i][col_i], then divide by EMB_DIM at the end
                S_ROW_MEAN: begin
                    // accumulate
                    row_mean <= row_mean + x_mem[row_i][col_i];
                    if(col_i < (EMB_DIM-1)) begin
                        col_i <= col_i + 1;
                    end else begin
                        // we've done EMB_DIM sums
                        // row_mean = row_mean / EMB_DIM
                        // do integer/fixed division or floating
                        row_mean <= row_mean / EMB_DIM;
                        col_i <= 0;
                    end
                end

                // S_ROW_VAR => partial sum of (x - row_mean)^2
                S_ROW_VAR: begin
                    // let’s do (x - row_mean)
                    // interpret row_mean in same format as x 
                    // naive approach
                    logic signed [31:0] diff;
                    diff = x_mem[row_i][col_i] - row_mean;

                    // accumulate diff^2
                    row_var <= row_var + (diff * diff);

                    if(col_i < (EMB_DIM-1)) begin
                        col_i <= col_i + 1;
                    end else begin
                        // done row
                        // row_var = row_var / EMB_DIM
                        row_var <= row_var / EMB_DIM;
                        col_i <= 0;
                    end
                end

                // S_ROW_NORM => for each col => out_mem[row_i][col] = LN formula
                // LN formula: (x - mean) / sqrt(var + eps) * gamma + beta
                // We do a naive sqrt or approximation
                S_ROW_NORM: begin
                    // do one element => col_i
                    logic signed [31:0] diff;
                    diff = x_mem[row_i][col_i] - row_mean;
                    logic [31:0] denom = row_var + EPSILON; // var + eps
                    logic [31:0] inv_std;
                    // approximate 1/sqrt(denom)
                    inv_std = approximate_inv_sqrt(denom);

                    // multiply => normalized = diff * inv_std
                    // then multiply by gamma_arr[col_i], add beta_arr[col_i]
                    logic signed [31:0] tmp_norm;
                    tmp_norm = diff * inv_std; // or do proper float multiply

                    // multiply by gamma
                    // we'll treat gamma_arr[col_i] also as 16-bit => sign-extend
                    logic signed [31:0] gamma_val = gamma_arr[col_i];
                    logic signed [31:0] beta_val  = beta_arr[col_i];

                    logic signed [31:0] after_gamma;
                    after_gamma = tmp_norm * gamma_val;

                    logic signed [31:0] final_val;
                    final_val = after_gamma + beta_val;

                    // store truncated to DATA_WIDTH
                    out_mem[row_i][col_i] <= final_val[DATA_WIDTH-1:0];

                    if(col_i < (EMB_DIM-1)) begin
                        col_i <= col_i + 1;
                    end else begin
                        // done this row
                        col_i <= 0;
                    end
                end

                S_NEXT_ROW: begin
                    // increment row
                    if(row_i < (SEQ_LEN-1))
                        row_i <= row_i + 1;

                    // reset row_mean, row_var
                    row_mean <= 32'd0;
                    row_var  <= 32'd0;
                end

                S_DONE: begin
                    done      <= 1'b1;
                    out_valid <= 1'b1;

                    // flatten out_mem => x_out
                    for(i=0; i<SEQ_LEN; i++) begin
                        for(j=0; j<EMB_DIM; j++) begin
                            x_out[ ((i*EMB_DIM)+j+1)*DATA_WIDTH -1 -: DATA_WIDTH ]
                                <= out_mem[i][j];
                        end
                    end
                end

                default: /* no-op */;
            endcase
        end
    end

    // --------------------------------------------------------------------------
    // 6) approximate_inv_sqrt( val ) function
    // --------------------------------------------------------------------------
    // TOTALLY a placeholder. Real design might use:
    //   - IP core for float sqrt
    //   - LUT for 1/sqrt
    //   - Newton-Raphson pipeline
    function logic [31:0] approximate_inv_sqrt(input logic [31:0] val);
        // placeholder: if val=0 => avoid /0
        // do a naive approach. For demonstration only.
        approximate_inv_sqrt = (val == 0) ? 32'd0 : 32'd10; 
        // you could implement a better approximation or call a vendor IP
    endfunction

endmodule
