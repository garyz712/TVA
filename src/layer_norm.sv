//------------------------------------------------------------------------------
// layer_norm.sv
// Implements layer normalization for a sequence of tokens with given embedding dimensions.
// Computes mean and variance per row, normalizes input, and applies gamma/beta scaling.
// Operates on flattened input/output arrays with fixed-point arithmetic.
//
// May 3 2025    Max Zhang      Initial version
// May 3 2025    Tianwei Liu    Fix syntax issue
// May 31 2025   Max Zhang      Modified to output rows in standard row-major order
//------------------------------------------------------------------------------
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
    input  logic [DATA_WIDTH*EMB_DIM-1:0]             gamma_in,
    input  logic [DATA_WIDTH*EMB_DIM-1:0]             beta_in,

    // Output shape (SEQ_LEN, EMB_DIM), LN(x), flattened
    output logic [DATA_WIDTH*SEQ_LEN*EMB_DIM-1:0]     x_out,
    output logic                                      out_valid
);

    // ------------------------------------------------------------------
    // State machine
    // ------------------------------------------------------------------
    typedef enum logic [2:0] { S_IDLE, S_ROW_MEAN, S_ROW_VAR,
                               S_ROW_NORM, S_NEXT_ROW, S_DONE } state_t;
    state_t curr_state, next_state;

    // Row / column indices
    logic [2:0] row_i, col_i;   // 0-7

    // ------------------------------------------------------------------
    // Memories
    // ------------------------------------------------------------------
    logic [DATA_WIDTH-1:0] x_mem [0:SEQ_LEN-1][0:EMB_DIM-1];
    logic [DATA_WIDTH-1:0] out_mem[0:SEQ_LEN-1][0:EMB_DIM-1];
    logic [DATA_WIDTH-1:0] gamma_arr [0:EMB_DIM-1];
    logic [DATA_WIDTH-1:0] beta_arr  [0:EMB_DIM-1];

    // ------------------------------------------------------------------
    // Accumulators
    // ------------------------------------------------------------------
    logic signed [31:0] row_mean, row_var;
    logic signed [31:0] mean_acc, var_acc;

    localparam logic [31:0] EPSILON = EPS;

    // ------------------------------------------------------------------
    // Helper: fetch flattened x_in  (width-clean)
    // ------------------------------------------------------------------
    function automatic logic [DATA_WIDTH-1:0] get_x(
        input logic [2:0] r,
        input logic [2:0] c
    );
        logic [31:0] r_ext = {29'b0, r};
        logic [31:0] c_ext = {29'b0, c};
        logic [31:0] flat_idx = (r_ext * EMB_DIM) + c_ext;
        return x_in[(flat_idx * DATA_WIDTH) +: DATA_WIDTH];
    endfunction

    // ------------------------------------------------------------------
    // Floor division helper (signed, truncates toward -∞)
    // ------------------------------------------------------------------
    function automatic logic signed [31:0] floor_div32(
        input logic signed [31:0] numer,
        input int                 denom        // positive constant (8)
    );
        logic signed [31:0] q  = numer / denom;   // truncate-to-zero
        logic signed [31:0] r  = numer % denom;   // remainder has same sign as numer
        if ((r != 0) && (numer < 0))  // adjust when numerator is negative and rem != 0
            q = q - 1;
        return q;
    endfunction

    // ------------------------------------------------------------------
    // State register
    // ------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) curr_state <= S_IDLE;
        else        curr_state <= next_state;
    end

    // Next-state logic
    always_comb begin
        next_state = curr_state;
        case (curr_state)
            S_IDLE     : if (start)         next_state = S_ROW_MEAN;
            S_ROW_MEAN : if (col_i == 3'd7) next_state = S_ROW_VAR;
            S_ROW_VAR  : if (col_i == 3'd7) next_state = S_ROW_NORM;
            S_ROW_NORM : if (col_i == 3'd7) next_state = S_NEXT_ROW;
            S_NEXT_ROW : begin
                if (row_i == 3'd7)
                    next_state = S_DONE;
                else
                    next_state = S_ROW_MEAN;
            end
            S_DONE     :                   next_state = S_IDLE;
            default: ;
        endcase
    end

    // ------------------------------------------------------------------
    // Datapath signals
    // ------------------------------------------------------------------
    logic signed [31:0] new_mean_acc;
    logic signed [31:0] diff;
    logic signed [31:0] new_var_acc;
    logic signed [31:0] denom;
    logic signed [31:0] inv_std;
    logic signed [31:0] tmp_norm;
    logic signed [31:0] gamma_val;
    logic signed [31:0] beta_val;
    logic signed [31:0] after_gamma;
    logic signed [31:0] final_val;

    logic [DATA_WIDTH-1:0] x_mem_temp [0:SEQ_LEN-1][0:EMB_DIM-1];

    // Combinational calculations
    always_comb begin
        new_mean_acc = mean_acc + {{16{x_mem[row_i][col_i][15]}}, x_mem[row_i][col_i]};
        diff = {{16{x_mem[row_i][col_i][15]}}, x_mem[row_i][col_i]} - row_mean;
        new_var_acc = var_acc + (diff * diff);
        denom = row_var + EPSILON;
        inv_std = approximate_inv_sqrt(denom);
        tmp_norm = diff * inv_std;
        gamma_val = {{16{gamma_arr[col_i][15]}}, gamma_arr[col_i]};
        beta_val = {{16{beta_arr[col_i][15]}}, beta_arr[col_i]};
        after_gamma = tmp_norm * gamma_val;
        final_val = after_gamma + beta_val;

        // Compute x_mem_temp combinationaly
        for (int i = 0; i < SEQ_LEN; i++)
            for (int j = 0; j < EMB_DIM; j++)
                x_mem_temp[i][j] = get_x(i[2:0], j[2:0]);
    end

    // ------------------------------------------------------------------
    // Datapath and control
    // ------------------------------------------------------------------
    integer i, j;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_i     <= 3'd0;
            col_i     <= 3'd0;
            done      <= 1'b0;
            out_valid <= 1'b0;
            row_mean  <= 32'd0;
            row_var   <= 32'd0;
            mean_acc  <= 32'd0;
            var_acc   <= 32'd0;
            for (i = 0; i < SEQ_LEN; i++)
                for (j = 0; j < EMB_DIM; j++) begin
                    x_mem [i][j] <= '0;
                    out_mem[i][j] <= '0;
                end
            for (j = 0; j < EMB_DIM; j++) begin
                gamma_arr[j] <= '0;
                beta_arr [j] <= '0;
            end
        end
        else begin
            done      <= 1'b0;
            out_valid <= 1'b0;

            case (curr_state)
                //------------------------------------------------------
                S_IDLE: begin
                    for (i = 0; i < SEQ_LEN; i++)
                        for (j = 0; j < EMB_DIM; j++)
                            x_mem[i][j] <= get_x(i[2:0], j[2:0]);

                    for (j = 0; j < EMB_DIM; j++) begin
                        gamma_arr[j] <= gamma_in[(j*DATA_WIDTH) +: DATA_WIDTH];
                        beta_arr [j] <= beta_in [(j*DATA_WIDTH) +: DATA_WIDTH];
                    end

                    row_i    <= 3'd0;
                    col_i    <= 3'd0;
                    mean_acc <= 32'd0;
                    var_acc  <= 32'd0;
                end

                //------------------------------------------------------
                // Mean accumulation
                // ------------------------------------------------------
                S_ROW_MEAN: begin
                    mean_acc <= new_mean_acc;

                    if (col_i == 3'd7) begin
                        row_mean <= floor_div32(new_mean_acc, EMB_DIM);
                        col_i    <= 3'd0;
                        mean_acc <= 32'd0;
                    end
                    else
                        col_i <= col_i + 1;
                end

                //------------------------------------------------------
                // Variance accumulation
                //------------------------------------------------------
                S_ROW_VAR: begin
                    var_acc <= new_var_acc;

                    if (col_i == 3'd7) begin
                        row_var <= floor_div32(new_var_acc, EMB_DIM);
                        col_i   <= 3'd0;
                        var_acc <= 32'd0;
                    end
                    else
                        col_i <= col_i + 1;
                end

                //------------------------------------------------------
                // Normalisation + γ/β
                //------------------------------------------------------
                S_ROW_NORM: begin
                    if (final_val > 32'sd32767)
                        out_mem[row_i][col_i] <= 16'sd32767;
                    else if (final_val < -32'sd32768)
                        out_mem[row_i][col_i] <= -16'sd32768;
                    else
                        out_mem[row_i][col_i] <= final_val[15:0];

                    if (col_i == 3'd7)
                        col_i <= 3'd0;
                    else
                        col_i <= col_i + 1;
                end

                //------------------------------------------------------
                S_NEXT_ROW: begin
                    if (row_i < 3'd7)
                        row_i <= row_i + 1;

                    row_mean <= 32'd0;
                    mean_acc <= 32'd0;
                    row_var  <= 32'd0;
                    var_acc  <= 32'd0;
                end

                //------------------------------------------------------
                S_DONE: begin
                    done      <= 1'b1;
                    out_valid <= 1'b1;
                    for (i = 0; i < SEQ_LEN; i++)
                        for (j = 0; j < EMB_DIM; j++)
                            x_out[(i*EMB_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= out_mem[i][j];
                end
                default: ;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // Cheap placeholder inv-sqrt
    // ------------------------------------------------------------------
    function automatic logic [31:0] approximate_inv_sqrt(input logic [31:0] val);
        return (val == 0) ? 32'd0 : 32'd10;
    endfunction
endmodule
