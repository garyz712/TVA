//------------------------------------------------------------------------------
// precision_assigner.sv
//
// precision_assigner
// This module analyzes a flattened 3D attention matrix A_in of shape (L, N, L)
// and assigns a 4-bit precision code to each token (along the last dimension L).
// It sums values across the batch (N) and input sequence (L) dimensions for
// each token, then determines a precision level based on thresholding the sum.
//
// The output is an array of 2-bit precision codes, one per token.
// The operation is driven by a start signal and signals completion via a done output.
// An internal FSM manages the steps: summing elements, thresholding sums, and updating outputs.
//
// Apr. 21 2025    Max Zhang      Initial version
// Apr. 26 2025    Tianwei Liu    Syntax fix and comments
// Apr. 28 2025    Tianwei Liu    Fix bug
//------------------------------------------------------------------------------
module precision_assigner #(
    parameter int DATA_WIDTH = 16,  // e.g. 16 bits per A element
    parameter int L = 8,           // number of tokens in the last dimension
    parameter int N = 1            // batch size
)(
    input  logic                              clk,
    input  logic                              rst_n,

    // Control
    input  logic                              start,
    output logic                              done,

    // Flattened attention matrix A_in: shape (L, N, L) => total L*N*L elements
    // Typically: A[l, n, l2].
    input  logic [DATA_WIDTH*L*N*L-1:0]       A_in,

    // Output: 2-bit precision code per token (l2), total L codes
    output logic [1:0]                        token_precision [0:L-1]
);

    // ---------------------------------------------------------------
    // 1) Local parameters for indexing
    // ---------------------------------------------------------------
    localparam int TOT_ROWS  = L * N;  // Summation dimension = # of (l,n) combos
    localparam int TOT_COLS  = L;      // # of tokens (the last dimension, l2)
    // So the total elements in A_in is TOT_ROWS * TOT_COLS = L*N*L

    // We'll keep a local array to store the 2-bit code for each l2
    logic [1:0] token_prec_array [0:TOT_COLS-1];

    // here we just directly output an array of 2-bit codes.

    // ---------------------------------------------------------------
    // 2) FSM states
    // ---------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_PREP,        // reset counters
        S_SUM_COL,     // accumulate sum over TOT_ROWS for one col_l2
        S_DECIDE,      // threshold logic => token_prec_array[col_l2]
        S_CHECK_DONE,  // check if we have more columns
        S_DONE
    } state_t;

    state_t curr_state, next_state;

    // ---------------------------------------------------------------
    // 3) Registers & counters
    // ---------------------------------------------------------------
    // col_l2: which column (token) we’re summing
    // row_i: which row in the flattened (L*N) dimension
    logic [$clog2(TOT_COLS)-1:0]  col_l2; 
    logic [$clog2(TOT_ROWS)-1:0] row_i;

    // We'll keep partial sums in a bigger width if L*N can be large
    // e.g., 32 bits is safer if data can accumulate.
    logic [31:0] sum_temp;

    // ---------------------------------------------------------------
    // 4) State register
    // ---------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n)
            curr_state <= S_IDLE;
        else
            curr_state <= next_state;
    end

    // ---------------------------------------------------------------
    // 5) Next-state logic (always_comb)
    // ---------------------------------------------------------------
    always_comb begin
        // Default
        next_state = curr_state;

        case(curr_state)

            S_IDLE: begin
                if(start)
                    next_state = S_PREP;
            end

            S_PREP: begin
                // once we set counters, move on
                next_state = S_SUM_COL;
            end

            // S_SUM_COL => accumulate TOT_ROWS elements for this col
            S_SUM_COL: begin
                if(row_i == (TOT_ROWS-1))
                    next_state = S_DECIDE;
            end

            // S_DECIDE => threshold logic => assign token_prec_array[col_l2]
            S_DECIDE: begin
                next_state = S_CHECK_DONE;
            end

            S_CHECK_DONE: begin
                // if col_l2 < TOT_COLS-1, move to next column
                // else done
                if(col_l2 == (TOT_COLS-1))
                    next_state = S_DONE;
                else
                    next_state = S_PREP; // set counters again for next col
            end

            S_DONE: begin
                // remain or go to IDLE
                next_state = S_IDLE;
            end

            default: next_state = S_IDLE;

        endcase
    end

    // ---------------------------------------------------------------
    // 6) Output logic + registers (always_ff)
    // ---------------------------------------------------------------
    // We'll handle summation in S_SUM_COL. 
    // We'll handle threshold in S_DECIDE.
    // We'll output `done` and fill `token_precision` array at S_DONE.
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            done      <= 1'b0;
            col_l2    <= '0;
            row_i     <= '0;
            sum_temp  <= 32'd0;

            // Initialize token_prec_array
            for(int i=0; i<TOT_COLS; i++) token_prec_array[i] <= 4'd0;

        end else begin
            // Defaults each cycle
            done <= 1'b0;

            case(curr_state)

                S_PREP: begin
                    // prepare counters for a new column
                    row_i    <= 0;
                    sum_temp <= 32'd0; // reset sum
                end

                S_SUM_COL: begin
                    // Add A[row_i, col_l2] to sum_temp
                    // We interpret row_i in [0.. TOT_ROWS-1], col_l2 in [0.. TOT_COLS-1].
                    // Flattening formula: index = (row_i*TOT_COLS + col_l2).
                    // Each element is DATA_WIDTH bits.

                    logic [DATA_WIDTH-1:0] a_val;
                    a_val = A_in[
                        ((row_i * TOT_COLS) + col_l2 + 1)*DATA_WIDTH -1 -: DATA_WIDTH
                    ];

                    sum_temp <= sum_temp + a_val;

                    // increment row_i
                    if(row_i < (TOT_ROWS-1))
                        row_i <= row_i + 1;
                end

                S_DECIDE: begin
                    // threshold logic => token_prec_array[col_l2]
                    if(sum_temp <  16384) // if less than 1/2
                        token_prec_array[col_l2] <= 2'd0; // e.g. int4
                    else if(sum_temp < 32768) // if less than 1
                        token_prec_array[col_l2] <= 2'd1; // e.g. int8
                    else
                        token_prec_array[col_l2] <= 2'd2; // e.g. fp16
                end

                S_CHECK_DONE: begin
                    // if more columns, increment col_l2
                    if(col_l2 < (TOT_COLS-1))
                        col_l2 <= col_l2 + 1;
                    else
                        col_l2 <= 0; // or keep it at TOT_COLS-1
                end

                S_DONE: begin
                    // Latch the final done 
                    done <= 1'b1;
                    // token_prec_array is stable. 
                    // We'll drive the token_precision output below.
                end

                default: /* do nothing */;
            endcase
        end
    end

    // ---------------------------------------------------------------
    // 7) Drive the output array
    // ---------------------------------------------------------------
    // If your tool supports returning a packed array of 4 bits,
    // you can simply tie the output to the array in always_comb 
    // or do it in a final state. We'll do it in a small always_comb:
    always_comb begin
        for(int c=0; c< L; c++) begin
            token_precision[c] = token_prec_array[c];
        end
    end

endmodule

