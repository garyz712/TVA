//------------------------------------------------------------------------------
// softmax_approx.sv
//
// Enhanced softmax_approx with LUT-based reciprocal
// This module computes a row-wise softmax approximation for attention scores
// with enhanced accuracy through a reciprocal lookup table.
//
// It takes an input matrix A_in of shape (L, N, L) and produces normalized
// attention weights A_out of the same shape. The implementation:
// 1. Computes row sums
// 2. Uses a lookup table for fast reciprocal approximation
// 3. Normalizes each element by multiplying with the reciprocal value
//
// May 05 2025    Tianwei Liu    Enhanced version of softmax with LUT
// May 25 2025    Tianwei Liu    Use ReLU activation
//------------------------------------------------------------------------------

module softmax_approx #(
    parameter int DATA_WIDTH = 16,
    parameter int L = 8,           // Sequence length
    parameter int N = 1,           // Number of attention heads
    parameter int FRAC_BITS = 8,   // Fractional bits for fixed-point
    parameter int LUT_ADDR_WIDTH = 8  // Controls LUT size (2^LUT_ADDR_WIDTH entries)
)(
    input  logic                        clk,
    input  logic                        rst_n,
    // Control signals
    input  logic                        start,
    output logic                        done,
    // Input: A_in of shape (L, N, L)
    input  logic [DATA_WIDTH*L*N*L-1:0] A_in,
    // Output: A_out of shape (L, N, L)
    output logic [DATA_WIDTH*L*N*L-1:0] A_out,
    output logic                        out_valid
);

    // Internal array storage
    logic [DATA_WIDTH-1:0] A_arr [L][N][L];
    logic [DATA_WIDTH-1:0] A_relu [L][N][L];    // After ReLU
    logic [DATA_WIDTH-1:0] A_out_arr [L][N][L]; // Final output

    // State machine
    typedef enum logic [2:0] {
        S_IDLE      = 3'd0,
        S_LOAD      = 3'd1,
        S_RELU      = 3'd2,
        S_SUM       = 3'd3,
        S_RECIP     = 3'd4,
        S_NORMALIZE = 3'd5,
        S_OUTPUT    = 3'd6,
        S_DONE      = 3'd7
    } state_t;
    state_t state, next_state;

    // Counters
    logic [$clog2(L)-1:0] row_cnt;
    logic [$clog2(N)-1:0] head_cnt;
    logic [$clog2(L)-1:0] col_cnt;
    
    // Flags
    logic load_done, relu_done, sum_done, recip_done, normalize_done, output_done;

    // Temporary registers for calculation
    logic [DATA_WIDTH+$clog2(L)-1:0] row_sum [L][N];  // Sum of ReLU values
    logic [DATA_WIDTH-1:0] reciprocal [L][N];
    logic [DATA_WIDTH*2-1:0] scaled_value;
    
    // Reciprocal Lookup Table
    logic [DATA_WIDTH-1:0] recip_lut [2**LUT_ADDR_WIDTH-1:0];
    
    // LUT address calculation
    logic [LUT_ADDR_WIDTH-1:0] recip_lut_addr;
    
    // LUT initialization
    initial begin        
        // Initialize reciprocal LUT with 1/x values
        for (int i = 1; i < 2**LUT_ADDR_WIDTH; i++) begin
            real recip_float = 1.0 / real'(i);
            recip_lut[i] = int'(recip_float * (2.0 ** FRAC_BITS));
        end
        recip_lut[0] = (1 << (DATA_WIDTH - 1)) - 1; // Max value for division by zero case
    end

    // Sequential state transition
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            out_valid <= 1'b0;
            row_cnt <= '0;
            head_cnt <= '0;
            col_cnt <= '0;
            load_done <= 1'b0;
            relu_done <= 1'b0;
            sum_done <= 1'b0;
            recip_done <= 1'b0;
            normalize_done <= 1'b0;
            output_done <= 1'b0;
        end else begin
            state <= next_state;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    out_valid <= 1'b0;
                    row_cnt <= '0;
                    head_cnt <= '0;
                    col_cnt <= '0;
                    load_done <= 1'b0;
                    relu_done <= 1'b0;
                    sum_done <= 1'b0;
                    recip_done <= 1'b0;
                    normalize_done <= 1'b0;
                    output_done <= 1'b0;
                end

                S_LOAD: begin
                    load_done <= 1'b1;
                end

                S_RELU: begin
                    // Increment counters for ReLU calculation
                    if (col_cnt == L-1) begin
                        col_cnt <= '0;
                        if (head_cnt == N-1) begin
                            head_cnt <= '0;
                            if (row_cnt == L-1) begin
                                row_cnt <= '0;
                                relu_done <= 1'b1;
                            end else begin
                                row_cnt <= row_cnt + 1;
                            end
                        end else begin
                            head_cnt <= head_cnt + 1;
                        end
                    end else begin
                        col_cnt <= col_cnt + 1;
                    end
                end

                S_SUM: begin
                    // Initialize row sums when entering this state
                    if (!sum_done && row_cnt == 0 && head_cnt == 0 && col_cnt == 0) begin
                        for (int i = 0; i < L; i++) begin
                            for (int j = 0; j < N; j++) begin
                                row_sum[i][j] <= '0;
                            end
                        end
                    end
                    
                    // Increment counters for row sum calculation
                    if (col_cnt == L-1) begin
                        col_cnt <= '0;
                        if (head_cnt == N-1) begin
                            head_cnt <= '0;
                            if (row_cnt == L-1) begin
                                row_cnt <= '0;
                                sum_done <= 1'b1;
                            end else begin
                                row_cnt <= row_cnt + 1;
                            end
                        end else begin
                            head_cnt <= head_cnt + 1;
                        end
                    end else begin
                        col_cnt <= col_cnt + 1;
                    end
                end
                
                S_RECIP: begin
                    // Increment counters for reciprocal calculation
                    if (head_cnt == N-1) begin
                        head_cnt <= '0;
                        if (row_cnt == L-1) begin
                            row_cnt <= '0;
                            recip_done <= 1'b1;
                        end else begin
                            row_cnt <= row_cnt + 1;
                        end
                    end else begin
                        head_cnt <= head_cnt + 1;
                    end
                end

                S_NORMALIZE: begin
                    // Increment counters for normalization
                    if (col_cnt == L-1) begin
                        col_cnt <= '0;
                        if (head_cnt == N-1) begin
                            head_cnt <= '0;
                            if (row_cnt == L-1) begin
                                row_cnt <= '0;
                                normalize_done <= 1'b1;
                            end else begin
                                row_cnt <= row_cnt + 1;
                            end
                        end else begin
                            head_cnt <= head_cnt + 1;
                        end
                    end else begin
                        col_cnt <= col_cnt + 1;
                    end
                end
                
                S_OUTPUT: begin
                    output_done <= 1'b1;
                end

                S_DONE: begin
                    out_valid <= 1'b1;
                    done <= 1'b1;
                end
            endcase
        end
    end

    // Next state logic
    always_comb begin
        next_state = state;
        case (state)
            S_IDLE:      if (start)           next_state = S_LOAD;
            S_LOAD:      if (load_done)       next_state = S_RELU;
            S_RELU:      if (relu_done)       next_state = S_SUM;
            S_SUM:       if (sum_done)        next_state = S_RECIP;
            S_RECIP:     if (recip_done)      next_state = S_NORMALIZE;
            S_NORMALIZE: if (normalize_done)  next_state = S_OUTPUT;
            S_OUTPUT:    if (output_done)     next_state = S_DONE;
            S_DONE:                           next_state = S_IDLE;
            default:                          next_state = S_IDLE;
        endcase
    end

    // Load input data
    always_ff @(posedge clk) begin
        if (state == S_LOAD) begin
            for (int l = 0; l < L; l++) begin
                for (int n = 0; n < N; n++) begin
                    for (int l2 = 0; l2 < L; l2++) begin
                        A_arr[l][n][l2] <= A_in[((l*N*L)+(n*L)+l2)*DATA_WIDTH +: DATA_WIDTH];
                    end
                end
            end
        end
    end

    // ReLU activation: max(0, x)
    always_ff @(posedge clk) begin
        if (state == S_RELU) begin
            // Check if the value is negative (MSB = 1 in signed representation)
            if (A_arr[row_cnt][head_cnt][col_cnt][DATA_WIDTH-1] == 1'b1) begin
                A_relu[row_cnt][head_cnt][col_cnt] <= '0; // Negative -> 0
            end else begin
                A_relu[row_cnt][head_cnt][col_cnt] <= A_arr[row_cnt][head_cnt][col_cnt]; // Positive -> keep
            end
        end
    end

    // Row sum calculation (sum of ReLU values)
    always_ff @(posedge clk) begin
        if (state == S_SUM) begin
            // Add current ReLU value to row sum
            row_sum[row_cnt][head_cnt] <= row_sum[row_cnt][head_cnt] + A_relu[row_cnt][head_cnt][col_cnt];
        end
    end
    
    // Reciprocal calculation using LUT
    always_ff @(posedge clk) begin
        if (state == S_RECIP) begin
            // Calculate LUT address by taking appropriate bits of the row sum
            // Scale the sum to fit in LUT address range
            logic [DATA_WIDTH+$clog2(L)-1:0] sum_val;
            sum_val = row_sum[row_cnt][head_cnt];
            
            if (sum_val == 0) begin
                reciprocal[row_cnt][head_cnt] <= recip_lut[0]; // Max reciprocal for zero sum
            end else if (sum_val >= (1 << LUT_ADDR_WIDTH)) begin
                recip_lut_addr <= (1 << LUT_ADDR_WIDTH) - 1; // Max index
                reciprocal[row_cnt][head_cnt] <= recip_lut[recip_lut_addr];
            end else begin
                recip_lut_addr <= sum_val[LUT_ADDR_WIDTH-1:0];
                reciprocal[row_cnt][head_cnt] <= recip_lut[recip_lut_addr];
            end
        end
    end

    // Normalization using reciprocal multiplication
    always_ff @(posedge clk) begin
        if (state == S_NORMALIZE) begin
            // Multiply each ReLU value by the reciprocal of the row sum
            scaled_value <= A_relu[row_cnt][head_cnt][col_cnt] * reciprocal[row_cnt][head_cnt];
            
            // Scale back to correct fixed-point representation and store
            A_out_arr[row_cnt][head_cnt][col_cnt] <= scaled_value[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS];
        end
    end

    // Output packing
    always_ff @(posedge clk) begin
        if (state == S_OUTPUT) begin
            for (int l = 0; l < L; l++) begin
                for (int n = 0; n < N; n++) begin
                    for (int l2 = 0; l2 < L; l2++) begin
                        A_out[((l*N*L)+(n*L)+l2)*DATA_WIDTH +: DATA_WIDTH] <= A_out_arr[l][n][l2];
                    end
                end
            end
        end
    end

endmodule