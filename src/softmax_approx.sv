//------------------------------------------------------------------------------
// softmax_approx.sv
//
// This module computes a row-wise softmax approximation for attention scores
//
// It takes an input matrix A_in of shape (L, N, L) and produces normalized
// attention weights A_out of the same shape.
//
// May 05 2025    Tianwei Liu    Enhanced version of softmax with LUT
// May 25 2025    Tianwei Liu    Use ReLU activation
// May 25 2025    Tianwei Liu    Use divide for now
// May 25 2025    Tianwei Liu    Fix Q15 logic
// May 26 2025    Max Zhang    Fix Indexing
//------------------------------------------------------------------------------

module softmax_approx #(
    parameter int DATA_WIDTH = 16,
    parameter int L = 8, // Sequence length
    parameter int N = 1  // Number of attention heads
)(
    input logic clk,
    input logic rst_n,
    // Control signals
    input logic start,
    output logic done,
    // Input: A_in of shape (L, N, L)
    input logic [DATA_WIDTH-1:0] A_in [0:L*N*L-1],
    // Output: A_out of shape (L, N, L)
    output logic [DATA_WIDTH-1:0] A_out [0:L*N*L-1],
    output logic out_valid
);

    // State machine states
    typedef enum logic [2:0] {
        IDLE,
        RELU_COMPUTE,
        SUM_COMPUTE,
        NORMALIZE,
        DONE_STATE
    } state_t;
    
    state_t current_state, next_state;
    
    // Internal registers and wires
    logic [DATA_WIDTH-1:0] relu_data [0:L*N*L-1];
    logic [DATA_WIDTH+$clog2(L)-1:0] row_sums [0:L*N-1]; // Extended width for sum
    logic [$clog2(L*N*L):0] element_counter;
    logic [$clog2(L*N):0] row_counter;
    logic [$clog2(L):0] col_counter;
    
    // Division result for normalization
    logic [DATA_WIDTH-1:0] div_result;
    logic div_valid;
    
    // Control signals
    logic relu_done, sum_done, norm_done;
    logic start_div;
    
    // State machine sequential logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    // State machine combinational logic
    always_comb begin
        next_state = current_state;
        case (current_state)
            IDLE: begin
                if (start) next_state = RELU_COMPUTE;
            end
            RELU_COMPUTE: begin
                if (relu_done) next_state = SUM_COMPUTE;
            end
            SUM_COMPUTE: begin
                if (sum_done) next_state = NORMALIZE;
            end
            NORMALIZE: begin
                if (norm_done) next_state = DONE_STATE;
            end
            DONE_STATE: begin
                next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end
    
    // ReLU computation stage
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            element_counter <= 0;
            relu_done <= 0;
            for (int i = 0; i < L*N*L; i++) begin
                relu_data[i] <= 0;
            end
        end else if (current_state == RELU_COMPUTE) begin
            // $display("starting relu compute");
            if (element_counter < L*N*L) begin
                // ReLU: max(0, x) - if MSB is 1 (negative), output 0
                if (A_in[element_counter][DATA_WIDTH-1] == 1'b0) begin
                    relu_data[element_counter] <= A_in[element_counter];
                end else begin
                    relu_data[element_counter] <= 0;
                end
                element_counter <= element_counter + 1;
            end else begin
                relu_done <= 1;
            end
        end else if (current_state == IDLE) begin
            element_counter <= 0;
            relu_done <= 0;
        end
    end
    
    // Sum computation stage - compute row sums for normalization
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_counter <= 0;
            col_counter <= 0;
            sum_done <= 0;
            for (int i = 0; i < L*N; i++) begin
                row_sums[i] <= 0;
            end
        end else if (current_state == SUM_COMPUTE) begin
            // $display("starting sum compute");
            if (row_counter < L*N) begin
                if (col_counter < L) begin
                    // Calculate index: row_counter * L + col_counter
                    row_sums[row_counter] <= row_sums[row_counter] + relu_data[row_counter * L + col_counter];
                    col_counter <= col_counter + 1;
                end else begin
                    col_counter <= 0;
                    row_counter <= row_counter + 1;
                    // $display("row counter: %d", row_counter);
                end
            end else begin
                sum_done <= 1;
            end
        end else if (current_state == IDLE) begin
            row_counter <= 0;
            col_counter <= 0;
            sum_done <= 0;
            for (int i = 0; i < L*N; i++) begin
                row_sums[i] <= 0;
            end
        end
    end
    
    // Normalization stage
    logic [$clog2(L*N*L):0] norm_counter;
    logic [$clog2(L*N)-1:0] current_row;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            norm_counter <= 0;
            norm_done <= 0;
            for (int i = 0; i < L*N*L; i++) begin
                A_out[i] <= 0;
            end
        end else if (current_state == NORMALIZE) begin
            if (norm_counter < L*N*L) begin
                current_row = norm_counter / L;
                
                // Simple division approximation: if sum is 0, output 0
                // Otherwise, scale the relu output
                if (row_sums[current_row] == 0) begin
                    A_out[norm_counter] <= 0;
                end else begin
                    // Simplified normalization: (relu_data * 2^(DATA_WIDTH-4)) / row_sum
                    // This maintains some precision while avoiding complex division
                    logic [DATA_WIDTH+16-1:0] scaled_numerator;
                    scaled_numerator = relu_data[norm_counter] << 15; // Scale by 15 to get Q15
                    if (scaled_numerator / row_sums[current_row] >= 16'h8000)
                        A_out[norm_counter] <= 16'h7FFF; // saturate
                    else
                        A_out[norm_counter] <= scaled_numerator / row_sums[current_row];
                end
                norm_counter <= norm_counter + 1;
            end else begin
                norm_done <= 1;
            end
        end else if (current_state == IDLE) begin
            norm_counter <= 0;
            norm_done <= 0;
        end
    end
    
    // Output control signals
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 0;
            out_valid <= 0;
        end else begin
            done <= (current_state == DONE_STATE);
            out_valid <= (current_state == DONE_STATE);
        end
    end

endmodule
