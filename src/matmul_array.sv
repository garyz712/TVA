//--------------------------------------------------------------------------------
// matmul_array.sv
// 
// Top-level of a 4x4 systolic array matrix multiplier.
// Parameterized for square matrices of dimension N, where N is a multiple of 4.
// We tile the NxN matrices A and B into (N/4)x(N/4) blocks of size 4x4.
// We serialize the multiplication of block-pairs: for each (bi,bj), we accumulate
// C_block(bi,bj) = sum_{bk=0 to M-1} A_block(bi,bk) * B_block(bk,bj)
// using our 4x4 PE array.
//
// May 9 2025    Tianwei Liu    Initial version
// May 13 2025   Tianwei Liu    update multiply logic
//--------------------------------------------------------------------------------
module matmul_array #(
    parameter int M = 16,        // Rows of matrix A
    parameter int K = 16,        // Columns of A, rows of B
    parameter int N = 16,        // Columns of matrix B
    parameter int WIDTH = 16     // Bit width of matrix elements
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  start,
    input  logic [WIDTH-1:0]      a_in [0:M*K-1],  // Flattened matrix A
    input  logic [WIDTH-1:0]      b_in [0:K*N-1],  // Flattened matrix B
    output logic [31:0]           c_out [0:M*N-1], // Flattened matrix C (32-bit to hold accumulation)
    output logic                  done
);

    // State machine states
    enum logic [1:0] {IDLE, COMPUTE, DONE} state, next_state;

    // Counters for block indices and data feeding
    logic [$clog2(M/4)-1:0] p_counter;  // Row block index for C
    logic [$clog2(N/4)-1:0] q_counter;  // Column block index for C
    logic [$clog2(K/4)-1:0] r_counter;  // Inner dimension block index
    logic [$clog2(4)-1:0]   inner_feed_counter; // Counter for feeding 4 elements per block

    // PE interconnect signals
    logic a_valid_inter [0:3][0:4]; // Valid signals for A (left to right)
    logic [WIDTH-1:0] a_inter [0:3][0:4];  // Data for A
    logic b_valid_inter [0:4][0:3]; // Valid signals for B (top to bottom)
    logic [WIDTH-1:0] b_inter [0:4][0:3];  // Data for B
    logic [31:0] c_pe [0:3][0:3];          // Accumulated results from PEs
    logic out16_valid_pe [0:3][0:3];       // Multiplier valid signals from PEs

    // Control signals
    logic reset_acc;  // Reset accumulators in PEs
    logic block_done; // Indicates when a block computation is complete

    // PE instantiation (4x4 grid)
    for (genvar i = 0; i < 4; i++) begin : row
        for (genvar j = 0; j < 4; j++) begin : col
            matmul_pe pe (
                .clk(clk),
                .rst_n(rst_n),
                .reset_acc(reset_acc),
                .a_valid_in(a_valid_inter[i][j]),
                .a_in(a_inter[i][j]),
                .b_valid_in(b_valid_inter[i][j]),
                .b_in(b_inter[i][j]),
                .a_valid_out(a_valid_inter[i][j+1]),
                .a_out(a_inter[i][j+1]),
                .b_valid_out(b_valid_inter[i+1][j]),
                .b_out(b_inter[i+1][j]),
                .c_out(c_pe[i][j]),
                .out16_valid(out16_valid_pe[i][j])
            );
        end
    end

    // Detect block completion using PE valid signals
    logic all_pe_valid;
    assign all_pe_valid = &{out16_valid_pe[0][0], out16_valid_pe[0][1], out16_valid_pe[0][2], out16_valid_pe[0][3],
                            out16_valid_pe[1][0], out16_valid_pe[1][1], out16_valid_pe[1][2], out16_valid_pe[1][3],
                            out16_valid_pe[2][0], out16_valid_pe[2][1], out16_valid_pe[2][2], out16_valid_pe[2][3],
                            out16_valid_pe[3][0], out16_valid_pe[3][1], out16_valid_pe[3][2], out16_valid_pe[3][3]};
    assign block_done = (state == COMPUTE) && (inner_feed_counter == 3) && all_pe_valid;

    // State machine logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            p_counter <= 0;
            q_counter <= 0;
            r_counter <= 0;
            inner_feed_counter <= 0;
            done <= 0;
        end else begin
            state <= next_state;
            case (state)
                IDLE: begin
                    if (start) begin
                        next_state <= COMPUTE;
                        inner_feed_counter <= 0;
                    end
                end
                COMPUTE: begin
                    if (inner_feed_counter < 3) begin
                        inner_feed_counter <= inner_feed_counter + 1;
                    end else if (block_done) begin
                        inner_feed_counter <= 0;
                        r_counter <= r_counter + 1;
                        if (r_counter == (K/4) - 1) begin
                            r_counter <= 0;
                            q_counter <= q_counter + 1;
                            if (q_counter == (N/4) - 1) begin
                                q_counter <= 0;
                                p_counter <= p_counter + 1;
                                if (p_counter == (M/4) - 1) begin
                                    next_state <= DONE;
                                end
                            end
                        end
                    end
                end
                DONE: begin
                    done <= 1;
                    if (!start) begin
                        next_state <= IDLE;
                        done <= 0;
                    end
                end
            endcase
        end
    end

    // Data feeding with skew
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 4; i++) begin
                for (int j = 0; j < 5; j++) begin
                    a_valid_inter[i][j] <= 0;
                    a_inter[i][j] <= 0;
                end
                for (int j = 0; j < 4; j++) begin
                    b_valid_inter[i][j] <= 0;
                    b_inter[i][j] <= 0;
                end
            end
        end else if (state == COMPUTE) begin
            for (int i = 0; i < 4; i++) begin
                // Skewed feeding for A (left edge)
                if (inner_feed_counter >= i && inner_feed_counter < 4) begin
                    a_valid_inter[i][0] <= 1;
                    a_inter[i][0] <= a_in[(p_counter * 4 + i) * K + (r_counter * 4) + (inner_feed_counter - i)];
                end else begin
                    a_valid_inter[i][0] <= 0;
                    a_inter[i][0] <= 0;
                end

                // Skewed feeding for B (top edge)
                if (inner_feed_counter >= i && inner_feed_counter < 4) begin
                    b_valid_inter[0][i] <= 1;
                    b_inter[0][i] <= b_in[(r_counter * 4 + (inner_feed_counter - i)) * N + (q_counter * 4 + i)];
                end else begin
                    b_valid_inter[0][i] <= 0;
                    b_inter[0][i] <= 0;
                end
            end
        end
    end

    // Accumulator reset and result collection
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reset_acc <= 1;
            for (int i = 0; i < M*N; i++) c_out[i] <= 0;
        end else begin
            reset_acc <= (state == COMPUTE && r_counter == 0 && inner_feed_counter == 0);
            if (block_done && r_counter == (K/4) - 1) begin
                for (int i = 0; i < 4; i++) begin
                    for (int j = 0; j < 4; j++) begin
                        c_out[(p_counter * 4 + i) * N + (q_counter * 4 + j)] <= c_pe[i][j];
                    end
                end
            end
        end
    end

endmodule