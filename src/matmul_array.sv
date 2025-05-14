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
    parameter int N = 16         // Columns of matrix B
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  start,
    input  logic signed [15:0]    a_in [0:M*K-1],  // Flattened matrix A
    input  logic signed [15:0]    b_in [0:K*N-1],  // Flattened matrix B
    output logic signed [31:0]    c_out [0:M*N-1], // Flattened matrix C
    output logic                  done
);

    // State machine states
    typedef enum logic [1:0] {
        IDLE  = 2'b00,
        LOAD  = 2'b01,
        COMPUTE = 2'b10,
        DONE  = 2'b11
    } state_t;

    state_t state, next_state;

    // Internal signals
    logic signed [15:0] a_reg [0:M*K-1]; // Register to hold matrix A
    logic signed [15:0] b_reg [0:K*N-1]; // Register to hold matrix B
    logic signed [31:0] c_temp [0:M*N-1]; // Temporary result storage
    logic [31:0] cycle_count; // Counter for computation cycles
    logic done_int; // Internal done signal

    // State machine: Sequential logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            cycle_count <= 0;
            done <= 0;
        end else begin
            state <= next_state;
            if (state == COMPUTE)
                cycle_count <= cycle_count + 1;
            else
                cycle_count <= 0;
            done <= done_int;
        end
    end

    // State machine: Next state and output logic
    always_comb begin
        next_state = state;
        done_int = 0;

        case (state)
            IDLE: begin
                if (start)
                    next_state = LOAD;
            end
            LOAD: begin
                next_state = COMPUTE;
            end
            COMPUTE: begin
                if (cycle_count == K-1)
                    next_state = DONE;
            end
            DONE: begin
                done_int = 1;
                next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end

    // Load input matrices into registers
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < M*K; i++)
                a_reg[i] <= 0;
            for (int i = 0; i < K*N; i++)
                b_reg[i] <= 0;
        end else if (state == LOAD) begin
            a_reg <= a_in;
            b_reg <= b_in;
        end
    end

    function automatic logic signed [31:0] sat_add32
            (input  logic signed [31:0] a,
            input  logic signed [31:0] b);

        logic signed [31:0] sum;
        logic               ovf;
        begin
            sum = a + b;                              // 32-bit two's-complement add
            ovf = (a[31] == b[31]) && (sum[31] != a[31]);

            if (!ovf) begin
                sat_add32 = sum;                      // no overflow → pass through
            end else if (a[31] == 0) begin            // operands were positive
                sat_add32 = 32'h7FFF_FFFF;            // clamp to +0.999 999 999 (Q2.30)
            end else begin                            // operands were negative
                sat_add32 = 32'h8000_0000;            // clamp to –1.0
            end
        end
    endfunction

    // Matrix multiplication logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < M*N; i++)
                c_temp[i] <= 0;
        end else if (state == LOAD) begin
            // Initialize c_temp to zero
            for (int i = 0; i < M*N; i++)
                c_temp[i] <= 0;
        end else if (state == COMPUTE) begin
            // Perform one step of the dot product for each element of C
            for (int i = 0; i < M; i++) begin
                for (int j = 0; j < N; j++) begin
                    // Compute c_temp[i*N+j] += a_reg[i*K+k] * b_reg[k*N+j]
                    c_temp[i*N+j] <= sat_add32 (c_temp[i*N+j],
                        a_reg[i*K+cycle_count[31:0]] * b_reg[cycle_count[31:0]*N+j]);
                end
            end
        end
    end

    // Assign output
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < M*N; i++)
                c_out[i] <= 0;
        end else if (state == DONE) begin
            c_out <= c_temp;
        end
    end

endmodule