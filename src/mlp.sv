//------------------------------------------------------------------------------
// mlp.sv
//
// SystemVerilog implementation of a Multilayer Perceptron (MLP) for
// Vision Transformer (ViT) inference. Processes one patch's feature vector
// using two linear layers with precise matrix multiplication via matmul_array,
// a piecewise linear GELU activation, and bias addition. Optimized for FPGA
// with 16-bit fixed-point arithmetic, pipelining, and Block RAM for
// weights/biases.
//
// May 10 2025    Max Zhang      Initial version
// May 18 2025    Tianwei Liu    modify to use correct matrix multiplier
//------------------------------------------------------------------------------
module mlp_vit #(
    parameter int HIDDEN_DIM = 768,    // Input/output dimension
    parameter int MLP_DIM = 3072,      // Intermediate dimension
    parameter int DATA_WIDTH = 16,     // Q8.8 fixed-point
    parameter int ACC_WIDTH = 32       // Accumulator width
) (
    input  logic                  clk,         // Clock
    input  logic                  rst,         // Active-high reset
    input  logic                  start,       // Start computation
    input  logic                  valid_in,    // Input data valid
    input  logic [DATA_WIDTH-1:0] x [HIDDEN_DIM], // Input vector
    output logic                  valid_out,   // Output data valid
    output logic                  done,        // Computation complete
    output logic [DATA_WIDTH-1:0] y [HIDDEN_DIM]  // Output vector
);

    // Typedefs
    typedef logic [DATA_WIDTH-1:0] data_t;
    typedef logic [ACC_WIDTH-1:0] acc_t;

    // State machine enum
    typedef enum logic [2:0] {
        IDLE      = 3'b000,
        FC1_START = 3'b001,
        FC1_WAIT  = 3'b010,
        GELU      = 3'b011,
        FC2_START = 3'b100,
        FC2_WAIT  = 3'b101,
        DONE      = 3'b110
    } state_t;

    // Weight and bias memories (Block RAM)
    data_t w1 [HIDDEN_DIM*MLP_DIM]; // fc1 weights (flattened)
    data_t b1 [MLP_DIM];            // fc1 biases
    data_t w2 [MLP_DIM*HIDDEN_DIM]; // fc2 weights (flattened)
    data_t b2 [HIDDEN_DIM];         // fc2 biases

    // Intermediate signals
    data_t z [MLP_DIM];         // fc1 output (post-bias)
    data_t a [MLP_DIM];         // GELU output
    acc_t  z_pre [MLP_DIM];     // fc1 matmul output
    acc_t  y_pre [HIDDEN_DIM];  // fc2 matmul output

    // Matmul control signals
    logic fc1_start, fc2_start;
    logic fc1_done, fc2_done;

    // State machine registers
    state_t state, next_state;

    // Matmul instance for fc1: x @ W1^T
    matmul_array #(
        .M(1),
        .K(HIDDEN_DIM),
        .N(MLP_DIM)
    ) fc1_matmul (
        .clk(clk),
        .rst_n(~rst),
        .start(fc1_start),
        .a_in(x),      // 1 × HIDDEN_DIM
        .b_in(w1),     // HIDDEN_DIM × MLP_DIM
        .c_out(z_pre), // 1 × MLP_DIM
        .done(fc1_done)
    );

    // Matmul instance for fc2: a @ W2^T
    matmul_array #(
        .M(1),
        .K(MLP_DIM),
        .N(HIDDEN_DIM)
    ) fc2_matmul (
        .clk(clk),
        .rst_n(~rst),
        .start(fc2_start),
        .a_in(a),      // 1 × MLP_DIM
        .b_in(w2),     // MLP_DIM × HIDDEN_DIM
        .c_out(y_pre), // 1 × HIDDEN_DIM
        .done(fc2_done)
    );

    // Initialize weights/biases -- for simulation put weights here
    initial begin
        // Load via external interface or ROM in practice
        // Example: w1 = '{default: 16'h1000}; // Q8.8 value 1.0
    end

    // State transition (sequential)
    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            state <= IDLE;
        else
            state <= next_state;
    end

    // Next state and control logic (combinatorial)
    always_comb begin
        next_state = state;
        valid_out = 1'b0;
        done = 1'b0;
        fc1_start = 1'b0;
        fc2_start = 1'b0;

        unique case (state)
            IDLE: begin
                if (start && valid_in)
                    next_state = FC1_START;
            end
            FC1_START: begin
                fc1_start = 1'b1;
                next_state = FC1_WAIT;
            end
            FC1_WAIT: begin
                if (fc1_done)
                    next_state = GELU;
            end
            GELU: begin
                next_state = FC2_START;
            end
            FC2_START: begin
                fc2_start = 1'b1;
                next_state = FC2_WAIT;
            end
            FC2_WAIT: begin
                if (fc2_done)
                    next_state = DONE;
            end
            DONE: begin
                valid_out = 1'b1;
                done = 1'b1;
                next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end

    // Bias addition and truncation for fc1 (sequential)
    always_ff @(posedge clk) begin
        if (state == FC1_WAIT && fc1_done) begin
            for (int i = 0; i < MLP_DIM; i++) begin
                z[i] <= z_pre[i][ACC_WIDTH-1:ACC_WIDTH-DATA_WIDTH] + b1[i];
            end
        end
    end

    // GELU computation (sequential)
    always_ff @(posedge clk) begin
        if (state == GELU) begin
            for (int i = 0; i < MLP_DIM; i++) begin
                // Piecewise linear GELU approximation
                if (z[i][DATA_WIDTH-1] && z[i] < -16'h2000) // z < -2.0
                    a[i] <= '0;
                else if (z[i] >= 16'h2000) // z >= 2.0
                    a[i] <= z[i];
                else // -2.0 <= z < 2.0
                    a[i] <= (z[i] >> 1) + 16'h0800; // Approx: 0.5*z + 0.5
            end
        end
    end

    // Bias addition and truncation for fc2 (sequential)
    always_ff @(posedge clk) begin
        if (state == FC2_WAIT && fc2_done) begin
            for (int i = 0; i < HIDDEN_DIM; i++) begin
                y[i] <= y_pre[i][ACC_WIDTH-1:ACC_WIDTH-DATA_WIDTH] + b2[i];
            end
        end
    end

    // // Assertion for input validity
    // assert property (@(posedge clk) disable iff (rst)
    //     start |-> valid_in)
    //     else $error("Start signal asserted without valid input");

endmodule
