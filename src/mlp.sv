//------------------------------------------------------------------------------
// mlp.sv
//
// SystemVerilog implementation of a Multilayer Perceptron (MLP) for
// Vision Transformer (ViT) inference. Processes one patch's feature vector
// using two linear layers with precise matrix multiplication via matmul_array,
// a piecewise linear RELU activation, and bias addition. Optimized for FPGA
// with 16-bit fixed-point arithmetic, pipelining, and Block RAM for
// weights/biases.
//
// May 10 2025    Max Zhang      Initial version
// May 18 2025    Tianwei Liu    modify to use correct matrix multiplier
//------------------------------------------------------------------------------
module mlp #(
    parameter int HIDDEN_DIM = 768,    // Input/output dimension
    parameter int MLP_DIM = 3072,      // Intermediate dimension
    parameter int DATA_WIDTH = 16,     // Q0.15 fixed-point
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
        RELU      = 3'b011,
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
    data_t a [MLP_DIM];         // RELU output
    acc_t  z_pre [MLP_DIM];     // fc1 matmul output
    acc_t  y_pre [HIDDEN_DIM];  // fc2 matmul output

    // Matmul control signals
    logic matmul_start;
    logic matmul_done;

    // Matmul input/output selection
    data_t a_in [MLP_DIM];      // Selected input vector (x or a)
    data_t b_in [MLP_DIM*HIDDEN_DIM]; // Selected weight matrix (w1 or w2)
    acc_t  c_out [MLP_DIM];     // Matmul output (z_pre or y_pre)
    logic is_fc1;               // Flag to select FC1 (1) or FC2 (0)

    // State machine registers
    state_t state, next_state;

    // Single matmul instance
    matmul_array #(
        .M(1),
        .K($max(HIDDEN_DIM, MLP_DIM)), // Max of input dimensions
        .N($max(MLP_DIM, HIDDEN_DIM))  // Max of output dimensions
    ) matmul (
        .clk(clk),
        .rst_n(~rst),
        .start(matmul_start),
        .a_in(a_in),   // Selected input
        .b_in(b_in),   // Selected weights
        .c_out(c_out), // Selected output
        .done(matmul_done)
    );

    // Initialize weights/biases
    initial begin
        // Load via external interface or ROM in practice
        // Example: w1 = '{default: 16'h1000};
    end

    // State transition (sequential)
    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            state <= IDLE;
        else
            state <= next_state;
    end

    // Input and weight selection
    always_comb begin
        if (is_fc1) begin
            // FC1: x @ W1^T (1 × HIDDEN_DIM × MLP_DIM)
            a_in = x; // HIDDEN_DIM
            b_in = w1; // HIDDEN_DIM × MLP_DIM
        end else begin
            // FC2: a @ W2^T (1 × MLP_DIM × HIDDEN_DIM)
            a_in = a; // MLP_DIM
            b_in = w2; // MLP_DIM × HIDDEN_DIM
        end
    end

    // Output routing
    always_ff @(posedge clk) begin
        if (matmul_done) begin
            if (is_fc1) begin
                for (int i = 0; i < MLP_DIM; i++) begin
                    z_pre[i] <= c_out[i];
                end
            end else begin
                for (int i = 0; i < HIDDEN_DIM; i++) begin
                    y_pre[i] <= c_out[i];
                end
            end
        end
    end

    // Next state and control logic (combinatorial)
    always_comb begin
        next_state = state;
        valid_out = 1'b0;
        done = 1'b0;
        matmul_start = 1'b0;
        is_fc1 = 1'b0;

        unique case (state)
            IDLE: begin
                if (start && valid_in)
                    next_state = FC1_START;
            end
            FC1_START: begin
                matmul_start = 1'b1;
                is_fc1 = 1'b1;
                next_state = FC1_WAIT;
            end
            FC1_WAIT: begin
                is_fc1 = 1'b1;
                if (matmul_done)
                    next_state = RELU;
            end
            RELU: begin
                next_state = FC2_START;
            end
            FC2_START: begin
                matmul_start = 1'b1;
                is_fc1 = 1'b0;
                next_state = FC2_WAIT;
            end
            FC2_WAIT: begin
                is_fc1 = 1'b0;
                if (matmul_done)
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
        if (state == FC1_WAIT && matmul_done) begin
            for (int i = 0; i < MLP_DIM; i++) begin
                z[i] <= z_pre[i][ACC_WIDTH-1:ACC_WIDTH-DATA_WIDTH] + b1[i];
            end
        end
    end

    // RELU computation (sequential)
    always_ff @(posedge clk) begin
        if (state == RELU) begin
            for (int i = 0; i < MLP_DIM; i++) begin
                a[i] <= z[i][DATA_WIDTH-1] ? '0 : z[i]; // z < 0 ? 0 : z
            end
        end
    end

    // Bias addition and truncation for fc2 (sequential)
    always_ff @(posedge clk) begin
        if (state == FC2_WAIT && matmul_done) begin
            for (int i = 0; i < HIDDEN_DIM; i++) begin
                y[i] <= y_pre[i][ACC_WIDTH-1:ACC_WIDTH-DATA_WIDTH] + b2[i];
            end
        end
    end

endmodule