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
// May 27 2025    Tianwei Liu    use 1 layer only
//------------------------------------------------------------------------------
module mlp #(
    parameter int HIDDEN_DIM = 16,    // Input dimension
    parameter int MLP_DIM = 64,      // Output dimension
    parameter int DATA_WIDTH = 16     // Q0.15 fixed-point
) (
    input  logic                  clk,         // Clock
    input  logic                  rst,         // Active-high reset
    input  logic                  start,       // Start computation
    input  logic                  valid_in,    // Input data valid
    input  logic [DATA_WIDTH-1:0] x [HIDDEN_DIM], // Input vector
    // Weight matrix: W (HIDDEN_DIM × MLP_DIM), flattened
    input  logic [DATA_WIDTH-1:0] W [HIDDEN_DIM*MLP_DIM],
    // Bias vector: b (MLP_DIM)
    input  logic [DATA_WIDTH-1:0] b [MLP_DIM],
    output logic                  valid_out,   // Output data valid
    output logic                  done,        // Computation complete
    output logic [DATA_WIDTH-1:0] y [MLP_DIM]    // Output vector
);

    function automatic logic signed [15:0] sat_add16
            (input  logic signed [15:0] a,
            input  logic signed [15:0] b);

        logic signed [15:0] sum;
        logic               ovf;
        begin
            sum = a + b;
            ovf = (a[15] == b[15]) && (sum[15] != a[15]);

            if (!ovf) begin
                sat_add16 = sum;
            end else if (a[15] == 0) begin
                sat_add16 = 16'h7FFF;
            end else begin
                sat_add16 = 16'h8000;
            end
        end
    endfunction

    // Typedefs
    typedef logic signed [DATA_WIDTH-1:0] data_t;
    typedef logic signed [31:0] acc_t;

    // State machine enum
    typedef enum logic [1:0] {
        IDLE       = 2'b00,
        FC_START   = 2'b01,
        FC_WAIT    = 2'b10,
        DONE       = 2'b11
    } state_t;

    // Intermediate signals
    acc_t  y_pre [MLP_DIM];  // Matmul output before bias
    logic  matmul_start;     // Matmul control signal
    logic  matmul_done;      // Matmul completion signal

    // State machine registers
    state_t state, next_state;

    // Matrix multiplication instance
    matmul_array #(
        .M(1),                    // Single vector
        .K(HIDDEN_DIM),           // Input dimension
        .N(MLP_DIM)               // Output dimension
    ) matmul (
        .clk(clk),
        .rst_n(~rst),             // Convert active-high to active-low
        .start(matmul_start),
        .a_in(x),                 // Input vector x (1 × HIDDEN_DIM)
        .b_in(W),                 // Weight matrix W (HIDDEN_DIM × MLP_DIM)
        .c_out(y_pre),            // Output vector (1 × MLP_DIM)
        .done(matmul_done)
    );

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
        matmul_start = 1'b0;

        unique case (state)
            IDLE: begin
                if (start && valid_in)
                    next_state = FC_START;
            end
            FC_START: begin
                matmul_start = 1'b1;
                next_state = FC_WAIT;
            end
            FC_WAIT: begin
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

    logic signed [15:0] result;

    // Bias addition and truncation (sequential)
    always_ff @(posedge clk) begin
        if (state == FC_WAIT && matmul_done) begin
            for (int i = 0; i < MLP_DIM; i++) begin
                if (y_pre[i] > 32'sh3FFFFFFF)
                    result = 16'sh7FFF;
                else if (y_pre[i] < 32'shC0000000)
                    result = 16'sh8000;
                else
                    result = {y_pre[i][31], y_pre[i][29:15]};
                y[i] <= sat_add16(result, b[i]);
            end
        end
    end

endmodule