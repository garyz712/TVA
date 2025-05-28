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
    parameter int MLP_DIM = 64,      // Hidden dimension
    parameter int OUT_DIM = 16,      // Output dimension (EMB_DIM)
    parameter int DATA_WIDTH = 16     // Q0.15 fixed-point
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  start,
    input  logic                  valid_in,
    input  logic [DATA_WIDTH-1:0] x [HIDDEN_DIM],
    input  logic [DATA_WIDTH-1:0] W1 [HIDDEN_DIM*MLP_DIM],
    input  logic [DATA_WIDTH-1:0] b1 [MLP_DIM],
    input  logic [DATA_WIDTH-1:0] W2 [MLP_DIM*OUT_DIM],
    input  logic [DATA_WIDTH-1:0] b2 [OUT_DIM],
    output logic                  valid_out,
    output logic                  done,
    output logic [DATA_WIDTH-1:0] y [OUT_DIM]
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

    // Intermediate signals
    acc_t  y_pre1 [MLP_DIM];  // First layer output before bias
    logic [DATA_WIDTH-1:0] y1 [MLP_DIM]; // First layer output after bias and ReLU
    acc_t  y_pre2 [OUT_DIM];  // Second layer output before bias
    logic  matmul1_start, matmul1_done;
    logic  matmul2_start, matmul2_done;
    logic  relu_done;

    // State machine enum
    typedef enum logic [2:0] {
        IDLE       = 3'b000,
        FC1_START  = 3'b001,
        FC1_WAIT   = 3'b010,
        RELU       = 3'b011,
        FC2_START  = 3'b100,
        FC2_WAIT   = 3'b101,
        DONE       = 3'b110
    } state_t;

    state_t state, next_state;

    // First layer matmul
    matmul_array #(
        .M(1),
        .K(HIDDEN_DIM),
        .N(MLP_DIM)
    ) matmul1 (
        .clk(clk),
        .rst_n(rst_n),
        .start(matmul1_start),
        .a_in(x),
        .b_in(W1),
        .c_out(y_pre1),
        .done(matmul1_done)
    );

    // Second layer matmul
    matmul_array #(
        .M(1),
        .K(MLP_DIM),
        .N(OUT_DIM)
    ) matmul2 (
        .clk(clk),
        .rst_n(rst_n),
        .start(matmul2_start),
        .a_in(y1),
        .b_in(W2),
        .c_out(y_pre2),
        .done(matmul2_done)
    );

    // State transition
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // Next state and control logic
    always_comb begin
        next_state = state;
        valid_out = 1'b0;
        done = 1'b0;
        matmul1_start = 1'b0;
        matmul2_start = 1'b0;

        unique case (state)
            IDLE: begin
                if (start && valid_in)
                    next_state = FC1_START;
            end
            FC1_START: begin
                matmul1_start = 1'b1;
                next_state = FC1_WAIT;
            end
            FC1_WAIT: begin
                if (matmul1_done)
                    next_state = RELU;
            end
            RELU: begin
                next_state = FC2_START;
            end
            FC2_START: begin
                matmul2_start = 1'b1;
                next_state = FC2_WAIT;
            end
            FC2_WAIT: begin
                if (matmul2_done)
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

    // Bias addition and ReLU for first layer
    always_ff @(posedge clk) begin
        if (state == FC1_WAIT && matmul1_done) begin
            for (int i = 0; i < MLP_DIM; i++) begin
                logic signed [15:0] result;
                if (y_pre1[i] > 32'sh3FFFFFFF)
                    result = 16'sh7FFF;
                else if (y_pre1[i] < 32'shC0000000)
                    result = 16'sh8000;
                else
                    result = {y_pre1[i][31], y_pre1[i][29:15]};
                y1[i] <= (sat_add16(result, b1[i]) > 0) ? sat_add16(result, b1[i]) : 0; // ReLU
            end
        end
    end

    // Bias addition for second layer
    always_ff @(posedge clk) begin
        if (state == FC2_WAIT && matmul2_done) begin
            for (int i = 0; i < OUT_DIM; i++) begin
                logic signed [15:0] result;
                if (y_pre2[i] > 32'sh3FFFFFFF)
                    result = 16'sh7FFF;
                else if (y_pre2[i] < 32'shC0000000)
                    result = 16'sh8000;
                else
                    result = {y_pre2[i][31], y_pre2[i][29:15]};
                y[i] <= sat_add16(result, b2[i]);
            end
        end
    end

endmodule