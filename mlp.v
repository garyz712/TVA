module mlp_block #(
    parameter DATA_WIDTH     = 16,   // e.g., FP16 or fixed-point bit width
    parameter EMBED_DIM      = 128,  // Input/Output feature size
    parameter HIDDEN_DIM     = 256,  // Hidden layer size
    parameter ACTIVATION_TYPE= 0     // e.g., 0=ReLU, 1=GeLU (can define externally)
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Control signals
    input  wire                         start,
    output reg                          done,
    
    // Input vector (one token's embedding)
    input  wire [DATA_WIDTH*EMBED_DIM-1:0]  in_vec,
    input  wire                         in_valid,
    
    // Output vector
    output reg  [DATA_WIDTH*EMBED_DIM-1:0]  out_vec,
    output reg                          out_valid
);

    //--------------------------------------------------------------------------
    // Internal Parameters/Variables
    //--------------------------------------------------------------------------
    
    // Memory or registers for W1 and W2
    // For demonstration, we declare arrays in RTL. In practice, you might:
    // (a) Initialize from a file
    // (b) Store in BRAM or external memory
    // (c) Use partial reconfiguration if large
    // 
    // W1 is HIDDEN_DIM x EMBED_DIM
    // W2 is EMBED_DIM x HIDDEN_DIM
    
    reg [DATA_WIDTH-1:0] W1 [0:HIDDEN_DIM-1][0:EMBED_DIM-1];
    reg [DATA_WIDTH-1:0] b1 [0:HIDDEN_DIM-1];
    
    reg [DATA_WIDTH-1:0] W2 [0:EMBED_DIM-1][0:HIDDEN_DIM-1];
    reg [DATA_WIDTH-1:0] b2 [0:EMBED_DIM-1];

    // Buffers to hold intermediate results: the hidden layer after FC1
    reg [DATA_WIDTH-1:0] hidden_reg [0:HIDDEN_DIM-1];

    // State machine
    reg [2:0] state;
    localparam S_IDLE   = 3'd0;
    localparam S_FC1    = 3'd1;
    localparam S_ACT    = 3'd2;
    localparam S_FC2    = 3'd3;
    localparam S_DONE   = 3'd4;

    integer i, j;
    
    //--------------------------------------------------------------------------
    // Load Some Dummy Weights (for Example Only)
    //--------------------------------------------------------------------------
    // In a real design, you’ll initialize these from an external source or memory.
    // This always block is just to illustrate storing something in W1/W2.
    // You should remove or replace with your actual memory initialization.
    initial begin
        for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
            for (j = 0; j < EMBED_DIM; j = j + 1) begin
                W1[i][j] = {DATA_WIDTH{1'b0}};  // e.g., 0
            end
            b1[i] = {DATA_WIDTH{1'b0}};
        end
        
        for (i = 0; i < EMBED_DIM; i = i + 1) begin
            for (j = 0; j < HIDDEN_DIM; j = j + 1) begin
                W2[i][j] = {DATA_WIDTH{1'b0}};
            end
            b2[i] = {DATA_WIDTH{1'b0}};
        end
    end

    //--------------------------------------------------------------------------
    // Finite State Machine
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state     <= S_IDLE;
            out_valid <= 1'b0;
            done      <= 1'b0;
        end else begin
            case(state)
                S_IDLE: begin
                    out_valid <= 1'b0;
                    done      <= 1'b0;
                    if(start && in_valid) begin
                        state <= S_FC1;
                    end
                end

                // 1) First Fully-Connected: hidden_reg = W1 * in_vec + b1
                S_FC1: begin
                    // We'll do a *sequential* matrix-vector multiply, for demonstration.
                    // If you want more speed, you can pipeline or parallelize.

                    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
                        // Initialize accumulation with bias
                        hidden_reg[i] = b1[i];

                        // Accumulate sum of multiplications
                        for (j = 0; j < EMBED_DIM; j = j + 1) begin
                            // Extract input
                            reg [DATA_WIDTH-1:0] x_elt;
                            x_elt = in_vec[(j+1)*DATA_WIDTH-1 -: DATA_WIDTH];

                            // Multiply & accumulate
                            // In reality, do proper float or fixed multiply & add
                            hidden_reg[i] = hidden_reg[i] + (W1[i][j] * x_elt);
                        end
                    end

                    state <= S_ACT;
                end

                // 2) Activation (e.g., ReLU)
                S_ACT: begin
                    // Example: ReLU, done in a single cycle (not pipelined)
                    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
                        if (ACTIVATION_TYPE == 0) begin
                            // ReLU
                            if (hidden_reg[i][DATA_WIDTH-1] == 1'b1) begin
                                // Negative if signed. For demonstration, zero it out.
                                hidden_reg[i] = {DATA_WIDTH{1'b0}};
                            end
                        end
                        else begin
                            // GeLU or other, not shown here. 
                            // Implementation depends on approximate polynomials or LUTs.
                            // hidden_reg[i] <= gelu(hidden_reg[i]);
                        end
                    end
                    state <= S_FC2;
                end

                // 3) Second Fully-Connected: out_vec = W2 * hidden_reg + b2
                S_FC2: begin
                    // For each output dimension in EMBED_DIM
                    for (i = 0; i < EMBED_DIM; i = i + 1) begin
                        reg [DATA_WIDTH-1:0] sum_elt;
                        sum_elt = b2[i];

                        // Multiply & accumulate
                        for (j = 0; j < HIDDEN_DIM; j = j + 1) begin
                            sum_elt = sum_elt + (W2[i][j] * hidden_reg[j]);
                        end

                        // Store into out_vec
                        out_vec[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = sum_elt;
                    end

                    out_valid <= 1'b1;
                    state     <= S_DONE;
                end

                S_DONE: begin
                    done      <= 1'b1;
                    // Optionally stay here or return to IDLE
                    // out_valid can remain 1 for 1 cycle, etc.
                    state     <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
