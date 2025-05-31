//------------------------------------------------------------------------------
// Module: residual
// Description: Performs element-wise addition of two input vectors
// (x_in and sub_in) to compute a residual connection using Q1.15 saturating addition.
// Uses an FSM to control sequential addition.
//
// May 3 2025    Max Zhang      Initial version
// May 3 2025    Tianwei Liu    Comments
// May 30 2025   Max Zhang      Added Q1.15 saturating adder
//------------------------------------------------------------------------------
module residual #(
    parameter int DATA_WIDTH = 16, // Bit width of each data element (Q1.15)
    parameter int SEQ_LEN    = 8,  // Sequence length of input data
    parameter int EMB_DIM    = 8   // Embedding dimension of input data
)(
    // control signals
    input  logic                                      clk,
    input  logic                                      rst_n,
    input  logic                                      start,
    output logic                                      done,
    // input vectors
    input  logic [DATA_WIDTH*SEQ_LEN*EMB_DIM -1:0]    x_in,
    input  logic [DATA_WIDTH*SEQ_LEN*EMB_DIM -1:0]    sub_in,
    // residual sum
    output logic [DATA_WIDTH*SEQ_LEN*EMB_DIM -1:0]    y_out,
    output logic                                      out_valid
);

    // FSM states
    enum logic [1:0] {S_IDLE, S_ADD, S_DONE} state, next_state;

    // loop variables
    int r, c;

    // functions to get individual elements from input vectors
    function automatic logic [DATA_WIDTH-1:0] get_x(
        input int r,
        input int c
    );
        int flat_idx = (r*EMB_DIM)+c;
        return x_in[((flat_idx+1)*DATA_WIDTH-1) -: DATA_WIDTH];
    endfunction

    function automatic logic [DATA_WIDTH-1:0] get_sub(
        input int r,
        input int c
    );
        int flat_idx = (r*EMB_DIM)+c;
        return sub_in[((flat_idx+1)*DATA_WIDTH-1) -: DATA_WIDTH];
    endfunction

    // Q1.15 saturating adder function
    function automatic logic signed [15:0] sat_add16
                (input logic signed [15:0] a,
                 input logic signed [15:0] b);
        logic signed [15:0] sum;
        logic               ovf;
        begin
            sum = a + b;                              // 16-bit two's-complement add
            ovf = (a[15] == b[15]) && (sum[15] != a[15]);

            if (!ovf) begin
                sat_add16 = sum;                      // no overflow → pass through
            end else if (a[15] == 0) begin            // operands were positive
                sat_add16 = 16'h7FFF;                 // clamp to +0.999969482421875 (Q1.15)
            end else begin                            // operands were negative
                sat_add16 = 16'h8000;                 // clamp to –1.0
            end
        end
    endfunction

    // state register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end
        else begin
            state <= next_state;
        end
    end

    // next state logic
    always_comb begin
        case (state)
            S_IDLE: begin
                if (start)
                    next_state = S_ADD;
                else
                    next_state = S_IDLE;
            end
            S_ADD: begin
                if (r == SEQ_LEN-1 && c == EMB_DIM-1)
                    next_state = S_DONE;
                else
                    next_state = S_ADD;
            end
            S_DONE: begin
                if (start)
                    next_state = S_ADD;
                else
                    next_state = S_IDLE;
            end
            default: next_state = S_IDLE;
        endcase
    end

    // datapath and output logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r <= 0;
            c <= 0;
            done <= 0;
            out_valid <= 0;
            y_out <= 0;
        end
        else begin
            case (state)
                S_IDLE: begin
                    r <= 0;
                    c <= 0;
                    done <= 0;
                    out_valid <= 0;
                end
                S_ADD: begin
                    // compute residual using saturating adder
                    y_out[((r*EMB_DIM+c+1)*DATA_WIDTH-1) -: DATA_WIDTH] <= sat_add16(get_x(r,c), get_sub(r,c));

                    // update loop variables
                    if (c == EMB_DIM-1) begin
                        c <= 0;
                        r <= r + 1;
                    end
                    else begin
                        c <= c + 1;
                    end
                end
                S_DONE: begin
                    done <= 1;
                    out_valid <= 1;
                end
                default: begin
                    r <= 0;
                    c <= 0;
                    done <= 0;
                    out_valid <= 0;
                end
            endcase
        end
    end

endmodule
