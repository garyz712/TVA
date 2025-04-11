module softmax_approx #(
    parameter DATA_WIDTH = 16,
    parameter L = 8,
    parameter N = 1
)(
    input  wire                            clk,
    input  wire                            rst_n,
    // Control
    input  wire                            start,
    output reg                             done,

    // A_in: shape (L, N, L)
    input  wire [DATA_WIDTH*L*N*L-1:0]     A_in,
    // A_out: shape (L, N, L)
    output reg  [DATA_WIDTH*L*N*L-1:0]     A_out,
    output reg                             out_valid
);

    // We'll treat each "row" as (l, n), with length L in the last dimension.
    reg [DATA_WIDTH-1:0] A_arr [0:L-1][0:N-1][0:L-1];

    integer l, n_, l2;
    reg [2:0] state;
    localparam S_IDLE=0, S_LOAD=1, S_SOFT=2, S_DONE=3;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state <= S_IDLE;
            done  <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            case(state)
                S_IDLE: begin
                    if(start) begin
                        done      <= 1'b0;
                        out_valid <= 1'b0;
                        state     <= S_LOAD;
                    end
                end

                // 1) LOAD
                S_LOAD: begin
                    for(l=0; l<L; l=l+1) begin
                        for(n_=0; n_<N; n_=n_+1) begin
                            for(l2=0; l2<L; l2=l2+1) begin
                                A_arr[l][n_][l2] = A_in[
                                  ((l*N*L)+(n_*L)+l2+1)*DATA_WIDTH -1 -: DATA_WIDTH
                                ];
                            end
                        end
                    end
                    state <= S_SOFT;
                end

                // 2) ROW-WISE SOFTMAX
                S_SOFT: begin
                    for(l=0; l<L; l=l+1) begin
                        for(n_=0; n_<N; n_=n_+1) begin
                            reg [DATA_WIDTH-1:0] row_sum;
                            row_sum = {DATA_WIDTH{1'b0}};
                            // sum
                            for(l2=0; l2<L; l2=l2+1) begin
                                row_sum = row_sum + A_arr[l][n_][l2];
                            end
                            // divide
                            for(l2=0; l2<L; l2=l2+1) begin
                                // Approx: A_arr[l][n_][l2] /= row_sum
                                A_arr[l][n_][l2] = A_arr[l][n_][l2]; // placeholder
                            end
                        end
                    end
                    state <= S_DONE;
                end

                // 3) DONE: pack out
                S_DONE: begin
                    for(l=0; l<L; l=l+1) begin
                        for(n_=0; n_<N; n_=n_+1) begin
                            for(l2=0; l2<L; l2=l2+1) begin
                                A_out[
                                  ((l*N*L)+(n_*L)+l2+1)*DATA_WIDTH -1 -: DATA_WIDTH
                                ] = A_arr[l][n_][l2];
                            end
                        end
                    end
                    out_valid <= 1'b1;
                    done      <= 1'b1;
                    state     <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
