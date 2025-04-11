module attention_score #(
    parameter DATA_WIDTH = 16,
    parameter L = 8,
    parameter N = 1,
    parameter E = 8
)(
    input  wire                            clk,
    input  wire                            rst_n,
    // Control
    input  wire                            start,
    output reg                             done,

    // Inputs Q, K: each (L, N, E)
    input  wire [DATA_WIDTH*L*N*E-1:0]     Q_in,
    input  wire [DATA_WIDTH*L*N*E-1:0]     K_in,

    // Output: A of shape (L, N, L)
    output reg  [DATA_WIDTH*L*N*L-1:0]     A_out,
    output reg                             out_valid
);

    // Convert Q_in/K_in to arrays
    reg [DATA_WIDTH-1:0] Q [0:L-1][0:N-1][0:E-1];
    reg [DATA_WIDTH-1:0] K [0:L-1][0:N-1][0:E-1];

    // Store A in a local 3D array [l, n, l2]
    reg [DATA_WIDTH-1:0] A [0:L-1][0:N-1][0:L-1];

    integer l, n_, l2, e_;
    reg [2:0] state;

    localparam S_IDLE = 3'd0,
               S_LOAD = 3'd1,
               S_COMPUTE=3'd2,
               S_DONE = 3'd3;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state     <= S_IDLE;
            done      <= 1'b0;
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
                            for(e_=0; e_<E; e_=e_+1) begin
                                Q[l][n_][e_] = Q_in[
                                  ((l*N*E)+(n_*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH
                                ];
                                K[l][n_][e_] = K_in[
                                  ((l*N*E)+(n_*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH
                                ];
                            end
                        end
                    end
                    state <= S_COMPUTE;
                end

                // 2) COMPUTE: A[l,n,l2] = dot(Q[l,n], K[l2,n]) / sqrt(E)
                S_COMPUTE: begin
                    for(l=0; l<L; l=l+1) begin
                        for(n_=0; n_<N; n_=n_+1) begin
                            for(l2=0; l2<L; l2=l2+1) begin
                                reg [DATA_WIDTH-1:0] sum_temp;
                                sum_temp = {DATA_WIDTH{1'b0}};
                                for(e_=0; e_<E; e_=e_+1) begin
                                    sum_temp = sum_temp + (Q[l][n_][e_] * K[l2][n_][e_]);
                                end
                                // Optional scale by 1/sqrt(E)
                                A[l][n_][l2] = sum_temp; 
                            end
                        end
                    end
                    state <= S_DONE;
                end

                // 3) DONE: pack A into A_out
                S_DONE: begin
                    for(l=0; l<L; l=l+1) begin
                        for(n_=0; n_<N; n_=n_+1) begin
                            for(l2=0; l2<L; l2=l2+1) begin
                                A_out[
                                  ((l*N*L)+(n_*L)+l2+1)*DATA_WIDTH -1 -: DATA_WIDTH
                                ] = A[l][n_][l2];
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
