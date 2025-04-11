module qkv_generator #(
    parameter DATA_WIDTH = 16,   // e.g., FP16 or fixed
    parameter L = 8,            // Sequence length
    parameter N = 1,            // Batch size
    parameter E = 8             // Embedding dimension
)(
    input  wire                               clk,
    input  wire                               rst_n,
    // Control
    input  wire                               start,
    output reg                                done,

    // Input x: (L, N, E)
    input  wire [DATA_WIDTH*L*N*E-1:0]        x_in,
    
    // Weights for Q, K, V: each (E x E), plus bias of length E
    // For brevity, we pass them in flattened. Real design likely in BRAM or memory.
    input  wire [DATA_WIDTH*E*E-1:0]          WQ_in,
    input  wire [DATA_WIDTH*E*E-1:0]          WK_in,
    input  wire [DATA_WIDTH*E*E-1:0]          WV_in,
    input  wire [DATA_WIDTH*E-1:0]            bQ_in,
    input  wire [DATA_WIDTH*E-1:0]            bK_in,
    input  wire [DATA_WIDTH*E-1:0]            bV_in,

    // Outputs Q, K, V (each L,N,E)
    output reg  [DATA_WIDTH*L*N*E-1:0]        Q_out,
    output reg  [DATA_WIDTH*L*N*E-1:0]        K_out,
    output reg  [DATA_WIDTH*L*N*E-1:0]        V_out,
    output reg                                out_valid
);

    // Internal arrays for x, WQ, WK, WV, bQ, bK, bV
    reg [DATA_WIDTH-1:0] x_arr  [0:L-1][0:N-1][0:E-1];
    reg [DATA_WIDTH-1:0] WQ_arr [0:E-1][0:E-1], WK_arr [0:E-1][0:E-1], WV_arr [0:E-1][0:E-1];
    reg [DATA_WIDTH-1:0] bQ_arr [0:E-1], bK_arr [0:E-1], bV_arr [0:E-1];

    // Q, K, V arrays
    reg [DATA_WIDTH-1:0] Q_arr [0:L-1][0:N-1][0:E-1];
    reg [DATA_WIDTH-1:0] K_arr [0:L-1][0:N-1][0:E-1];
    reg [DATA_WIDTH-1:0] V_arr [0:L-1][0:N-1][0:E-1];

    integer l, n_, e_, i;
    reg [2:0] state;

    localparam S_IDLE = 3'd0,
               S_LOAD = 3'd1,
               S_GENQ = 3'd2,
               S_GENK = 3'd3,
               S_GENV = 3'd4,
               S_DONE = 3'd5;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state     <= S_IDLE;
            done      <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            case(state)
                // Wait for start
                S_IDLE: begin
                    if(start) begin
                        done      <= 1'b0;
                        out_valid <= 1'b0;
                        state     <= S_LOAD;
                    end
                end

                // 1) LOAD: parse x_in, WQ_in, WK_in, WV_in, etc.
                S_LOAD: begin
                    // (L, N, E) -> x_arr
                    for(l=0; l<L; l=l+1) begin
                        for(n_=0; n_<N; n_ = n_+1) begin
                            for(e_=0; e_<E; e_=e_+1) begin
                                x_arr[l][n_][e_] = x_in[
                                    ((l*N*E)+(n_*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH
                                ];
                            end
                        end
                    end
                    // (E, E) -> WQ_arr
                    for(i=0; i<E; i=i+1) begin
                        for(e_=0; e_<E; e_ = e_+1) begin
                            WQ_arr[i][e_] = WQ_in[ ((i*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH ];
                            WK_arr[i][e_] = WK_in[ ((i*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH ];
                            WV_arr[i][e_] = WV_in[ ((i*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH ];
                        end
                        bQ_arr[i] = bQ_in[ (i+1)*DATA_WIDTH -1 -: DATA_WIDTH ];
                        bK_arr[i] = bK_in[ (i+1)*DATA_WIDTH -1 -: DATA_WIDTH ];
                        bV_arr[i] = bV_in[ (i+1)*DATA_WIDTH -1 -: DATA_WIDTH ];
                    end
                    state <= S_GENQ;
                end

                // 2) GENQ: Q[l,n] = x[l,n] * WQ + bQ
                S_GENQ: begin
                    for(l=0; l<L; l=l+1) begin
                        for(n_=0; n_<N; n_=n_+1) begin
                            for(i=0; i<E; i=i+1) begin
                                reg [DATA_WIDTH-1:0] sum_temp;
                                sum_temp = bQ_arr[i];
                                for(e_=0; e_<E; e_=e_+1) begin
                                    // sum_temp += x_arr[l][n_][e_] * WQ_arr[i][e_]
                                    sum_temp = sum_temp + (x_arr[l][n_][e_] * WQ_arr[i][e_]);
                                end
                                Q_arr[l][n_][i] = sum_temp;
                            end
                        end
                    end
                    state <= S_GENK;
                end

                // 3) GENK: K[l,n] = x[l,n] * WK + bK
                S_GENK: begin
                    for(l=0; l<L; l=l+1) begin
                        for(n_=0; n_<N; n_=n_+1) begin
                            for(i=0; i<E; i=i+1) begin
                                reg [DATA_WIDTH-1:0] sum_temp;
                                sum_temp = bK_arr[i];
                                for(e_=0; e_<E; e_=e_+1) begin
                                    sum_temp = sum_temp + (x_arr[l][n_][e_] * WK_arr[i][e_]);
                                end
                                K_arr[l][n_][i] = sum_temp;
                            end
                        end
                    end
                    state <= S_GENV;
                end

                // 4) GENV: V[l,n] = x[l,n] * WV + bV
                S_GENV: begin
                    for(l=0; l<L; l=l+1) begin
                        for(n_=0; n_<N; n_=n_+1) begin
                            for(i=0; i<E; i=i+1) begin
                                reg [DATA_WIDTH-1:0] sum_temp;
                                sum_temp = bV_arr[i];
                                for(e_=0; e_<E; e_=e_+1) begin
                                    sum_temp = sum_temp + (x_arr[l][n_][e_] * WV_arr[i][e_]);
                                end
                                V_arr[l][n_][i] = sum_temp;
                            end
                        end
                    end
                    state <= S_DONE;
                end

                // 5) DONE: pack Q, K, V => Q_out, K_out, V_out
                S_DONE: begin
                    for(l=0; l<L; l=l+1) begin
                        for(n_=0; n_<N; n_=n_+1) begin
                            for(e_=0; e_<E; e_=e_+1) begin
                                Q_out[
                                  ((l*N*E)+(n_*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH
                                ] = Q_arr[l][n_][e_];

                                K_out[
                                  ((l*N*E)+(n_*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH
                                ] = K_arr[l][n_][e_];

                                V_out[
                                  ((l*N*E)+(n_*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH
                                ] = V_arr[l][n_][e_];
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
