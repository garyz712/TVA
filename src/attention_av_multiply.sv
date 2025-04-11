module attention_av_multiply #(
    parameter DATA_WIDTH = 16,
    parameter L = 8,
    parameter N = 1,
    parameter E = 8
)(
    input  wire                                       clk,
    input  wire                                       rst_n,
    // Control
    input  wire                                       start,
    output reg                                        done,

    // Input A: shape (L, N, L)
    input  wire [DATA_WIDTH*L*N*L-1:0]                A_in,
    // Input V: shape (L, N, E)
    input  wire [DATA_WIDTH*L*N*E-1:0]                V_in,

    // Per-token precision codes: length L
    input  wire [3:0]                                  token_precision [0:L-1],

    // Output Z: shape (L, N, E)
    output reg  [DATA_WIDTH*L*N*E-1:0]                Z_out,
    output reg                                        out_valid
);

    reg [DATA_WIDTH-1:0] A_arr [0:L-1][0:N-1][0:L-1];
    reg [DATA_WIDTH-1:0] V_arr [0:L-1][0:N-1][0:E-1];
    reg [DATA_WIDTH-1:0] Z_arr [0:L-1][0:N-1][0:E-1];

    integer l, n_, l2, e_;
    reg [2:0] state;
    localparam S_IDLE=0, S_LOAD=1, S_MUL=2, S_DONE=3;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state <= S_IDLE;
            done  <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            case(state)
                S_IDLE: begin
                    if(start) begin
                        out_valid <= 1'b0;
                        done      <= 1'b0;
                        state     <= S_LOAD;
                    end
                end

                // 1) LOAD: parse A_in, V_in
                S_LOAD: begin
                    // A
                    for(l=0; l<L; l=l+1) begin
                        for(n_=0; n_<N; n_=n_+1) begin
                            for(l2=0; l2<L; l2=l2+1) begin
                                A_arr[l][n_][l2] = A_in[
                                  ((l*N*L)+(n_*L)+l2+1)*DATA_WIDTH -1 -: DATA_WIDTH
                                ];
                            end
                        end
                    end
                    // V
                    for(l2=0; l2<L; l2=l2+1) begin
                        for(n_=0; n_<N; n_=n_+1) begin
                            for(e_=0; e_<E; e_=e_+1) begin
                                V_arr[l2][n_][e_] = V_in[
                                  ((l2*N*E)+(n_*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH
                                ];
                            end
                        end
                    end
                    // Z init
                    for(l=0; l<L; l=l+1) begin
                        for(n_=0; n_<N; n_=n_+1) begin
                            for(e_=0; e_<E; e_=e_+1) begin
                                Z_arr[l][n_][e_] = {DATA_WIDTH{1'b0}};
                            end
                        end
                    end
                    state <= S_MUL;
                end

                // 2) MUL: Z[l,n,e] = ∑_{l2} A[l,n,l2] * downcast( V[l2,n,e] )
                S_MUL: begin
                    for(l=0; l<L; l=l+1) begin
                        for(n_=0; n_<N; n_=n_+1) begin
                            for(l2=0; l2<L; l2=l2+1) begin
                                reg [3:0] prec_code;
                                prec_code = token_precision[l2];
                                for(e_=0; e_<E; e_=e_+1) begin
                                    reg [DATA_WIDTH-1:0] valV_down;
                                    reg [DATA_WIDTH-1:0] product;

                                    // Downcast
                                    case(prec_code)
                                        4'd0: begin
                                            // INT4 => lower 4 bits
                                            valV_down = {
                                                {(DATA_WIDTH-4){1'b0}},
                                                V_arr[l2][n_][e_][3:0]
                                            };
                                        end
                                        4'd1: begin
                                            // INT8 => lower 8 bits
                                            valV_down = {
                                                {(DATA_WIDTH-8){1'b0}},
                                                V_arr[l2][n_][e_][7:0]
                                            };
                                        end
                                        4'd2: begin
                                            // FP16 or full 16 bits
                                            valV_down = V_arr[l2][n_][e_];
                                        end
                                        default: begin
                                            valV_down = V_arr[l2][n_][e_];
                                        end
                                    endcase

                                    // Multiply
                                    product = A_arr[l][n_][l2] * valV_down;

                                    // Accumulate
                                    Z_arr[l][n_][e_] = Z_arr[l][n_][e_] + product;
                                end
                            end
                        end
                    end
                    state <= S_DONE;
                end

                // 3) DONE: pack Z_arr => Z_out
                S_DONE: begin
                    for(l=0; l<L; l=l+1) begin
                        for(n_=0; n_<N; n_=n_+1) begin
                            for(e_=0; e_<E; e_=e_+1) begin
                                Z_out[
                                  ((l*N*E)+(n_*E)+e_+1)*DATA_WIDTH -1 -: DATA_WIDTH
                                ] = Z_arr[l][n_][e_];
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
