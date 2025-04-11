module token_precision_analyzer #(
    parameter DATA_WIDTH = 16,
    parameter L = 8,
    parameter N = 1
)(
    input                         clk,
    input                         rst_n,
    input                         start,
    output logic                  done,

    // A_in: shape (L, N, L)
    input  [DATA_WIDTH*L*N*L-1:0] A_in,
    
    // One 4-bit precision code per "key token" => total L codes
    output logic [3:0]            token_precision [0:L-1],
    output logic                  out_valid
);

    logic   [DATA_WIDTH-1:0] A_arr [0:L-1][0:N-1][0:L-1];
    integer l, n_, l2;
    logic   [2:0] state;

    typedef enum { S_IDLE, S_LOAD, S_ANALYZE, S_DONE } STATES;

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
                    state <= S_ANALYZE;
                end

                // 2) ANALYZE: sum over (l,n) for each l2
                S_ANALYZE: begin
                    for(l2=0; l2<L; l2=l2+1) begin
                        reg [DATA_WIDTH-1:0] col_sum;
                        col_sum = {DATA_WIDTH{1'b0}};
                        for(l=0; l<L; l=l+1) begin
                            for(n_=0; n_<N; n_=n_+1) begin
                                col_sum = col_sum + A_arr[l][n_][l2];
                            end
                        end
                        // Decide code
                        if(col_sum < 100)
                            token_precision[l2] = 4'd0; // INT4
                        else if (col_sum < 200)
                            token_precision[l2] = 4'd1; // INT8
                        else
                            token_precision[l2] = 4'd2; // FP16
                    end
                    state <= S_DONE;
                end

                S_DONE: begin
                    out_valid <= 1'b1;
                    done      <= 1'b1;
                    state     <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
