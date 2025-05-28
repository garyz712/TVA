// October 2023    Tianwei Liu
module ISR(
    input reset,
    input [63:0] value,
    input clock,
    output logic [31:0] result,
    output logic done
);
    logic [1:0] state; // 4 state, 0b00 = init, 0b01 = 
    logic [1:0] next_state;
    logic [63:0] val;

    logic [31:0] bit_mask;
    logic [31:0] next_bit_mask;
    logic [31:0] next_result;


    logic mul_reset;
    logic [63:0] mul_p;
    logic mul_start, mul_done;

    mult mul(.clock(clock), .reset(mul_reset), .mcand({32'b0, next_result}), .mplier({32'b0, next_result}), .start(mul_start), .product(mul_p), .done(mul_done));



    always_comb begin
        case(state)
            2'd0: begin
                next_result = 32'b0;
                done = 1'b0;
                mul_reset = 1'b1;
                mul_start = 1'b0;
                next_state = 1'b1;
                next_bit_mask = bit_mask;
            end
            2'd1: begin
                done = 1'b0;
                next_result = result | bit_mask;
                mul_reset = 1'b0;
                mul_start = 1'b1;
                next_state = 2'd2;
                next_bit_mask = bit_mask;
            end
            2'd2: begin
                mul_start = 1'b0;
                if(mul_done) begin
                    if(mul_p > val) next_result = result ^ bit_mask;
                    else next_result = result;
                    if(bit_mask == 0) begin
                        next_bit_mask = bit_mask;
                        done = 1'b1;
                        next_state = 0;
                    end else begin
                        next_bit_mask = bit_mask >> 1;
                        done = 1'b0;
                        next_state = 1;
                    end
                end else begin
                    done = 1'b0;
                    next_state = 2'd2;
                    next_result = result;
                    next_bit_mask = bit_mask;
                end
            end
            default: begin
                next_result = result;
                done = 1'b0;
                mul_reset = 1'b0;
                mul_start = 1'b0;
                next_state = 1'b0;
                next_bit_mask = bit_mask;
            end
        endcase
    end


    //synopsys sync_set_reset “reset”
    always_ff @(posedge clock) begin
        if(reset) begin
            val <= #1 value;
            state <= #1 2'b0;
            bit_mask <= #1 32'h8000_0000;
            result <= #1 32'h0000_0000;
        end else begin
            state <= #1 next_state;
            bit_mask <= #1 next_bit_mask;
            result <= #1 next_result;
        end
    end

endmodule
