//------------------------------------------------------------------------------
// mult.sv
//
// This is one stage of an 4 stage (5 depending on how you look at it)
// pipelined multiplier that multiplies 2 16-bit integers and returns
// the low 16 bits of the result.  This is not an ideal multiplier but
// is sufficient to allow a faster clock period than straight *
//
// Feb. 10 2023    Tianwei Liu    Initial version
// Apr. 30 2025    Tianwei Liu    Configure to 16 bit
//------------------------------------------------------------------------------

`define STAGE 4
parameter bit [(16/`STAGE)-1:0] ZERO = 'b0;

module mult_stage(
                    input clock, reset, start,
                    input [15:0] product_in, mplier_in, mcand_in,

                    output logic done,
                    output logic [15:0] product_out, mplier_out, mcand_out
                );



    logic [15:0] prod_in_reg, partial_prod_reg;
    logic [15:0] partial_product, next_mplier, next_mcand;

    assign product_out = prod_in_reg + partial_prod_reg;

    assign partial_product = mplier_in[(16/`STAGE)-1:0] * mcand_in;

    assign next_mplier = {ZERO,mplier_in[15:(16/`STAGE)]};
    assign next_mcand = {mcand_in[16-(16/`STAGE)-1:0],ZERO};

    //synopsys sync_set_reset "reset"
    always_ff @(posedge clock) begin
        prod_in_reg      <= product_in;
        partial_prod_reg <= partial_product;
        mplier_out       <= next_mplier;
        mcand_out        <= next_mcand;
    end

    // synopsys sync_set_reset "reset"
    always_ff @(posedge clock) begin
        if(reset)
            done <= 1'b0;
        else
            done <= start;
    end

endmodule
