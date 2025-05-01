//------------------------------------------------------------------------------
// mult.sv
//
// This is an 4 stage (5 depending on how you look at it) pipelined 
// multiplier that multiplies 2 16-bit integers and returns the low 64 bits 
// of the result.  This is not an ideal multiplier but is sufficient to 
// allow a faster clock period than straight *
// This module instantiates 4 pipeline stages as an array of submodules.
//
// Feb. 10 2023    Tianwei Liu    Initial version
// Apr. 30 2025    Tianwei Liu    Configure to 16 bit
//------------------------------------------------------------------------------

`define STAGE 4

module mult(
                input clock, reset,
                input [15:0] mcand, mplier,
                input start,

                output [15:0] product,
                output done
            );

  logic [15:0] mcand_out, mplier_out;
  logic [((`STAGE-1)*16)-1:0] internal_products, internal_mcands, internal_mpliers;
  logic [(`STAGE-2):0] internal_dones;
  
    mult_stage mstage [(`STAGE-1):0]  (
        .clock(clock),
        .reset(reset),
        .product_in({internal_products,16'h0}),
        .mplier_in({internal_mpliers,mplier}),
        .mcand_in({internal_mcands,mcand}),
        .start({internal_dones,start}),
        .product_out({product,internal_products}),
        .mplier_out({mplier_out,internal_mpliers}),
        .mcand_out({mcand_out,internal_mcands}),
        .done({done,internal_dones})
    );

endmodule