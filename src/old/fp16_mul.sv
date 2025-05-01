//------------------------------------------------------------------------------
// fp16_mul.sv
//
// A Synthesizable IEEE 754 Half-Precision (FP16) Floating Point Multiplier
// Maybe use IP later, but this is needed for now.
//
// Apr. 29 2025    Tianwei Liu    Initial Version
//------------------------------------------------------------------------------
module fp16_mul (
    input  logic [15:0] A,
    input  logic [15:0] B,
    output logic [15:0] result
);
    logic sign_a, sign_b, sign_res;
    logic [4:0] exp_a, exp_b, exp_res;
    logic [10:0] man_a, man_b;
    logic [21:0] man_mul;
    logic [4:0] exp_sum;

    always_comb begin
        // Extract sign, exponent, mantissa
        sign_a = A[15];
        sign_b = B[15];
        exp_a = A[14:10];
        exp_b = B[14:10];
        man_a = {1'b1, A[9:0]}; // add implicit 1
        man_b = {1'b1, B[9:0]};

        // Multiply mantissas
        man_mul = man_a * man_b; // 11x11 -> 22 bits

        // Add exponents and subtract bias
        exp_sum = exp_a + exp_b - 5'd15;

        // Normalize
        if (man_mul[21]) begin
            man_mul = man_mul >> 1;
            exp_sum = exp_sum + 1;
        end

        // Assign final values (TODO: rounding/overflow handling)
        result[15]   = sign_a ^ sign_b;
        result[14:10]= exp_sum;
        result[9:0]  = man_mul[19:10]; // take 10 bits after normalization
    end
endmodule
