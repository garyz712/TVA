//------------------------------------------------------------------------------
// patch_embedding.sv
//
// This module takes an input image and converts it into a sequence of patch
// embeddings suitable for transformer processing. Each patch of size PH x PW
// from the input image is linearly projected into an E-dimensional embedding
// space using learned weights (W_patch) and biases (b_patch).
//
// Apr. 15 2025    Max Zhang      Initial version
// Apr. 21 2025    Tianwei Liu    Syntax fix and comments
// Apr. 25 2025    Tianwei Liu    Changed default parameter
//------------------------------------------------------------------------------
module patch_embedding #(
    parameter int DATA_WIDTH  = 4,  // e.g., 16 bits per pixel or intermediate
    parameter int IMG_H       = 32,
    parameter int IMG_W       = 32,
    parameter int C           = 3,   // input channels
    parameter int PH          = 16,  // patch height
    parameter int PW          = 16,  // patch width
    parameter int E           = 8  // embedding dimension
)(
    input  logic                                          clk,
    input  logic                                          rst_n,

    // Control signals
    input  logic                                          start,
    output logic                                          done,

    // Flattened image input: shape (IMG_H, IMG_W, C)
    // => total IMG_H*IMG_W*C elements, each DATA_WIDTH bits
    input  logic [DATA_WIDTH*IMG_H*IMG_W*C -1:0]          image_in,

    // Weight & bias for patch embedding
    // W_patch_in: shape (PATCH_SIZE x E) => (PH*PW*C x E)
    // Flattened => DATA_WIDTH*(PH*PW*C*E)
    input  logic [DATA_WIDTH*PH*PW*C*E -1:0]              W_patch_in,
    // b_patch_in: shape (E)
    input  logic [DATA_WIDTH*E -1:0]                      b_patch_in,

    // Output: (NUM_PATCHES, E), flattened
    // Where NUM_PATCHES = (IMG_H/PH)*(IMG_W/PW)
    output logic [DATA_WIDTH*((IMG_H/PH)*(IMG_W/PW))*E -1:0] patch_out,

    output logic                                          out_valid
);

    //--------------------------------------------------------------------------
    // 1) Parameters
    //--------------------------------------------------------------------------
    localparam int PATCH_SIZE  = PH*PW*C;
    localparam int PATCH_H_CNT = IMG_H / PH;   // #vertical patches
    localparam int PATCH_W_CNT = IMG_W / PW;   // #horizontal patches
    localparam int NUM_PATCHES = PATCH_H_CNT * PATCH_W_CNT;  // total patches

    //--------------------------------------------------------------------------
    // 2) State definitions
    //--------------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_LOAD,
        S_SETUP,       // prepare for dot-product
        S_DOT,         // accumulate sum for one embedding dimension
        S_STORE,       // store partial result
        S_CHECK_DONE,  // check if done all dims/patches
        S_DONE
    } state_t;

    state_t curr_state, next_state;

    //--------------------------------------------------------------------------
    // 3) Internal Storage
    //--------------------------------------------------------------------------
    // For time-multiplexing:
    // - We'll keep a small "dot product" logic that accumulates the multiplication
    //   of flatten(patch) with W_patch_in for each embedding dimension "e_idx".

    // We can store the entire image_in in a local array or read it directly from the
    // flattened bus. Below, we do a partial approach.

    // We store W_patch in a 2D array: [PATCH_SIZE][E].
    // We store b_patch in 1D array: [E].
    logic [DATA_WIDTH-1:0] W_patch [0:PATCH_SIZE-1][0:E-1];
    logic [DATA_WIDTH-1:0] b_patch [0:E-1];

    // We'll store the final embedded patch vectors in:
    // patch_out_mem[patch_idx][dim_e], each of DATA_WIDTH bits
    logic [DATA_WIDTH-1:0] patch_out_mem [0:NUM_PATCHES-1][0:E-1];

    // We'll handle partial sums in a larger width to avoid overflow:
    logic [31:0] sum_temp;

    // We'll keep counters:
    // patch_idx => 0 .. (NUM_PATCHES-1)
    // dim_e     => 0 .. (E-1)
    // px_count  => 0 .. (PATCH_SIZE-1) => for partial dot products
    logic [$clog2(NUM_PATCHES):0] patch_idx;
    logic [$clog2(E):0]           dim_e;
    logic [$clog2(PATCH_SIZE):0]  px_count;

    // Some signals for partial image indexing
    // We'll compute patch_y, patch_x, local pixel offset, etc. if we want to retrieve from image_in.

    //--------------------------------------------------------------------------
    // 4) State register
    //--------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n)
            curr_state <= S_IDLE;
        else
            curr_state <= next_state;
    end

    //--------------------------------------------------------------------------
    // 5) Next-state logic (always_comb)
    //--------------------------------------------------------------------------
    always_comb begin
        // default
        next_state = curr_state;

        case(curr_state)
            S_IDLE: begin
                if(start)
                    next_state = S_LOAD;
            end

            // S_LOAD => read W_patch_in, b_patch_in into local arrays
            //           or read if not already stored.
            S_LOAD: begin
                next_state = S_SETUP;
            end

            // S_SETUP => init counters, sum_temp, prepare first dot-product
            S_SETUP: begin
                next_state = S_DOT;
            end

            // S_DOT => partial accumulate: sum_temp += patch_pixel * W_patch[px_count, dim_e]
            //         once px_count reaches PATCH_SIZE-1 => S_STORE
            S_DOT: begin
                if(px_count == PATCH_SIZE-1)
                    next_state = S_STORE;
            end

            // S_STORE => store sum_temp => patch_out_mem[patch_idx][dim_e]
            // then next_state => S_CHECK_DONE
            S_STORE: begin
                next_state = S_CHECK_DONE;
            end

            // S_CHECK_DONE => see if we've done all E dims for this patch,
            //  and if we've done all patches => S_DONE, else go S_SETUP
            S_CHECK_DONE: begin
                if ((patch_idx==(NUM_PATCHES-1)) && (dim_e==(E-1)))
                    next_state = S_DONE;
                else
                    next_state = S_SETUP;
            end

            S_DONE: begin
                next_state = S_IDLE; // or stay in DONE
            end

            default: next_state = S_IDLE;
        endcase
    end

    //--------------------------------------------------------------------------
    // 6) Datapath / Output logic
    //--------------------------------------------------------------------------

    // 6a) Unpack W_patch_in, b_patch_in in always_comb or a small sub-FSM
    // We'll do a simple always_comb to read them
    always_comb begin
        for (int i = 0; i < E; i++) begin
            b_patch[i] = b_patch_in[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH];
        end
        for (int i = 0; i < PATCH_SIZE; i++) begin
            for (int j = 0; j < E; j++) begin
                W_patch[i][j] = W_patch_in[ ((i*E)+j+1)*DATA_WIDTH -1 -: DATA_WIDTH];
            end
        end
    end

    // 6b) Wires for extracted pixel from image_in
    // We'll do a function get_pixel() to read from flattened image_in
    // based on patch_idx, px_count, etc.

    function logic [DATA_WIDTH-1:0] get_pixel(
        input logic [$clog2(NUM_PATCHES):0] p_idx,
        input logic [$clog2(PATCH_SIZE):0] px
    );
        // We'll decode p_idx => (patch_row, patch_col)
        //   patch_row = p_idx / PATCH_W_CNT
        //   patch_col = p_idx % PATCH_W_CNT
        // Then px => (local_h, local_w, ch)
        //   local_h = px / (PW*C)
        //   local_w = (px % (PW*C)) / C
        //   ch      = px % C
        // Then global y = patch_row*PH + local_h
        //      global x = patch_col*PW + local_w
        // Flattened index => (y*IMG_W + x)*C + ch
        // We'll read from image_in.
        logic [DATA_WIDTH-1:0] pix_val;
        logic [$clog2(IMG_H):0] gy, gx;
        logic [$clog2(PH):0]   ly;
        logic [$clog2(PW):0]   lx;
        logic [$clog2(C):0]    cc;
        logic [$clog2(PATCH_W_CNT):0] pcx;
        logic [$clog2(PATCH_H_CNT):0] pcy;

        logic [$clog2(IMG_H*IMG_W*C):0] flat_idx;

        pcy = p_idx / PATCH_W_CNT;
        pcx = p_idx % PATCH_W_CNT;
        ly  = px / (PW*C);
        lx  = (px % (PW*C)) / C;
        cc  = px % C;

        gy  = pcy*PH + ly;
        gx  = pcx*PW + lx;

        flat_idx = (((gy*IMG_W) + gx)*C + cc); // flatten to 1D

        pix_val = image_in[(flat_idx+1)*DATA_WIDTH -1 -: DATA_WIDTH];
        return pix_val;
    endfunction

    // 6c) The main sequential logic for sum_temp, counters, storing results
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            done        <= 1'b0;
            out_valid   <= 1'b0;
            patch_idx   <= 0;
            dim_e       <= 0;
            px_count    <= 0;
            sum_temp    <= 32'd0;

            // init patch_out_mem to zero
            for(int i=0; i<NUM_PATCHES; i++) begin
                for(int j=0; j<E; j++) begin
                    patch_out_mem[i][j] <= '0;
                end
            end

        end else begin
            // defaults
            done      <= 1'b0;
            out_valid <= 1'b0;

            case(curr_state)

                S_IDLE: begin
                    // no action
                end

                S_LOAD: begin
                    // (already have W_patch, b_patch assigned in always_comb)
                    // no action if all done combinationally
                end

                S_SETUP: begin
                    px_count <= 0;
                    // init sum_temp with bias
                    sum_temp <= {16'd0, b_patch[dim_e]}; 
                    // or sign-extend/zero-extend as needed
                end

                S_DOT: begin
                    // partial accumulate
                    logic [DATA_WIDTH-1:0] pix_val;
                    pix_val = get_pixel(patch_idx, px_count);

                    // multiply pix_val * W_patch[px_count][dim_e]
                    // Convert them to bigger int or do proper FP. 
                    sum_temp <= sum_temp + (pix_val * W_patch[px_count][dim_e]);

                    if(px_count < PATCH_SIZE-1)
                        px_count <= px_count + 1;
                end

                S_STORE: begin
                    // store sum_temp (truncated to DATA_WIDTH or do saturate) 
                    // in patch_out_mem[patch_idx][dim_e]
                    patch_out_mem[patch_idx][dim_e] <= sum_temp[DATA_WIDTH-1:0];
                end

                S_CHECK_DONE: begin
                    // move to next dimension or next patch
                    if(dim_e < E-1) begin
                        dim_e <= dim_e + 1;
                    end else begin
                        dim_e <= 0;
                        if(patch_idx < NUM_PATCHES-1)
                            patch_idx <= patch_idx + 1;
                    end
                end

                S_DONE: begin
                    done      <= 1'b1;
                    out_valid <= 1'b1;

                    // Flatten patch_out_mem into patch_out
                    // we do that either right here or in an always_comb
                    for(int p=0; p<NUM_PATCHES; p++) begin
                        for(int d_=0; d_<E; d_++) begin
                            patch_out[ ((p*E)+ d_ +1)*DATA_WIDTH -1 -: DATA_WIDTH ]
                                <= patch_out_mem[p][d_];
                        end
                    end
                end

                default: /* no-op */;
            endcase
        end
    end

endmodule