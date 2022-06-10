#include <hls_stream.h>
#include <ap_int.h>
#include <assert.h>

#include "common.h"

void load_duplicate(
    const PACKED_VAL_T *packed_dense_vector,              // in
    const unsigned num_cols,                              // in
    const unsigned num_row_tiles,                         // in
    const unsigned num_col_tiles,                         // in
    hls::stream<VEC_AXIS_T> duplicate[3]                  // out
) {
    #pragma HLS array_partition variable=duplicate complete
    unsigned num_col_last_col_tile;
    if (num_cols % LOGICAL_VB_SIZE == 0) {
        num_col_last_col_tile = LOGICAL_VB_SIZE;
    } else {
        num_col_last_col_tile = num_cols % LOGICAL_VB_SIZE;
    }

    vector_loader_over_row_tiles:
    for (unsigned row_tile_id = 0; row_tile_id < num_row_tiles; row_tile_id++) {

        vector_loader_over_col_tiles:
        for (unsigned col_tile_id = 0; col_tile_id < num_col_tiles; col_tile_id++)  {
            #pragma HLS pipeline off

            // attach switch column tile (partition) token
            VEC_AXIS_T pout_sod;
            pout_sod.user = SOD;
            pout_sod.data = 0;
            for (unsigned k = 0; k < 3; k++) {
                #pragma HLS unroll
                duplicate[k].write(pout_sod);
            }

            unsigned part_len = LOGICAL_VB_SIZE;
            if (col_tile_id == num_col_tiles - 1) {
                part_len = num_col_last_col_tile;
            }

            assert(part_len % PACK_SIZE == 0);

            loop_load_vector_packets:
            for (unsigned i = 0; i < part_len / PACK_SIZE; i++) {
                #pragma HLS pipeline II=1
                IDX_T dv_idx = i + col_tile_id * VB_PER_CLUSTER / PACK_SIZE;
                PACKED_VAL_T dv_pkt = packed_dense_vector[dv_idx];
                VEC_AXIS_T pout[3];
                for (unsigned x = 0; x < 3; x++) {
                    #pragma HLS unroll
                    for (unsigned k = 0; k < PACK_SIZE; k++) {
                        #pragma HLS unroll
                        VEC_AXIS_VAL(pout[x], k) = VAL_T_BITCAST(dv_pkt.data[k]);
                    }
                    pout[x].user = 0;
                    // TODO: use the column index within the tile width, i.e. mod
                    // tile_width, to simplify the operations in VAU side
                    VEC_AXIS_PKT_IDX(pout[x]) = dv_idx;
                    duplicate[x].write(pout[x]);
                }
            }

            // attach switch column partition token
            VEC_AXIS_T pout_eod;
            pout_eod.user = EOD;
            pout_eod.data = 0;
            for (unsigned k = 0; k < 3; k++) {
                #pragma HLS unroll
                duplicate[k].write(pout_eod);
            }

        }

    }

    // attach last token
    VEC_AXIS_T pout_eos;
    pout_eos.user = EOS;
    pout_eos.data = 0;
    for (unsigned k = 0; k < 3; k++) {
        #pragma HLS unroll
        duplicate[k].write(pout_eos);
    }

}

void write_k2ks(
    hls::stream<VEC_AXIS_T> &in,                              // in
    hls::stream<VEC_AXIS_IF_T> &out                           // out
) {
    bool exit = false;
    loop_fifo2axis:
    while (!exit) {
        #pragma HLS pipeline II=1
        VEC_AXIS_T pkt = in.read();
        VEC_AXIS_IF_T pkt_if;
        pkt_if.data = pkt.data;
        pkt_if.user = pkt.user;
        out.write(pkt_if);
        exit = (pkt.user == EOS);
    }
}

extern "C" {
void spmv_vector_loader(
    const PACKED_VAL_T *packed_dense_vector,                  // in
    const unsigned num_cols,                                  // in
    const unsigned num_row_tiles,                             // in
    const unsigned num_col_tiles,                             // in
    hls::stream<VEC_AXIS_IF_T> &to_SLR0,                      // out
    hls::stream<VEC_AXIS_IF_T> &to_SLR1,                      // out
    hls::stream<VEC_AXIS_IF_T> &to_SLR2                       // out
) {
    #pragma HLS interface m_axi port=packed_dense_vector offset=slave bundle=spmv_vin
    #pragma HLS interface s_axilite port=packed_dense_vector bundle=control
    #pragma HLS interface s_axilite port=num_cols bundle=control
    #pragma HLS interface s_axilite port=num_row_tiles bundle=control
    #pragma HLS interface s_axilite port=num_col_tiles bundle=control
    #pragma HLS interface s_axilite port=return bundle=control

    #pragma HLS interface axis register both port=to_SLR0
    #pragma HLS interface axis register both port=to_SLR1
    #pragma HLS interface axis register both port=to_SLR2

    #pragma HLS dataflow
    hls::stream<VEC_AXIS_T> duplicate[3];
    #pragma HLS stream variable=duplicate depth=8
    load_duplicate(packed_dense_vector, num_cols, num_row_tiles, num_col_tiles, duplicate);
    write_k2ks(duplicate[0], to_SLR0);
    write_k2ks(duplicate[1], to_SLR1);
    write_k2ks(duplicate[2], to_SLR2);

} // kernel
} // extern "C"
