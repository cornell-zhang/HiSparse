#include <hls_stream.h>
#include <ap_int.h>
#include <assert.h>

#include "common.h"

void load_duplicate(
    const PACKED_VAL_T *packed_dense_vector,              // in
    const unsigned num_cols,                              // in
    hls::stream<VEC_AXIS_T> duplicate[3],                 // out
    const unsigned logical_vb_size
) {
    #pragma HLS array_partition variable=duplicate complete
    unsigned num_col_partitions = (num_cols + logical_vb_size - 1) / logical_vb_size;
    unsigned num_col_last_partition;
    if (num_cols % logical_vb_size == 0) {
        num_col_last_partition = logical_vb_size;
    } else {
        num_col_last_partition = num_cols % logical_vb_size;
    }

    vector_loader_over_col_partitions:
    for (unsigned part_id = 0; part_id < num_col_partitions; part_id++)  {
        #pragma HLS pipeline off

        // attach switch column partition token
        VEC_AXIS_T pout_sod;
        pout_sod.user = SOD;
        pout_sod.data = 0;
        for (unsigned k = 0; k < 3; k++) {
            #pragma HLS unroll
            duplicate[k].write(pout_sod);
        }

        unsigned part_len = logical_vb_size;
        if (part_id == num_col_partitions - 1) {
            part_len = num_col_last_partition;
        }

        assert(part_len % PACK_SIZE == 0);

        loop_load_vector_packets:
        for (unsigned i = 0; i < part_len / PACK_SIZE; i++) {
            #pragma HLS pipeline II=1
            IDX_T dv_idx = i + part_id * logical_vb_size / PACK_SIZE;
            PACKED_VAL_T dv_pkt = packed_dense_vector[dv_idx];
            VEC_AXIS_T pout[3];
            for (unsigned x = 0; x < 3; x++) {
                #pragma HLS unroll
                for (unsigned k = 0; k < PACK_SIZE; k++) {
                    #pragma HLS unroll
                    VEC_AXIS_VAL(pout[x], k) = VAL_T_BITCAST(dv_pkt.data[k]);
                }
                pout[x].user = 0;
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
    hls::stream<VEC_AXIS_T> &in,                      // in
    hls::stream<VEC_AXIS_T> &out                      // out
) {
    bool exit = false;
    loop_fifo2axis:
    while (!exit) {
        #pragma HLS pipeline II=1
        VEC_AXIS_T pkt = in.read();
        out.write(pkt);
        exit = (pkt.user == EOS);
    }
}

extern "C" {
void spmv_vector_loader(
    const PACKED_VAL_T *packed_dense_vector,               // in
    const unsigned num_cols,                               // in
    hls::stream<VEC_AXIS_T> &to_SLR0,                      // out
    hls::stream<VEC_AXIS_T> &to_SLR1,                      // out
    hls::stream<VEC_AXIS_T> &to_SLR2,                      // out
    const unsigned logical_vb_size
) {
    #pragma HLS interface m_axi port=packed_dense_vector offset=slave bundle=spmv_vin
    #pragma HLS interface s_axilite port=packed_dense_vector bundle=control
    #pragma HLS interface s_axilite port=num_cols bundle=control
    #pragma HLS interface s_axilite port=return bundle=control

    #pragma HLS interface axis register both port=to_SLR0
    #pragma HLS interface axis register both port=to_SLR1
    #pragma HLS interface axis register both port=to_SLR2

    #pragma HLS dataflow
    hls::stream<VEC_AXIS_T> duplicate[3];
    #pragma HLS stream variable=duplicate depth=8
    load_duplicate(packed_dense_vector, num_cols, duplicate, logical_vb_size);
    write_k2ks(duplicate[0], to_SLR0);
    write_k2ks(duplicate[1], to_SLR1);
    write_k2ks(duplicate[2], to_SLR2);

} // kernel
} // extern "C"
