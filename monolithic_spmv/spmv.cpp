#include <hls_stream.h>
#include <ap_fixed.h>
#include <assert.h>

#include "common.h"
#include "stream_utils.h"
#include "spmv_cluster.h"

#ifndef __SYNTHESIS__
#define SPMV_LINE_TRACING
// #define RESULT_DRAIN_LINE_TRACING
#endif

void vector_loader(
    const PACKED_VAL_T *packed_dense_vector,              // in
    const unsigned num_cols,                              // in
    hls::stream<VEC_AXIS_T> duplicate[16]                 // out
) {
    unsigned num_col_partitions = (num_cols + LOGICAL_VB_SIZE - 1) / LOGICAL_VB_SIZE;
    unsigned num_col_last_partition;
    if (num_cols % LOGICAL_VB_SIZE == 0) {
        num_col_last_partition = LOGICAL_VB_SIZE;
    } else {
        num_col_last_partition = num_cols % LOGICAL_VB_SIZE;
    }

    vector_loader_over_col_partitions:
    for (unsigned part_id = 0; part_id < num_col_partitions; part_id++)  {
        #pragma HLS pipeline off

        // attach switch column partition token
        VEC_AXIS_T pout_sod;
        pout_sod.user = SOD;
        pout_sod.data = 0;
        for (unsigned k = 0; k < 16; k++) {
            #pragma HLS unroll
            duplicate[k].write(pout_sod);
        }

        unsigned part_len = LOGICAL_VB_SIZE;
        if (part_id == num_col_partitions - 1) {
            part_len = num_col_last_partition;
        }

        assert(part_len % PACK_SIZE == 0);

        loop_load_vector_packets:
        for (unsigned i = 0; i < part_len / PACK_SIZE; i++) {
            #pragma HLS pipeline II=1
            IDX_T dv_idx = i + part_id * VB_PER_CLUSTER / PACK_SIZE;
            PACKED_VAL_T dv_pkt = packed_dense_vector[dv_idx];
            VEC_AXIS_T pout[16];
            for (unsigned x = 0; x < 16; x++) {
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
        for (unsigned k = 0; k < 16; k++) {
            #pragma HLS unroll
            duplicate[k].write(pout_eod);
        }

    }

    // attach last token
    VEC_AXIS_T pout_eos;
    pout_eos.user = EOS;
    pout_eos.data = 0;
    for (unsigned k = 0; k < 16; k++) {
        #pragma HLS unroll
        duplicate[k].write(pout_eos);
    }
}

void result_drain(
    PACKED_VAL_T *packed_dense_result,      // out
    const unsigned row_part_id,             // in
    hls::stream<VEC_AXIS_T> from_clusters[16]     // in
) {
    // write back
    char current_input = 0;
    ap_uint<16> finished = 0;
    unsigned write_counter = 0;
    bool exit = false;
    unsigned pkt_idx_offset = row_part_id * LOGICAL_OB_SIZE / PACK_SIZE;
    result_drain_main_loop:
    while (!exit) {
        #pragma HLS pipeline II=1
        VEC_AXIS_T pkt;
        bool do_write = false;

        if (!finished[current_input]) {
            pkt = from_clusters[current_input].read();
#ifdef RESULT_DRAIN_LINE_TRACING
            std::cout << "RD got pkt from cluster " << current_input << ": " << pkt;
#endif
            if (pkt.user == EOS) {
                finished[current_input] = true;
                do_write = false;
            } else if (pkt.user != SOD && pkt.user != EOD) {
                do_write = true;
            }
        } else {
            do_write = false;
        }
        current_input = (current_input + 1) % 16;

        exit = finished.and_reduce();

        unsigned abs_pkt_idx = write_counter + pkt_idx_offset;
        if (do_write) {
            PACKED_VAL_T rout;
            for (unsigned k = 0; k < PACK_SIZE; k++) {
                #pragma HLS unroll
                VAL_T_BITCAST(rout.data[k]) = VEC_AXIS_VAL(pkt, k);
            }
            write_counter++;
            packed_dense_result[abs_pkt_idx] = rout;
        }

#ifdef RESULT_DRAIN_LINE_TRACING
        if (do_write) {
            std::cout << ", written to " << abs_pkt_idx << std::endl;
        } else {
            std::cout << std::endl;
        }
#endif

    } // while
}

extern "C" {
void spmv(
    const SPMV_MAT_PKT_T *matrix_hbm_0,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_1,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_2,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_3,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_4,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_5,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_6,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_7,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_8,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_9,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_10,      // in
    const SPMV_MAT_PKT_T *matrix_hbm_11,      // in
    const SPMV_MAT_PKT_T *matrix_hbm_12,      // in
    const SPMV_MAT_PKT_T *matrix_hbm_13,      // in
    const SPMV_MAT_PKT_T *matrix_hbm_14,      // in
    const SPMV_MAT_PKT_T *matrix_hbm_15,      // in
    const PACKED_VAL_T *packed_dense_vector,  // in
    PACKED_VAL_T *packed_dense_result,        // out
    const unsigned num_rows,                  // in
    const unsigned num_columns                // in
) {
    #pragma HLS interface m_axi port=matrix_hbm_0 offset=slave bundle=spmv_mat0
    #pragma HLS interface m_axi port=matrix_hbm_1 offset=slave bundle=spmv_mat1
    #pragma HLS interface m_axi port=matrix_hbm_2 offset=slave bundle=spmv_mat2
    #pragma HLS interface m_axi port=matrix_hbm_3 offset=slave bundle=spmv_mat3
    #pragma HLS interface m_axi port=matrix_hbm_4 offset=slave bundle=spmv_mat4
    #pragma HLS interface m_axi port=matrix_hbm_5 offset=slave bundle=spmv_mat5
    #pragma HLS interface m_axi port=matrix_hbm_6 offset=slave bundle=spmv_mat6
    #pragma HLS interface m_axi port=matrix_hbm_7 offset=slave bundle=spmv_mat7
    #pragma HLS interface m_axi port=matrix_hbm_8 offset=slave bundle=spmv_mat8
    #pragma HLS interface m_axi port=matrix_hbm_9 offset=slave bundle=spmv_mat9
    #pragma HLS interface m_axi port=matrix_hbm_10 offset=slave bundle=spmv_mat10
    #pragma HLS interface m_axi port=matrix_hbm_11 offset=slave bundle=spmv_mat11
    #pragma HLS interface m_axi port=matrix_hbm_12 offset=slave bundle=spmv_mat12
    #pragma HLS interface m_axi port=matrix_hbm_13 offset=slave bundle=spmv_mat13
    #pragma HLS interface m_axi port=matrix_hbm_14 offset=slave bundle=spmv_mat14
    #pragma HLS interface m_axi port=matrix_hbm_15 offset=slave bundle=spmv_mat15

    #pragma HLS interface s_axilite port=matrix_hbm_0 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_1 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_2 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_3 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_4 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_5 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_6 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_7 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_8 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_9 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_10 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_11 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_12 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_13 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_14 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_15 bundle=control

    #pragma HLS interface s_axilite port=num_rows bundle=control
    #pragma HLS interface s_axilite port=num_columns bundle=control
    #pragma HLS interface s_axilite port=return bundle=control

    unsigned rows_per_ch_in_last_row_part;
    if (num_rows % LOGICAL_OB_SIZE == 0) {
        rows_per_ch_in_last_row_part = LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
    } else {
        rows_per_ch_in_last_row_part = num_rows % LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
    }

    unsigned num_row_partitions = (num_rows + LOGICAL_OB_SIZE - 1) / LOGICAL_OB_SIZE;
    unsigned num_col_partitions = (num_columns + LOGICAL_VB_SIZE - 1) / LOGICAL_VB_SIZE;
    unsigned num_partitions = num_row_partitions * num_col_partitions;

    for (unsigned row_partition_idx = 0; row_partition_idx < num_row_partitions; row_partition_idx++) {

        #pragma HLS dataflow
        hls::stream<VEC_AXIS_T> vec_dup[16];
        hls::stream<VEC_AXIS_T> res[16];
        #pragma HLS stream variable=vec_dup depth=FIFO_DEPTH
        #pragma HLS stream variable=res     depth=FIFO_DEPTH
        #pragma HLS bind_storage variable=vec_dup type=FIFO impl=SRL
        #pragma HLS bind_storage variable=res     type=FIFO impl=SRL

        // number of rows per cluster in this row partition
        unsigned rows_per_c_in_partition = LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
        if (row_partition_idx == num_row_partitions - 1) {
            rows_per_c_in_partition = rows_per_ch_in_last_row_part;
        }

#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : SpMV Kernel Started: row partition " << row_partition_idx
                  << " with " << rows_per_c_in_partition << " rows per cluster" << std::endl;
#endif

        vector_loader(packed_dense_vector, num_columns, vec_dup);

#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Vector Loader Complete" << std::endl;
#endif
        spmv_cluster<0>(
            matrix_hbm_0,
            vec_dup[0],
            res[0],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );

#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 0 Complete" << std::endl;
#endif

        spmv_cluster<1>(
            matrix_hbm_1,
            vec_dup[1],
            res[1],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );
#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 1 Complete" << std::endl;
#endif
        spmv_cluster<2>(
            matrix_hbm_2,
            vec_dup[2],
            res[2],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );
#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 2 Complete" << std::endl;
#endif
        spmv_cluster<3>(
            matrix_hbm_3,
            vec_dup[3],
            res[3],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );
#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 3 Complete" << std::endl;
#endif
        spmv_cluster<4>(
            matrix_hbm_4,
            vec_dup[4],
            res[4],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );
#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 4 Complete" << std::endl;
#endif
        spmv_cluster<5>(
            matrix_hbm_5,
            vec_dup[5],
            res[5],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );
#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 5 Complete" << std::endl;
#endif
        spmv_cluster<6>(
            matrix_hbm_6,
            vec_dup[6],
            res[6],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );
#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 6 Complete" << std::endl;
#endif
        spmv_cluster<7>(
            matrix_hbm_7,
            vec_dup[7],
            res[7],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );
#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 7 Complete" << std::endl;
#endif
        spmv_cluster<8>(
            matrix_hbm_8,
            vec_dup[8],
            res[8],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );
#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 8 Complete" << std::endl;
#endif
        spmv_cluster<9>(
            matrix_hbm_9,
            vec_dup[9],
            res[9],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );
#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 9 Complete" << std::endl;
#endif
        spmv_cluster<10>(
            matrix_hbm_10,
            vec_dup[10],
            res[10],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );
#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 10 Complete" << std::endl;
#endif
        spmv_cluster<11>(
            matrix_hbm_11,
            vec_dup[11],
            res[11],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );
#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 11 Complete" << std::endl;
#endif
        spmv_cluster<12>(
            matrix_hbm_12,
            vec_dup[12],
            res[12],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );
#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 12 Complete" << std::endl;
#endif
        spmv_cluster<13>(
            matrix_hbm_13,
            vec_dup[13],
            res[13],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );
#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 13 Complete" << std::endl;
#endif
        spmv_cluster<14>(
            matrix_hbm_14,
            vec_dup[14],
            res[14],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );
#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 14 Complete" << std::endl;
#endif
        spmv_cluster<15>(
            matrix_hbm_15,
            vec_dup[15],
            res[15],
            row_partition_idx,
            rows_per_c_in_partition,
            num_col_partitions,
            num_partitions
        );
#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Cluster 15 Complete" << std::endl;
#endif
        result_drain(packed_dense_result, row_partition_idx, res);

#ifdef SPMV_LINE_TRACING
        std::cout << "INFO : Result Drain Complete" << std::endl;
#endif

    } // iterate through all row partitions

} // kernel
} // extern "C"
