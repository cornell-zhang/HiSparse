#include <hls_stream.h>
#include <ap_fixed.h>

#include "common.h"
#include "stream_utils.h"
#include "spmv_cluster.h"

#ifndef __SYNTHESIS__
// #define SPMV_SK0_LINE_TRACING
#endif

extern "C" {
unsigned long long spmv_sk0(
    const SPMV_MAT_PKT_T *matrix_hbm_0,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_1,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_2,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_3,       // in
    hls::stream<VEC_AXIS_T> &vec_in,          // in
    hls::stream<VEC_AXIS_T> &res_out,         // out
    const unsigned row_partition_idx,         // in
    const unsigned rows_per_c_in_partition,   // in
    const unsigned num_col_partitions,        // in
    const unsigned num_partitions,            // in
    const unsigned vb_bank_size,              // in
    const unsigned ob_bank_size               // in
) {
    #pragma HLS interface m_axi port=matrix_hbm_0 offset=slave bundle=spmv_mat0
    #pragma HLS interface m_axi port=matrix_hbm_1 offset=slave bundle=spmv_mat1
    #pragma HLS interface m_axi port=matrix_hbm_2 offset=slave bundle=spmv_mat2
    #pragma HLS interface m_axi port=matrix_hbm_3 offset=slave bundle=spmv_mat3
    #pragma HLS interface s_axilite port=matrix_hbm_0 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_1 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_2 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_3 bundle=control

    #pragma HLS interface s_axilite port=row_partition_idx bundle=control
    #pragma HLS interface s_axilite port=rows_per_c_in_partition bundle=control
    #pragma HLS interface s_axilite port=num_col_partitions bundle=control
    #pragma HLS interface s_axilite port=num_partitions bundle=control
    #pragma HLS interface s_axilite port=return bundle=control

    #pragma HLS interface axis register both port=vec_in
    #pragma HLS interface axis register both port=res_out

    #pragma HLS dataflow

    hls::stream<VEC_AXIS_T> vec_dup[4];
    hls::stream<VEC_AXIS_T> res[4];
    #pragma HLS stream variable=vec_dup depth=FIFO_DEPTH
    #pragma HLS stream variable=res     depth=FIFO_DEPTH
    #pragma HLS bind_storage variable=vec_dup type=FIFO impl=SRL
    #pragma HLS bind_storage variable=res     type=FIFO impl=SRL

    unsigned long long sf1_iter_cnt[4];
    axis_duplicate<4>(vec_in, vec_dup);

#ifdef SPMV_SK0_LINE_TRACING
    std::cout << "INFO : [Sub-kernel0] vector duplication complete!" << std::endl;
#endif

    sf1_iter_cnt[0] = spmv_cluster<0>(
        matrix_hbm_0,
        vec_dup[0],
        res[0],
        row_partition_idx,
        rows_per_c_in_partition,
        num_col_partitions,
        num_partitions,
        vb_bank_size,
        ob_bank_size
    );

#ifdef SPMV_SK0_LINE_TRACING
    std::cout << "INFO : [Sub-kernel0] cluster 0 complete!" << std::endl;
#endif

    sf1_iter_cnt[1] = spmv_cluster<1>(
        matrix_hbm_1,
        vec_dup[1],
        res[1],
        row_partition_idx,
        rows_per_c_in_partition,
        num_col_partitions,
        num_partitions,
        vb_bank_size,
        ob_bank_size
    );

#ifdef SPMV_SK0_LINE_TRACING
    std::cout << "INFO : [Sub-kernel0] cluster 1 complete!" << std::endl;
#endif

    sf1_iter_cnt[2] = spmv_cluster<2>(
        matrix_hbm_2,
        vec_dup[2],
        res[2],
        row_partition_idx,
        rows_per_c_in_partition,
        num_col_partitions,
        num_partitions,
        vb_bank_size,
        ob_bank_size
    );

#ifdef SPMV_SK0_LINE_TRACING
    std::cout << "INFO : [Sub-kernel0] cluster 2 complete!" << std::endl;
#endif

    sf1_iter_cnt[3] = spmv_cluster<3>(
        matrix_hbm_3,
        vec_dup[3],
        res[3],
        row_partition_idx,
        rows_per_c_in_partition,
        num_col_partitions,
        num_partitions,
        vb_bank_size,
        ob_bank_size
    );

#ifdef SPMV_SK0_LINE_TRACING
    std::cout << "INFO : [Sub-kernel0] cluster 3 complete!" << std::endl;
#endif

    axis_merge<4>(res, res_out);

#ifdef SPMV_SK0_LINE_TRACING
    std::cout << "INFO : [Sub-kernel0] result merging complete!" << std::endl;
#endif
    return ull_max<4>(sf1_iter_cnt);

} // kernel
} // extern "C"
