#include <hls_stream.h>
#include <ap_fixed.h>

#include "common.h"
#include "stream_utils.h"
#include "spmv_cluster.h"


extern "C" {
void spmv_sk1(
    const SPMV_MAT_PKT_T *matrix_hbm_4,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_5,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_6,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_7,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_8,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_9,       // in
    hls::stream<VEC_AXIS_T> &vec_in,          // in
    hls::stream<VEC_AXIS_T> &res_out,         // out
    const unsigned row_partition_idx,         // in
    const unsigned rows_per_c_in_partition,   // in
    const unsigned num_col_partitions,        // in
    const unsigned num_partitions             // in

) {
    #pragma HLS interface m_axi port=matrix_hbm_4 offset=slave bundle=spmv_mat4
    #pragma HLS interface m_axi port=matrix_hbm_5 offset=slave bundle=spmv_mat5
    #pragma HLS interface m_axi port=matrix_hbm_6 offset=slave bundle=spmv_mat6
    #pragma HLS interface m_axi port=matrix_hbm_7 offset=slave bundle=spmv_mat7
    #pragma HLS interface m_axi port=matrix_hbm_8 offset=slave bundle=spmv_mat8
    #pragma HLS interface m_axi port=matrix_hbm_9 offset=slave bundle=spmv_mat9
    #pragma HLS interface s_axilite port=matrix_hbm_4 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_5 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_6 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_7 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_8 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_9 bundle=control

    #pragma HLS interface s_axilite port=row_partition_idx bundle=control
    #pragma HLS interface s_axilite port=rows_per_c_in_partition bundle=control
    #pragma HLS interface s_axilite port=num_col_partitions bundle=control
    #pragma HLS interface s_axilite port=num_partitions bundle=control
    #pragma HLS interface s_axilite port=return bundle=control

    #pragma HLS interface axis register both port=vec_in
    #pragma HLS interface axis register both port=res_out

    #pragma HLS dataflow

    hls::stream<VEC_AXIS_T> vec_dup[6];
    hls::stream<VEC_AXIS_T> res[6];
    #pragma HLS stream variable=vec_dup depth=FIFO_DEPTH
    #pragma HLS stream variable=res     depth=FIFO_DEPTH
    #pragma HLS bind_storage variable=vec_dup type=FIFO impl=SRL
    #pragma HLS bind_storage variable=res     type=FIFO impl=SRL

    axis_duplicate<6>(vec_in, vec_dup);

    spmv_cluster<4>(
        matrix_hbm_4,
        vec_dup[0],
        res[0],
        row_partition_idx,
        rows_per_c_in_partition,
        num_col_partitions,
        num_partitions
    );

    spmv_cluster<5>(
        matrix_hbm_5,
        vec_dup[1],
        res[1],
        row_partition_idx,
        rows_per_c_in_partition,
        num_col_partitions,
        num_partitions
    );

    spmv_cluster<6>(
        matrix_hbm_6,
        vec_dup[2],
        res[2],
        row_partition_idx,
        rows_per_c_in_partition,
        num_col_partitions,
        num_partitions
    );

    spmv_cluster<7>(
        matrix_hbm_7,
        vec_dup[3],
        res[3],
        row_partition_idx,
        rows_per_c_in_partition,
        num_col_partitions,
        num_partitions
    );

    spmv_cluster<8>(
        matrix_hbm_8,
        vec_dup[4],
        res[4],
        row_partition_idx,
        rows_per_c_in_partition,
        num_col_partitions,
        num_partitions
    );

    spmv_cluster<9>(
        matrix_hbm_9,
        vec_dup[5],
        res[5],
        row_partition_idx,
        rows_per_c_in_partition,
        num_col_partitions,
        num_partitions
    );

    axis_merge<6>(res, res_out);

} // kernel
} // extern "C"
