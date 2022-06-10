#include <hls_stream.h>
#include <ap_fixed.h>

#include "common.h"
#include "stream_utils.h"
#include "spmv_cluster.h"

extern "C" {
void spmv_sk2(
    const SPMV_MAT_PKT_T *matrix_hbm_10,      // in
    const SPMV_MAT_PKT_T *matrix_hbm_11,      // in
    const SPMV_MAT_PKT_T *matrix_hbm_12,      // in
    const SPMV_MAT_PKT_T *matrix_hbm_13,      // in
    const SPMV_MAT_PKT_T *matrix_hbm_14,      // in
    const SPMV_MAT_PKT_T *matrix_hbm_15,      // in
    const unsigned num_row_tiles,             // in
    const unsigned num_col_tiles,             // in
    hls::stream<VEC_AXIS_IF_T> &vec_in,       // in
    hls::stream<VEC_AXIS_IF_T> &res_out       // out
) {
    #pragma HLS interface m_axi port=matrix_hbm_10 offset=slave bundle=spmv_mat10
    #pragma HLS interface m_axi port=matrix_hbm_11 offset=slave bundle=spmv_mat11
    #pragma HLS interface m_axi port=matrix_hbm_12 offset=slave bundle=spmv_mat12
    #pragma HLS interface m_axi port=matrix_hbm_13 offset=slave bundle=spmv_mat13
    #pragma HLS interface m_axi port=matrix_hbm_14 offset=slave bundle=spmv_mat14
    #pragma HLS interface m_axi port=matrix_hbm_15 offset=slave bundle=spmv_mat15
    #pragma HLS interface s_axilite port=matrix_hbm_10 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_11 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_12 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_13 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_14 bundle=control
    #pragma HLS interface s_axilite port=matrix_hbm_15 bundle=control

    #pragma HLS interface s_axilite port=num_row_tiles bundle=control
    #pragma HLS interface s_axilite port=num_col_tiles bundle=control
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

    spmv_cluster<10>(
        matrix_hbm_10,
        vec_dup[0],
        res[0],
        num_row_tiles,
        num_col_tiles
    );

    spmv_cluster<11>(
        matrix_hbm_11,
        vec_dup[1],
        res[1],
        num_row_tiles,
        num_col_tiles
    );

    spmv_cluster<12>(
        matrix_hbm_12,
        vec_dup[2],
        res[2],
        num_row_tiles,
        num_col_tiles
    );

    spmv_cluster<13>(
        matrix_hbm_13,
        vec_dup[3],
        res[3],
        num_row_tiles,
        num_col_tiles
    );

    spmv_cluster<14>(
        matrix_hbm_14,
        vec_dup[4],
        res[4],
        num_row_tiles,
        num_col_tiles
    );

    spmv_cluster<15>(
        matrix_hbm_15,
        vec_dup[5],
        res[5],
        num_row_tiles,
        num_col_tiles
    );

    axis_merge<6>(res, res_out);

} // kernel
} // extern "C"
