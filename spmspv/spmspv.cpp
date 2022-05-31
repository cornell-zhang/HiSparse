#include "./libfpga/hisparse.h"

#include "./kernel_spmspv_impl.h"

extern "C" {

void spmspv(
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)
    const SPMSPV_MAT_PKT_T *mat_0,  // in,  HBM[0]
    const IDX_T *mat_indptr_0,      // in,  HBM[0]
    const IDX_T *mat_partptr_0,     // in,  HBM[0]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 2)
    const SPMSPV_MAT_PKT_T *mat_1,  // in,  HBM[1]
    const IDX_T *mat_indptr_1,      // in,  HBM[1]
    const IDX_T *mat_partptr_1,     // in,  HBM[1]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 4)
    const SPMSPV_MAT_PKT_T *mat_2,  // in,  HBM[2]
    const IDX_T *mat_indptr_2,      // in,  HBM[2]
    const IDX_T *mat_partptr_2,     // in,  HBM[2]
    const SPMSPV_MAT_PKT_T *mat_3,  // in,  HBM[3]
    const IDX_T *mat_indptr_3,      // in,  HBM[3]
    const IDX_T *mat_partptr_3,     // in,  HBM[3]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 8)
    const SPMSPV_MAT_PKT_T *mat_4,  // in,  HBM[4]
    const IDX_T *mat_indptr_4,      // in,  HBM[4]
    const IDX_T *mat_partptr_4,     // in,  HBM[4]
    const SPMSPV_MAT_PKT_T *mat_5,  // in,  HBM[5]
    const IDX_T *mat_indptr_5,      // in,  HBM[5]
    const IDX_T *mat_partptr_5,     // in,  HBM[5]
    const SPMSPV_MAT_PKT_T *mat_6,  // in,  HBM[6]
    const IDX_T *mat_indptr_6,      // in,  HBM[6]
    const IDX_T *mat_partptr_6,     // in,  HBM[6]
    const SPMSPV_MAT_PKT_T *mat_7,  // in,  HBM[7]
    const IDX_T *mat_indptr_7,      // in,  HBM[7]
    const IDX_T *mat_partptr_7,     // in,  HBM[7]
#endif
    IDX_VAL_T *vector,              // inout, HBM[20]
    IDX_VAL_T *result,              // out,   HBM[22]
    IDX_T num_rows,                 // in
    IDX_T num_cols                  // in
) {

/*----------------- arguments for SpMSpV -------------------*/
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)

    #pragma HLS interface m_axi port=mat_0         offset=slave bundle=spmspv_gmem0_0
    #pragma HLS interface m_axi port=mat_indptr_0  offset=slave bundle=spmspv_gmem1_0
    #pragma HLS interface m_axi port=mat_partptr_0 offset=slave bundle=spmspv_gmem2_0

    #pragma HLS interface s_axilite port=mat_0         bundle=control
    #pragma HLS interface s_axilite port=mat_indptr_0  bundle=control
    #pragma HLS interface s_axilite port=mat_partptr_0 bundle=control

#endif

#if (SPMSPV_NUM_HBM_CHANNEL >= 2)

    #pragma HLS interface m_axi port=mat_1         offset=slave bundle=spmspv_gmem0_1
    #pragma HLS interface m_axi port=mat_indptr_1  offset=slave bundle=spmspv_gmem1_1
    #pragma HLS interface m_axi port=mat_partptr_1 offset=slave bundle=spmspv_gmem2_1

    #pragma HLS interface s_axilite port=mat_1         bundle=control
    #pragma HLS interface s_axilite port=mat_indptr_1  bundle=control
    #pragma HLS interface s_axilite port=mat_partptr_1 bundle=control

#endif

#if (SPMSPV_NUM_HBM_CHANNEL >= 4)

      #pragma HLS interface m_axi port=mat_2         offset=slave bundle=spmspv_gmem0_2
      #pragma HLS interface m_axi port=mat_indptr_2  offset=slave bundle=spmspv_gmem1_2
      #pragma HLS interface m_axi port=mat_partptr_2 offset=slave bundle=spmspv_gmem2_2

      #pragma HLS interface s_axilite port=mat_2         bundle=control
      #pragma HLS interface s_axilite port=mat_indptr_2  bundle=control
      #pragma HLS interface s_axilite port=mat_partptr_2 bundle=control

      #pragma HLS interface m_axi port=mat_3         offset=slave bundle=spmspv_gmem0_3
      #pragma HLS interface m_axi port=mat_indptr_3  offset=slave bundle=spmspv_gmem1_3
      #pragma HLS interface m_axi port=mat_partptr_3 offset=slave bundle=spmspv_gmem2_3

      #pragma HLS interface s_axilite port=mat_3         bundle=control
      #pragma HLS interface s_axilite port=mat_indptr_3  bundle=control
      #pragma HLS interface s_axilite port=mat_partptr_3 bundle=control

#endif

#if (SPMSPV_NUM_HBM_CHANNEL >= 8)

      #pragma HLS interface m_axi port=mat_4         offset=slave bundle=spmspv_gmem0_4
      #pragma HLS interface m_axi port=mat_indptr_4  offset=slave bundle=spmspv_gmem1_4
      #pragma HLS interface m_axi port=mat_partptr_4 offset=slave bundle=spmspv_gmem2_4

      #pragma HLS interface s_axilite port=mat_4         bundle=control
      #pragma HLS interface s_axilite port=mat_indptr_4  bundle=control
      #pragma HLS interface s_axilite port=mat_partptr_4 bundle=control

      #pragma HLS interface m_axi port=mat_5         offset=slave bundle=spmspv_gmem0_5
      #pragma HLS interface m_axi port=mat_indptr_5  offset=slave bundle=spmspv_gmem1_5
      #pragma HLS interface m_axi port=mat_partptr_5 offset=slave bundle=spmspv_gmem2_5

      #pragma HLS interface s_axilite port=mat_5         bundle=control
      #pragma HLS interface s_axilite port=mat_indptr_5  bundle=control
      #pragma HLS interface s_axilite port=mat_partptr_5 bundle=control

      #pragma HLS interface m_axi port=mat_6         offset=slave bundle=spmspv_gmem0_6
      #pragma HLS interface m_axi port=mat_indptr_6  offset=slave bundle=spmspv_gmem1_6
      #pragma HLS interface m_axi port=mat_partptr_6 offset=slave bundle=spmspv_gmem2_6

      #pragma HLS interface s_axilite port=mat_6         bundle=control
      #pragma HLS interface s_axilite port=mat_indptr_6  bundle=control
      #pragma HLS interface s_axilite port=mat_partptr_6 bundle=control

      #pragma HLS interface m_axi port=mat_7         offset=slave bundle=spmspv_gmem0_7
      #pragma HLS interface m_axi port=mat_indptr_7  offset=slave bundle=spmspv_gmem1_7
      #pragma HLS interface m_axi port=mat_partptr_7 offset=slave bundle=spmspv_gmem2_7

      #pragma HLS interface s_axilite port=mat_7         bundle=control
      #pragma HLS interface s_axilite port=mat_indptr_7  bundle=control
      #pragma HLS interface s_axilite port=mat_partptr_7 bundle=control

#endif

    #pragma HLS interface m_axi port=vector         offset=slave bundle=spmspv_gmem_vec
    #pragma HLS interface m_axi port=result         offset=slave bundle=spmspv_gmem_out

    #pragma HLS interface s_axilite port=vector     bundle=control
    #pragma HLS interface s_axilite port=result     bundle=control

    #pragma HLS interface s_axilite port=num_rows   bundle=control
    #pragma HLS interface s_axilite port=num_cols   bundle=control

    #pragma HLS interface s_axilite port=return bundle=control

    #pragma HLS inline off

    IDX_T vec_num_nnz = vector[0].index;

    // result Nnz counter
    IDX_T result_Nnz = 0;

    // total number of parts
    IDX_T num_parts = (num_rows + SPMSPV_OUT_BUF_LEN - 1) / SPMSPV_OUT_BUF_LEN;

    // number of rows in the last part
    IDX_T num_rows_last_part = (num_rows % SPMSPV_OUT_BUF_LEN) ? (num_rows % SPMSPV_OUT_BUF_LEN) : SPMSPV_OUT_BUF_LEN;

    IDX_T num_cols_each_channel[SPMSPV_NUM_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=num_cols_each_channel complete
    for (int k = 0; k < SPMSPV_NUM_HBM_CHANNEL; k++) {
        #pragma HLS unroll
        num_cols_each_channel[k] = (num_cols - 1 - k) / SPMSPV_NUM_HBM_CHANNEL + 1;
    }

    IDX_T mat_indptr_base_each_channel[SPMSPV_NUM_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=mat_indptr_base_each_channel complete

    // loop over parts
    loop_over_parts:
    for (unsigned int part_id = 0; part_id < num_parts; part_id++) {
        #pragma HLS pipeline off
        IDX_T num_rows_this_part = (part_id == (num_parts - 1)) ? num_rows_last_part : SPMSPV_OUT_BUF_LEN;
        IDX_T mat_row_id_base = SPMSPV_OUT_BUF_LEN * part_id;
        for (int k = 0; k < SPMSPV_NUM_HBM_CHANNEL; k++) {
            #pragma HLS unroll
            mat_indptr_base_each_channel[k] = (num_cols_each_channel[k] + 1) * part_id;
        }
        #ifndef __SYNTHESIS__
        if (line_tracing_spmspv) {
            std::cout << "INFO: [Kernel SpMSpV] Partition " << part_id <<" start" << std::endl << std::flush;
            std::cout << "  # of rows this part: " << num_rows_this_part << std::endl << std::flush;
            std::cout << "          row id base: " << mat_row_id_base << std::endl << std::flush;
        }
        #endif
        spmspv_core(
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)
            mat_0,
            mat_indptr_0,
            mat_partptr_0,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 2)
            mat_1,
            mat_indptr_1,
            mat_partptr_1,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 4)
            mat_2,
            mat_indptr_2,
            mat_partptr_2,
            mat_3,
            mat_indptr_3,
            mat_partptr_3,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 8)
            mat_4,
            mat_indptr_4,
            mat_partptr_4,
            mat_5,
            mat_indptr_5,
            mat_partptr_5,
            mat_6,
            mat_indptr_6,
            mat_partptr_6,
            mat_7,
            mat_indptr_7,
            mat_partptr_7,
#endif
            vector,
            result,
            vec_num_nnz,
            mat_indptr_base_each_channel,
            mat_row_id_base,
            part_id,
            result_Nnz,
            (num_rows_this_part + PACK_SIZE - 1) / PACK_SIZE
        );
        #ifndef __SYNTHESIS__
        if (line_tracing_spmspv) {
            std::cout << "INFO: [Kernel SpMSpV] Partition " << part_id
                      << " complete" << std::endl << std::flush;
            std::cout << "     Nnz written back: " << result_Nnz << std::endl << std::flush;
        }
        #endif
    }

    // attach head
    IDX_VAL_T result_head;
    result_head.index = result_Nnz;
    result_head.val = /*Zero*/0;
    result[0] = result_head;
    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO: [Kernel SpMSpV] Kernel Finish" << std::endl << std::flush;
        std::cout << "  Result Nnz = " << result_Nnz << std::endl << std::flush;
    }
    #endif

}

}  // extern "C"
