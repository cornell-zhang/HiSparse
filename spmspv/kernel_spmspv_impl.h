#include "libfpga/hisparse.h"
#include "libfpga/shuffle.h"
#include "libfpga/pe.h"

#include <ap_fixed.h>
#include <hls_stream.h>

#ifndef __SYNTHESIS__
#include <iostream>
#include <iomanip>
static bool line_tracing_spmspv = false;
static bool line_tracing_spmspv_load_data = false;
static bool line_tracing_spmspv_write_back = false;
#endif


// vector loader for spmspv
static void load_vector_from_gmem(
    // vector data, row_id
    const IDX_VAL_T *vector,
    // number of non-zeros
    IDX_T vec_num_nnz,
    // fifo
    hls::stream<IDX_VAL_INST_T> VL_to_ML_stream[SPMSPV_NUM_HBM_CHANNEL]
) {
    for (unsigned int k = 0; k < SPMSPV_NUM_HBM_CHANNEL; k++) {
        #pragma HLS unroll
        VL_to_ML_stream[k].write(IDX_VAL_INST_SOD);
    }

    loop_over_vector_values:
    for (unsigned int vec_nnz_cnt = 0; vec_nnz_cnt < vec_num_nnz; vec_nnz_cnt++) {
        #pragma HLS pipeline II=1
        IDX_VAL_INST_T instruction_to_ml;
        IDX_T index = vector[vec_nnz_cnt + 1].index;
        instruction_to_ml.index = index / SPMSPV_NUM_HBM_CHANNEL;
        instruction_to_ml.val = vector[vec_nnz_cnt + 1].val;
        instruction_to_ml.inst = 0;
        VL_to_ML_stream[index % SPMSPV_NUM_HBM_CHANNEL].write(instruction_to_ml);
    }

    for (unsigned int k = 0; k < SPMSPV_NUM_HBM_CHANNEL; k++) {
        #pragma HLS unroll
        VL_to_ML_stream[k].write(IDX_VAL_INST_EOD);
        VL_to_ML_stream[k].write(IDX_VAL_INST_EOS);
    }

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv_load_data) {
        std::cout << "INFO: [kernel SpMSpV] Load vector finished, vec_num_nnz = "
                  << vec_num_nnz << std::endl << std::flush;
    }
    #endif
}


// data loader for SpMSpV from one HBM channel
static void load_matrix_from_gmem(
    // matrix data, row_id
    const SPMSPV_MAT_PKT_T *matrix,
    // matrix indptr
    const IDX_T *mat_indptr,
    // matrix part ptr
    const IDX_T *mat_partptr,
    // partition base
    IDX_T mat_indptr_base,
    IDX_T mat_row_id_base,
    // current part id
    IDX_T part_id,
    // fifos
    hls::stream<IDX_VAL_INST_T> &VL_to_ML_stream,
    hls::stream<INST_T> &DL_to_MG_inst,
    hls::stream<UPDATE_PLD_T> DL_to_MG_stream[PACK_SIZE]
) {
    IDX_T mat_addr_base = mat_partptr[part_id];
    bool exit = false;

    DL_to_MG_inst.write(SOD); // no need to fill `DL_to_MG_stream` with SOD anymore

    // loop over all active columns
    loop_over_active_columns_ML:
    while (!exit) {

        // slice out the current column out of the active columns
        IDX_VAL_INST_T pld = VL_to_ML_stream.read();
        if (pld.inst == EOS) {
            exit = true;
        } else if (pld.inst != SOD && pld.inst != EOD) {

            IDX_T current_column_id = pld.index;
            VAL_T vec_val = pld.val;
            // [0] for start, [1] for end
            // write like this to make sure it uses burst read
            IDX_T col_slice[2];
            #pragma HLS array_partition variable=col_slice complete

            loop_get_column_len_ML:
            for (unsigned int i = 0; i < 2; i++) {
                #pragma HLS pipeline II=1
                col_slice[i] = mat_indptr[current_column_id + mat_indptr_base + i];
            }

            loop_over_pkts_ML:
            for (unsigned int i = 0; i < (col_slice[1] - col_slice[0]); i++) {
                #pragma HLS pipeline II=1
                SPMSPV_MAT_PKT_T packet_from_mat = matrix[i + mat_addr_base + col_slice[0]];
                DL_to_MG_inst.write(0);

                loop_unpack_ML_unroll:
                for (unsigned int k = 0; k < PACK_SIZE; k++) {
                    #pragma HLS unroll
                    UPDATE_PLD_T input_to_MG;
                    input_to_MG.mat_val = packet_from_mat.vals.data[k];
                    input_to_MG.vec_val = vec_val;
                    input_to_MG.row_idx = packet_from_mat.indices.data[k] - mat_row_id_base;
                    input_to_MG.inst = 0;
                    // discard paddings is done in data merger
                    DL_to_MG_stream[k].write(input_to_MG);
                }
            }

        }

    }

    // no need to fill `DL_to_MG_stream` with end inst anymore
    DL_to_MG_inst.write(EOD);
    DL_to_MG_inst.write(EOS);

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv_load_data) {
        std::cout << "INFO: [kernel SpMSpV] Load matrix finished, part_id = "
                  << part_id << std::endl << std::flush;
    }
    #endif
}

// merge streams of matrix loader in different HBM, and forward the available
// output to shuffle stream
static void merge_load_streams(
    hls::stream<INST_T> ML_to_MG_insts[SPMSPV_NUM_HBM_CHANNEL],
    hls::stream<UPDATE_PLD_T> ML_to_MG_streams[SPMSPV_NUM_HBM_CHANNEL][PACK_SIZE],
    hls::stream<UPDATE_PLD_T> MG_to_SF_streams[PACK_SIZE]
    // VAL_T Zero
) {
    for (unsigned int k = 0; k < PACK_SIZE; k++) {
        #pragma HLS unroll
        MG_to_SF_streams[k].write(UPDATE_PLD_SOD);
    }

    bool exit = false;
    char current_input = 0; // read from multiple matrix loader streams
    ap_uint<SPMSPV_NUM_HBM_CHANNEL> finished = 0;

    spmspv_merge_load_streams_loop:
    while (!exit) {
        #pragma HLS pipeline II=1

        INST_T ctrl;
        if (!finished[current_input] && ML_to_MG_insts[current_input].read_nb(ctrl)) {
            if (ctrl == EOS) {
                finished[current_input] = true;
            } else if (ctrl != SOD && ctrl != EOD) {

                forward_mat_pkt_MG_unroll:
                for (unsigned int k = 0; k < PACK_SIZE; k++) {
                    #pragma HLS unroll
                    UPDATE_PLD_T pld_to_SF = ML_to_MG_streams[current_input][k].read();
                    if (pld_to_SF.mat_val != /*Zero*/0) {
                        // only forward non-zero payload, and SOD/EOS/EOD to shuffle
                        // needs to take care manually
                        MG_to_SF_streams[k].write(pld_to_SF);
                    }
                }

            }
        }

        exit = finished.and_reduce();

        if ( (++current_input) == SPMSPV_NUM_HBM_CHANNEL) {
            current_input = 0;
        }
    }

    for (unsigned int k = 0; k < PACK_SIZE; k++) {
        #pragma HLS unroll
        MG_to_SF_streams[k].write(UPDATE_PLD_EOD);
        MG_to_SF_streams[k].write(UPDATE_PLD_EOS);
    }

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv_load_data) {
        std::cout << "INFO: [kernel SpMSpV] Merge streams finished" << std::endl
                  << std::flush;
    }
    #endif
}

// write back to gemm
static void write_back_results (
    hls::stream<VEC_PLD_T> PE_to_WB_stream[PACK_SIZE],
    IDX_VAL_T *res_out,
    // const VAL_T *mask,
    IDX_T mat_row_id_base,
    IDX_T &Nnz
    // VAL_T Zero,
    // MASK_T mask_type
) {
    IDX_T res_idx = Nnz;
    bool exit = false;
    char current_input = 0; // read from multiple PE output streams
    ap_uint<PACK_SIZE> finished = 0;

    spmspv_write_back_loop:
    while (!exit) {
        #pragma HLS pipeline II=1
        #pragma HLS dependence variable=res_out inter false

        VEC_PLD_T pld;
        if (!finished[current_input] && PE_to_WB_stream[current_input].read_nb(pld)) {
            if (pld.inst == EOS) {
                finished[current_input] = true;
            } else if (pld.inst != SOD && pld.inst != EOD) {
                IDX_T index = mat_row_id_base + pld.idx;
                // bool do_write = false;
                // switch (mask_type) {
                //     case NOMASK:
                //         do_write = true;
                //         break;
                //     case WRITETOONE:
                //         do_write = (mask[index] != 0);
                //         break;
                //     case WRITETOZERO:
                //         do_write = (mask[index] == 0);
                //         break;
                //     default:
                //         do_write = false;
                //         break;
                // }
                // if (do_write) {
                IDX_VAL_T res_pld;
                res_pld.index = index;
                res_pld.val = pld.val;
                res_idx++;
                res_out[res_idx] = res_pld;
                #ifndef __SYNTHESIS__
                if (line_tracing_spmspv_write_back) {
                    std::cout << "INFO: [kernel SpMSpV] Write results"
                                << " non-zero " << pld.val
                                << " found at " << pld.idx
                                << " mapped to " << index << std::endl << std::flush;
                }
                #endif
                // }
            }
        }

        exit = finished.and_reduce();

        if ( (++current_input) == PACK_SIZE) {
            current_input = 0;
        }
    }
    Nnz = res_idx;
}

// abbreviation for matrix arguments, `x` is the index of HBM channel
#define SPMSPV_MAT_ARGS(x) \
const SPMSPV_MAT_PKT_T *mat_##x, \
const IDX_T *mat_indptr_##x, \
const IDX_T *mat_partptr_##x

// vec loader -> mat loader -> stream merger -> shuffle -> PE -> write back
static void spmspv_core(
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)
    SPMSPV_MAT_ARGS(0),
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 2)
    SPMSPV_MAT_ARGS(1),
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 4)
    SPMSPV_MAT_ARGS(2),
    SPMSPV_MAT_ARGS(3),
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 8)
    SPMSPV_MAT_ARGS(4),
    SPMSPV_MAT_ARGS(5),
    SPMSPV_MAT_ARGS(6),
    SPMSPV_MAT_ARGS(7),
#endif
    const IDX_VAL_T *vector,
    // const VAL_T *mask,
    IDX_VAL_T *result_out,
    IDX_T vec_num_nnz,
    IDX_T mat_indptr_base_each_channel[SPMSPV_NUM_HBM_CHANNEL],
    IDX_T mat_row_id_base,
    IDX_T part_id,
    IDX_T &Nnz,
    // OP_T Op,
    // VAL_T Zero,
    // MASK_T mask_type,
    const unsigned used_buf_len_per_pe
) {
    // fifos

    hls::stream<IDX_VAL_INST_T> VL_to_ML_stream[SPMSPV_NUM_HBM_CHANNEL];
    hls::stream<INST_T> ML_to_MG_inst[SPMSPV_NUM_HBM_CHANNEL];
    hls::stream<UPDATE_PLD_T> ML_to_MG_stream[SPMSPV_NUM_HBM_CHANNEL][PACK_SIZE];
    hls::stream<UPDATE_PLD_T> MG_to_SF_stream[PACK_SIZE];
    hls::stream<UPDATE_PLD_T> SF_to_PE_stream[PACK_SIZE];
    hls::stream<VEC_PLD_T> PE_to_WB_stream[PACK_SIZE];
    #pragma HLS stream variable=VL_to_ML_stream depth=FIFO_DEPTH
    #pragma HLS stream variable=ML_to_MG_inst   depth=FIFO_DEPTH
    #pragma HLS stream variable=ML_to_MG_stream depth=FIFO_DEPTH
    #pragma HLS stream variable=MG_to_SF_stream depth=FIFO_DEPTH
    #pragma HLS stream variable=SF_to_PE_stream depth=FIFO_DEPTH
    #pragma HLS stream variable=PE_to_WB_stream depth=FIFO_DEPTH
    #pragma HLS bind_storage variable=VL_to_ML_stream type=FIFO impl=SRL
    #pragma HLS bind_storage variable=ML_to_MG_inst   type=FIFO impl=SRL
    #pragma HLS bind_storage variable=ML_to_MG_stream type=FIFO impl=SRL
    #pragma HLS bind_storage variable=MG_to_SF_stream type=FIFO impl=SRL
    #pragma HLS bind_storage variable=SF_to_PE_stream type=FIFO impl=SRL
    #pragma HLS bind_storage variable=PE_to_WB_stream type=FIFO impl=SRL

    // dataflow pipeline
    #pragma HLS dataflow
    load_vector_from_gmem(
        vector,
        vec_num_nnz,
        VL_to_ML_stream
    );

#define LOAD_MAT_FROM_HBM(x) \
load_matrix_from_gmem( \
    mat_##x, \
    mat_indptr_##x, \
    mat_partptr_##x, \
    mat_indptr_base_each_channel[x], \
    mat_row_id_base, \
    part_id, \
    VL_to_ML_stream[x], \
    ML_to_MG_inst[x], \
    ML_to_MG_stream[x] \
)

#if (SPMSPV_NUM_HBM_CHANNEL >= 1)
    LOAD_MAT_FROM_HBM(0);
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 2)
    LOAD_MAT_FROM_HBM(1);
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 4)
    LOAD_MAT_FROM_HBM(2);
    LOAD_MAT_FROM_HBM(3);
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 8)
    LOAD_MAT_FROM_HBM(4);
    LOAD_MAT_FROM_HBM(5);
    LOAD_MAT_FROM_HBM(6);
    LOAD_MAT_FROM_HBM(7);
#endif
    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO: [Kernel SpMSpV] Data Loader complete" << std::endl << std::flush;
    }
    #endif

    merge_load_streams(ML_to_MG_inst, ML_to_MG_stream, MG_to_SF_stream/*, Zero*/);

    shuffler<UPDATE_PLD_T, PACK_SIZE>(MG_to_SF_stream, SF_to_PE_stream);

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO: [Kernel SpMSpV] Shuffler complete" << std::endl << std::flush;
    }
    #endif

#define PE_SPARSE(x) \
pe_bram_sparse<(x), SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>( \
    SF_to_PE_stream[x], \
    PE_to_WB_stream[x], \
    used_buf_len_per_pe \
)
    PE_SPARSE(0);
    PE_SPARSE(1);
    PE_SPARSE(2);
    PE_SPARSE(3);
    PE_SPARSE(4);
    PE_SPARSE(5);
    PE_SPARSE(6);
    PE_SPARSE(7);

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO: [Kernel SpMSpV] Process Elements complete" << std::endl << std::flush;
    }
    #endif

    write_back_results(
        PE_to_WB_stream,
        result_out,
        // mask,
        mat_row_id_base,
        Nnz
        // Zero,
        // mask_type
    );

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO: [Kernel SpMSpV] Result writeback complete" << std::endl << std::flush;
    }
    #endif
}
