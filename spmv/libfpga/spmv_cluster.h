#ifndef SPMV_CLUSTER_H_
#define SPMV_CLUSTER_H_

#include <hls_stream.h>
#include <ap_fixed.h>

#include "common.h"
#include "shuffle.h"
#include "vecbuf_access_unit.h"
#include "pe.h"

#include <hls_stream.h>
#include <ap_fixed.h>

#include "common.h"

#ifndef __SYNTHESIS__
// #define SPMV_CLUSTER_H_LINE_TRACING
#include <iostream>
#endif

template<typename T, unsigned len>
T array_max(T array[len]) {
    #pragma HLS inline
    // #pragma HLS expression_balance
    T result = 0;
    for (unsigned i = 0; i < len; i++) {
        #pragma HLS unroll
        result = (array[i] > result)? array[i] : result;
    }
    return result;
}

static void coo_matrix_loader(
    const SPMV_MAT_PKT_T *matrix_hbm,                      // in
    hls::stream<EDGE_PLD_T> ML_to_SF_1_stream[PACK_SIZE]   // out
) {
    // prepare to read header, data[0] is the last index of streams, data[1] is
    // the flush size of the last partition in each PE
    PACKED_IDX_T header = matrix_hbm[0].indices;
    IDX_T aligned_stream_last_index = header.data[0];
    IDX_T last_flush_size_each_pe = header.data[1];

    // attach start-of-data
    for (unsigned k = 0; k < PACK_SIZE; k++) {
        #pragma HLS UNROLL
        ML_to_SF_1_stream[k].write(EDGE_PLD_SOD);
    }

    // TODO: maunally control the burst length will help?
    loop_matrix_loader:
    for (unsigned i = 1; i <= aligned_stream_last_index; i++) {
        #pragma HLS PIPELINE II=1
        SPMV_MAT_PKT_T mat_pkt = matrix_hbm[i];
        for (unsigned k = 0; k < PACK_SIZE; k++) {
            #pragma HLS UNROLL
            // Format of stream data, `...` means normal non-zero matrix data:
            // [last index of the stream] ... /col switch/ ... /row switch/ ...
            // Note: there is no row switch marker in the last position
            VAL_T value_in_pkt = mat_pkt.vals.data[k];
            IDX_T index_in_pkt = mat_pkt.indices.data[k];
            if (value_in_pkt == VAL_MARKER) {
                switch (index_in_pkt) {
                    case IDX_COL_TILE_MARKER:
                        ML_to_SF_1_stream[k].write(EDGE_PLD_EOD);
                        ML_to_SF_1_stream[k].write(EDGE_PLD_SOD);
                    break;
                    case IDX_ROW_TILE_MARKER:
                        // if adjusting the tile height to some value not equals
                        // to OB_PER_CLUSTER (which means PE output buffer cannot
                        // be fully utilized), change the arg here to some value
                        // passed the same way as `last_flush_size_each_pe`
                        ML_to_SF_1_stream[k].write(EDGE_PLD_EOD_FLUSH(OB_BANK_SIZE));
                        ML_to_SF_1_stream[k].write(EDGE_PLD_SOD);
                    break;
                    default:
                    break;
                }
            } else {
                EDGE_PLD_T input_to_SF_1;
                input_to_SF_1.mat_val = value_in_pkt;
                // the range of `row_idx` and `col_idx` (i.e., the tile dimension)
                // is [ 0, 65535 ]
                input_to_SF_1.row_idx = index_in_pkt >> 16;
                input_to_SF_1.col_idx = index_in_pkt & 0x0000ffff;
                ML_to_SF_1_stream[k].write(input_to_SF_1);
            }
        }
    }

    // attach end-of-stream
    for (unsigned k = 0; k < PACK_SIZE; k++) {
        #pragma HLS UNROLL
        ML_to_SF_1_stream[k].write(EDGE_PLD_EOD_FLUSH(last_flush_size_each_pe));
        ML_to_SF_1_stream[k].write(EDGE_PLD_EOS);
    }
}

static void spmv_vector_unpacker (
    hls::stream<VEC_AXIS_T> &vec_in,
    hls::stream<VEC_PLD_T> vec_out[PACK_SIZE]
) {
    bool exit = false;
    while (!exit) {
        #pragma HLS pipeline II=1
        VEC_AXIS_T pkt = vec_in.read();
        for (unsigned k = 0; k < PACK_SIZE; k++) {
            #pragma HLS unroll
            VEC_PLD_T p;
            VAL_T_BITCAST(p.val) = VEC_AXIS_VAL(pkt, k);
            p.idx = VEC_AXIS_PKT_IDX(pkt) * PACK_SIZE + k;
            p.inst = pkt.user;
            vec_out[k].write(p);
        }
        exit = (pkt.user == EOS);
    }
}

#ifndef __SYNTHESIS__
// #define SPMV_RESULT_PACKER_LINE_TRACING
#endif

static void spmv_result_packer (
    hls::stream<VEC_PLD_T> res_in[PACK_SIZE],
    hls::stream<VEC_AXIS_T> &res_out
) {
    bool exit = false;
    unsigned pkt_idx = 0;
    while (!exit) {
        #pragma HLS pipeline II=1
        ap_uint<PACK_SIZE> got_SOD = 0;
        ap_uint<PACK_SIZE> got_EOD = 0;
        ap_uint<PACK_SIZE> got_EOS = 0;
        VEC_AXIS_T pkt;
        for (unsigned k = 0; k < PACK_SIZE; k++) {
            #pragma HLS unroll
            VEC_PLD_T p = res_in[k].read();
            VEC_AXIS_VAL(pkt, k) = VAL_T_BITCAST(p.val);
            switch (p.inst) {
                case SOD:
                    got_SOD[k] = 1;
                    got_EOD[k] = 0;
                    got_EOS[k] = 0;
                    break;
                case EOD:
                    got_SOD[k] = 0;
                    got_EOD[k] = 1;
                    got_EOS[k] = 0;
                    break;
                case EOS:
                    got_SOD[k] = 0;
                    got_EOD[k] = 0;
                    got_EOS[k] = 1;
                    break;
                default:
                    got_SOD[k] = 0;
                    got_EOD[k] = 0;
                    got_EOS[k] = 0;
                    break;
            }
        }

        if (got_SOD.and_reduce()) {
            pkt.user = SOD;
            VEC_AXIS_PKT_IDX(pkt) = 0;
        } else if (got_EOD.and_reduce()) {
            pkt.user = EOD;
            VEC_AXIS_PKT_IDX(pkt) = 0;
        } else if (got_EOS.and_reduce()) {
            pkt.user = EOS;
            VEC_AXIS_PKT_IDX(pkt) = 0;
            exit = true;
        } else {
            pkt.user = 0;
            VEC_AXIS_PKT_IDX(pkt) = pkt_idx;
            pkt_idx++;
        }
        res_out.write(pkt);
#ifdef SPMV_RESULT_PACKER_LINE_TRACING
        std::cout << "SpMV Result Packer write output: " << pkt << std::endl;
#endif
    }
}


// one computational cluster
template<int cluster_id>
void spmv_cluster(
    const SPMV_MAT_PKT_T *matrix_hbm,       // in
    hls::stream<VEC_AXIS_T> &vec_in,        // in
    hls::stream<VEC_AXIS_T> &res_out,       // out
    const unsigned num_row_tiles,       // in
    const unsigned num_col_tiles      // in
) {

    hls::stream<EDGE_PLD_T> ML2SF[PACK_SIZE];
    hls::stream<EDGE_PLD_T> SF2VAU[PACK_SIZE];
    hls::stream<UPDATE_PLD_T> VAU2SF[PACK_SIZE];
    hls::stream<UPDATE_PLD_T> SF2PE[PACK_SIZE];
    hls::stream<VEC_PLD_T> PE2PK[PACK_SIZE];
    hls::stream<VEC_PLD_T> UPK2VAU[PACK_SIZE];
    #pragma HLS stream variable=ML2SF   depth=FIFO_DEPTH
    #pragma HLS stream variable=SF2VAU  depth=FIFO_DEPTH
    #pragma HLS stream variable=VAU2SF  depth=FIFO_DEPTH
    #pragma HLS stream variable=SF2PE   depth=FIFO_DEPTH
    #pragma HLS stream variable=PE2PK   depth=FIFO_DEPTH
    #pragma HLS stream variable=UPK2VAU depth=FIFO_DEPTH

    #pragma HLS bind_storage variable=ML2SF   type=FIFO impl=SRL
    #pragma HLS bind_storage variable=SF2VAU  type=FIFO impl=SRL
    #pragma HLS bind_storage variable=VAU2SF  type=FIFO impl=SRL
    #pragma HLS bind_storage variable=SF2PE   type=FIFO impl=SRL
    #pragma HLS bind_storage variable=PE2PK   type=FIFO impl=SRL
    #pragma HLS bind_storage variable=UPK2VAU type=FIFO impl=SRL

    #pragma HLS dataflow

    spmv_vector_unpacker(
        vec_in,
        UPK2VAU
    );
#ifdef SPMV_CLUSTER_H_LINE_TRACING
    std::cout << "INFO : [SpMV cluster] Vector Unpacker complete" << std::endl;
#endif

    coo_matrix_loader(
        matrix_hbm,
        ML2SF
    );

#ifdef SPMV_CLUSTER_H_LINE_TRACING
    std::cout << "INFO : [SpMV cluster] Matrix Loader complete" << std::endl;
#endif

    shuffler_flushable<EDGE_PLD_T, PACK_SIZE>(
        ML2SF,
        SF2VAU
    );

#ifdef SPMV_CLUSTER_H_LINE_TRACING
    std::cout << "INFO : [SpMV cluster] Shuffler 1 complete" << std::endl;
#endif

    vecbuf_access_unit<0, VB_BANK_SIZE, PACK_SIZE>(
        SF2VAU[0],
        UPK2VAU[0],
        VAU2SF[0],
        num_row_tiles,
        num_col_tiles
    );
    vecbuf_access_unit<1, VB_BANK_SIZE, PACK_SIZE>(
        SF2VAU[1],
        UPK2VAU[1],
        VAU2SF[1],
        num_row_tiles,
        num_col_tiles
    );
    vecbuf_access_unit<2, VB_BANK_SIZE, PACK_SIZE>(
        SF2VAU[2],
        UPK2VAU[2],
        VAU2SF[2],
        num_row_tiles,
        num_col_tiles
    );
    vecbuf_access_unit<3, VB_BANK_SIZE, PACK_SIZE>(
        SF2VAU[3],
        UPK2VAU[3],
        VAU2SF[3],
        num_row_tiles,
        num_col_tiles
    );
    vecbuf_access_unit<4, VB_BANK_SIZE, PACK_SIZE>(
        SF2VAU[4],
        UPK2VAU[4],
        VAU2SF[4],
        num_row_tiles,
        num_col_tiles
    );
    vecbuf_access_unit<5, VB_BANK_SIZE, PACK_SIZE>(
        SF2VAU[5],
        UPK2VAU[5],
        VAU2SF[5],
        num_row_tiles,
        num_col_tiles
    );
    vecbuf_access_unit<6, VB_BANK_SIZE, PACK_SIZE>(
        SF2VAU[6],
        UPK2VAU[6],
        VAU2SF[6],
        num_row_tiles,
        num_col_tiles
    );
    vecbuf_access_unit<7, VB_BANK_SIZE, PACK_SIZE>(
        SF2VAU[7],
        UPK2VAU[7],
        VAU2SF[7],
        num_row_tiles,
        num_col_tiles
    );

#ifdef SPMV_CLUSTER_H_LINE_TRACING
    std::cout << "INFO : [SpMV cluster] Vector Access Unit complete" << std::endl;
#endif

    shuffler_flushable<UPDATE_PLD_T, PACK_SIZE>(
        VAU2SF,
        SF2PE
    );

#ifdef SPMV_CLUSTER_H_LINE_TRACING
    std::cout << "INFO : [SpMV cluster] Shuffler 2 complete" << std::endl;
#endif

    pe_flushable<0, OB_BANK_SIZE, PACK_SIZE>(
        SF2PE[0],
        PE2PK[0]
    );
    pe_flushable<1, OB_BANK_SIZE, PACK_SIZE>(
        SF2PE[1],
        PE2PK[1]
    );
    pe_flushable<2, OB_BANK_SIZE, PACK_SIZE>(
        SF2PE[2],
        PE2PK[2]
    );
    pe_flushable<3, OB_BANK_SIZE, PACK_SIZE>(
        SF2PE[3],
        PE2PK[3]
    );
    pe_flushable<4, OB_BANK_SIZE, PACK_SIZE>(
        SF2PE[4],
        PE2PK[4]
    );
    pe_flushable<5, OB_BANK_SIZE, PACK_SIZE>(
        SF2PE[5],
        PE2PK[5]
    );
    pe_flushable<6, OB_BANK_SIZE, PACK_SIZE>(
        SF2PE[6],
        PE2PK[6]
    );
    pe_flushable<7, OB_BANK_SIZE, PACK_SIZE>(
        SF2PE[7],
        PE2PK[7]
    );

#ifdef SPMV_CLUSTER_H_LINE_TRACING
    std::cout << "INFO : [SpMV cluster] Process Elements complete" << std::endl;
#endif

    spmv_result_packer(
        PE2PK,
        res_out
    );

#ifdef SPMV_CLUSTER_H_LINE_TRACING
    std::cout << "INFO : [SpMV cluster] Result Packer complete" << std::endl;
#endif
}

#endif // SPMV_CLUSTER_H_
