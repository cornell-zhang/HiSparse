#include "pe_cluster_tb.h"
#include "pe.h"
#include "hls_stream.h"

#ifndef __SYNTHESIS__
#include <iostream>
#include <iomanip>
#include <vector>
#endif

template<unsigned num_PE>
void data_feeder(
    UPDATE_PLD_T input_buffer[num_PE][IN_BUF_SIZE],
    hls::stream<UPDATE_PLD_T> to_pes[num_PE]
) {
    for (unsigned PEid = 0; PEid < num_PE; PEid++)  {
        #pragma HLS unroll
        bool loop_exit = false;
        unsigned i = 0;
        loop_data_feeder:
        while (!loop_exit) {
            #pragma HLS pipeline II=1
            UPDATE_PLD_T p = input_buffer[PEid][i];
            to_pes[PEid].write(p);
            if (p.inst == EOS) {
                loop_exit = true;
            }
            i++;
        }
    }
}

template<unsigned num_PE>
void data_drain(
    hls::stream<VEC_PLD_T> from_pe[num_PE],
    VEC_PLD_T result_buffer[num_PE][BANK_SIZE]
) {
    for (unsigned PEid = 0; PEid < num_PE; PEid++)  {
        #pragma HLS unroll
        bool loop_exit = false;
        unsigned i = 0;
        loop_data_drain:
        while (!loop_exit) {
            #pragma HLS pipeline II=1
            VEC_PLD_T p;
            if (from_pe[PEid].read_nb(p)) {
                if (p.inst != EOS) {
                    result_buffer[PEid][i] = p;
                    i++;
                } else {
                    loop_exit = true;
                }
            }
        }
    }
}


void main_dataflow(
    UPDATE_PLD_T input_buffer[8][IN_BUF_SIZE],
    VEC_PLD_T result_buffer[8][BANK_SIZE],
    const VAL_T zero,
    const OP_T op
) {
    hls::stream<UPDATE_PLD_T> feeder_to_pe[8];
    hls::stream<VEC_PLD_T> pe_to_drain[8];
    #pragma HLS stream variable=feeder_to_pe depth=8
    #pragma HLS stream variable=pe_to_drain depth=8

    VAL_T pe_output_buffer[8][BANK_SIZE];
    #pragma HLS array_partition variable=pe_output_buffer complete dim=1

    #pragma HLS dataflow

    data_feeder<8>(input_buffer, feeder_to_pe);

    ufixed_pe<0, BANK_SIZE, 8>(
        feeder_to_pe[0],
        pe_to_drain[0],
        pe_output_buffer[0],
        zero,
        op
    );
    ufixed_pe<1, BANK_SIZE, 8>(
        feeder_to_pe[1],
        pe_to_drain[1],
        pe_output_buffer[1],
        zero,
        op
    );
    ufixed_pe<2, BANK_SIZE, 8>(
        feeder_to_pe[2],
        pe_to_drain[2],
        pe_output_buffer[2],
        zero,
        op
    );
    ufixed_pe<3, BANK_SIZE, 8>(
        feeder_to_pe[3],
        pe_to_drain[3],
        pe_output_buffer[3],
        zero,
        op
    );
    ufixed_pe<4, BANK_SIZE, 8>(
        feeder_to_pe[4],
        pe_to_drain[4],
        pe_output_buffer[4],
        zero,
        op
    );
    ufixed_pe<5, BANK_SIZE, 8>(
        feeder_to_pe[5],
        pe_to_drain[5],
        pe_output_buffer[5],
        zero,
        op
    );
    ufixed_pe<6, BANK_SIZE, 8>(
        feeder_to_pe[6],
        pe_to_drain[6],
        pe_output_buffer[6],
        zero,
        op
    );
    ufixed_pe<7, BANK_SIZE, 8>(
        feeder_to_pe[7],
        pe_to_drain[7],
        pe_output_buffer[7],
        zero,
        op
    );

    data_drain<8>(pe_to_drain, result_buffer);
}

void merge_results(
    VEC_PLD_T result_buffer[8][BANK_SIZE],
    VAL_T *result_gmem
) {
    for (unsigned PEid = 0; PEid < 8; PEid++)  {
        #pragma HLS unroll
        loop_merge_wb:
        for (unsigned i = 0; i < BANK_SIZE; i++) {
            #pragma HLS pipeline II=1
            VEC_PLD_T p = result_buffer[PEid][i];
            result_gmem[p.idx] = p.val;
        }
    }
}

extern "C" {
void pe_cluster_tb(
    const IDX_T *test_addr_gmem, //0
    const VAL_T *test_mat_gmem,  //1
    const VAL_T *test_vec_gmem,  //2
    VAL_T *result_gmem,          //3
    const IDX_T *length_table,   //4
    const OP_T op                //5
) {
    #pragma HLS interface m_axi port=test_addr_gmem offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=test_mat_gmem  offset=slave bundle=gmem1
    #pragma HLS interface m_axi port=test_vec_gmem  offset=slave bundle=gmem2
    #pragma HLS interface m_axi port=result_gmem    offset=slave bundle=gmem3
    #pragma HLS interface m_axi port=length_table   offset=slave bundle=gmem3

    #pragma HLS interface s_axilite port=test_addr_gmem bundle=control
    #pragma HLS interface s_axilite port=test_mat_gmem  bundle=control
    #pragma HLS interface s_axilite port=test_vec_gmem  bundle=control
    #pragma HLS interface s_axilite port=result_gmem    bundle=control
    #pragma HLS interface s_axilite port=length_table   bundle=control

    #pragma HLS interface s_axilite port=op     bundle=control

    #pragma HLS interface s_axilite port=return bundle=control

    VAL_T zero;
    switch (op) {
        case MULADD:
            zero = MulAddZero;
            break;
        case ANDOR:
            zero = AndOrZero;
            break;
        case ADDMIN:
            zero = AddMinZero;
            break;
        default:
            zero = MulAddZero;
            break;
    }

    // input buffer
    UPDATE_PLD_T input_buffer[8][IN_BUF_SIZE];
    #pragma HLS array_partition variable=input_buffer dim=1 complete
    #pragma HLS resource variable=input_buffer core=RAM_1P

    // result buffer
    VEC_PLD_T result_buffer[8][BANK_SIZE];
    #pragma HLS array_partition variable=result_buffer dim=1 complete
    #pragma HLS resource variable=result_buffer core=RAM_2P latency=2

    // reset result buffer
    loop_reset_ob:
    for (unsigned i = 0; i < BANK_SIZE; i++) {
        #pragma HLS pipeline II=1
        for (unsigned PEid = 0; PEid < 8; PEid++) {
            #pragma HLS unroll
            VEC_PLD_T out_pld;
            out_pld.val = 0;
            out_pld.idx = 0;
            out_pld.inst = 0;
            result_buffer[PEid][i] = out_pld;
        }
    }

    // initialize input buffer
    unsigned offset = 0;
    loop_ini_ib_banks:
    for (unsigned PEid = 0; PEid < 8; PEid++) {
        unsigned len = length_table[PEid];
        input_buffer[PEid][0].mat_val = 0;
        input_buffer[PEid][0].vec_val = 0;
        input_buffer[PEid][0].row_idx = 0;
        input_buffer[PEid][0].inst = SOD;
        input_buffer[PEid][len].mat_val = 0;
        input_buffer[PEid][len].vec_val = 0;
        input_buffer[PEid][len].row_idx = 0;
        input_buffer[PEid][len].inst = EOD;
        input_buffer[PEid][len + 1].mat_val = 0;
        input_buffer[PEid][len + 1].vec_val = 0;
        input_buffer[PEid][len + 1].row_idx = 0;
        input_buffer[PEid][len + 1].inst = EOS;
        loop_ini_ib:
        for (unsigned i = 0; i < len; i++) {
            #pragma HLS pipeline II=1
            input_buffer[PEid][i + 1].mat_val = test_mat_gmem[i + offset];
            input_buffer[PEid][i + 1].vec_val = test_vec_gmem[i + offset];
            input_buffer[PEid][i + 1].row_idx = test_addr_gmem[i + offset];
            input_buffer[PEid][i + 1].inst = 0;
        }
        offset += len;
    }

    // run main dataflow
    main_dataflow(input_buffer, result_buffer, zero, op);

    // write back to results
    merge_results(result_buffer, result_gmem);

} // extern "C"
} // kernel
