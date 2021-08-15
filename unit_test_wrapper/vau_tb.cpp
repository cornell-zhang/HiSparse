#include "vau_tb.h"
#include "vecbuf_access_unit.h"
#include "hls_stream.h"
#include "./overlay.h"
#include "ap_fixed.h"

#ifndef __SYNTHESIS__
#include <iostream>
#include <iomanip>
#include <vector>
#endif

void data_feeder(
    const IDX_T *test_col_idx_gmem,
    const unsigned *partition_table,
    hls::stream<EDGE_PLD_T> &to_vau,
    const unsigned num_partitions
) {
    unsigned offset = 0;
    loop_DF_outer:
    for (unsigned i = 0; i < num_partitions; i++) {
        #pragma HLS pipeline off
        to_vau.write(EDGE_PLD_SOD);
        unsigned len = partition_table[i];

        loop_data_feeder:
        for (unsigned j = 0; j < len; j++) {
            #pragma HLS pipeline II=1
            EDGE_PLD_T pld;
            IDX_T abs_idx = test_col_idx_gmem[offset + j];
            pld.mat_val = 1;
            pld.row_idx = abs_idx;
            pld.col_idx = abs_idx;
            pld.inst = 0x0;
            to_vau.write(pld);
        }

        to_vau.write(EDGE_PLD_EOD);
        offset += len;
    }
    to_vau.write(EDGE_PLD_EOS);
}

void vector_loader(
    const VAL_T *test_vector_gmem,
    hls::stream<VEC_PLD_T> &out,
    const unsigned num_partitions
) {
    loop_VL_outer:
    for (unsigned i = 0; i < num_partitions; i++) {
        #pragma HLS pipeline off
        out.write(VEC_PLD_SOD);

        loop_vector_loader:
        for (unsigned j = 0; j < BANK_SIZE; j++) {
            #pragma HLS pipeline II=1
            VEC_PLD_T pld;
            IDX_T abs_idx = i * BANK_SIZE + j;
            VAL_T vec_val = test_vector_gmem[abs_idx];
            pld.idx = abs_idx;
            pld.val = vec_val;
            pld.inst = 0;
            out.write(pld);
        }

        out.write(VEC_PLD_EOD);
    }

    out.write(VEC_PLD_EOS);
}

void data_drain(
    hls::stream<UPDATE_PLD_T> &from_vau,
    IDX_T *result_idx,
    VAL_T *result_val
) {
    bool vin_SOD = false;
    while (!vin_SOD) {
        #pragma HLS pipeline II=1
        UPDATE_PLD_T p = from_vau.read();
        vin_SOD = (p.inst == SOD);
#ifndef __SYNTHESIS__
            std::cout << "data drain got payload: " << p << std::endl;
#endif
    }

    bool loop_exit = false;
    unsigned i = 0;
    loop_data_drain:
    while (!loop_exit) {
        #pragma HLS pipeline II=1
        UPDATE_PLD_T p = from_vau.read();
#ifndef __SYNTHESIS__
            std::cout << "data drain got payload: " << p << std::endl;
#endif
        if (p.inst == EOS) {
            loop_exit = true;
        } else if (p.inst == SOD) {
            continue;
        } else if (p.inst == EOD) {
            continue;
        } else {
            result_val[i] = p.vec_val;
            result_idx[i] = p.row_idx;
            i++;
#ifndef __SYNTHESIS__
            std::cout << "output count =  " << i-1 << std::endl;
            std::cout << "  result_val: " << result_val[i-1] << std::endl;
            std::cout << "  result_idx: " << result_idx[i-1] << std::endl;
#endif
        }
    }
}

extern "C" {
void vau_tb (
    const IDX_T *test_col_idx_gmem,   //0
    const VAL_T *test_vector_gmem,    //1
    const unsigned *partition_table,  //2
    IDX_T *result_idx,                //3
    VAL_T *result_val,                //4
    const unsigned num_partitions     //5
) {
    #pragma HLS interface m_axi port=test_col_idx_gmem offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=test_vector_gmem  offset=slave bundle=gmem1
    #pragma HLS interface m_axi port=partition_table   offset=slave bundle=gmem2
    #pragma HLS interface m_axi port=result_idx        offset=slave bundle=gmem3
    #pragma HLS interface m_axi port=result_val        offset=slave bundle=gmem4

    #pragma HLS interface s_axilite port=test_col_idx_gmem bundle=control
    #pragma HLS interface s_axilite port=test_vector_gmem  bundle=control
    #pragma HLS interface s_axilite port=partition_table   bundle=control
    #pragma HLS interface s_axilite port=result_idx        bundle=control
    #pragma HLS interface s_axilite port=result_val        bundle=control

    #pragma HLS interface s_axilite port=num_partitions    bundle=control

    #pragma HLS interface s_axilite port=return bundle=control

    #pragma HLS dataflow

    hls::stream<EDGE_PLD_T> data_feeder_to_vau;
    hls::stream<VEC_PLD_T> vector_loader_to_vau;
    hls::stream<UPDATE_PLD_T> vau_to_data_drain;
    #pragma HLS stream depth=8 variable=data_feeder_to_vau
    #pragma HLS stream depth=8 variable=vector_loader_to_vau
    #pragma HLS stream depth=8 variable=vau_to_data_drain

#ifndef __SYNTHESIS__
    std::cout << "Testbench started" << std::endl;
#endif

    data_feeder(
        test_col_idx_gmem,
        partition_table,
        data_feeder_to_vau,
        num_partitions
    );

#ifndef __SYNTHESIS__
    std::cout << "data_feeder complete" << std::endl;
#endif

    vector_loader(
        test_vector_gmem,
        vector_loader_to_vau,
        num_partitions
    );

#ifndef __SYNTHESIS__
    std::cout << "vector_loader complete" << std::endl;
#endif

    vecbuf_access_unit<0, BANK_SIZE, 1>(
        data_feeder_to_vau,
        vector_loader_to_vau,
        vau_to_data_drain,
        num_partitions
    );

#ifndef __SYNTHESIS__
    std::cout << "vecbuf_access_unit complete" << std::endl;
#endif

    data_drain(
        vau_to_data_drain,
        result_idx,
        result_val
    );

#ifndef __SYNTHESIS__
    std::cout << "data_drain complete" << std::endl;
#endif


} // kernel
} // extern "C"
