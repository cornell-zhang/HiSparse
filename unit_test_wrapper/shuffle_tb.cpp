#include "shuffle_tb.h"
#include "shuffle.h"
#include "hls_stream.h"

#ifndef __SYNTHESIS__
#include <iostream>
#include <iomanip>
#include <vector>
#endif

void data_feeder(
    EDGE_PLD_T input_buffer[BUF_SIZE],
    hls::stream<EDGE_PLD_T> &to_sf
) {
    bool finish = 0;
    unsigned idx = 0;

    loop_data_feeder:
    while (!finish) {
        #pragma HLS pipeline II=1
        if (!finish) {
            EDGE_PLD_T pout = input_buffer[idx];
            if (pout.inst == EOS) {
                finish = true;
            }
            idx++;
            to_sf.write(pout);
        }
    }
}

template<unsigned num_lanes>
void data_drain(
    hls::stream<EDGE_PLD_T> from_sf[num_lanes],
    EDGE_PLD_T result_buffer[num_lanes][num_lanes * BUF_SIZE]
) {

    ap_uint<num_lanes> finish = 0;
    unsigned idx[num_lanes];
    for (unsigned i = 0; i < num_lanes; i++) {
        #pragma HLS unroll
        idx[i] = 0;
    }

    loop_data_drain:
    while (!finish.and_reduce()) {
        #pragma HLS pipeline II=1
        for (unsigned OLid = 0; OLid < num_lanes; OLid++)  {
            #pragma HLS unroll
            if (!finish[OLid]) {
                EDGE_PLD_T p;
                if (from_sf[OLid].read_nb(p)) {
                    result_buffer[OLid][idx[OLid]] = p;
                    idx[OLid]++;
                    if (p.inst == EOS) {
                        finish[OLid] = true;
                    }
                }
            }
        }
    }

}


void main_dataflow(
    EDGE_PLD_T input_buffer[8][BUF_SIZE],
    EDGE_PLD_T result_buffer[8][8 * BUF_SIZE]
) {
    hls::stream<EDGE_PLD_T> feeder_to_sf[8];
    hls::stream<EDGE_PLD_T> sf_to_drain[8];
    #pragma HLS stream variable=feeder_to_sf depth=8
    #pragma HLS stream variable=sf_to_drain depth=8

    #pragma HLS dataflow

    data_feeder(input_buffer[0], feeder_to_sf[0]);
    data_feeder(input_buffer[1], feeder_to_sf[1]);
    data_feeder(input_buffer[2], feeder_to_sf[2]);
    data_feeder(input_buffer[3], feeder_to_sf[3]);
    data_feeder(input_buffer[4], feeder_to_sf[4]);
    data_feeder(input_buffer[5], feeder_to_sf[5]);
    data_feeder(input_buffer[6], feeder_to_sf[6]);
    data_feeder(input_buffer[7], feeder_to_sf[7]);
#ifndef __SYNTHESIS__
    std::cout << "data feeder complete" << std::endl;
#endif

    shuffler<EDGE_PLD_T, 8>(feeder_to_sf, sf_to_drain);
#ifndef __SYNTHESIS__
    std::cout << "shuffler complete" << std::endl;
#endif

    data_drain<8>(sf_to_drain, result_buffer);
#ifndef __SYNTHESIS__
    std::cout << "data drain complete" << std::endl;
#endif
}

extern "C" {
void shuffle_tb(
    const EDGE_PLD_T *input_packets,   //0
    EDGE_PLD_T *output_packets         //1
) {
    #pragma HLS interface m_axi port=input_packets offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=output_packets  offset=slave bundle=gmem1

    #pragma HLS interface s_axilite port=input_packets bundle=control
    #pragma HLS interface s_axilite port=output_packets  bundle=control

    #pragma HLS interface s_axilite port=return bundle=control

    // input buffer
    EDGE_PLD_T input_buffer[8][BUF_SIZE];
    #pragma HLS array_partition variable=input_buffer dim=1 complete
    #pragma HLS resource variable=input_buffer core=RAM_1P

    // result buffer
    EDGE_PLD_T result_buffer[8][8 * BUF_SIZE];
    #pragma HLS array_partition variable=result_buffer dim=1 complete
    #pragma HLS resource variable=result_buffer core=RAM_2P latency=2

    // reset result buffer
    loop_reset_ob:
    for (unsigned i = 0; i < BUF_SIZE; i++) {
        #pragma HLS pipeline II=1
        for (unsigned k = 0; k < 8; k++) {
            #pragma HLS unroll
            EDGE_PLD_T p;
            p.mat_val = 0;
            p.row_idx = 0;
            p.col_idx = 0;
            p.inst = 0;
            result_buffer[k][i] = p;
        }
    }
#ifndef __SYNTHESIS__
    std::cout << "reset result buffer complete" << std::endl;
#endif

    // initialize input buffer
    unsigned offset = 0;
    loop_ini_ib:
    for (unsigned ILid = 0; ILid < 8; ILid++) {
        #pragma HLS pipeline off
        bool exit = false;
        unsigned i = 0;
        loop_ini_ib_inner:
        while (!exit) {
            #pragma HLS pipeline
            EDGE_PLD_T p = input_packets[i + offset];
            input_buffer[ILid][i] = p;
            i++;
            if (p.inst == EOS) {
                exit = true;
            }
        }
        offset += i;
    }

#ifndef __SYNTHESIS__
    std::cout << "initialize input buffer complete" << std::endl;
#endif

    // run main dataflow
    main_dataflow(input_buffer, result_buffer);

#ifndef __SYNTHESIS__
    std::cout << "main dataflow complete" << std::endl;
#endif

    // write back results
    offset = 0;
    loop_write_back:
    for (unsigned OLid = 0; OLid < 8; OLid++) {
        #pragma HLS pipeline off
        bool exit = false;
        unsigned i = 0;
// #ifndef __SYNTHESIS__
//         std::cout << "Output Lane: " << OLid << std::endl;
// #endif
        loop_wb_inner:
        while (!exit) {
            #pragma HLS pipeline
            EDGE_PLD_T p = result_buffer[OLid][i];
            output_packets[i + offset] = p;
// #ifndef __SYNTHESIS__
//             std::cout << "  res[" << i + offset << "] <= " << p << std::endl;
// #endif
            i++;
            if (p.inst == EOS) {
                exit = true;
            }
        }
        offset += i;
    }

#ifndef __SYNTHESIS__
    std::cout << "kernel retruning" << std::endl;
#endif

} // kernel
} // extern "C"
