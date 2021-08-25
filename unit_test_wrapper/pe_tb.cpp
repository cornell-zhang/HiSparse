#include "pe_tb.h"
#include "pe.h"
#include "hls_stream.h"

#ifndef __SYNTHESIS__
#include <iostream>
#include <iomanip>
#include <vector>
#endif

void data_feeder(
    UPDATE_PLD_T input_buffer[IN_BUF_SIZE],
    hls::stream<UPDATE_PLD_T> &to_pe
) {
    bool loop_exit = false;
    unsigned i = 0;
    loop_data_feeder:
    while (!loop_exit) {
        #pragma HLS pipeline II=1
// #ifndef __SYNTHESIS__
//         std::cout << "Tracing: Iteration " << i << std::flush;
// #endif
        UPDATE_PLD_T p = input_buffer[i];

// #ifndef __SYNTHESIS__
//         std::cout << "  input payload: " << p << std::endl;
// #endif
        to_pe.write(p);
        if (p.inst == EOS) {
            loop_exit = true;
        }
        i++;
    }
}

void data_drain(
    hls::stream<VEC_PLD_T> &from_pe,
    VEC_PLD_T result_buffer[BANK_SIZE]
) {
    bool loop_exit = false;
    unsigned i = 0;
    loop_data_drain:
    while (!loop_exit) {
        #pragma HLS pipeline II=1
        VEC_PLD_T p;
        if (from_pe.read_nb(p)) {
            if (p.inst != EOS) {
                if (p.inst != SOD && p.inst != EOD) {
                    result_buffer[i] = p;
                    i++;
                }
            } else {
                loop_exit = true;
            }
        }
    }
}


void main_dataflow(
    UPDATE_PLD_T input_buffer[IN_BUF_SIZE],
    VEC_PLD_T result_buffer[BANK_SIZE]
) {
#ifndef __SYNTHESIS__
    std::cout << "Main dataflow: started." << std::endl;
#endif

    hls::stream<UPDATE_PLD_T> feeder_to_pe;
    hls::stream<VEC_PLD_T> pe_to_drain;
    #pragma HLS stream variable=feeder_to_pe depth=8
    #pragma HLS stream variable=pe_to_drain depth=8

    #pragma HLS dataflow

    data_feeder(input_buffer, feeder_to_pe);

#ifndef __SYNTHESIS__
    std::cout << "Main dataflow: data feeder complete" << std::endl;
#endif

    pe<0, BANK_SIZE, 1>(
        feeder_to_pe,
        pe_to_drain,
        BANK_SIZE
    );

#ifndef __SYNTHESIS__
    std::cout << "Main dataflow: DUT \"ufixed_pe<0, BANK_SIZE, 1>\" complete" << std::endl;
#endif

    data_drain(pe_to_drain, result_buffer);

#ifndef __SYNTHESIS__
    std::cout << "Main dataflow: data drain complete" << std::endl;
#endif
}

extern "C" {
void pe_tb (
    const IDX_T *test_addr_gmem, //0
    const VAL_T *test_mat_gmem,  //1
    const VAL_T *test_vec_gmem,  //2
    VAL_T *result_gmem           //3
) {
    #pragma HLS interface m_axi port=test_addr_gmem offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=test_mat_gmem  offset=slave bundle=gmem1
    #pragma HLS interface m_axi port=test_vec_gmem  offset=slave bundle=gmem2
    #pragma HLS interface m_axi port=result_gmem    offset=slave bundle=gmem3

    #pragma HLS interface s_axilite port=test_addr_gmem bundle=control
    #pragma HLS interface s_axilite port=test_mat_gmem  bundle=control
    #pragma HLS interface s_axilite port=test_vec_gmem  bundle=control
    #pragma HLS interface s_axilite port=result_gmem    bundle=control

    #pragma HLS interface s_axilite port=return bundle=control

#ifndef __SYNTHESIS__
    std::cout << "Testbench started (" << std::endl;
    std::cout << "\tconst IDX_T *test_addr_gmem = " << std::hex << test_addr_gmem << std::endl;
    std::cout << "\tconst VAL_T *test_mat_gmem = " << std::hex << test_addr_gmem << std::endl;
    std::cout << "\tconst VAL_T *test_vec_gmem = " << std::hex << test_addr_gmem << std::endl;
    std::cout << "\tVAL_T *result_gmem = " << std::hex << test_addr_gmem << std::endl;
    std::cout << std::dec << std::endl;
#endif

    // input buffer
    UPDATE_PLD_T input_buffer[IN_BUF_SIZE];
    // result buffer
    VEC_PLD_T result_buffer[BANK_SIZE];

    // reset result buffer
    loop_reset_results:
    for (unsigned i = 0; i < BANK_SIZE; i++) {
        #pragma HLS pipeline II=1
        VEC_PLD_T out_pld;
        out_pld.val = 0;
        out_pld.idx = 0;
        out_pld.inst = 0;
        result_buffer[i] = out_pld;
    }

#ifndef __SYNTHESIS__
    std::cout << "Testbench reset output buffer complete." << std::endl;
#endif

    // initialize input buffer
    input_buffer[0].mat_val = 0;
    input_buffer[0].vec_val = 0;
    input_buffer[0].row_idx = 0;
    input_buffer[0].inst = SOD;
    input_buffer[TEST_LEN + 1].mat_val = 0;
    input_buffer[TEST_LEN + 1].vec_val = 0;
    input_buffer[TEST_LEN + 1].row_idx = 0;
    input_buffer[TEST_LEN + 1].inst = EOD;
    input_buffer[TEST_LEN + 2].mat_val = 0;
    input_buffer[TEST_LEN + 2].vec_val = 0;
    input_buffer[TEST_LEN + 2].row_idx = 0;
    input_buffer[TEST_LEN + 2].inst = EOS;

#ifndef __SYNTHESIS__
    std::cout << "Testbench initialize input buffer complete." << std::endl;
#endif

    loop_read_inputs:
    for (unsigned i = 0; i < TEST_LEN; i++) {
        #pragma HLS pipeline II=1
        input_buffer[i + 1].mat_val = test_mat_gmem[i];
        input_buffer[i + 1].vec_val = test_vec_gmem[i];
        input_buffer[i + 1].row_idx = test_addr_gmem[i];
        input_buffer[i + 1].inst = 0;
    }
#ifndef __SYNTHESIS__
    std::cout << "Testbench read input complete." << std::endl;
#endif

#ifndef __SYNTHESIS__
    std::cout << "Testbench invoking the main dataflow... " << std::endl;
#endif
    // run main dataflow
    main_dataflow(input_buffer, result_buffer);

#ifndef __SYNTHESIS__
    std::cout << "Testbench main dataflow finished" << std::endl;
#endif

    // write back to results
    loop_wb:
    for (unsigned i = 0; i < BANK_SIZE; i++) {
        #pragma HLS pipeline II=1
        VEC_PLD_T p = result_buffer[i];
        result_gmem[p.idx] = p.val;
    }

#ifndef __SYNTHESIS__
    std::cout << "Testbench write back results complete." << std::endl;
    std::cout << "Testbench finished." << std::endl;
#endif

} // extern "C"
} // kernel
