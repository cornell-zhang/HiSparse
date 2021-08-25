#ifndef SPMV_FLOAT_PE_H_
#define SPMV_FLOAT_PE_H_

#include <hls_stream.h>
#include <ap_int.h>

#include "common.h"

#ifdef __SYNTHESIS__
#include "utils/x_hls_utils.h" // for reg() function
#else
#ifndef REG_FOR_SW_EMU
#define REG_FOR_SW_EMU
template<typename T>
T reg(T in) {
    return in;
}
#endif
#endif

#ifndef __SYNTHESIS__
// #define PE_LINE_TRACING
#endif

//----------------------------------------------------------------
// pe processing pipeline
//----------------------------------------------------------------
template<int id, unsigned bank_size, unsigned pack_size>
void float_pe_process(
    hls::stream<UPDATE_PLD_T> &input,
    VAL_T output_buffer[DEP_DISTANCE][bank_size]
) {
    bool exit = false;

    // bool stall = false;
    UPDATE_PLD_T pld;
    bool valid = true;
    unsigned pb_idx = 0;

    pe_process_loop:
    while (!exit) {
        #pragma HLS pipeline II=1
        #pragma HLS dependence variable=output_buffer inter false

        pld = input.read();
#ifdef PE_LINE_TRACING
        std::cout << "  input payload: " << pld << std::endl;
#endif
        if (pld.inst == EOD) {
            exit = true;
            valid = false;
        } else {
            exit = false;
            valid = true;
        }

        if (valid) {
            IDX_T bank_addr = pld.row_idx / pack_size;
            VAL_T incr = pld.mat_val * pld.vec_val;
            VAL_T q = output_buffer[pb_idx][bank_addr];
            VAL_T new_q = q + incr;
            #pragma HLS bind_op variable=new_q op=fadd impl=fulldsp latency=FPADD_LATENCY
            output_buffer[pb_idx][bank_addr] = new_q;
        }

        pb_idx = (pb_idx + 1) % DEP_DISTANCE;
    }
}

//----------------------------------------------------------------
// pe output pipeline
//----------------------------------------------------------------
template<int id, unsigned bank_size, unsigned pack_size>
void float_pe_output(
    hls::stream<VEC_PLD_T> &output,
    VAL_T output_buffer[DEP_DISTANCE][bank_size],
    const unsigned used_buf_len
) {
    bool exit = false;
    unsigned dump_count = 0;
    pe_output_loop:
    for (unsigned dump_count = 0; dump_count < used_buf_len; dump_count++) {
        #pragma HLS pipeline II=1

        // VAL_T q = output_buffer[dump_count];
        VAL_T q = 0;
        for (unsigned d = 0; d < DEP_DISTANCE; d++) {
            #pragma HLS unroll
            q += output_buffer[d][dump_count];
        }

        VEC_PLD_T out_pld;
        out_pld.val = q;
        out_pld.idx = dump_count * pack_size + id;
        out_pld.inst = 0x0;
        output.write(out_pld);
#ifdef PE_LINE_TRACING
        std::cout << "  write output: " << VEC_PLD_EOD << std::endl;
#endif
    }
}

//----------------------------------------------------------------
// floating-point pe
//----------------------------------------------------------------
template<int id, unsigned bank_size, unsigned pack_size>
void pe(
    hls::stream<UPDATE_PLD_T> &input,
    hls::stream<VEC_PLD_T> &output,
    const unsigned used_buf_len
) {
    VAL_T output_buffer[DEP_DISTANCE][bank_size];
    #pragma HLS bind_storage variable=output_buffer type=RAM_2P impl=BRAM latency=1
    #pragma HLS array_partition variable=output_buffer dim=1 complete

    // reset output buffer before doing anything
    loop_reset_ob:
    for (unsigned i = 0; i < used_buf_len; i++) {
        #pragma HLS pipeline II=1
        for (unsigned d = 0; d < DEP_DISTANCE; d++) {
            #pragma HLS unroll
            output_buffer[d][i] = 0;
        }
    }

    // wait on the first SOD
    bool got_SOD = false;
    pe_sync_SOD:
    while (!got_SOD) {
        #pragma HLS pipeline II=1
        UPDATE_PLD_T p = input.read();
        got_SOD = (p.inst == SOD);
    }

    // start processing
    bool exit = false;
    pe_main_loop:
    while (!exit) {
        #pragma HLS pipeline off
        // this function will exit upon EOD
        float_pe_process<id, bank_size, pack_size>(input, output_buffer);

        // read the next payload and decide whether continue processing or exit
        bool got_valid_pld = false;
        pe_sync_SODEOS:
        while (!got_valid_pld) {
            #pragma HLS pipeline II=1
            UPDATE_PLD_T p = input.read();
            if (p.inst == SOD) {
                got_valid_pld = true;
                exit = false;
            } else if (p.inst == EOS) {
                got_valid_pld = true;
                exit = true;
            } else {
                got_valid_pld = false;
                exit = false;
            }
        }
    }

    // dump results
    output.write(VEC_PLD_SOD);
    float_pe_output<id, bank_size, pack_size>(output, output_buffer, used_buf_len);
    output.write(VEC_PLD_EOD);
    output.write(VEC_PLD_EOS);
}

#endif  // SPMV_FLOAT_PE_H_
