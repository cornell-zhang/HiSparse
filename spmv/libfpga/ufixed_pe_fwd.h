#ifndef SPMV_UFIXED_PE_H_
#define SPMV_UFIXED_PE_H_

#include <hls_stream.h>
#include <ap_int.h>

#include "math_constants.h"
#include "common.h"

#define MIN(a, b) ((a < b)? a : b)

#ifndef __SYNTHESIS__
// #define PE_LINE_TRACING
#endif

//----------------------------------------------------------------
// ALUs
//----------------------------------------------------------------

VAL_T pe_ufixed_mul_alu(VAL_T a, VAL_T b, const VAL_T z, const OP_T op) {
    #pragma HLS pipeline II=1
    #pragma HLS latency min=2 max=2
    VAL_T out;
    switch (op) {
        case MULADD:
            out = a * b;
            break;
        case ANDOR:
            out = a && b;
            break;
        case ADDMIN:
            out = a + b;
            break;
        default:
            out = z;  // z is the zero value in this semiring
            break;
    }
    return out;
}

VAL_T pe_ufixed_add_alu(VAL_T a, VAL_T b, const OP_T op) {
    #pragma HLS pipeline II=1
    #pragma HLS latency min=0 max=0
    VAL_T out;
    switch (op) {
        case MULADD:
            out = a + b;
            break;
        case ANDOR:
            out = a || b;
            break;
        case ADDMIN:
            out = MIN(a, b);
            break;
        default:
            out = a;
            break;
    }
    return out;
}

//----------------------------------------------------------------
// pe processing pipeline
//----------------------------------------------------------------
struct IN_FLIGHT_WRITE {
    bool valid;
    IDX_T addr;
    VAL_T value;
};

template<int id, unsigned bank_size, unsigned pack_size>
void ufixed_pe_process(
    hls::stream<UPDATE_PLD_T> &input,
    VAL_T output_buffer[bank_size],
    const VAL_T zero,
    const OP_T op
) {
    bool exit = false;

    // in-flight write queue for data-forwarding
    // the maximum write latency of URAM is 2
    IN_FLIGHT_WRITE ifwq[2];
    #pragma HLS array_partition variable=ifwq complete;
    ifwq[0] = (IN_FLIGHT_WRITE){false, 0, 0};
    ifwq[1] = (IN_FLIGHT_WRITE){false, 0, 0};

    pe_process_loop:
    while (!exit) {
        #pragma HLS pipeline II=1
        #pragma HLS dependence variable=output_buffer false
        UPDATE_PLD_T pld = input.read();
        bool valid = true;
#ifdef PE_LINE_TRACING
        std::cout << "  input payload: " << pld << std::endl;
#endif
        if (pld.inst == EOD) {
            exit = true;
            valid = false;
        }

        if (valid) {
            IDX_T bank_addr = pld.row_idx / pack_size;
            VAL_T incr = pe_ufixed_mul_alu(
                pld.mat_val, pld.vec_val,
                zero, op
            );
            VAL_T q = output_buffer[bank_addr];
            VAL_T q_fwd = ((bank_addr == ifwq[0].addr) && ifwq[0].valid) ? ifwq[0].value :
                          ((bank_addr == ifwq[1].addr) && ifwq[1].valid) ? ifwq[1].value :
                          q;
            VAL_T new_q = pe_ufixed_add_alu(q_fwd, incr, op);
            output_buffer[bank_addr] = new_q;
            ifwq[1] = ifwq[0];
            ifwq[0] = (IN_FLIGHT_WRITE){true, bank_addr, new_q};
        } else {
            ifwq[1] = ifwq[0];
            ifwq[0] = (IN_FLIGHT_WRITE){false, 0, 0};
        }

    }
}

//----------------------------------------------------------------
// pe output pipeline
//----------------------------------------------------------------
template<int id, unsigned bank_size, unsigned pack_size>
void ufixed_pe_output(
    hls::stream<VEC_PLD_T> &output,
    VAL_T output_buffer[bank_size],
    const unsigned used_buf_len
) {
    bool exit = false;
    unsigned dump_count = 0;
    pe_output_loop:
    for (unsigned dump_count = 0; dump_count < used_buf_len; dump_count++) {
        #pragma HLS pipeline II=1
        VAL_T q = output_buffer[dump_count];
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
// unsigned fixed-point pe
//----------------------------------------------------------------
template<int id, unsigned bank_size, unsigned pack_size>
void ufixed_pe(
    hls::stream<UPDATE_PLD_T> &input,
    hls::stream<VEC_PLD_T> &output,
    const unsigned used_buf_len,
    const VAL_T zero,
    const OP_T op
) {
    VAL_T output_buffer[bank_size];
    #pragma HLS bind_storage variable=output_buffer type=RAM_2P impl=URAM latency=3

    // reset output buffer before doing anything
    loop_reset_ob:
    for (unsigned i = 0; i < used_buf_len; i++) {
        #pragma HLS pipeline II=1
        output_buffer[i] = zero;
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
        ufixed_pe_process<id, bank_size, pack_size>(input, output_buffer, zero, op);

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
    ufixed_pe_output<id, bank_size, pack_size>(output, output_buffer, used_buf_len);
    output.write(VEC_PLD_EOD);
    output.write(VEC_PLD_EOS);
}

#endif  // SPMV_UFIXED_PE_H_
