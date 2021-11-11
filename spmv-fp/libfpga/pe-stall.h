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
struct IN_FLIGHT_WRITE {
    bool valid;
    IDX_T addr;
    VAL_T value;
};

struct IN_FLIGHT_ADD {
    bool valid;
    IDX_T addr;
};

void float_pe_mul(
   hls::stream<UPDATE_PLD_T> &input,
   hls::stream<VEC_PLD_T> &output
) {
    bool exit = false;
    while (!exit) {
        #pragma HLS pipeline II=1
        UPDATE_PLD_T pld = input.read();
        if (pld.inst == EOD) {
            exit = true;
            output.write(VEC_PLD_EOD);
        } else {
            exit = false;
            VEC_PLD_T po;
            po.val = pld.mat_val * pld.vec_val;
            po.idx = pld.row_idx;
            po.inst = 0;
            output.write(po);
        }
    }
}


template<unsigned bank_size, unsigned pack_size>
void float_pe_acc(
    hls::stream<VEC_PLD_T> &input,
    VAL_T output_buffer[bank_size]
) {
    bool exit = false;

    // in-flight add queue for floating add stalling
    // designed for URAM latnecy=2 (RDL=2, WRL=2), FPADD latnecy=4 (FAL=4)
    IN_FLIGHT_ADD ifaq[8];
    #pragma HLS array_partition variable=ifaq complete
    ifaq[0] = (IN_FLIGHT_ADD){false, 0};
    ifaq[1] = (IN_FLIGHT_ADD){false, 0};
    ifaq[2] = (IN_FLIGHT_ADD){false, 0};
    ifaq[3] = (IN_FLIGHT_ADD){false, 0};
    ifaq[4] = (IN_FLIGHT_ADD){false, 0};
    ifaq[5] = (IN_FLIGHT_ADD){false, 0};
    ifaq[6] = (IN_FLIGHT_ADD){false, 0};
    ifaq[7] = (IN_FLIGHT_ADD){false, 0};
    // ifaq[8] = (IN_FLIGHT_ADD){false, 0};

    bool stall = false;
    VEC_PLD_T pld = (VEC_PLD_T){0,0,0};

    pe_process_loop:
    while (!exit) {
        #pragma HLS pipeline II=1
        #pragma HLS dependence variable=output_buffer inter false
        #pragma HLS dependence variable=ifaq intra true

        bool valid;
        if (!stall) {
            pld = input.read();
            if (pld.inst == EOD) {
                valid = false;
            } else {
                valid = true;
            }
        } else {
            pld = pld;
            valid = true;
        }
        if (pld.inst == EOD) {
            exit = true;
        } else {
            exit = false;
        }

#ifdef PE_LINE_TRACING
        std::cout << "  input payload: " << pld << std::endl;
#endif

        IDX_T bank_addr = pld.idx / pack_size;
        VAL_T incr = pld.val;
        VAL_T q = output_buffer[bank_addr];

        // check stall
        stall = valid && (
                    ((bank_addr == ifaq[0].addr) && ifaq[0].valid) ||
                    ((bank_addr == ifaq[1].addr) && ifaq[1].valid) ||
                    ((bank_addr == ifaq[2].addr) && ifaq[2].valid) ||
                    ((bank_addr == ifaq[3].addr) && ifaq[3].valid) ||
                    ((bank_addr == ifaq[4].addr) && ifaq[4].valid) ||
                    ((bank_addr == ifaq[5].addr) && ifaq[5].valid) ||
                    ((bank_addr == ifaq[6].addr) && ifaq[6].valid) ||
                    ((bank_addr == ifaq[7].addr) && ifaq[7].valid)
                    // ((bank_addr == ifaq[8].addr) && ifaq[8].valid)
                );

        // update ifaq
        for(unsigned i = 7; i > 0; i--) {
            ifaq[i] = ifaq[i - 1];
        }
        ifaq[0] = (valid && !stall) ? (IN_FLIGHT_ADD){true, bank_addr} : (IN_FLIGHT_ADD){false, 0};

        if (valid && !stall) {
            VAL_T new_q = q + incr;
            #pragma HLS bind_op variable=new_q op=fadd impl=fulldsp latency=4
            output_buffer[bank_addr] = new_q;
        }
    } // while()
}

template<int id, unsigned bank_size, unsigned pack_size>
void float_pe_process(
    hls::stream<UPDATE_PLD_T> &input,
    VAL_T output_buffer[bank_size]
) {
    hls::stream<VEC_PLD_T> incr_stream;
    #pragma HLS stream variable=incr_stream depth=4
    #pragma HLS dataflow
    float_pe_mul(input, incr_stream);
    float_pe_acc<bank_size, pack_size>(incr_stream, output_buffer);
}

//----------------------------------------------------------------
// pe output pipeline
//----------------------------------------------------------------
template<int id, unsigned bank_size, unsigned pack_size>
void float_pe_output(
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
// floating-point pe
//----------------------------------------------------------------
template<int id, unsigned bank_size, unsigned pack_size>
void pe(
    hls::stream<UPDATE_PLD_T> &input,
    hls::stream<VEC_PLD_T> &output,
    const unsigned used_buf_len
) {
    VAL_T output_buffer[bank_size];
    #pragma HLS bind_storage variable=output_buffer type=RAM_2P impl=URAM latency=2

    // reset output buffer before doing anything
    loop_reset_ob:
    for (unsigned i = 0; i < used_buf_len; i++) {
        #pragma HLS pipeline II=1
        output_buffer[i] = 0;
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
