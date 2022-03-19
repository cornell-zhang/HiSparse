#include <hls_stream.h>
#include <ap_int.h>

#include "common.h"

extern "C" {
void k2k_relay(
    hls::stream<VEC_AXIS_IF_T> &in,                      // in
    hls::stream<VEC_AXIS_IF_T> &out                      // out
) {
    #pragma HLS interface ap_ctrl_none port=return

    #pragma HLS interface axis register both port=in
    #pragma HLS interface axis register both port=out

    while (1) {
        #pragma HLS pipeline II=1
        VEC_AXIS_IF_T pkt = in.read();
        out.write(pkt);
    }

} // kernel
} // extern "C"
