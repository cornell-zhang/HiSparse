#ifndef GRAPHLILY_HW_CONFIG_H_
#define GRAPHLILY_HW_CONFIG_H_

// !! Below kernel configurations will be overwritten by the compiler. For more
// details, please see the implementation in graphlily::synthesizer.
const unsigned SPMV_OUT_BUF_LEN = 1024;
const unsigned SPMSPV_OUT_BUF_LEN = 256 * 1024;
const unsigned SPMV_VEC_BUF_LEN = 256;
#define NUM_HBM_CHANNEL 16 // TODO: tag SpMV prefix to old NUM_HBM_CHANNEL

#ifdef USE_DDR_CHANNEL
#define SPMSPV_NUM_HBM_CHANNEL 1
#else
#define SPMSPV_NUM_HBM_CHANNEL 8
#endif

#endif  // GRAPHLILY_HW_CONFIG_H_
