#ifndef SPMV_MATH_CONSTANTS_H_
#define SPMV_MATH_CONSTANTS_H_

#include "ap_fixed.h"

const unsigned UINT_INF = 0xffffffff;
const ap_ufixed<32, 8, AP_RND, AP_SAT> UFIXED_INF = 255;
const float FLOAT_INF = 999999999;

#endif  // SPMV_MATH_CONSTANTS_H_
