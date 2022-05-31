#ifndef GRAPHLILY_HW_UTIL_H_
#define GRAPHLILY_HW_UTIL_H_


template<typename T, unsigned len>
void array_shift_left(T array[len], T array_dest[len], unsigned rotate) {
    #pragma HLS inline
    // #pragma HLS latency min=0 max=0
    #pragma HLS array_partition variable=array complete
    #pragma HLS array_partition variable=array_dest complete
    for (unsigned i = 0; i < len; i++) {
        #pragma HLS unroll
        array_dest[i] = array[(i + rotate) % len];
    }
}


template<typename T, unsigned len, unsigned maximum>
void array_cyclic_add(T array[len], bool array_valid[len], unsigned inc) {
    #pragma HLS inline
    // #pragma HLS latency min=0 max=0
    #pragma HLS array_partition variable=array complete
    for (unsigned i = 0; i < len; i++) {
        #pragma HLS unroll
        if (array_valid[i]) {
            array[i] = (array[i] + inc) % maximum;
        }
    }
}


template<unsigned len>
bool array_and_reduction(bool array[len]) {
    #pragma HLS inline
    #pragma HLS expression_balance
    bool result = true;
    for (unsigned i = 0; i < len; i++) {
        #pragma HLS unroll
        result = result && array[i];
    }
    return result;
}


template<unsigned len>
bool array_or_reduction(bool array[len]) {
    #pragma HLS inline
    #pragma HLS expression_balance
    bool result = false;
    for (unsigned i = 0; i < len; i++) {
        #pragma HLS unroll
        result = result || array[i];
    }
    return result;
}


template<unsigned len>
unsigned array_popcount(bool array[len]) {
    #pragma HLS pipeline II = 1
    #pragma HLS latency min=1 max=1
    #pragma HLS array_partition variable=array complete
    #pragma HLS expression_balance
    unsigned cnt = 0;
    for (unsigned i = 0; i < len; i++) {
        #pragma HLS unroll
        if (array[i]) {
            cnt++;
        }
    }
    return cnt;
}


template<typename T, unsigned len>
T array_sum(T array[len]) {
    #pragma HLS inline
    #pragma HLS expression_balance
    T result = 0;
    for (unsigned i = 0; i < len; i++) {
        #pragma HLS unroll
        result += array[i];
    }
    return result;
}


template<typename T, unsigned len>
T array_max(T array[len]) {
    #pragma HLS inline
    #pragma HLS expression_balance
    T result = 0;
    for (unsigned i = 0; i < len; i++) {
        #pragma HLS unroll
        result = (array[i] > result)? array[i] : result;
    }
    return result;
}


// force a register
template<typename T>
T HLS_REG(T in) {
#pragma HLS pipeline
#pragma HLS inline off
#pragma HLS interface port=return register
    return in;
}


// // Cyclic partitioning
// unsigned get_bank_idx(unsigned full_addr) {
//     return full_addr & BANK_ID_MASK;
// }


// // Cyclic partitioning
// unsigned get_bank_address(unsigned full_addr) {
//     return full_addr >> BANK_ID_NBITS;
// }

#endif  // GRAPHLILY_HW_UTIL_H_
