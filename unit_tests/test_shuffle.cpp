#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include <vector>
#include <ap_fixed.h>
#include <ap_int.h>

#include "shuffle_tb.h"
#include "common.h"
#include "xcl2.hpp"

#define DISP_EXE_CMD(cmd)\
std::cout << cmd << std::endl;\
system(cmd.c_str());

// device memory channels
#define MAX_HBM_CHANNEL_COUNT 32
#define CHANNEL_NAME(n) n | XCL_MEM_TOPOLOGY
const int HBM[MAX_HBM_CHANNEL_COUNT] = {
    CHANNEL_NAME(0),  CHANNEL_NAME(1),  CHANNEL_NAME(2),  CHANNEL_NAME(3),  CHANNEL_NAME(4),
    CHANNEL_NAME(5),  CHANNEL_NAME(6),  CHANNEL_NAME(7),  CHANNEL_NAME(8),  CHANNEL_NAME(9),
    CHANNEL_NAME(10), CHANNEL_NAME(11), CHANNEL_NAME(12), CHANNEL_NAME(13), CHANNEL_NAME(14),
    CHANNEL_NAME(15), CHANNEL_NAME(16), CHANNEL_NAME(17), CHANNEL_NAME(18), CHANNEL_NAME(19),
    CHANNEL_NAME(20), CHANNEL_NAME(21), CHANNEL_NAME(22), CHANNEL_NAME(23), CHANNEL_NAME(24),
    CHANNEL_NAME(25), CHANNEL_NAME(26), CHANNEL_NAME(27), CHANNEL_NAME(28), CHANNEL_NAME(29),
    CHANNEL_NAME(30), CHANNEL_NAME(31)};

const int DDR[2] = {CHANNEL_NAME(32), CHANNEL_NAME(33)};


#define CL_CREATE_EXT_PTR(name, data, channel)\
cl_mem_ext_ptr_t name;\
name.obj = data;\
name.param = 0;\
name.flags = channel;

std::string target = "hw_emu";
const unsigned num_lanes = 8;

using test_pld_t = EDGE_PLD_T;

//--------------------------------------------------------------------------------------------------
// reference and verify utils
//--------------------------------------------------------------------------------------------------

void compute_ref(std::vector<test_pld_t> &test_input,
                 std::vector<test_pld_t> &ref_results
) {
    std::vector<test_pld_t> input_buckets[num_lanes];
    for (unsigned k = 0; k < num_lanes; k++) {
        input_buckets[k].clear();
    }
    // std::cout << "Compute ref: temporary input buffer allocated" << std::endl;

    // split to input buckets
    unsigned ILid = 0;
    for (auto p : test_input) {
        input_buckets[ILid].push_back(p);
        if (p.inst == EOS) {
            ILid++;
            if (ILid == num_lanes) {
                break;
            }
        }
    }

    // std::cout << "Compute ref: temporary input buffer initialized" << std::endl;

    std::vector<test_pld_t> output_buckets[num_lanes];
    for (unsigned k = 0; k < num_lanes; k++) {
        output_buckets[k].clear();
    }
    // std::cout << "Compute ref: temporary output buffer allocated & initialized" << std::endl;

    // do the actual shuffling
    unsigned ILidx[num_lanes];
    for (unsigned i = 0; i < num_lanes; i++) {
        ILidx[i] = 0;
    }
    int state = 0; // 0: idle, 1: working, 2: done
    bool exit = false;
    ap_uint<num_lanes> got_SOD = 0;
    ap_uint<num_lanes> got_EOD = 0;
    ap_uint<num_lanes> got_EOS = 0;
    while (!exit) {
        // std::cout << "State: " << state << ", "
        //           << "SOD: " << std::hex << got_SOD << std::dec << ", "
        //           << "EOD: " << std::hex << got_EOD << std::dec << ", "
        //           << "EOS: " << std::hex << got_EOS << std::dec << ", "
        //           << "Activities:" << std::endl;

        switch (state) {
        case 0: // idle: sync on SOD
            if (!got_SOD.and_reduce()) {
                for (unsigned ILid = 0; ILid < num_lanes; ILid++) {
                    if (!got_SOD[ILid]) {
                        test_pld_t p = input_buckets[ILid][ILidx[ILid]];
                        ILidx[ILid]++;
                        // std::cout << "  ILane " << ILid << " read: " << p << std::endl;
                        if (p.inst == SOD) {
                            got_SOD[ILid] = 1;
                        }
                    }
                }
            } else {
                for (unsigned OLid = 0; OLid < num_lanes; OLid++) {
                    output_buckets[OLid].push_back(test_pld_t({0,0,0,SOD}));
                    // std::cout << "  OLane " << OLid << " write: " << test_pld_t({0,0,0,SOD}) << std::endl;
                }
                got_SOD = 0;
                got_EOD = 0;
                got_EOS = 0;
                state = 1;
            }
            break;

        case 1: // working: sync on EOD
            if (!got_EOD.and_reduce()) {
                for (unsigned ILid = 0; ILid < num_lanes; ILid++) {
                    if (!got_EOD[ILid]) {
                        test_pld_t p = input_buckets[ILid][ILidx[ILid]];
                        ILidx[ILid]++;
                        // std::cout << "  ILane " << ILid << " read: " << p << std::endl;
                        if (p.inst == EOD) {
                            got_EOD[ILid] = 1;
                        } else {
                            output_buckets[p.col_idx % num_lanes].push_back(p);
                            // std::cout << "  OLane " << p.col_idx % num_lanes << " write: " << p << std::endl;
                        }
                    }
                }
            } else {
                for (unsigned OLid = 0; OLid < num_lanes; OLid++) {
                    output_buckets[OLid].push_back(test_pld_t({0,0,0,EOD}));
                    // std::cout << "  OLane " << OLid << " write: " << test_pld_t({0,0,0,EOD}) << std::endl;
                }
                got_SOD = 0;
                got_EOD = 0;
                got_EOS = 0;
                state = 2;
            }
            break;

        case 2: // done: sync on EOS/SOD
            if (!(got_SOD.and_reduce() || got_EOS.and_reduce())) {
                for (unsigned ILid = 0; ILid < num_lanes; ILid++) {
                    if (!(got_SOD[ILid] || got_EOS[ILid])) {
                        test_pld_t p = input_buckets[ILid][ILidx[ILid]];
                        // std::cout << "  ILane " << ILid << " read: " << p << std::endl;
                        ILidx[ILid]++;
                        if (p.inst == SOD) {
                            got_SOD[ILid] = 1;
                        }
                        if (p.inst == EOS) {
                            got_EOS[ILid] = 1;
                        }
                    }
                }
            } else if (got_SOD.and_reduce()) {
                for (unsigned OLid = 0; OLid < num_lanes; OLid++) {
                    output_buckets[OLid].push_back(test_pld_t({0,0,0,SOD}));
                    // std::cout << "  OLane " << OLid << " write: " << test_pld_t({0,0,0,SOD}) << std::endl;
                }
                got_SOD = 0;
                got_EOD = 0;
                got_EOS = 0;
                state = 1;
            } else if (got_EOS.and_reduce()) {
                for (unsigned OLid = 0; OLid < num_lanes; OLid++) {
                    output_buckets[OLid].push_back(test_pld_t({0,0,0,EOS}));
                    // std::cout << "  OLane " << OLid << " write: " << test_pld_t({0,0,0,EOS}) << std::endl;
                }
                exit = true;
            }
            break;

        default:
            break;
        } // switch (state)
    } // while (!exit)
    // std::cout << "Compute ref: main processing logic finished" << std::endl;

    // merge output buckets
    unsigned offset = 0;
    for (unsigned OLid = 0; OLid < num_lanes; OLid++) {
        // std::cout << "output_buckets[" << OLid << "]:\n" << output_buckets[OLid] << std::endl;
        // std::cout << "output_buckets[" << OLid << "] size: " << output_buckets[OLid].size() << std::endl;
        for (unsigned i = 0; i < output_buckets[OLid].size(); i++) {
            test_pld_t p = output_buckets[OLid][i];
            ref_results[i + offset] = p;
        }
        offset += output_buckets[OLid].size();
    }
}

bool find_payload(std::vector<test_pld_t> a, bool* checkout, test_pld_t p) {
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].mat_val == p.mat_val && !checkout[i]) {
            checkout[i] = true;
            return true;
        }
    }
    return false;
}

bool and_reduction(bool* arr, unsigned size) {
    for (size_t i = 0; i < size; i++) {
        if (!arr[i]) { return false; }
    }
    return true;
}

bool stream_match_ooo(std::vector<test_pld_t> ref, std::vector<test_pld_t> src) {
    if (ref.size() != src.size()) {
        std::cout << __FILE__ << ":" << __LINE__ << ": Error" << std::endl;
        std::cout << "  stream size mismatch!\n"
                  << "  ref = " << ref.size() << "\n"
                  << "  res = " << src.size() << "\n";
        return false;
    }
    unsigned size = ref.size();
    bool checkout[size];
    for (size_t i = 0; i < size; i++) {
        checkout[i] = false;
    }

    for (size_t i = 0; i < size; i++) {
        bool found = find_payload(ref, checkout, src[i]);
        if (!found) { return false; }
    }

    bool all_checkout = and_reduction(checkout, size);
    if (!all_checkout) { return false; }

    return true;
}

bool verify(std::vector<test_pld_t> ref_results,
            std::vector<test_pld_t> kernel_results) {
    if (ref_results.size() != kernel_results.size()) {
        std::cout << "Error: Size mismatch"
                      << std::endl;
        std::cout   << "  Reference result size: " << ref_results.size()
                    << "  Kernel result size: " << kernel_results.size()
                    << std::endl;
        return false;
    }
    unsigned ref_idx = 0;
    unsigned res_idx = 0;
    bool passing = true;
    for (size_t i = 0; i < num_lanes; i++) {
        std::vector<test_pld_t> ref_stream;
        std::vector<test_pld_t> res_stream;
        bool ref_eos = false;
        bool res_eos = false;
        ref_stream.clear();
        res_stream.clear();
        while (!(ref_eos && res_eos)) {
            if (!ref_eos) {
                test_pld_t rp = ref_results[ref_idx];
                ref_idx++;
                ref_eos = (rp.inst == EOS);
                if (rp.inst != EOD && rp.inst != SOD && rp.inst != EOS) {
                    ref_stream.push_back(rp);
                }
            }
            if (!res_eos) {
                test_pld_t sp = kernel_results[res_idx];
                res_idx++;
                res_eos = (sp.inst == EOS);
                if (sp.inst != EOD && sp.inst != SOD && sp.inst != EOS) {
                    res_stream.push_back(sp);
                }
            }
        }
        // std::cout << "Ref: \n" << ref_stream;
        // std::cout << "Res: \n" << res_stream;

        std::cout << "Out Lane: " << i << std::flush;
        bool match = stream_match_ooo(ref_stream, res_stream);
        std::cout << (match ? "  passed." : "  failed!") << std::endl;
        passing = passing && match;
    }
    if (!passing) {
        return false;
    }
    return true;
}

//--------------------------------------------------------------------------------------------------
// test harness
//--------------------------------------------------------------------------------------------------
bool _test_shuffle(std::vector<test_pld_t> &test_input) {
    // set up runtime
    cl_int err;
    if (target == "sw_emu" || target == "hw_emu") {
        setenv("XCL_EMULATION_MODE", target.c_str(), true);
    }
    cl::Device device;
    bool found_device = false;
    auto devices = xcl::get_xil_devices();
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i].getInfo<CL_DEVICE_NAME>() == "xilinx_u280_xdma_201920_3") {
            device = devices[i];
            found_device = true;
            break;
        }
    }
    if (!found_device) {
        std::cout << "Failed to find " << "xilinx_u280_xdma_201920_3" << ", exit!\n";
        exit(EXIT_FAILURE);
    }
    cl::Context context = cl::Context(device, NULL, NULL, NULL);
    auto file_buf = xcl::read_binary_file("../unit_test_wrapper/shuffle_tb_build_dir." + target + "/shuffle_tb.xclbin");
    cl::Program::Binaries binaries{{file_buf.data(), file_buf.size()}};
    cl::Program program(context, {device}, binaries, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device with xclbin file\n";
    } else {
        std::cout << "Successfully programmed device with xclbin file\n";
    }
    cl::Kernel kernel;
    OCL_CHECK(err, kernel = cl::Kernel(program, "shuffle_tb", &err));

    cl::CommandQueue command_queue;
    OCL_CHECK(err, command_queue = cl::CommandQueue(context,
                                                    device,
                                                    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
                                                    &err));

    // prepare space for results
    std::vector<test_pld_t> kernel_result;
    kernel_result.resize(BUF_SIZE * num_lanes);
    std::fill(kernel_result.begin(), kernel_result.end(), test_pld_t({0,0,0,0}));

    // allocate memory

    CL_CREATE_EXT_PTR(test_input_ext, test_input.data(), DDR[0]);
    CL_CREATE_EXT_PTR(kernel_result_ext, kernel_result.data(), DDR[0]);

    cl::Buffer test_input_buf;
    cl::Buffer kernel_result_buf;

    OCL_CHECK(err, test_input_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(test_pld_t) * test_input.size(),
        &test_input_ext,
        &err));

    OCL_CHECK(err, kernel_result_buf = cl::Buffer(context,
        CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(test_pld_t) * kernel_result.size(),
        &kernel_result_ext,
        &err));

    // migrate data
    std::cout << "Moving data to device..." << std::endl;
    OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects(
        {test_input_buf}, 0 /* 0 means from host*/));
    command_queue.finish();

    // set arguments
    OCL_CHECK(err, err = kernel.setArg(0, test_input_buf));
    OCL_CHECK(err, err = kernel.setArg(1, kernel_result_buf));

    // launch kernel
    std::cout << "Invoking test bench..." << std::endl;
    OCL_CHECK(err, err = command_queue.enqueueTask(kernel));
    command_queue.finish();
    std::cout << "Test bench finished successfully!" << std::endl;

    // collect results
    OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects({kernel_result_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
    command_queue.finish();

    // std::cout << kernel_result << std::endl;

    // compute reference
    std::vector<test_pld_t> ref_result;
    ref_result.resize(BUF_SIZE * num_lanes);
    std::fill(ref_result.begin(), ref_result.end(), test_pld_t({0,0,0,0}));
    compute_ref(test_input, ref_result);
    std::cout << "Reference Compute Success!" << std::endl;


    // std::cout << "REF ALL:\n" << ref_result;
    // std::cout << "RES ALL:\n" << kernel_result;
    // verify
    return verify(ref_result, kernel_result);
}

//--------------------------------------------------------------------------------------------------
// Test cases
//--------------------------------------------------------------------------------------------------

// uniform traffic
void uniform_gen(
    std::vector<test_pld_t> &test_input,
    unsigned len,
    unsigned parts,
    unsigned rotate
) {
    // the input buffer size is 1024
    assert(1 + (len + 2) * parts < BUF_SIZE);
    test_input.resize(BUF_SIZE * num_lanes);
    std::vector<test_pld_t> input_buckets[num_lanes];
    for (unsigned ILid = 0; ILid < num_lanes; ILid++) {
        input_buckets[ILid].resize(1 + (len + 2) * parts);
    }

    for (unsigned ILid = 0; ILid < num_lanes; ILid++) {
        for (unsigned part_id = 0; part_id < parts; part_id++) {
            unsigned offset = part_id * (len + 2);
            input_buckets[ILid][offset] = test_pld_t({0,0,0,SOD});
            for (unsigned i = 0; i < len; i++) {
                input_buckets[ILid][offset + i + 1]
                    = test_pld_t({
                        0,
                        (ILid + rotate) % num_lanes + (i + offset) * num_lanes,
                        (ILid + rotate) % num_lanes + (i + offset) * num_lanes,
                        0
                    });
            }
            input_buckets[ILid][offset + len + 1] = test_pld_t({0,0,0,EOD});
        }
        input_buckets[ILid][(len + 2) * parts] = test_pld_t({0,0,0,EOS});
    }

    unsigned ti_idx = 0;
    for (unsigned ILid = 0; ILid < num_lanes; ILid++) {
        bool eos = false;
        unsigned i = 0;
        while (!eos) {
            test_pld_t p = input_buckets[ILid][i];
            test_input[ti_idx] = p;
            i++;
            ti_idx++;
            eos = (p.inst == EOS);
        }
    }
}

// conflict traffic
void conflict_gen(
    std::vector<test_pld_t> &test_input,
    unsigned len,
    unsigned parts,
    unsigned num_olanes
) {
    // the input buffer size is 1024
    assert(1 + (len + 2) * parts < BUF_SIZE);
    test_input.resize(BUF_SIZE * num_lanes);
    std::vector<test_pld_t> input_buckets[num_lanes];
    for (unsigned ILid = 0; ILid < num_lanes; ILid++) {
        input_buckets[ILid].resize(1 + (len + 2) * parts);
    }
    unsigned UUid = 1;
    for (unsigned ILid = 0; ILid < num_lanes; ILid++) {
        for (unsigned part_id = 0; part_id < parts; part_id++) {
            unsigned offset = part_id * (len + 2);
            input_buckets[ILid][offset] = test_pld_t({0,0,0,SOD});
            for (unsigned i = 0; i < len; i++) {
                input_buckets[ILid][offset + i + 1]
                    = test_pld_t({
                        0,
                        UUid,
                        ILid % num_olanes + (i + offset) * num_lanes,
                        0
                    });
                UUid++;
            }
            input_buckets[ILid][offset + len + 1] = test_pld_t({0,0,0,EOD});
        }
        input_buckets[ILid][(len + 2) * parts] = test_pld_t({0,0,0,EOS});
    }

    unsigned ti_idx = 0;
    for (unsigned ILid = 0; ILid < num_lanes; ILid++) {
        bool eos = false;
        unsigned i = 0;
        while (!eos) {
            test_pld_t p = input_buckets[ILid][i];
            test_input[ti_idx] = p;
            i++;
            ti_idx++;
            eos = (p.inst == EOS);
        }
    }
}


// random traffic
void random_gen(
    std::vector<test_pld_t> &test_input,
    unsigned avg_len,
    unsigned parts
) {
    unsigned len_var = avg_len / 2;
    // the input buffer size is 1024
    assert(1 + (avg_len + len_var + 2) * parts < BUF_SIZE);
    test_input.resize(BUF_SIZE * num_lanes);
    std::vector<test_pld_t> input_buckets[num_lanes];
    unsigned len_table[num_lanes];
    for (unsigned ILid = 0; ILid < num_lanes; ILid++) {
        int sign = (rand() % 2) ? 1 : -1;
        len_table[ILid] = avg_len + sign * (rand() % len_var);
        input_buckets[ILid].resize(1 + (len_table[ILid] + 2) * parts);
    }

    for (unsigned ILid = 0; ILid < num_lanes; ILid++) {
        for (unsigned part_id = 0; part_id < parts; part_id++) {
            unsigned offset = part_id * (len_table[ILid] + 2);
            input_buckets[ILid][offset] = test_pld_t({0,0,0,SOD});
            for (unsigned i = 0; i < len_table[ILid]; i++) {
                input_buckets[ILid][offset + i + 1]
                    = test_pld_t({
                        0,
                        rand() % num_lanes + (i + offset) * num_lanes,
                        rand() % num_lanes + (i + offset) * num_lanes,
                        0
                    });
            }
            input_buckets[ILid][offset + len_table[ILid] + 1] = test_pld_t({0,0,0,EOD});
        }
        input_buckets[ILid][(len_table[ILid] + 2) * parts] = test_pld_t({0,0,0,EOS});
    }

    unsigned ti_idx = 0;
    for (unsigned ILid = 0; ILid < num_lanes; ILid++) {
        bool eos = false;
        unsigned i = 0;
        while (!eos) {
            test_pld_t p = input_buckets[ILid][i];
            test_input[ti_idx] = p;
            i++;
            ti_idx++;
            eos = (p.inst == EOS);
        }
    }
}


bool test_unifrom_direct() {
    std::cout << "------ Running test: uniform direct " << std::endl;
    std::vector<test_pld_t> test_input;
    uniform_gen(test_input, 8, 4, 0);
    if (_test_shuffle(test_input)) {
        std::cout << "Test passed" << std::endl;
        return true;
    } else {
        std::cout << "Test Failed" << std::endl;
        return false;
    }
}

bool test_uniform_rotated(unsigned rotate) {
    std::cout << "------ Running test: uniform rotate " << rotate << " " << std::endl;
    std::vector<test_pld_t> test_input;
    uniform_gen(test_input, 16, 4, rotate);
    if (_test_shuffle(test_input)) {
        std::cout << "Test passed" << std::endl;
        return true;
    } else {
        std::cout << "Test Failed" << std::endl;
        return false;
    }
}

bool test_conflict(unsigned out_lanes) {
    std::cout << "------ Running test: conflict " << out_lanes << " out lanes" << std::endl;
    std::vector<test_pld_t> test_input;
    conflict_gen(test_input, 16, 4, out_lanes);
    if (_test_shuffle(test_input)) {
        std::cout << "Test passed" << std::endl;
        return true;
    } else {
        std::cout << "Test Failed" << std::endl;
        return false;
    }
}

bool test_random() {
    std::cout << "------ Running test: random " << std::endl;
    std::vector<test_pld_t> test_input;
    random_gen(test_input, 64, 4);
    if (_test_shuffle(test_input)) {
        std::cout << "Test passed" << std::endl;
        return true;
    } else {
        std::cout << "Test Failed" << std::endl;
        return false;
    }
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main(int argc, char ** argv) {
    bool passed = true;
    passed = passed && test_unifrom_direct();
    passed = passed && test_uniform_rotated(1);
    passed = passed && test_uniform_rotated(2);
    passed = passed && test_uniform_rotated(5);
    passed = passed && test_uniform_rotated(7);
    passed = passed && test_conflict(1);
    passed = passed && test_conflict(2);
    passed = passed && test_conflict(4);
    passed = passed && test_conflict(7);
    passed = passed && test_random();

    std::cout << (passed ? "===== All Test Passed! =====" : "===== Test FAILED! =====") << std::endl;
    return passed ? 0 : 1;
}

#pragma GCC diagnostic pop
