#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include <vector>
#include <ap_fixed.h>

#include "pe_tb.h"
#include "common.h"

#include "xcl2.hpp"

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

#define DISP_EXE_CMD(cmd)\
std::cout << cmd << std::endl;\
system(cmd.c_str());

#define CL_CREATE_EXT_PTR(name, data, channel)\
cl_mem_ext_ptr_t name;\
name.obj = data;\
name.param = 0;\
name.flags = channel;

std::string target = "hw_emu";

typedef struct {
    VAL_T mat_val;
    VAL_T vec_val;
    IDX_T row_idx;
} test_pld_t;


void compute_ref(std::vector<test_pld_t> &test_input,
                 std::vector<VAL_T> &ref_result_host
) {
    // output buffer
    VAL_T output_buffer[BANK_SIZE];

    // reset output buffer
    for (unsigned i = 0; i < BANK_SIZE; i++) {
        output_buffer[i] = 0;
    }

    // compute
    for (unsigned i = 0; i < TEST_LEN; i++) {
        VAL_T incr = test_input[i].mat_val * test_input[i].vec_val;
        VAL_T q = output_buffer[test_input[i].row_idx];
        VAL_T new_q = q + incr;
        output_buffer[test_input[i].row_idx] = new_q;
    }

    // write back to results
    for (unsigned i = 0; i < BANK_SIZE; i++) {
        ref_result_host[i] = output_buffer[i];
    }
}

bool verify(std::vector<VAL_T> reference_results,
            std::vector<VAL_T, aligned_allocator<VAL_T>> kernel_results) {
    float epsilon = 0.0001;
    if (reference_results.size() != kernel_results.size()) {
        std::cout << "Error: Size mismatch"
                      << std::endl;
        std::cout   << "  Reference result size: " << reference_results.size()
                    << "  Kernel result size: " << kernel_results.size()
                    << std::endl;
        return false;
    }
    for (size_t i = 0; i < BANK_SIZE; i++) {
        bool match = abs(float(kernel_results[i] - reference_results[i])) < epsilon;
        if (!match) {
            std::cout << "Error: Result mismatch"
                      << std::endl;
            std::cout << "i = " << i
                      << " Reference result = " << reference_results[i]
                      << " Kernel result = " << kernel_results[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

//--------------------------------------------------------------------------------------------------
// test harness
//--------------------------------------------------------------------------------------------------
bool _test_pe(
    std::vector<test_pld_t> &test_input
) {
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
    auto file_buf = xcl::read_binary_file("../unit_test_wrapper/pe_tb_build_dir." + target + "/pe_tb.xclbin");
    cl::Program::Binaries binaries{{file_buf.data(), file_buf.size()}};
    cl::Program program(context, {device}, binaries, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device with xclbin file\n";
    } else {
        std::cout << "Successfully programmed device with xclbin file\n";
    }
    cl::Kernel kernel;
    OCL_CHECK(err, kernel = cl::Kernel(program, "pe_tb", &err));

    cl::CommandQueue command_queue;
    OCL_CHECK(err, command_queue = cl::CommandQueue(context,
                                                    device,
                                                    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
                                                    &err));

    // prepare space for results
    std::vector<VAL_T, aligned_allocator<VAL_T>> kernel_result;
    kernel_result.resize(BANK_SIZE);
    std::fill(kernel_result.begin(), kernel_result.end(), 0);

    // allocate memory
    std::cout << "Allocating memory on device..." << std::endl;
    std::vector<IDX_T, aligned_allocator<IDX_T>> test_addr;
    std::vector<VAL_T, aligned_allocator<VAL_T>> test_mat;
    std::vector<VAL_T, aligned_allocator<VAL_T>> test_vec;
    test_addr.resize(test_input.size());
    test_mat.resize(test_input.size());
    test_vec.resize(test_input.size());
    for (size_t i = 0; i < test_input.size(); i++) {
        test_mat[i] = test_input[i].mat_val;
        test_vec[i] = test_input[i].vec_val;
        test_addr[i] = test_input[i].row_idx;
    }

    CL_CREATE_EXT_PTR(test_addr_ext, test_addr.data(), DDR[0]);
    CL_CREATE_EXT_PTR(test_mat_ext, test_mat.data(), DDR[0]);
    CL_CREATE_EXT_PTR(test_vec_ext, test_vec.data(), DDR[0]);
    CL_CREATE_EXT_PTR(kernel_result_ext, kernel_result.data(), DDR[1]);

    cl::Buffer test_addr_buf;
    cl::Buffer test_mat_buf;
    cl::Buffer test_vec_buf;
    cl::Buffer kernel_result_buf;

    OCL_CHECK(err, test_addr_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(IDX_T) * test_input.size(),
        &test_addr_ext,
        &err));

    OCL_CHECK(err, test_mat_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(VAL_T) * test_input.size(),
        &test_mat_ext,
        &err));

    OCL_CHECK(err, test_vec_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(VAL_T) * test_input.size(),
        &test_vec_ext,
        &err));

    OCL_CHECK(err, kernel_result_buf = cl::Buffer(context,
        CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(VAL_T) * kernel_result.size(),
        &kernel_result_ext,
        &err));

    // migrate data
    std::cout << "Moving data to device..." << std::endl;
    OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects(
        {test_addr_buf, test_mat_buf, test_vec_buf}, 0 /* 0 means from host*/));
    command_queue.finish();

    // set arguments
    OCL_CHECK(err, err = kernel.setArg(0, test_addr_buf));
    OCL_CHECK(err, err = kernel.setArg(1, test_mat_buf));
    OCL_CHECK(err, err = kernel.setArg(2, test_vec_buf));
    OCL_CHECK(err, err = kernel.setArg(3, kernel_result_buf));

    // launch kernel
    std::cout << "Invoking test bench..." << std::endl;
    OCL_CHECK(err, err = command_queue.enqueueTask(kernel));
    command_queue.finish();
    std::cout << " test bench finished successfully!" << std::endl;

    // collect results
    OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects({kernel_result_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
    command_queue.finish();

    // compute reference
    std::vector<VAL_T> ref_result;
    ref_result.resize(BANK_SIZE);
    std::fill(ref_result.begin(), ref_result.end(), 0);
    compute_ref(test_input, ref_result);

    // verify
    return verify(ref_result, kernel_result);
}

//--------------------------------------------------------------------------------------------------
// Test cases
//--------------------------------------------------------------------------------------------------

// generate payloads with a certain dependence distance.
// distance < 0 means no dependency.
void testcase_gen(
    int distance,
    std::vector<test_pld_t> &test_input
) {
    test_input.resize(TEST_LEN);
    for (size_t j = 0; j < TEST_LEN; j++) {
        if (distance < 0) {
            test_input[j].row_idx = j % BANK_SIZE;
        } else {
            test_input[j].row_idx = j % distance;
        }
        test_input[j].mat_val = 1;
        test_input[j].vec_val = 1;
    }
}

bool test_nodep() {
    std::cout << "------ Running test: no dependence " << std::endl;
    std::vector<test_pld_t> test_input;
    testcase_gen(-1, test_input);
    if (_test_pe(test_input)) {
        std::cout << "Test passed" << std::endl;
        return true;
    } else {
        std::cout << "Test Failed" << std::endl;
        return false;
    }
}

bool test_dep(int distance) {
    std::cout << "------ Running test: RAW distance " << distance << " " << std::endl;
    std::vector<test_pld_t> test_input;
    testcase_gen(distance, test_input);
    if (_test_pe(test_input)) {
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
    test_input.resize(TEST_LEN);
    for (size_t j = 0; j < TEST_LEN; j++) {
        test_input[j].mat_val = VAL_T((rand() % 9 + 1) / 10.0);
        test_input[j].vec_val = VAL_T((rand() % 9 + 1) / 10.0);
        test_input[j].row_idx = rand() % BANK_SIZE;
    }
    if (_test_pe(test_input)) {
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
    // passed = passed && test_nodep();
    passed = passed && test_dep(1);
    passed = passed && test_dep(2);
    passed = passed && test_dep(3);
    passed = passed && test_dep(4);
    passed = passed && test_dep(5);
    passed = passed && test_dep(6);
    passed = passed && test_dep(7);
    // passed = passed && test_dep(8);
    // passed = passed && test_dep(9);
    // passed = passed && test_dep(10);
    // passed = passed && test_dep(11);
    passed = passed && test_random();

    std::cout << (passed ? "===== All Test Passed! =====" : "===== Test FAILED! =====") << std::endl;
    return passed ? 0 : 1;
}
#pragma GCC diagnostic pop
