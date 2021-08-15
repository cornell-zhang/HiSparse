#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include <vector>
#include <ap_fixed.h>
#include <gtest/gtest.h>

#include "graphlily/global.h"

#define DISP_EXE_CMD(cmd)\
std::cout << cmd << std::endl;\
system(cmd.c_str());

#define CL_CREATE_EXT_PTR(name, data, channel)\
cl_mem_ext_ptr_t name;\
name.obj = data;\
name.param = 0;\
name.flags = channel;

std::string target = "hw_emu";
uint32_t test_len = 64;
uint32_t bank_size = 1024;
uint32_t num_PE = 8;

typedef struct {
    graphlily::val_t mat_val;
    graphlily::val_t vec_val;
    graphlily::idx_t row_idx;
} test_pld_t;

//--------------------------------------------------------------------------------------------------
// clean stuff
//--------------------------------------------------------------------------------------------------

void clean_proj_folder() {
    std::string command = "rm -rf ./" + graphlily::proj_folder_name;
    DISP_EXE_CMD(command);
}

//--------------------------------------------------------------------------------------------------
// reference and verify utils
//--------------------------------------------------------------------------------------------------

#define MULADD 0
#define ANDOR  1
#define ADDMIN 2

const graphlily::val_t MulAddZero = 0;
const graphlily::val_t AndOrZero  = 0;
const graphlily::val_t AddMinZero = 255;

graphlily::val_t mul_ref(
    graphlily::val_t a,
    graphlily::val_t b,
    const graphlily::val_t z,
    const char op
) {
    graphlily::val_t out;
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

#define MIN(a, b) ((a < b)? a : b)
graphlily::val_t add_ref(
    graphlily::val_t a,
    graphlily::val_t b,
    const char op
) {
    graphlily::val_t out;
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

void compute_ref(std::vector<test_pld_t> &test_input,
                 std::vector<graphlily::idx_t> &length_table,
                 std::vector<graphlily::val_t> &ref_result_host,
                 graphlily::val_t zero,
                 char op
) {
    // output buffer
    graphlily::val_t output_buffer[bank_size];

    // reset output buffer
    for (unsigned i = 0; i < bank_size; i++) {
        output_buffer[i] = zero;
    }

    // compute
    unsigned offset = 0;
    for (unsigned PEid = 0; PEid < num_PE; PEid++) {
        unsigned len = length_table[PEid];
        for (unsigned i = 0; i < len; i++) {
            graphlily::val_t incr = mul_ref(
                test_input[i + offset].mat_val,
                test_input[i + offset].vec_val,
                zero, op
            );
            graphlily::val_t q = output_buffer[test_input[i + offset].row_idx];
            graphlily::val_t new_q = add_ref(q, incr, op);
            output_buffer[test_input[i + offset].row_idx] = new_q;
        }
        offset += len;
    }

    // write back to results
    for (unsigned i = 0; i < bank_size; i++) {
        ref_result_host[i] = output_buffer[i];
    }
}

void verify(std::vector<graphlily::val_t> reference_results,
            std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> kernel_results) {
    float epsilon = 0.0001;
    ASSERT_EQ(reference_results.size(), kernel_results.size());
    for (size_t i = 0; i < bank_size; i++) {
        bool match = abs(float(kernel_results[i] - reference_results[i])) < epsilon;
        if (!match) {
            std::cout << "Error: Result mismatch"
                      << std::endl;
            std::cout << "i = " << i
                      << " Reference result = " << reference_results[i]
                      << " Kernel result = " << kernel_results[i]
                      << std::endl;
            ASSERT_TRUE(match);
        }
    }
}

//--------------------------------------------------------------------------------------------------
// synthesizer
//--------------------------------------------------------------------------------------------------
void synthesize_tb() {
    // create proj directory
    std::string command = "mkdir -p " + graphlily::proj_folder_name;
    DISP_EXE_CMD(command);

    // copy source code
    command = "cp " + graphlily::root_path + "/graphlily/hw/ufixed_pe_fwd.h"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);
    command = "cp " + graphlily::root_path + "/graphlily/hw/math_constants.h"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);
    command = "cp " + graphlily::root_path + "/graphlily/hw/util.h"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);
    command = "cp " + graphlily::root_path + "/graphlily/hw/overlay.h"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);
    command = "cp " + graphlily::root_path + "/tests/testbench/pe_cluster_tb.h"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);
    command = "cp " + graphlily::root_path + "/tests/testbench/pe_cluster_tb.cpp"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);

    // put configuration into pe_cluster_tb.h
    std::ofstream header_tb(graphlily::proj_folder_name + "/pe_cluster_tb.h", std::ios_base::app);
    header_tb << "const unsigned BANK_SIZE = " << bank_size << ";" << std::endl;
    header_tb << "#endif // GRAPHLILY_TEST_TESTBENCH_PE_CLUSTER_TB_H_" << std::endl;
    header_tb.close();

    // close the include guard in overlay.h
    // we do not use any configuration here, we just need the typedefs
    std::ofstream header_gll(graphlily::proj_folder_name + "/overlay.h", std::ios_base::app);
    header_gll << "#endif // GRAPHLILY_HW_OVERLAY_H_" << std::endl;
    header_gll.close();

    // generate pe_tb.ini
    std::ofstream ini(graphlily::proj_folder_name + "/pe_cluster_tb.ini");
    ini << "[connectivity]" << std::endl;
    ini << "sp=pe_cluster_tb_1.test_addr_gmem:DDR[0]" << std::endl;
    ini << "sp=pe_cluster_tb_1.test_mat_gmem:DDR[0]" << std::endl;
    ini << "sp=pe_cluster_tb_1.test_vec_gmem:DDR[0]" << std::endl;
    ini << "sp=pe_cluster_tb_1.length_table:DDR[0]" << std::endl;
    ini << "sp=pe_cluster_tb_1.result_gmem:DDR[1]" << std::endl;
    ini.close();

    // generate makefile
    std::ofstream makefile(graphlily::proj_folder_name + "/makefile");
    std::string makefile_body;
    makefile_body += "LDCLFLAGS += --config pe_cluster_tb.ini\n";
    makefile_body += "KERNEL_OBJS += $(TEMP_DIR)/pe_cluster_tb.xo\n";
    makefile_body += "\n";
    makefile_body += "$(TEMP_DIR)/pe_cluster_tb.xo: pe_cluster_tb.cpp\n";
    makefile_body += "\tmkdir -p $(TEMP_DIR)\n";
    makefile_body += "\t$(VPP) $(CLFLAGS) --temp_dir $(TEMP_DIR) -c -k pe_cluster_tb -I'$(<D)' -o'$@' '$<'\n";
    makefile_body += "\n";
    makefile << "TARGET := " << target << "\n" << std::endl;
    makefile << graphlily::makefile_prologue << makefile_body << graphlily::makefile_epilogue;
    makefile.close();

    // switch to build folder and build
    command = "cd " + graphlily::proj_folder_name + "; " + "make build";
    DISP_EXE_CMD(command);
    if (target == "sw_emu" || target == "hw_emu") {
        command = "cp " + graphlily::proj_folder_name + "/emconfig.json " + ".";
        DISP_EXE_CMD(command);
    }
}

//--------------------------------------------------------------------------------------------------
// test harness
//--------------------------------------------------------------------------------------------------
void _test_pe_cluster(
    std::vector<test_pld_t> &test_input,
    std::vector<graphlily::idx_t> &length_table,
    graphlily::val_t zero,
    char op
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
        if (devices[i].getInfo<CL_DEVICE_NAME>() == graphlily::device_name) {
            device = devices[i];
            found_device = true;
            break;
        }
    }
    if (!found_device) {
        std::cout << "Failed to find " << graphlily::device_name << ", exit!\n";
        exit(EXIT_FAILURE);
    }
    cl::Context context = cl::Context(device, NULL, NULL, NULL);
    auto file_buf = xcl::read_binary_file("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");
    cl::Program::Binaries binaries{{file_buf.data(), file_buf.size()}};
    cl::Program program(context, {device}, binaries, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device with xclbin file\n";
    } else {
        std::cout << "Successfully programmed device with xclbin file\n";
    }
    cl::Kernel kernel;
    OCL_CHECK(err, kernel = cl::Kernel(program, "pe_cluster_tb", &err));

    cl::CommandQueue command_queue;
    OCL_CHECK(err, command_queue = cl::CommandQueue(context,
                                                    device,
                                                    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
                                                    &err));

    // prepare space for results
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> kernel_result;
    kernel_result.resize(bank_size * num_PE);
    std::fill(kernel_result.begin(), kernel_result.end(), zero);

    // allocate memory
    std::vector<graphlily::idx_t, aligned_allocator<graphlily::idx_t>> test_addr;
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> test_mat;
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> test_vec;
    test_addr.resize(test_input.size());
    test_mat.resize(test_input.size());
    test_vec.resize(test_input.size());
    for (size_t i = 0; i < test_input.size(); i++) {
        test_mat[i] = test_input[i].mat_val;
        test_vec[i] = test_input[i].vec_val;
        test_addr[i] = test_input[i].row_idx;
    }

    CL_CREATE_EXT_PTR(test_addr_ext, test_addr.data(), graphlily::DDR[0]);
    CL_CREATE_EXT_PTR(test_mat_ext, test_mat.data(), graphlily::DDR[0]);
    CL_CREATE_EXT_PTR(test_vec_ext, test_vec.data(), graphlily::DDR[0]);
    CL_CREATE_EXT_PTR(length_table_ext, length_table.data(), graphlily::DDR[0]);
    CL_CREATE_EXT_PTR(kernel_result_ext, kernel_result.data(), graphlily::DDR[1]);

    cl::Buffer test_addr_buf;
    cl::Buffer test_mat_buf;
    cl::Buffer test_vec_buf;
    cl::Buffer length_table_buf;
    cl::Buffer kernel_result_buf;

    OCL_CHECK(err, test_addr_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(graphlily::idx_t) * test_input.size(),
        &test_addr_ext,
        &err));

    OCL_CHECK(err, test_mat_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(graphlily::val_t) * test_input.size(),
        &test_mat_ext,
        &err));

    OCL_CHECK(err, test_vec_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(graphlily::val_t) * test_input.size(),
        &test_vec_ext,
        &err));

    OCL_CHECK(err, length_table_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(graphlily::idx_t) * num_PE,
        &length_table_ext,
        &err));

    OCL_CHECK(err, kernel_result_buf = cl::Buffer(context,
        CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(graphlily::val_t) * kernel_result.size(),
        &kernel_result_ext,
        &err));

    // migrate data
    std::cout << "Moving data to device..." << std::endl;
    OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects(
        {test_addr_buf, test_mat_buf, test_vec_buf, length_table_buf}, 0 /* 0 means from host*/));
    command_queue.finish();

    // set arguments
    OCL_CHECK(err, err = kernel.setArg(0, test_addr_buf));
    OCL_CHECK(err, err = kernel.setArg(1, test_mat_buf));
    OCL_CHECK(err, err = kernel.setArg(2, test_vec_buf));
    OCL_CHECK(err, err = kernel.setArg(3, kernel_result_buf));
    OCL_CHECK(err, err = kernel.setArg(4, length_table_buf));
    OCL_CHECK(err, err = kernel.setArg(5, op));

    // launch kernel
    std::cout << "Invoking test bench..." << std::flush;
    OCL_CHECK(err, err = command_queue.enqueueTask(kernel));
    command_queue.finish();
    std::cout << " test bench finished successfully!" << std::endl;

    // collect results
    OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects({kernel_result_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
    command_queue.finish();

    // compute reference
    std::vector<graphlily::val_t> ref_result;
    ref_result.resize(bank_size * num_PE);
    std::fill(ref_result.begin(), ref_result.end(), zero);
    compute_ref(test_input, length_table, ref_result, zero, op);

    // verify
    verify(ref_result, kernel_result);
}

//--------------------------------------------------------------------------------------------------
// Test cases
//--------------------------------------------------------------------------------------------------

// generate payloads with a certain dependence distance.
// distance < 0 means no dependency.
void testcase_gen(
    int distance,
    std::vector<test_pld_t> &test_input,
    std::vector<graphlily::idx_t> &length_table
) {
    unsigned offset = 0;
    length_table.resize(num_PE);
    for (size_t PEid = 0; PEid < num_PE; PEid++) {

        graphlily::idx_t bank_address[test_len];
        graphlily::val_t mat_vals[test_len];
        graphlily::val_t vec_vals[test_len];

        for (size_t j = 0; j < test_len; j++) {
            if (distance < 0) {
                bank_address[j] = j % bank_size;
            } else {
                bank_address[j] = j % distance;
            }
            mat_vals[j] = 1;
            vec_vals[j] = 1;
        }

        test_input.resize(offset + test_len);
        for (size_t j = 0; j < test_len; j++) {
            test_input[j + offset].mat_val = mat_vals[j];
            test_input[j + offset].vec_val = vec_vals[j];
            test_input[j + offset].row_idx = bank_address[j];
        }
        length_table[PEid] = test_len;
        offset += test_len;
    }
}

TEST(Build, Synthesize) {
    synthesize_tb();
}

TEST(MulAdd, NoDep) {
    std::vector<test_pld_t> test_input;
    std::vector<graphlily::idx_t> length_table;
    testcase_gen(-1, test_input, length_table);
    _test_pe_cluster(test_input, length_table, MulAddZero, MULADD);
}

TEST(CleanUp, CleanProjDir) {
    clean_proj_folder();
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main(int argc, char ** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#pragma GCC diagnostic pop
