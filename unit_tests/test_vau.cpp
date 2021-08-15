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
uint32_t partition_len = 32;
uint32_t bank_size = 128;
uint32_t num_partitions = 8;

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

void verify(std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> vector,
            std::vector<graphlily::idx_t, aligned_allocator<graphlily::idx_t>> kernel_results_idx,
            std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> kernel_results_val
) {
    float epsilon = 0.0001;
    ASSERT_EQ(kernel_results_idx.size(), kernel_results_val.size());
    for (size_t i = 0; i < kernel_results_idx.size(); i++) {
        bool match = abs(float(kernel_results_val[i] - vector[kernel_results_idx[i]])) < epsilon;
        if (!match) {
            std::cout << "Error: Result mismatch"
                      << std::endl;
            std::cout << "idx = " << kernel_results_idx[i]
                      << " Reference result = " << vector[kernel_results_idx[i]]
                      << " Kernel result = " << kernel_results_val[i]
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
    command = "cp " + graphlily::root_path + "/graphlily/hw/vecbuf_access_unit.h"
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
    command = "cp " + graphlily::root_path + "/tests/testbench/vau_tb.h"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);
    command = "cp " + graphlily::root_path + "/tests/testbench/vau_tb.cpp"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);

    // put configuration into vau_tb.h
    std::ofstream header_tb(graphlily::proj_folder_name + "/vau_tb.h", std::ios_base::app);
    header_tb << "const unsigned BANK_SIZE = " << bank_size << ";" << std::endl;
    header_tb << "#endif // GRAPHLILY_TEST_TESTBENCH_VAU_TB_H_" << std::endl;
    header_tb.close();

    // close the include guard in overlay.h
    // we do not use any configuration here, we just need the typedefs
    std::ofstream header_gll(graphlily::proj_folder_name + "/overlay.h", std::ios_base::app);
    header_gll << "#endif // GRAPHLILY_HW_OVERLAY_H_" << std::endl;
    header_gll.close();

    // generate vau_tb.ini
    std::ofstream ini(graphlily::proj_folder_name + "/vau_tb.ini");
    ini << "[connectivity]" << std::endl;
    ini << "sp=vau_tb_1.test_col_idx_gmem:DDR[0]" << std::endl;
    ini << "sp=vau_tb_1.test_vector_gmem:DDR[0]" << std::endl;
    ini << "sp=vau_tb_1.partition_table:DDR[0]" << std::endl;
    ini << "sp=vau_tb_1.result_idx:DDR[1]" << std::endl;
    ini << "sp=vau_tb_1.result_val:DDR[1]" << std::endl;
    ini.close();

    // generate makefile
    std::ofstream makefile(graphlily::proj_folder_name + "/makefile");
    std::string makefile_body;
    makefile_body += "LDCLFLAGS += --config vau_tb.ini\n";
    makefile_body += "KERNEL_OBJS += $(TEMP_DIR)/vau_tb.xo\n";
    makefile_body += "\n";
    makefile_body += "$(TEMP_DIR)/vau_tb.xo: vau_tb.cpp\n";
    makefile_body += "\tmkdir -p $(TEMP_DIR)\n";
    makefile_body += "\t$(VPP) $(CLFLAGS) --temp_dir $(TEMP_DIR) -c -k vau_tb -I'$(<D)' -o'$@' '$<'\n";
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
void _test_vau(
    std::vector<graphlily::idx_t, aligned_allocator<graphlily::idx_t>> &col_idxes,
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> &vector,
    std::vector<unsigned> &partition_table
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
    OCL_CHECK(err, kernel = cl::Kernel(program, "vau_tb", &err));

    cl::CommandQueue command_queue;
    OCL_CHECK(err, command_queue = cl::CommandQueue(context,
                                                    device,
                                                    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
                                                    &err));

    // prepare space for results
    std::vector<graphlily::idx_t, aligned_allocator<graphlily::idx_t>> kernel_result_idx;
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> kernel_result_val;
    kernel_result_idx.resize(partition_len * num_partitions);
    kernel_result_val.resize(partition_len * num_partitions);
    std::fill(kernel_result_idx.begin(), kernel_result_idx.end(), 0);
    std::fill(kernel_result_val.begin(), kernel_result_val.end(), 0);

    // allocate memory
    std::cout << "Allocating memory on device..." << std::endl;
    CL_CREATE_EXT_PTR(test_col_idx_gmem_ext, col_idxes.data(), graphlily::DDR[0]);
    CL_CREATE_EXT_PTR(test_vector_gmem_ext, vector.data(), graphlily::DDR[0]);
    CL_CREATE_EXT_PTR(partition_table_ext, partition_table.data(), graphlily::DDR[0]);
    CL_CREATE_EXT_PTR(result_idx_ext, kernel_result_idx.data(), graphlily::DDR[1]);
    CL_CREATE_EXT_PTR(result_val_ext, kernel_result_val.data(), graphlily::DDR[1]);

    cl::Buffer test_col_idx_gmem_buf;
    cl::Buffer test_vector_gmem_buf;
    cl::Buffer partition_table_buf;
    cl::Buffer result_idx_buf;
    cl::Buffer result_val_buf;

    OCL_CHECK(err, test_col_idx_gmem_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(graphlily::idx_t) * col_idxes.size(),
        &test_col_idx_gmem_ext,
        &err));

    OCL_CHECK(err, test_vector_gmem_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(graphlily::val_t) * vector.size(),
        &test_vector_gmem_ext,
        &err));

    OCL_CHECK(err, partition_table_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(unsigned) * partition_table.size(),
        &partition_table_ext,
        &err));

    OCL_CHECK(err, result_idx_buf = cl::Buffer(context,
        CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(graphlily::idx_t) * kernel_result_idx.size(),
        &result_idx_ext,
        &err));

    OCL_CHECK(err, result_val_buf = cl::Buffer(context,
        CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(graphlily::val_t) * kernel_result_val.size(),
        &result_val_ext,
        &err));

    // migrate data
    std::cout << "Moving data to device..." << std::endl;
    OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects(
        {test_col_idx_gmem_buf, test_vector_gmem_buf, partition_table_buf}, 0 /* 0 means from host*/));
    command_queue.finish();

    // set arguments
    OCL_CHECK(err, err = kernel.setArg(0, test_col_idx_gmem_buf));
    OCL_CHECK(err, err = kernel.setArg(1, test_vector_gmem_buf));
    OCL_CHECK(err, err = kernel.setArg(2, partition_table_buf));
    OCL_CHECK(err, err = kernel.setArg(3, result_idx_buf));
    OCL_CHECK(err, err = kernel.setArg(4, result_val_buf));
    OCL_CHECK(err, err = kernel.setArg(5, num_partitions));

    // launch kernel
    std::cout << "Invoking test bench..." << std::flush;
    OCL_CHECK(err, err = command_queue.enqueueTask(kernel));
    command_queue.finish();
    std::cout << " test bench finished successfully!" << std::endl;

    // collect results
    OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects(
        {result_idx_buf, result_val_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
    command_queue.finish();

    // verify
    verify(vector, kernel_result_idx, kernel_result_val);
}

//--------------------------------------------------------------------------------------------------
// Test cases
//--------------------------------------------------------------------------------------------------

void testcase_gen(
    std::vector<graphlily::idx_t, aligned_allocator<graphlily::idx_t>> &col_idxes,
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> &vector,
    std::vector<unsigned> &partition_table
) {
    unsigned vector_len = bank_size * num_partitions;
    vector.resize(vector_len);
    // std::cout << "Vector: " << std::endl;
    for (size_t i = 0; i < vector_len; i++) {
        vector[i] = rand() % 9 + 1;
        // std::cout << "  i = " << i << ": " << vector[i] << '\n';
    }

    unsigned partition_offset = 0;
    col_idxes.resize(partition_len * num_partitions);
    partition_table.resize(num_partitions);
    // std::cout << "Request Col idxes: " << std::endl;
    for (size_t i = 0; i < num_partitions; i++) {
        for (size_t j = 0; j < partition_len; j++) {
            col_idxes[i * partition_len + j] = rand() % bank_size + partition_offset;
            // std::cout << "  " << col_idxes[i * partition_len + j] << '\n';
        }
        partition_offset += bank_size;
        partition_table[i] = partition_len;
    }
}

TEST(Build, Synthesize) {
    synthesize_tb();
}

TEST(Basic, basic) {
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> vector;
    std::vector<graphlily::idx_t, aligned_allocator<graphlily::idx_t>> col_idxes;
    std::vector<unsigned> partition_table;
    testcase_gen(col_idxes, vector, partition_table);
    _test_vau(col_idxes, vector, partition_table);
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
