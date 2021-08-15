#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include <vector>
#include <ap_fixed.h>
#include <gtest/gtest.h>

#include "graphlily/global.h"
#include "graphlily/io/data_loader.h"
#include "graphlily/io/data_formatter.h"

#define DISP_EXE_CMD(cmd)\
std::cout << cmd << std::endl;\
system(cmd.c_str());

#define CL_CREATE_EXT_PTR(name, data, channel)\
cl_mem_ext_ptr_t name;\
name.obj = data;\
name.param = 0;\
name.flags = channel;

std::string target = "sw_emu";
// change this according to the kernel! (in overlay.h)
const unsigned PACK_SIZE = 8;
const unsigned OB_BANK_SIZE = 1024 * 8;
const unsigned OB_PER_CLUSTER = OB_BANK_SIZE * PACK_SIZE;
const unsigned SK0_CLUSTER = 4;
const unsigned SK1_CLUSTER = 6;
const unsigned SK2_CLUSTER = 6;
const unsigned VB_BANK_SIZE = 1024 * 3;
const unsigned VB_PER_CLUSTER = VB_BANK_SIZE * PACK_SIZE;

const unsigned LOGICAL_OB_SIZE = (SK0_CLUSTER + SK1_CLUSTER + SK2_CLUSTER) * OB_PER_CLUSTER;
const unsigned LOGICAL_VB_SIZE = VB_PER_CLUSTER;

using aligned_dv_t = std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>>;
using aligned_fdv_t = std::vector<float, aligned_allocator<float>>;

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

// only support MULADD now!
void compute_ref(
    graphlily::io::CSRMatrix<float> &mat,
    aligned_fdv_t &vector,
    aligned_fdv_t &ref_result
) {
    ref_result.resize(mat.num_rows);
    std::fill(ref_result.begin(), ref_result.end(), 0);
    for (size_t row_idx = 0; row_idx < mat.num_rows; row_idx++) {
        graphlily::idx_t start = mat.adj_indptr[row_idx];
        graphlily::idx_t end = mat.adj_indptr[row_idx + 1];
        for (size_t i = start; i < end; i++) {
            graphlily::idx_t idx = mat.adj_indices[i];
            ref_result[row_idx] += mat.adj_data[i] * vector[idx];
        }
    }
}

void verify(aligned_fdv_t reference_results,
            aligned_dv_t kernel_results) {
    float epsilon = 0.0001;
    ASSERT_EQ(reference_results.size(), kernel_results.size());
    for (size_t i = 0; i < reference_results.size(); i++) {
        bool match = abs(float(kernel_results[i]) - reference_results[i]) < epsilon;
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

    // copy design files
    std::vector<std::string> design_files = {
        "hw/shuffle.h",
        "hw/vecbuf_access_unit.h",
        "hw/ufixed_pe_fwd.h",
        "hw/spmv_cluster.h",
        "hw/stream_utils.h",
        "hw/math_constants.h",
        "hw/util.h",
        "hw/overlay.h",
        "hw/k2k_relay.cpp",
        "hw/spmv_result_drain.cpp",
        "hw/spmv_vector_loader.cpp",
        "hw/spmv_sk0.cpp",
        "hw/spmv_sk1.cpp",
        "hw/spmv_sk2.cpp",
    };

    for (auto file : design_files) {
        command = "cp " + graphlily::root_path + "/graphlily/" + file
                        + " " + graphlily::proj_folder_name + "/";
        DISP_EXE_CMD(command);
    }

    // close the include guard in overlay.h
    // std::ofstream header_gll(graphlily::proj_folder_name + "/overlay.h", std::ios_base::app);
    // header_gll << "#endif // GRAPHLILY_HW_OVERLAY_H_" << std::endl;
    // header_gll.close();

    // generate spmv.ini
    std::ofstream ini(graphlily::proj_folder_name + "/spmv.ini");
    ini << "[connectivity]" << std::endl;
    ini << "nk=spmv_sk0:1:SK0" << std::endl;
    ini << "nk=spmv_sk1:1:SK1" << std::endl;
    ini << "nk=spmv_sk2:1:SK2" << std::endl;
    ini << "nk=spmv_vector_loader:1:VL" << std::endl;
    ini << "nk=spmv_result_drain:1:RD" << std::endl;
    ini << "nk=k2k_relay:2:relay_SK2_vin.relay_SK2_rout" << std::endl;
    ini << "slr=SK0:SLR0" << std::endl;
    ini << "slr=SK1:SLR1" << std::endl;
    ini << "slr=VL:SLR2" << std::endl;
    ini << "slr=RD:SLR0" << std::endl;
    ini << "slr=relay_SK2_vin:SLR1" << std::endl;
    ini << "slr=relay_SK2_rout:SLR1" << std::endl;
    ini << "sp=SK0.matrix_hbm_0:HBM[0]" << std::endl;
    ini << "sp=SK0.matrix_hbm_1:HBM[1]" << std::endl;
    ini << "sp=SK0.matrix_hbm_2:HBM[2]" << std::endl;
    ini << "sp=SK0.matrix_hbm_3:HBM[3]" << std::endl;
    ini << "sp=SK1.matrix_hbm_4:HBM[4]" << std::endl;
    ini << "sp=SK1.matrix_hbm_5:HBM[5]" << std::endl;
    ini << "sp=SK1.matrix_hbm_6:HBM[6]" << std::endl;
    ini << "sp=SK1.matrix_hbm_7:HBM[7]" << std::endl;
    ini << "sp=SK1.matrix_hbm_8:HBM[8]" << std::endl;
    ini << "sp=SK1.matrix_hbm_9:HBM[9]" << std::endl;
    ini << "sp=SK2.matrix_hbm_10:HBM[10]" << std::endl;
    ini << "sp=SK2.matrix_hbm_11:HBM[11]" << std::endl;
    ini << "sp=SK2.matrix_hbm_12:HBM[12]" << std::endl;
    ini << "sp=SK2.matrix_hbm_13:HBM[13]" << std::endl;
    ini << "sp=SK2.matrix_hbm_14:HBM[14]" << std::endl;
    ini << "sp=SK2.matrix_hbm_15:HBM[15]" << std::endl;
    ini << "sp=VL.packed_dense_vector:HBM[20]" << std::endl;
    ini << "sp=RD.packed_dense_result:HBM[20]" << std::endl;
    ini << "sc=VL.to_SLR0:SK0.vec_in [:32]" << std::endl;
    ini << "sc=VL.to_SLR1:SK1.vec_in [:32]" << std::endl;
    ini << "sc=VL.to_SLR2:relay_SK2_vin.in [:32]" << std::endl;
    ini << "sc=relay_SK2_vin.out:SK2.vec_in [:32]" << std::endl;
    ini << "sc=SK0.res_out:RD.from_SLR0 [:32]" << std::endl;
    ini << "sc=SK1.res_out:RD.from_SLR1 [:32]" << std::endl;
    ini << "sc=SK2.res_out:relay_SK2_rout.in [:32]" << std::endl;
    ini << "sc=relay_SK2_rout.out:RD.from_SLR2 [:32]" << std::endl;
    ini.close();

    // generate makefile
    std::ofstream makefile(graphlily::proj_folder_name + "/makefile");
    std::string makefile_body;
    makefile_body += "LDCLFLAGS += --config spmv.ini\n";
    makefile_body += graphlily::add_kernel_to_makefile("spmv_sk0");
    makefile_body += graphlily::add_kernel_to_makefile("spmv_sk1");
    makefile_body += graphlily::add_kernel_to_makefile("spmv_sk2");
    makefile_body += graphlily::add_kernel_to_makefile("spmv_vector_loader");
    makefile_body += graphlily::add_kernel_to_makefile("spmv_result_drain");
    makefile_body += graphlily::add_kernel_to_makefile("k2k_relay");
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
// void _test_pe(
//     std::vector<test_pld_t> &test_input,
//     graphlily::val_t zero,
//     char op
// ) {
//     // set up runtime
//     cl_int err;
//     if (target == "sw_emu" || target == "hw_emu") {
//         setenv("XCL_EMULATION_MODE", target.c_str(), true);
//     }
//     cl::Device device;
//     bool found_device = false;
//     auto devices = xcl::get_xil_devices();
//     for (size_t i = 0; i < devices.size(); i++) {
//         if (devices[i].getInfo<CL_DEVICE_NAME>() == graphlily::device_name) {
//             device = devices[i];
//             found_device = true;
//             break;
//         }
//     }
//     if (!found_device) {
//         std::cout << "Failed to find " << graphlily::device_name << ", exit!\n";
//         exit(EXIT_FAILURE);
//     }
//     cl::Context context = cl::Context(device, NULL, NULL, NULL);
//     auto file_buf = xcl::read_binary_file("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");
//     cl::Program::Binaries binaries{{file_buf.data(), file_buf.size()}};
//     cl::Program program(context, {device}, binaries, NULL, &err);
//     if (err != CL_SUCCESS) {
//         std::cout << "Failed to program device with xclbin file\n";
//     } else {
//         std::cout << "Successfully programmed device with xclbin file\n";
//     }
//     cl::Kernel kernel;
//     OCL_CHECK(err, kernel = cl::Kernel(program, "pe_tb", &err));

//     cl::CommandQueue command_queue;
//     OCL_CHECK(err, command_queue = cl::CommandQueue(context,
//                                                     device,
//                                                     CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
//                                                     &err));

//     // prepare space for results
//     std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> kernel_result;
//     kernel_result.resize(bank_size);
//     std::fill(kernel_result.begin(), kernel_result.end(), zero);

//     // allocate memory
//     std::cout << "Allocating memory on device..." << std::endl;
//     std::vector<graphlily::idx_t, aligned_allocator<graphlily::idx_t>> test_addr;
//     std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> test_mat;
//     std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> test_vec;
//     test_addr.resize(test_input.size());
//     test_mat.resize(test_input.size());
//     test_vec.resize(test_input.size());
//     for (size_t i = 0; i < test_input.size(); i++) {
//         test_mat[i] = test_input[i].mat_val;
//         test_vec[i] = test_input[i].vec_val;
//         test_addr[i] = test_input[i].row_idx;
//     }

//     CL_CREATE_EXT_PTR(test_addr_ext, test_addr.data(), graphlily::DDR[0]);
//     CL_CREATE_EXT_PTR(test_mat_ext, test_mat.data(), graphlily::DDR[0]);
//     CL_CREATE_EXT_PTR(test_vec_ext, test_vec.data(), graphlily::DDR[0]);
//     CL_CREATE_EXT_PTR(kernel_result_ext, kernel_result.data(), graphlily::DDR[1]);

//     cl::Buffer test_addr_buf;
//     cl::Buffer test_mat_buf;
//     cl::Buffer test_vec_buf;
//     cl::Buffer kernel_result_buf;

//     OCL_CHECK(err, test_addr_buf = cl::Buffer(context,
//         CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
//         sizeof(graphlily::idx_t) * test_input.size(),
//         &test_addr_ext,
//         &err));

//     OCL_CHECK(err, test_mat_buf = cl::Buffer(context,
//         CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
//         sizeof(graphlily::val_t) * test_input.size(),
//         &test_mat_ext,
//         &err));

//     OCL_CHECK(err, test_vec_buf = cl::Buffer(context,
//         CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
//         sizeof(graphlily::val_t) * test_input.size(),
//         &test_vec_ext,
//         &err));

//     OCL_CHECK(err, kernel_result_buf = cl::Buffer(context,
//         CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
//         sizeof(graphlily::val_t) * kernel_result.size(),
//         &kernel_result_ext,
//         &err));

//     // migrate data
//     std::cout << "Moving data to device..." << std::endl;
//     OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects(
//         {test_addr_buf, test_mat_buf, test_vec_buf}, 0 /* 0 means from host*/));
//     command_queue.finish();

//     // set arguments
//     OCL_CHECK(err, err = kernel.setArg(0, test_addr_buf));
//     OCL_CHECK(err, err = kernel.setArg(1, test_mat_buf));
//     OCL_CHECK(err, err = kernel.setArg(2, test_vec_buf));
//     OCL_CHECK(err, err = kernel.setArg(3, kernel_result_buf));
//     OCL_CHECK(err, err = kernel.setArg(4, op));

//     // launch kernel
//     std::cout << "Invoking test bench..." << std::endl;
//     OCL_CHECK(err, err = command_queue.enqueueTask(kernel));
//     command_queue.finish();
//     std::cout << " test bench finished successfully!" << std::endl;

//     // collect results
//     OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects({kernel_result_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
//     command_queue.finish();

//     // compute reference
//     std::vector<graphlily::val_t> ref_result;
//     ref_result.resize(bank_size);
//     std::fill(ref_result.begin(), ref_result.end(), zero);
//     compute_ref(test_input, ref_result, zero, op);

//     // verify
//     verify(ref_result, kernel_result);
// }

//--------------------------------------------------------------------------------------------------
// Test cases
//--------------------------------------------------------------------------------------------------

TEST(Build, Synthesize) {
    synthesize_tb();
}

// TEST(CleanUp, CleanProjDir) {
//     clean_proj_folder();
// }

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main(int argc, char ** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#pragma GCC diagnostic pop
