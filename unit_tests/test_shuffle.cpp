#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include <vector>
#include <ap_fixed.h>
#include <gtest/gtest.h>

#include "graphlily/global.h"

#include <ap_int.h>

#define DISP_EXE_CMD(cmd)\
std::cout << cmd << std::endl;\
system(cmd.c_str());

#define CL_CREATE_EXT_PTR(name, data, channel)\
cl_mem_ext_ptr_t name;\
name.obj = data;\
name.param = 0;\
name.flags = channel;

std::string target = "hw_emu";
const unsigned num_lanes = 8;

struct test_pld_t {
    graphlily::val_t mat_val;
    graphlily::idx_t row_idx;
    graphlily::idx_t col_idx;
    ap_uint<2> inst;
};
#define SOD 0x1 // start-of-data
#define EOD 0x2 // end-of-data
#define EOS 0x3 // end-of-stream

std::string inst2str(ap_uint<2> inst) {
    switch (inst) {
        case SOD: return std::string("SOD");
        case EOD: return std::string("EOD");
        case EOS: return std::string("EOS");
        default:  return std::string(std::to_string((int)inst));
    }
}

std::ostream& operator<<(std::ostream& os, const test_pld_t &p) {
    os << '{'
        << "mat val: " << p.mat_val << '|'
        << "row idx: " << p.row_idx << '|'
        << "col idx: " << p.col_idx << '|'
        << "inst: "  << inst2str(p.inst) << '}';
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<test_pld_t> &p) {
    for (size_t i = 0; i < p.size(); i++) {
        os << "i = " << i << ": " << p[i] << std::endl;
    }
    return os;
}

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
        }
    }

    std::vector<test_pld_t> output_buckets[num_lanes];
    for (unsigned k = 0; k < num_lanes; k++) {
        output_buckets[k].clear();
    }
    // std::cout << "Compute ref: temporary output buffer allocated" << std::endl;

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

void verify(std::vector<test_pld_t> ref_results,
            std::vector<test_pld_t> kernel_results) {
    ASSERT_EQ(ref_results.size(), kernel_results.size());
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
    ASSERT_TRUE(passing);
}

//--------------------------------------------------------------------------------------------------
// synthesizer
//--------------------------------------------------------------------------------------------------
void synthesize_tb(bool setup_only) {
    // create proj directory
    std::string command = "mkdir -p " + graphlily::proj_folder_name;
    DISP_EXE_CMD(command);

    // copy source code
    command = "cp " + graphlily::root_path + "/graphlily/hw/shuffle.h"
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
    command = "cp " + graphlily::root_path + "/tests/testbench/shuffle_tb.h"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);
    command = "cp " + graphlily::root_path + "/tests/testbench/shuffle_tb.cpp"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);

    // put configuration into pe_cluster_tb.h
    std::ofstream header_tb(graphlily::proj_folder_name + "/shuffle_tb.h", std::ios_base::app);
    header_tb << "#endif // GRAPHLILY_TEST_TESTBENCH_SHUFFLE_TB_H_" << std::endl;
    header_tb.close();

    // close the include guard in overlay.h
    // we do not use any configuration here, we just need the typedefs
    std::ofstream header_gll(graphlily::proj_folder_name + "/overlay.h", std::ios_base::app);
    header_gll << "#endif // GRAPHLILY_HW_OVERLAY_H_" << std::endl;
    header_gll.close();

    // generate pe_tb.ini
    std::ofstream ini(graphlily::proj_folder_name + "/shuffle_tb.ini");
    ini << "[connectivity]" << std::endl;
    ini << "sp=shuffle_tb_1.input_packets:DDR[0]" << std::endl;
    ini << "sp=shuffle_tb_1.output_packets:DDR[0]" << std::endl;
    ini.close();

    // generate makefile
    std::ofstream makefile(graphlily::proj_folder_name + "/makefile");
    std::string makefile_body;
    makefile_body += "LDCLFLAGS += --config shuffle_tb.ini\n";
    makefile_body += "KERNEL_OBJS += $(TEMP_DIR)/shuffle_tb.xo\n";
    makefile_body += "\n";
    makefile_body += "$(TEMP_DIR)/shuffle_tb.xo: shuffle_tb.cpp\n";
    makefile_body += "\tmkdir -p $(TEMP_DIR)\n";
    makefile_body += "\t$(VPP) $(CLFLAGS) --temp_dir $(TEMP_DIR) -c -k shuffle_tb -I'$(<D)' -o'$@' '$<'\n";
    makefile_body += "\n";
    makefile << "TARGET := " << target << "\n" << std::endl;
    makefile << graphlily::makefile_prologue << makefile_body << graphlily::makefile_epilogue;
    makefile.close();

    // switch to build folder and build
    if (!setup_only) {
        command = "cd " + graphlily::proj_folder_name + "; " + "make build";
        DISP_EXE_CMD(command);
        if (target == "sw_emu" || target == "hw_emu") {
            command = "cp " + graphlily::proj_folder_name + "/emconfig.json " + ".";
            DISP_EXE_CMD(command);
        }
    }
}

//--------------------------------------------------------------------------------------------------
// test harness
//--------------------------------------------------------------------------------------------------
void _test_shuffle(std::vector<test_pld_t> &test_input) {
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
    OCL_CHECK(err, kernel = cl::Kernel(program, "shuffle_tb", &err));

    cl::CommandQueue command_queue;
    OCL_CHECK(err, command_queue = cl::CommandQueue(context,
                                                    device,
                                                    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
                                                    &err));

    // prepare space for results
    std::vector<test_pld_t> kernel_result;
    kernel_result.resize(1024 * num_lanes);
    std::fill(kernel_result.begin(), kernel_result.end(), test_pld_t({0,0,0,0}));

    // allocate memory

    CL_CREATE_EXT_PTR(test_input_ext, test_input.data(), graphlily::DDR[0]);
    CL_CREATE_EXT_PTR(kernel_result_ext, kernel_result.data(), graphlily::DDR[0]);

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
    ref_result.resize(1024 * num_lanes);
    std::fill(ref_result.begin(), ref_result.end(), test_pld_t({0,0,0,0}));
    compute_ref(test_input, ref_result);
    std::cout << "Reference Compute Success!" << std::endl;


    // std::cout << "REF ALL:\n" << ref_result;
    // std::cout << "RES ALL:\n" << kernel_result;
    // verify
    verify(ref_result, kernel_result);
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
    assert(1 + (len + 2) * parts < 1024);
    test_input.resize(1024 * num_lanes);
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
    assert(1 + (len + 2) * parts < 1024);
    test_input.resize(1024 * num_lanes);
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
    assert(1 + (avg_len + len_var + 2) * parts < 1024);
    test_input.resize(1024 * num_lanes);
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


// TEST(Build, Synthesize) {
//     synthesize_tb(false);
// }

// TEST(Build, Setup) {
//     synthesize_tb(true);
// }

// TEST(Uniform, Direct) {
//     std::vector<test_pld_t> test_input;
//     uniform_gen(test_input, 64, 4, 0);
//     _test_shuffle(test_input);
// }

// TEST(Uniform, Rotated) {
//     std::vector<test_pld_t> test_input;
//     uniform_gen(test_input, 16, 4, 1);
//     _test_shuffle(test_input);
//     uniform_gen(test_input, 16, 4, 3);
//     _test_shuffle(test_input);
//     uniform_gen(test_input, 16, 4, 7);
//     _test_shuffle(test_input);
// }

// TEST(Uniform, Conflict) {
//     std::vector<test_pld_t> test_input;
//     conflict_gen(test_input, 16, 4, 1);
//     _test_shuffle(test_input);
//     conflict_gen(test_input, 16, 4, 3);
//     _test_shuffle(test_input);
//     conflict_gen(test_input, 32, 4, 5);
//     _test_shuffle(test_input);
// }

TEST(Random, A) {
    std::vector<test_pld_t> test_input;
    random_gen(test_input, 64, 4);
    _test_shuffle(test_input);
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
