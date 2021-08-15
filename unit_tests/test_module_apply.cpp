#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include "graphlily/synthesizer/overlay_synthesizer.h"

#include "graphlily/module/assign_vector_dense_module.h"
#include "graphlily/module/assign_vector_sparse_module.h"
#include "graphlily/module/add_scalar_vector_dense_module.h"

#include <ap_fixed.h>
#include <gtest/gtest.h>

#include "graphlily/global.h"
#include "graphlily/io/data_loader.h"
#include "graphlily/io/data_formatter.h"


std::string target = "sw_emu";
uint32_t spmv_out_buf_len = 1024;
uint32_t spmspv_out_buf_len = 512;
uint32_t vec_buf_len = 256;


void clean_proj_folder() {
    std::string command = "rm -rf ./" + graphlily::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
}


template<typename data_t>
void verify(std::vector<float, aligned_allocator<float>> &reference_results,
            std::vector<data_t, aligned_allocator<data_t>> &kernel_results) {
    ASSERT_EQ(reference_results.size(), kernel_results.size());
    float epsilon = 0.0001;
    for (size_t i = 0; i < reference_results.size(); i++) {
        ASSERT_TRUE(abs(float(kernel_results[i]) - reference_results[i]) < epsilon);
    }
}


TEST(Synthesize, NULL) {
    graphlily::synthesizer::OverlaySynthesizer synthesizer(graphlily::num_hbm_channels,
                                                           spmv_out_buf_len,
                                                           spmspv_out_buf_len,
                                                           vec_buf_len);
    synthesizer.set_target(target);
    synthesizer.synthesize();
}


TEST(AddScalarVectorDense, Basic) {
    graphlily::module::eWiseAddModule<graphlily::val_t> module;
    module.set_target(target);
    module.set_up_runtime("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    uint32_t length = 128;
    graphlily::val_t val = 1;
    float val_float = float(val);
    std::vector<float, aligned_allocator<float>> in_float(length);
    std::generate(in_float.begin(), in_float.end(), [&](){return float(rand() % 10) / 100;});
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> in(in_float.begin(), in_float.end());

    module.send_in_host_to_device(in);
    module.allocate_out_buf(length);
    module.run(length, val);
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> kernel_out =
        module.send_out_device_to_host();
    std::vector<float, aligned_allocator<float>> reference_out =
        module.compute_reference_results(in_float, length, val_float);

    verify<graphlily::val_t>(reference_out, kernel_out);
}


TEST(AssignVectorDense, Basic) {
    graphlily::module::AssignVectorDenseModule<graphlily::val_t> module;
    module.set_target(target);
    module.set_up_runtime("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    uint32_t length = 128;
    graphlily::val_t val = 23;
    float val_float = float(val);
    std::vector<float, aligned_allocator<float>> mask_float(length);
    std::generate(mask_float.begin(), mask_float.end(), [&](){return float(rand() % 2);});
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> mask(mask_float.begin(),
                                                                            mask_float.end());
    std::vector<float, aligned_allocator<float>> reference_inout(length);
    std::generate(reference_inout.begin(), reference_inout.end(), [&](){return float(rand() % 2);});
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> kernel_inout(reference_inout.begin(),
                                                                                    reference_inout.end());

    module.set_mask_type(graphlily::kMaskWriteToOne);
    module.send_mask_host_to_device(mask);
    module.send_inout_host_to_device(kernel_inout);
    module.run(length, val);
    kernel_inout = module.send_inout_device_to_host();
    module.compute_reference_results(mask_float, reference_inout, length, val_float);

    verify<graphlily::val_t>(reference_inout, kernel_inout);
}


TEST(AssignVectorSparseNoNewFrontier, Basic) {
    bool generate_new_frontier = false;
    graphlily::module::AssignVectorSparseModule<graphlily::val_t,
        graphlily::idx_val_t> module(generate_new_frontier);
    module.set_target(target);
    module.set_up_runtime("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    float mask_sparsity = 0.9;
    uint32_t inout_size = 8192;
    graphlily::val_t val = 3;
    float val_float = float(val);
    unsigned length = (unsigned)floor(inout_size * (1 - mask_sparsity));
    unsigned mask_indices_increment = inout_size / length;
    graphlily::aligned_sparse_float_vec_t mask_float(length + 1);
    for (size_t i = 0; i < length; i++) {
        mask_float[i + 1].val = float(rand() % 10);
        mask_float[i + 1].index = i * mask_indices_increment;
    }
    mask_float[0].val = 0;
    mask_float[0].index = length;
    std::vector<graphlily::idx_val_t, aligned_allocator<graphlily::idx_val_t>> mask(length + 1);
    for (size_t i = 0; i < length + 1; i++) {
        mask[i].val = mask_float[i].val;
        mask[i].index = mask_float[i].index;
    }
    graphlily::aligned_dense_float_vec_t reference_inout(inout_size);
    std::generate(reference_inout.begin(), reference_inout.end(), [&](){return (rand() % 10);});
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> kernel_inout(reference_inout.begin(),
                                                                                    reference_inout.end());

    module.send_mask_host_to_device(mask);
    module.send_inout_host_to_device(kernel_inout);
    module.run(val);
    kernel_inout = module.send_inout_device_to_host();
    module.compute_reference_results(mask_float, reference_inout, val_float);

    verify<graphlily::val_t>(reference_inout, kernel_inout);
}


TEST(AssignVectorSparseNewFrontier, Basic) {
    bool generate_new_frontier = true;
    graphlily::module::AssignVectorSparseModule<graphlily::val_t,
        graphlily::idx_val_t> module(generate_new_frontier);
    module.set_target(target);
    module.set_up_runtime("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    float mask_sparsity = 0.9;
    uint32_t inout_size = 128;
    float inf;
    if (std::is_same<graphlily::val_t, float>::value) {
        inf = float(graphlily::FLOAT_INF);
    } else if (std::is_same<graphlily::val_t, unsigned>::value) {
        inf = float(graphlily::UINT_INF);
    } else {
        inf = float(graphlily::UFIXED_INF);
    }
    unsigned length = (unsigned)floor(inout_size * (1 - mask_sparsity));
    unsigned mask_indices_increment = inout_size / length;
    graphlily::aligned_sparse_float_vec_t mask_float(length + 1);
    for (size_t i = 0; i < length; i++) {
        mask_float[i + 1].val = float(rand() % 10);
        mask_float[i + 1].index = i * mask_indices_increment;
    }
    mask_float[0].val = 0;
    mask_float[0].index = length;
    std::vector<graphlily::idx_val_t, aligned_allocator<graphlily::idx_val_t>> mask(length + 1);
    for (size_t i = 0; i < length + 1; i++) {
        mask[i].val = mask_float[i].val;
        mask[i].index = mask_float[i].index;
    }

    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> kernel_inout(inout_size);
    std::generate(kernel_inout.begin(), kernel_inout.end(),
        [&](){return (((rand() % 10) > 5) ? 5 : inf);});
    graphlily::aligned_dense_float_vec_t reference_inout(kernel_inout.begin(), kernel_inout.end());

    std::vector<graphlily::idx_val_t, aligned_allocator<graphlily::idx_val_t>> kernel_new_frontier;
    graphlily::aligned_sparse_float_vec_t reference_new_frontier;

    module.send_mask_host_to_device(mask);
    module.send_inout_host_to_device(kernel_inout);
    module.run();
    kernel_inout = module.send_inout_device_to_host();
    kernel_new_frontier = module.send_new_frontier_device_to_host();
    module.compute_reference_results(mask_float, reference_inout, reference_new_frontier);

    // Verify kernel_inout
    verify<graphlily::val_t>(reference_inout, kernel_inout);

    // Verify kernel_new_frontier
    graphlily::aligned_dense_float_vec_t dense_ref_nf =
        graphlily::convert_sparse_vec_to_dense_vec<graphlily::aligned_sparse_float_vec_t,
            graphlily::aligned_dense_float_vec_t, float>(reference_new_frontier, inout_size, 0);
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> dense_knl_nf =
        graphlily::convert_sparse_vec_to_dense_vec<
            std::vector<graphlily::idx_val_t, aligned_allocator<graphlily::idx_val_t>>,
            std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>>, graphlily::val_t>(
                kernel_new_frontier, inout_size, 0);
    verify<graphlily::val_t>(dense_ref_nf, dense_knl_nf);
}


TEST(CopyBufferBindBuffer, Basic) {
    graphlily::module::AssignVectorDenseModule<graphlily::val_t> module;
    module.set_target(target);
    module.set_up_runtime("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    uint32_t length = 128;
    std::vector<float, aligned_allocator<float>> mask_float(length);
    std::generate(mask_float.begin(), mask_float.end(), [&](){return float(rand() % 2);});
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> mask(mask_float.begin(),
                                                                            mask_float.end());
    std::vector<float, aligned_allocator<float>> inout_float(length);
    std::fill(inout_float.begin(), inout_float.end(), 0);
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> inout(inout_float.begin(),
                                                                             inout_float.end());

    module.set_mask_type(graphlily::kMaskWriteToOne);

    /*----------------------------- Copy buffer -------------------------------*/
    {
    module.send_mask_host_to_device(mask);
    module.send_inout_host_to_device(inout);
    module.copy_buffer_device_to_device(module.mask_buf, module.inout_buf, sizeof(graphlily::val_t) * length);
    inout = module.send_inout_device_to_host();
    verify<graphlily::val_t>(mask_float, inout);
    }

    /*----------------------------- Bind buffer -------------------------------*/
    {
    std::vector<float, aligned_allocator<float>> x_float(length);
    std::fill(x_float.begin(), x_float.end(), 0);
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> x(x_float.begin(), x_float.end());
    cl_mem_ext_ptr_t x_ext;
    x_ext.obj = x.data();
    x_ext.param = 0;
    x_ext.flags = graphlily::HBM[graphlily::num_hbm_channels + 1];
    cl::Device device = graphlily::find_device();
    cl::Context context = cl::Context(device, NULL, NULL, NULL);
    cl::Buffer x_buf = cl::Buffer(context,
                                  CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                  sizeof(graphlily::val_t) * length,
                                  &x_ext);
    cl::CommandQueue command_queue = cl::CommandQueue(context, device);

    module.send_mask_host_to_device(mask);
    module.bind_inout_buf(x_buf);
    module.run(length, 2);
    command_queue.enqueueMigrateMemObjects({x_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    command_queue.finish();

    module.compute_reference_results(mask_float, inout_float, length, 2);
    verify<graphlily::val_t>(inout_float, x);
    }
}


TEST(Clean, NULL) {
    clean_proj_folder();
}


int main(int argc, char ** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#pragma GCC diagnostic pop
