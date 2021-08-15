#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include "graphlily/synthesizer/overlay_synthesizer.h"

#include "graphlily/module/spmv_module.h"
#include "graphlily/module/spmspv_module.h"

#include <ap_fixed.h>
#include <gtest/gtest.h>

#include "graphlily/global.h"
#include "graphlily/io/data_loader.h"
#include "graphlily/io/data_formatter.h"


std::string target = "hw_emu";
uint32_t spmv_out_buf_len = graphlily::num_hbm_channels * 64;
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


std::string gen_test_case_name_spmspv(graphlily::SemiringType semiring,
                                      graphlily::MaskType mask_type,
                                      std::string matrix_name,
                                      float vector_sparsity) {
    std::string name = matrix_name + "_";
    switch (semiring.op) {
        case graphlily::kMulAdd:
            name += "Arithmetic_";
            break;
        case graphlily::kLogicalAndOr:
            name += "Logical_";
            break;
        case graphlily::kAddMin:
            name += "Tropical_";
            break;
        default:
            std::cout << "ERROR! Unsupported Semiring!" << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
    switch (mask_type) {
        case graphlily::kNoMask:
            name += "NoMask_";
            break;
        case graphlily::kMaskWriteToOne:
            name += "WriteToOne_";
            break;
        case graphlily::kMaskWriteToZero:
            name += "WriteToZero_";
            break;
        default:
            std::cout << "ERROR! Unsupported Mask Type!" << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
    float spv = vector_sparsity * 100;
    char c[20];
    sprintf(c, "%3.4f%%", spv);
    name += c;
    return name;
}


TEST(Synthesize, NULL) {
    graphlily::synthesizer::OverlaySynthesizer synthesizer(graphlily::num_hbm_channels,
                                                           spmv_out_buf_len,
                                                           spmspv_out_buf_len,
                                                           vec_buf_len);
    synthesizer.set_target(target);
    synthesizer.synthesize();
}


void _test_spmv_module(graphlily::module::SpMVModule<graphlily::val_t, graphlily::val_t> &module,
                       graphlily::SemiringType semiring,
                       graphlily::MaskType mask_type,
                       CSRMatrix<float> const &csr_matrix,
                       bool skip_empty_rows) {
    module.set_semiring(semiring);
    module.set_mask_type(mask_type);

    std::vector<float, aligned_allocator<float>> vector_float(csr_matrix.num_cols);
    std::generate(vector_float.begin(), vector_float.end(), [&](){return float(rand() % 2);});
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> vector(vector_float.begin(),
                                                                              vector_float.end());

    std::vector<float, aligned_allocator<float>> mask_float(csr_matrix.num_cols);
    std::generate(mask_float.begin(), mask_float.end(), [&](){return float(rand() % 2);});
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> mask(mask_float.begin(),
                                                                            mask_float.end());

    module.load_and_format_matrix(csr_matrix, skip_empty_rows);
    module.send_matrix_host_to_device();
    module.send_vector_host_to_device(vector);
    module.send_mask_host_to_device(mask);
    module.run();

    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> kernel_results =
        module.send_results_device_to_host();
    std::vector<float, aligned_allocator<float>> reference_results;
    if (mask_type == graphlily::kNoMask) {
        reference_results = module.compute_reference_results(vector_float);
    } else {
        reference_results = module.compute_reference_results(vector_float, mask_float);
    }

    // for (int i = 0; i < 10; i++) {
    //     std::cout << reference_results[i] << " " <<kernel_results[i] << std::endl;
    // }

    verify<graphlily::val_t>(reference_results, kernel_results);
}


TEST(SpMV, MultipleCases) {
    graphlily::module::SpMVModule<graphlily::val_t, graphlily::val_t> module(graphlily::num_hbm_channels,
                                                                             spmv_out_buf_len,
                                                                             vec_buf_len);
    module.set_target(target);
    module.set_up_runtime("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    std::string csr_float_npz_path = "/work/shared/common/project_build/graphblas/"
                                     "data/sparse_matrix_graph/dense_32_csr_float32.npz";
    CSRMatrix<float> dense_32_matrix = graphlily::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
    graphlily::io::util_round_csr_matrix_dim(
        dense_32_matrix,
        graphlily::num_hbm_channels * graphlily::pack_size,
        graphlily::pack_size);
    for (auto &x : dense_32_matrix.adj_data) x = 1.0 / dense_32_matrix.num_rows;

    _test_spmv_module(module, graphlily::ArithmeticSemiring, graphlily::kNoMask, dense_32_matrix, false);
    _test_spmv_module(module, graphlily::LogicalSemiring, graphlily::kNoMask, dense_32_matrix, false);
    _test_spmv_module(module, graphlily::ArithmeticSemiring, graphlily::kMaskWriteToZero, dense_32_matrix, false);
    _test_spmv_module(module, graphlily::LogicalSemiring, graphlily::kMaskWriteToZero, dense_32_matrix, false);
    _test_spmv_module(module, graphlily::ArithmeticSemiring, graphlily::kMaskWriteToOne, dense_32_matrix, false);
    _test_spmv_module(module, graphlily::LogicalSemiring, graphlily::kMaskWriteToOne, dense_32_matrix, false);

    _test_spmv_module(module, graphlily::ArithmeticSemiring, graphlily::kNoMask, dense_32_matrix, true);
    _test_spmv_module(module, graphlily::LogicalSemiring, graphlily::kNoMask, dense_32_matrix, true);
    _test_spmv_module(module, graphlily::ArithmeticSemiring, graphlily::kMaskWriteToZero, dense_32_matrix, true);
    _test_spmv_module(module, graphlily::LogicalSemiring, graphlily::kMaskWriteToZero, dense_32_matrix, true);
    _test_spmv_module(module, graphlily::ArithmeticSemiring, graphlily::kMaskWriteToOne, dense_32_matrix, true);
    _test_spmv_module(module, graphlily::LogicalSemiring, graphlily::kMaskWriteToOne, dense_32_matrix, true);

    csr_float_npz_path = "/work/shared/common/project_build/graphblas/"
                         "data/sparse_matrix_graph/uniform_10K_10_csr_float32.npz";
    CSRMatrix<float> uniform_10K_matrix = graphlily::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
    graphlily::io::util_round_csr_matrix_dim(
        uniform_10K_matrix,
        graphlily::num_hbm_channels * graphlily::pack_size,
        graphlily::pack_size);
    for (auto &x : uniform_10K_matrix.adj_data) x = 1.0 / uniform_10K_matrix.num_rows;
    _test_spmv_module(module, graphlily::ArithmeticSemiring, graphlily::kNoMask, uniform_10K_matrix, true);
    _test_spmv_module(module, graphlily::ArithmeticSemiring, graphlily::kMaskWriteToZero, uniform_10K_matrix, true);
    _test_spmv_module(module, graphlily::ArithmeticSemiring, graphlily::kMaskWriteToOne, uniform_10K_matrix, true);
}


void _test_spmspv_module(graphlily::module::SpMSpVModule<graphlily::val_t,
                                                         graphlily::val_t,
                                                         graphlily::idx_val_t> &module,
                         graphlily::SemiringType semiring,
                         graphlily::MaskType mask_type,
                         std::string matrix_id,
                         CSCMatrix<float> const &csc_matrix,
                         float vector_sparsity) {
    std::cout << gen_test_case_name_spmspv(semiring, mask_type, matrix_id, vector_sparsity) << std::endl;
    using aligned_dense_vec_t = graphlily::aligned_dense_vec_t;
    using aligned_sparse_vec_t = graphlily::aligned_sparse_vec_t;

    module.set_semiring(semiring);
    module.set_mask_type(mask_type);

    // generate vector
    unsigned vector_length = csc_matrix.num_cols;
    unsigned vector_nnz_cnt = (unsigned)floor(vector_length * (1 - vector_sparsity));
    unsigned vector_indices_increment = vector_length / vector_nnz_cnt;

    graphlily::aligned_sparse_float_vec_t vector_float(vector_nnz_cnt);
    for (size_t i = 0; i < vector_nnz_cnt; i++) {
        vector_float[i].val = (float)(rand() % 10) / 10;
        vector_float[i].index = i * vector_indices_increment;
    }
    graphlily::idx_float_t vector_head;
    vector_head.index = vector_nnz_cnt;
    vector_head.val = 0;
    vector_float.insert(vector_float.begin(), vector_head);
    aligned_sparse_vec_t vector(vector_float.size());
    for (size_t i = 0; i < vector[0].index + 1; i++) {
        vector[i].index = vector_float[i].index;
        vector[i].val = vector_float[i].val;
    }

    // generate mask
    unsigned mask_length = csc_matrix.num_rows;
    graphlily::aligned_dense_float_vec_t mask_float(mask_length, 0);
    for (size_t i = 0; i < mask_length; i++) {
        mask_float[i] = (float)(rand() % 2);
    }
    aligned_dense_vec_t mask;
    std::copy(mask_float.begin(), mask_float.end(), std::back_inserter(mask));

    aligned_sparse_vec_t kernel_results(csc_matrix.num_rows + 1);
    graphlily::aligned_dense_float_vec_t reference_results(csc_matrix.num_rows + 1);

    // run the kernel
    module.load_and_format_matrix(csc_matrix);
    module.send_matrix_host_to_device();
    module.send_mask_host_to_device(mask);
    module.send_vector_host_to_device(vector);
    module.run();
    kernel_results = module.send_results_device_to_host();
    reference_results = module.compute_reference_results(vector_float, mask_float);
    aligned_dense_vec_t kernel_results_dense =
        graphlily::convert_sparse_vec_to_dense_vec<aligned_sparse_vec_t, aligned_dense_vec_t,
            graphlily::val_t>(kernel_results, vector_length, semiring.zero);

    verify<graphlily::val_t>(reference_results, kernel_results_dense);
}


// TEST(SpMSpV, MultipleCases) {
//     graphlily::module::SpMSpVModule<graphlily::val_t,
//                                     graphlily::val_t,
//                                     graphlily::idx_val_t> module(spmspv_out_buf_len);
//     module.set_target(target);
//     module.set_up_runtime("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

//     std::string dataset_folder= "/work/shared/common/project_build/graphblas/data/sparse_matrix_graph/";

//     // dense 1K x 1K
//     CSCMatrix<float> csc_matrix_dense1K = graphlily::io::csr2csc(
//         graphlily::io::load_csr_matrix_from_float_npz(dataset_folder + "dense_1K_csr_float32.npz"));
//     for (auto &x : csc_matrix_dense1K.adj_data) x = 1.0 / csc_matrix_dense1K.num_rows;

//     // uniform (10K x 10K avg. degree 10)
//     CSCMatrix<float> csc_matrix_uniform10K10 = graphlily::io::csr2csc(
//         graphlily::io::load_csr_matrix_from_float_npz(dataset_folder + "uniform_10K_10_csr_float32.npz"));
//     for (auto &x : csc_matrix_uniform10K10.adj_data) x = 1.0 / csc_matrix_uniform10K10.num_rows;

//     // google plus (108K x 108K, 13M Nnz)
//     CSCMatrix<float> csc_matrix_gpuls = graphlily::io::csr2csc(
//         graphlily::io::load_csr_matrix_from_float_npz(dataset_folder + "gplus_108K_13M_csr_float32.npz"));
//     for (auto &x : csc_matrix_gpuls.adj_data) x = 1.0 / csc_matrix_gpuls.num_rows;

//     // bank conflict test case
//     CSCMatrix<float> csc_matrix_conflict;
//     unsigned conflict_matirx_size = 1024;
//     csc_matrix_conflict.num_rows = conflict_matirx_size;
//     csc_matrix_conflict.num_cols = conflict_matirx_size;
//     csc_matrix_conflict.adj_data.resize(conflict_matirx_size/8*conflict_matirx_size);
//     for (auto &x : csc_matrix_conflict.adj_data) x = 1.0 / csc_matrix_conflict.num_rows;
//     csc_matrix_conflict.adj_indices.resize(conflict_matirx_size/8*conflict_matirx_size);
//     for (size_t i = 0; i < conflict_matirx_size; i++) {
//         for (size_t j = 0; j < conflict_matirx_size/8; j++) {
//             csc_matrix_conflict.adj_indices[i * (conflict_matirx_size/8) + j] = j * 8 + i % 8;
//         }
//     }
//     csc_matrix_conflict.adj_indptr.resize(conflict_matirx_size + 1);
//     for (size_t i = 0; i < conflict_matirx_size + 1; i++) {
//         csc_matrix_conflict.adj_indptr[i] = i * (conflict_matirx_size / 8);
//     }

//     _test_spmspv_module(module, graphlily::ArithmeticSemiring, graphlily::kNoMask,
//         "conflict" + std::to_string(conflict_matirx_size), csc_matrix_conflict, 0.00);
//     _test_spmspv_module(module, graphlily::ArithmeticSemiring, graphlily::kNoMask,
//         "dense1K", csc_matrix_dense1K, 0.50);

//     _test_spmspv_module(module, graphlily::ArithmeticSemiring, graphlily::kNoMask,
//         "uniform10K10", csc_matrix_uniform10K10, 0.50);
//     _test_spmspv_module(module, graphlily::ArithmeticSemiring, graphlily::kNoMask,
//         "google+", csc_matrix_gpuls, 0.99);

//     _test_spmspv_module(module, graphlily::ArithmeticSemiring, graphlily::kNoMask,
//         "dense1K", csc_matrix_dense1K, 0.99);
//     _test_spmspv_module(module, graphlily::LogicalSemiring,    graphlily::kNoMask,
//         "dense1K", csc_matrix_dense1K, 0.99);
//     _test_spmspv_module(module, graphlily::TropicalSemiring,   graphlily::kNoMask,
//         "dense1K", csc_matrix_dense1K, 0.99);
//     _test_spmspv_module(module, graphlily::ArithmeticSemiring, graphlily::kMaskWriteToZero,
//         "dense1K", csc_matrix_dense1K, 0.99);
//     _test_spmspv_module(module, graphlily::LogicalSemiring,    graphlily::kMaskWriteToZero,
//         "dense1K", csc_matrix_dense1K, 0.99);
//     _test_spmspv_module(module, graphlily::TropicalSemiring,   graphlily::kMaskWriteToZero,
//         "dense1K", csc_matrix_dense1K, 0.99);
//     _test_spmspv_module(module, graphlily::ArithmeticSemiring, graphlily::kMaskWriteToOne,
//         "dense1K", csc_matrix_dense1K, 0.99);
//     _test_spmspv_module(module, graphlily::LogicalSemiring,    graphlily::kMaskWriteToOne,
//         "dense1K", csc_matrix_dense1K, 0.99);
//     _test_spmspv_module(module, graphlily::TropicalSemiring,   graphlily::kMaskWriteToOne,
//         "dense1K", csc_matrix_dense1K, 0.99);
// }


TEST(Clean, NULL) {
    clean_proj_folder();
}


int main(int argc, char ** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#pragma GCC diagnostic pop
