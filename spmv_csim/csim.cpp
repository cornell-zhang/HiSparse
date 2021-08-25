#include "common.h"
#include "k2k_relay.h"
#include "spmv_result_drain.h"
#include "spmv_vector_loader.h"
#include "spmv_sk0.h"
#include "spmv_sk1.h"
#include "spmv_sk2.h"

#include "data_loader.h"
#include "data_formatter.h"

#include <hls_stream.h>

#include <iostream>
#include <iomanip>
#include <assert.h>

//---------------------------------------------------------------
// C-simulation wrapper
//---------------------------------------------------------------

void top_wrapper (
    const SPMV_MAT_PKT_T *matrix_hbm_0,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_1,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_2,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_3,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_4,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_5,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_6,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_7,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_8,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_9,       // in
    const SPMV_MAT_PKT_T *matrix_hbm_10,      // in
    const SPMV_MAT_PKT_T *matrix_hbm_11,      // in
    const SPMV_MAT_PKT_T *matrix_hbm_12,      // in
    const SPMV_MAT_PKT_T *matrix_hbm_13,      // in
    const SPMV_MAT_PKT_T *matrix_hbm_14,      // in
    const SPMV_MAT_PKT_T *matrix_hbm_15,      // in
    const PACKED_VAL_T *packed_dense_vector,  // in
    PACKED_VAL_T *packed_dense_result,        // out
    const unsigned row_part_id,               // in
    const unsigned part_len,                  // in
    const unsigned num_col_partitions,        // in
    const unsigned num_partitions,            // in
    const unsigned num_cols                   // in
) {
    hls::stream<VEC_AXIS_T> vec_VL_to_SK0, vec_VL_to_SK1, vec_VL_to_relay, vec_relay_to_SK2;
    hls::stream<VEC_AXIS_T> res_SK0_to_RD, res_SK1_to_RD, res_SK2_to_relay, res_relay_to_RD;

    std::cout << "INFO : [top wrapper] SpMV Kernel Started: row partition " << row_part_id
              << " with " << part_len << " rows per cluster" << std::endl;

    spmv_vector_loader (
        packed_dense_vector,
        num_cols,
        vec_VL_to_SK0,
        vec_VL_to_SK1,
        vec_VL_to_relay
    );

    std::cout << "INFO : [top wrapper] Vector Loader Complete" << std::endl;

    spmv_sk0 (
        matrix_hbm_0,
        matrix_hbm_1,
        matrix_hbm_2,
        matrix_hbm_3,
        vec_VL_to_SK0,
        res_SK0_to_RD,
        row_part_id,
        part_len,
        num_col_partitions,
        num_partitions
    );

    std::cout << "INFO : [top wrapper] Sub-kernel 0 Complete" << std::endl;

    spmv_sk1 (
        matrix_hbm_4,
        matrix_hbm_5,
        matrix_hbm_6,
        matrix_hbm_7,
        matrix_hbm_8,
        matrix_hbm_9,
        vec_VL_to_SK1,
        res_SK1_to_RD,
        row_part_id,
        part_len,
        num_col_partitions,
        num_partitions
    );

    std::cout << "INFO : [top wrapper] Sub-kernel 1 Complete" << std::endl;

    k2k_relay (
        vec_VL_to_relay,
        vec_relay_to_SK2
    );

    std::cout << "INFO : [top wrapper] Relay: VL->SK2 Complete" << std::endl;

    spmv_sk2 (
        matrix_hbm_10,
        matrix_hbm_11,
        matrix_hbm_12,
        matrix_hbm_13,
        matrix_hbm_14,
        matrix_hbm_15,
        vec_relay_to_SK2,
        res_SK2_to_relay,
        row_part_id,
        part_len,
        num_col_partitions,
        num_partitions
    );

    std::cout << "INFO : [top wrapper] Sub-kernel 2 Complete" << std::endl;

    k2k_relay (
        res_SK2_to_relay,
        res_relay_to_RD
    );

    std::cout << "INFO : [top wrapper] Relay: SK2->RD Complete" << std::endl;

    spmv_result_drain (
        packed_dense_result,
        row_part_id,
        res_SK0_to_RD,
        res_SK1_to_RD,
        res_relay_to_RD
    );

    std::cout << "INFO : [top wrapper] Result Drain Complete" << std::endl;
    std::cout << "INFO : [top wrapper] SpMV Kernel Complete" << std::endl;
}

//--------------------------------------------------------------------------------------------------
// reference and verify utils
//--------------------------------------------------------------------------------------------------

// TODO: only support MULADD now!
void compute_ref(
    spmv::io::CSRMatrix<float> &mat,
    std::vector<float> &vector,
    std::vector<float> &ref_result
) {
    ref_result.resize(mat.num_rows);
    std::fill(ref_result.begin(), ref_result.end(), 0);
    for (size_t row_idx = 0; row_idx < mat.num_rows; row_idx++) {
        IDX_T start = mat.adj_indptr[row_idx];
        IDX_T end = mat.adj_indptr[row_idx + 1];
        for (size_t i = start; i < end; i++) {
            IDX_T idx = mat.adj_indices[i];
            ref_result[row_idx] += mat.adj_data[i] * vector[idx];
        }
    }
}

bool verify(std::vector<float> reference_results,
            std::vector<VAL_T> kernel_results) {
    float epsilon = 0.0001;
    if (reference_results.size() != kernel_results.size()) {
        std::cout << "Error: Size mismatch"
                      << std::endl;
        std::cout   << "  Reference result size: " << reference_results.size()
                    << "  Kernel result size: " << kernel_results.size()
                    << std::endl;
        return false;
    }
    for (size_t i = 0; i < reference_results.size(); i++) {
        bool match = abs(float(kernel_results[i]) - reference_results[i]) < epsilon;
        if (!match) {
            std::cout << "Error: Result mismatch"
                      << std::endl;
            std::cout << "  i = " << i
                      << "  Reference result = " << reference_results[i]
                      << "  Kernel result = " << kernel_results[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

void unpack_vector(
    std::vector<PACKED_VAL_T> &pdv,
    std::vector<VAL_T> &dv
) {
    dv.resize(pdv.size() * PACK_SIZE);
    for (size_t i = 0; i < pdv.size(); i++) {
        for (size_t k = 0; k < PACK_SIZE; k++) {
            dv[i * PACK_SIZE + k] = pdv[i].data[k];
        }
    }
}


//---------------------------------------------------------------
// test harness
//---------------------------------------------------------------

bool spmv_test_harness (
    spmv::io::CSRMatrix<float> &ext_matrix,
    bool skip_empty_rows
) {
    using namespace spmv::io;

    //--------------------------------------------------------------------
    // load and format the matrix
    //--------------------------------------------------------------------
    std::cout << "INFO : Test started" << std::endl;
    util_round_csr_matrix_dim<float>(ext_matrix, PACK_SIZE * NUM_HBM_CHANNELS, PACK_SIZE);
    CSRMatrix<VAL_T> mat = csr_matrix_convert_from_float<VAL_T>(ext_matrix);

    size_t num_row_partitions = (mat.num_rows + LOGICAL_OB_SIZE - 1) / LOGICAL_OB_SIZE;
    size_t num_col_partitions = (mat.num_cols + LOGICAL_VB_SIZE - 1) / LOGICAL_VB_SIZE;
    size_t num_partitions = num_row_partitions * num_col_partitions;
    CPSRMatrix<PACKED_VAL_T, PACKED_IDX_T, PACK_SIZE> cpsr_matrix
        = csr2cpsr<PACKED_VAL_T, PACKED_IDX_T, VAL_T, IDX_T, PACK_SIZE>(
            mat,
            IDX_MARKER,
            LOGICAL_OB_SIZE,
            LOGICAL_VB_SIZE,
            NUM_HBM_CHANNELS,
            skip_empty_rows
        );
    using partition_indptr_t = struct {IDX_T start; PACKED_IDX_T nnz;};
    using ch_partition_indptr_t = std::vector<partition_indptr_t>;
    using ch_packed_idx_t = std::vector<PACKED_IDX_T>;
    using ch_packed_val_t = std::vector<PACKED_VAL_T>;
    using ch_mat_pkt_t = std::vector<SPMV_MAT_PKT_T>;
    std::vector<ch_partition_indptr_t> channel_partition_indptr(NUM_HBM_CHANNELS);
    std::vector<ch_packed_idx_t> channel_indices(NUM_HBM_CHANNELS);
    std::vector<ch_packed_val_t> channel_vals(NUM_HBM_CHANNELS);
    std::vector<ch_mat_pkt_t> channel_packets(NUM_HBM_CHANNELS);
    for (size_t c = 0; c < NUM_HBM_CHANNELS; c++) {
        channel_partition_indptr[c].resize(num_partitions);
        channel_partition_indptr[c][0].start = 0;
    }
    // Iterate the channels
    for (size_t c = 0; c < NUM_HBM_CHANNELS; c++) {
        for (size_t j = 0; j < num_row_partitions; j++) {
            for (size_t i = 0; i < num_col_partitions; i++) {
                auto indices_partition = cpsr_matrix.get_packed_indices(j, i, c);
                channel_indices[c].insert(channel_indices[c].end(),
                    indices_partition.begin(), indices_partition.end());
                auto vals_partition = cpsr_matrix.get_packed_data(j, i, c);
                channel_vals[c].insert(channel_vals[c].end(),
                    vals_partition.begin(), vals_partition.end());
                assert(indices_partition.size() == vals_partition.size());
                auto indptr_partition = cpsr_matrix.get_packed_indptr(j, i, c);
                if (!((j == (num_row_partitions - 1)) && (i == (num_col_partitions - 1)))) {
                    channel_partition_indptr[c][j*num_col_partitions + i + 1].start =
                        channel_partition_indptr[c][j*num_col_partitions + i].start
                        + indices_partition.size();
                }
                channel_partition_indptr[c][j*num_col_partitions + i].nnz = indptr_partition.back();

            }
        }
        assert(channel_indices[c].size() == channel_vals[c].size());
        channel_packets[c].resize(2*num_partitions + channel_indices[c].size());
        // partition indptr
        for (size_t i = 0; i < num_partitions; i++) {
            channel_packets[c][2*i].indices.data[0] = channel_partition_indptr[c][i].start;
            channel_packets[c][2*i + 1].indices = channel_partition_indptr[c][i].nnz;
        }
        // matrix indices and vals
        for (size_t i = 0; i < channel_indices[c].size(); i++) {
            channel_packets[c][2*num_partitions + i].indices = channel_indices[c][i];
            channel_packets[c][2*num_partitions + i].vals = channel_vals[c][i];
        }
    }
    std::cout << "INFO : Matrix loading/preprocessing complete!" << std::endl;

    //--------------------------------------------------------------------
    // generate input vector
    //--------------------------------------------------------------------
    std::vector<float> vector_f(ext_matrix.num_cols);
    std::generate(vector_f.begin(), vector_f.end(), [&](){return float(rand() % 2);});
    std::vector<PACKED_VAL_T> vector(mat.num_cols / PACK_SIZE);
    for (size_t i = 0; i < vector.size(); i++) {
        for (size_t k = 0; k < PACK_SIZE; k++) {
            vector[i].data[k] = VAL_T(vector_f[i*PACK_SIZE + k]);
        }
    }

    //--------------------------------------------------------------------
    // allocate space for results
    //--------------------------------------------------------------------
    std::vector<PACKED_VAL_T> result(mat.num_rows / PACK_SIZE);
    for (size_t i = 0; i < result.size(); i++) {
        for (size_t k = 0; k < PACK_SIZE; k++) {
            result[i].data[k] = 0;
        }
    }
    std::cout << "INFO : Input/result initialization complete!" << std::endl;

    //--------------------------------------------------------------------
    // invoke kernel
    //--------------------------------------------------------------------
    std::cout << "INFO : Invoking kernel:" << std::endl;
    std::cout << "  row_partitions: " << num_row_partitions << std::endl;
    std::cout << "  col_partitions: " << num_col_partitions << std::endl;

    size_t rows_per_ch_in_last_row_part;
    if (mat.num_rows % LOGICAL_OB_SIZE == 0) {
        rows_per_ch_in_last_row_part = LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
    } else {
        rows_per_ch_in_last_row_part = mat.num_rows % LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
    }
    for (size_t row_part_id = 0; row_part_id < num_row_partitions; row_part_id++) {
        unsigned part_len = LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
        if (row_part_id == num_row_partitions - 1) {
            part_len = rows_per_ch_in_last_row_part;
        }
        top_wrapper (
            channel_packets[0].data(),
            channel_packets[1].data(),
            channel_packets[2].data(),
            channel_packets[3].data(),
            channel_packets[4].data(),
            channel_packets[5].data(),
            channel_packets[6].data(),
            channel_packets[7].data(),
            channel_packets[8].data(),
            channel_packets[9].data(),
            channel_packets[10].data(),
            channel_packets[11].data(),
            channel_packets[12].data(),
            channel_packets[13].data(),
            channel_packets[14].data(),
            channel_packets[15].data(),
            vector.data(),
            result.data(),
            row_part_id,
            part_len,
            num_col_partitions,
            num_partitions,
            mat.num_cols
        );
    }
    std::cout << "INFO : SpMV kernel complete!" << std::endl;

    //--------------------------------------------------------------------
    // compute reference
    //--------------------------------------------------------------------
    std::vector<float> ref_result;
    compute_ref(ext_matrix, vector_f, ref_result);
    std::cout << "INFO : Compute reference complete!" << std::endl;

    //--------------------------------------------------------------------
    // verify
    //--------------------------------------------------------------------
    std::vector<VAL_T> upk_result;
    unpack_vector(result, upk_result);
    return verify(ref_result, upk_result);
}

//---------------------------------------------------------------
// test case utils
//---------------------------------------------------------------

spmv::io::CSRMatrix<float> create_uniform_sparse_CSR (
    unsigned num_rows,
    unsigned num_cols,
    unsigned nnz_per_row
) {
    spmv::io::CSRMatrix<float> mat_f;
    mat_f.num_rows = num_rows;
    mat_f.num_cols = num_cols;
    mat_f.adj_data.resize(num_rows * nnz_per_row);
    mat_f.adj_indices.resize(num_rows * nnz_per_row);
    mat_f.adj_indptr.resize(num_rows + 1);

    for (auto &x : mat_f.adj_data) {x = 1;}

    unsigned indice_step = num_cols / nnz_per_row;
    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < nnz_per_row; j++) {
            mat_f.adj_indices[i*nnz_per_row + j] = (indice_step*j + i) % num_cols;
        }
    }
    for (size_t i = 0; i < num_rows + 1; i++) {
        mat_f.adj_indptr[i] = nnz_per_row*i;
    }
    return mat_f;
}

//---------------------------------------------------------------
// test cases
//---------------------------------------------------------------
bool test_basic() {
    std::cout << "------ Running test: on basic dense matrix " << std::endl;
    spmv::io::CSRMatrix<float> mat_f =
        spmv::io::load_csr_matrix_from_float_npz(
            "/work/shared/common/project_build/graphblas/"
            "data/sparse_matrix_graph/dense_128_csr_float32.npz"
        );
    for (auto &x : mat_f.adj_data) {x = 1;}
    if (spmv_test_harness(mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_basic_sparse() {
    std::cout << "------ Running test: on basic sparse matrix " << std::endl;
    spmv::io::CSRMatrix<float> mat_f = create_uniform_sparse_CSR(1000, 1024, 10);
    if (spmv_test_harness(mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_large_sparse() {
    std::cout << "------ Running test: on uniform 100K 10 (100K, 1M) " << std::endl;
    spmv::io::CSRMatrix<float> mat_f =
        spmv::io::load_csr_matrix_from_float_npz(
            "/work/shared/common/project_build/graphblas/"
            "data/sparse_matrix_graph/uniform_100K_10_csr_float32.npz"
        );
    for (auto &x : mat_f.adj_data) {x = 1;}
    if (spmv_test_harness(mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_gplus() {
    std::cout << "------ Running test: on google_plus (108K, 13M) " << std::endl;
    spmv::io::CSRMatrix<float> mat_f =
        spmv::io::load_csr_matrix_from_float_npz(
            "/work/shared/common/project_build/graphblas/"
            "data/sparse_matrix_graph/gplus_108K_13M_csr_float32.npz"
        );
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_ogbl_ppa() {
    std::cout << "------ Running test: on ogbl_ppa (576K, 42M) " << std::endl;
    spmv::io::CSRMatrix<float> mat_f =
        spmv::io::load_csr_matrix_from_float_npz(
            "/work/shared/common/project_build/graphblas/"
            "data/sparse_matrix_graph/ogbl_ppa_576K_42M_csr_float32.npz"
        );
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_orkut() {
    std::cout << "------ Running test: on orkut (3M, 213M) " << std::endl;
    spmv::io::CSRMatrix<float> mat_f =
        spmv::io::load_csr_matrix_from_float_npz(
            "/work/shared/common/project_build/graphblas/"
            "data/sparse_matrix_graph/orkut_3M_213M_csr_float32.npz"
        );
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(mat_f, true)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

//---------------------------------------------------------------
// main
//---------------------------------------------------------------

int main (int argc, char** argv) {
    bool passed = true;
    passed = passed && test_basic();
    passed = passed && test_basic_sparse();
    passed = passed && test_large_sparse();
    passed = passed && test_gplus();
    passed = passed && test_ogbl_ppa();
    passed = passed && test_orkut();

    std::cout << (passed ? "===== All Test Passed! =====" : "===== Test FAILED! =====") << std::endl;
    return passed ? 0 : 1;
}
