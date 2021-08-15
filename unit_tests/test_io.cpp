#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include "graphlily/io/data_loader.h"
#include "graphlily/io/data_formatter.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <gtest/gtest.h>


using namespace graphlily::io;


template<typename T>
void check_vector_equal(std::vector<T> const& vec1, std::vector<T> const& vec2) {
    ASSERT_EQ(vec1.size(), vec2.size());
    ASSERT_TRUE(std::equal(vec1.begin(), vec1.end(), vec2.begin()));
}


template<typename T, uint32_t pack_size>
void check_packed_vector_equal(std::vector<T> const& vec1, std::vector<T> const& vec2) {
    ASSERT_EQ(vec1.size(), vec2.size());
    auto predicate = [](T a, T b) {
        for (uint32_t i = 0; i < pack_size; i++) {
            if (a.data[i] != b.data[i]) {
                return false;
            }
        }
        return true;
    };
    ASSERT_TRUE(std::equal(vec1.begin(), vec1.end(), vec2.begin(), predicate));
}


// csr_matrix_1 is:
//     [[1, 2, 3, 4],
//      [5, 0, 6, 0],
//      [0, 7, 0, 0],
//      [0, 0, 0, 8]]
const static CSRMatrix<float> csr_matrix_1 = {
    .num_rows=4,
    .num_cols=4,
    .adj_data=std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8},
    .adj_indices=std::vector<uint32_t>{0, 1, 2, 3, 0, 2, 1, 3},
    .adj_indptr=std::vector<uint32_t>{0, 4, 6, 7, 8},
};


// csr_matrix_2 is:
//     [[1, 2, 3, 4, 1, 2, 3, 4],
//      [5, 0, 6, 0, 5, 0, 6, 0],
//      [0, 7, 0, 0, 0, 7, 0, 0],
//      [0, 0, 0, 8, 0, 0, 0, 8]
const static CSRMatrix<float> csr_matrix_2 = {
    .num_rows=4,
    .num_cols=8,
    .adj_data=std::vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 5, 6, 7, 7, 8, 8},
    .adj_indices=std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 1, 5, 3, 7},
    .adj_indptr=std::vector<uint32_t>{0, 8, 12, 14, 16},
};


TEST(DataLoader, CreateCSRMatrix) {
    uint32_t num_rows = 5;
    uint32_t num_cols = 5;
    std::vector<float> adj_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<uint32_t> adj_indices = {0, 1, 2, 3, 0, 2, 1, 3, 2};
    std::vector<uint32_t> adj_indptr = {0, 4, 6, 7, 8, 9};
    CSRMatrix<float> csr_matrix = create_csr_matrix(num_rows, num_cols, adj_data, adj_indices, adj_indptr);
    ASSERT_EQ(csr_matrix.num_rows, num_rows);
    ASSERT_EQ(csr_matrix.num_cols, num_cols);
    check_vector_equal<float>(csr_matrix.adj_data, adj_data);
    check_vector_equal<uint32_t>(csr_matrix.adj_indices, adj_indices);
    check_vector_equal<uint32_t>(csr_matrix.adj_indptr, adj_indptr);
}


TEST(DataLoader, LoadCSRMatrixFromFloatNpz) {
    CSRMatrix<float> csr_matrix = load_csr_matrix_from_float_npz("../test_data/eye_10_csr_float32.npz");
    ASSERT_EQ(csr_matrix.num_rows, uint32_t(10));
    ASSERT_EQ(csr_matrix.num_cols, uint32_t(10));
    check_vector_equal<float>(csr_matrix.adj_data,
                              std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    check_vector_equal<uint32_t>(csr_matrix.adj_indices,
                                 std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    check_vector_equal<uint32_t>(csr_matrix.adj_indptr,
                                 std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
}


TEST(DataLoader, CSRMatrixConvertFromFloat) {
    CSRMatrix<float> csr_matrix_float = load_csr_matrix_from_float_npz("../test_data/eye_10_csr_float32.npz");
    CSRMatrix<int> csr_matrix_int = csr_matrix_convert_from_float<int>(csr_matrix_float);
    ASSERT_EQ(csr_matrix_int.num_rows, uint32_t(10));
    ASSERT_EQ(csr_matrix_int.num_cols, uint32_t(10));
    check_vector_equal<int>(csr_matrix_int.adj_data,
                            std::vector<int>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    check_vector_equal<uint32_t>(csr_matrix_int.adj_indices,
                                 std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    check_vector_equal<uint32_t>(csr_matrix_int.adj_indptr,
                                 std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
}


TEST(DataLoader, Csr2CSC) {
    CSRMatrix<float> csr_matrix = csr_matrix_1;
    CSCMatrix<float> csc_matrix = csr2csc<float>(csr_matrix);
    ASSERT_EQ(csc_matrix.num_rows, uint32_t(4));
    ASSERT_EQ(csc_matrix.num_cols, uint32_t(4));
    check_vector_equal<float>(csc_matrix.adj_data, std::vector<float>{1, 5, 2, 7, 3, 6, 4, 8});
    check_vector_equal<uint32_t>(csc_matrix.adj_indices, std::vector<uint32_t>{0, 1, 0, 2, 0, 1, 0, 3});
    check_vector_equal<uint32_t>(csc_matrix.adj_indptr, std::vector<uint32_t>{0, 2, 4, 6, 8, });
}


TEST(DataFormatter, RoundCSRMatrixDim) {
    CSRMatrix<float> M = csr_matrix_1;
    ASSERT_EQ(M.num_rows, uint32_t(4));
    ASSERT_EQ(M.num_cols, uint32_t(4));
    uint32_t row_divisor = 3;
    uint32_t col_divisor = 5;
    util_round_csr_matrix_dim(M, row_divisor, col_divisor);
    ASSERT_EQ(M.num_rows, uint32_t(6));
    ASSERT_EQ(M.num_cols, uint32_t(5));
}


TEST(DataFormatter, NormalizeCSRMatrixByOutdegree) {
    CSRMatrix<float> M = csr_matrix_1;
    util_normalize_csr_matrix_by_outdegree(M);
    ASSERT_EQ(M.adj_data[0], 0.5);
    ASSERT_EQ(M.adj_data[1], 0.5);
    ASSERT_EQ(M.adj_data[2], 0.5);
    ASSERT_EQ(M.adj_data[3], 0.5);
}


TEST(DataFormatter, ConvertCsr2DDS) {
    CSRMatrix<float> M = csr_matrix_1;
    uint32_t num_cols_per_partition = 3;
    uint32_t num_col_partitions = (M.num_cols + num_cols_per_partition - 1) / num_cols_per_partition;
    std::vector<float> partitioned_data[num_col_partitions];
    std::vector<uint32_t> partitioned_indices[num_col_partitions];
    std::vector<uint32_t> partitioned_indptr[num_col_partitions];

    util_convert_csr_to_dds<float>(M.num_rows,
                                   M.num_cols,
                                   M.adj_data.data(),
                                   M.adj_indices.data(),
                                   M.adj_indptr.data(),
                                   num_cols_per_partition,
                                   partitioned_data,
                                   partitioned_indices,
                                   partitioned_indptr);

    std::vector<float> reference_partition_1_data = {1, 2, 3, 5, 6, 7};
    std::vector<float> reference_partition_2_data = {4, 8};
    std::vector<uint32_t> reference_partition_1_indices = {0, 1, 2, 0, 2, 1};
    std::vector<uint32_t> reference_partition_2_indices = {0, 0};
    std::vector<uint32_t> reference_partition_1_indptr = {0, 3, 5, 6, 6};
    std::vector<uint32_t> reference_partition_2_indptr = {0, 1, 1, 1, 2};

    check_vector_equal<float>(partitioned_data[0], reference_partition_1_data);
    check_vector_equal<float>(partitioned_data[1], reference_partition_2_data);
    check_vector_equal<uint32_t>(partitioned_indices[0], reference_partition_1_indices);
    check_vector_equal<uint32_t>(partitioned_indices[1], reference_partition_2_indices);
    check_vector_equal<uint32_t>(partitioned_indptr[0], reference_partition_1_indptr);
    check_vector_equal<uint32_t>(partitioned_indptr[1], reference_partition_2_indptr);
}


TEST(DataFormatter, ReorderRowsAscendingNnz) {
    CSRMatrix<float> M = csr_matrix_1;
    std::vector<float> reordered_data;
    std::vector<uint32_t> reordered_indices;
    std::vector<uint32_t> reordered_indptr;

    util_reorder_rows_ascending_nnz<float>(M.adj_data,
                                           M.adj_indices,
                                           M.adj_indptr,
                                           reordered_data,
                                           reordered_indices,
                                           reordered_indptr);

    // After reordering, the sparse matrix is:
    //     [[0, 7, 0, 0],
    //      [0, 0, 0, 8],
    //      [5, 0, 6, 0],
    //      [1, 2, 3, 4]]

    std::vector<float> reference_reordered_data = {7, 8, 5, 6, 1, 2, 3, 4};
    std::vector<uint32_t> reference_reordered_indices = {1, 3, 0, 2, 0, 1, 2, 3};
    std::vector<uint32_t> reference_reordered_indptr = {0, 1, 2, 4, 8};

    check_vector_equal<float>(reordered_data, reference_reordered_data);
    check_vector_equal<uint32_t>(reordered_indices, reference_reordered_indices);
    check_vector_equal<uint32_t>(reordered_indptr, reference_reordered_indptr);
}


TEST(DataFormatter, PackRows) {
    CSRMatrix<float> M = csr_matrix_1;
    const uint32_t num_hbm_channels = 2;
    const uint32_t num_PEs_per_hbm_channel = 2;
    typedef struct packed_val_t_ {float data[num_PEs_per_hbm_channel];} packed_val_t;
    typedef struct packed_idx_t_ {uint32_t data[num_PEs_per_hbm_channel];} packed_idx_t;

    std::vector<packed_val_t> packed_data[num_hbm_channels];
    std::vector<packed_idx_t> packed_indices[num_hbm_channels];
    std::vector<packed_idx_t> packed_indptr[num_hbm_channels];

    util_pack_rows<float, packed_val_t, packed_idx_t>(M.adj_data,
                                                             M.adj_indices,
                                                             M.adj_indptr,
                                                             num_hbm_channels,
                                                             num_PEs_per_hbm_channel,
                                                             packed_data,
                                                             packed_indices,
                                                             packed_indptr);

    std::vector<packed_val_t> reference_packed_data_channel_1 = {{1, 5}, {2, 6}, {3, 0}, {4, 0}};
    std::vector<packed_idx_t> reference_packed_indices_channel_1 = {{0, 0}, {1, 2}, {2, 0}, {3, 0}};
    std::vector<packed_idx_t> reference_packed_indptr_channel_1 = {{0, 0}, {4, 2}};
    std::vector<packed_val_t> reference_packed_data_channel_2 = {{7, 8}};
    std::vector<packed_idx_t> reference_packed_indices_channel_2 = {{1, 3}};
    std::vector<packed_idx_t> reference_packed_indptr_channel_2 = {{0, 0}, {1, 1}};

    check_packed_vector_equal<packed_val_t, num_PEs_per_hbm_channel>(
        packed_data[0], reference_packed_data_channel_1);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        packed_indices[0], reference_packed_indices_channel_1);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        packed_indptr[0], reference_packed_indptr_channel_1);
    check_packed_vector_equal<packed_val_t, num_PEs_per_hbm_channel>(
        packed_data[1], reference_packed_data_channel_2);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        packed_indices[1], reference_packed_indices_channel_2);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        packed_indptr[1], reference_packed_indptr_channel_2);
}


TEST(DataFormatter, Csr2CpsrColPartitioning) {
    CSRMatrix<float> M = csr_matrix_2;
    const uint32_t out_buf_len = 4;
    const uint32_t vec_buf_len = 4;
    const uint32_t num_hbm_channels = 2;
    const uint32_t num_PEs_per_hbm_channel = 2;
    unsigned idx_marker = std::numeric_limits<unsigned>::max();
    const bool skip_empty_rows = false;

    CPSRMatrix<float, num_PEs_per_hbm_channel> cpsr_matrix = csr2cpsr<float, num_PEs_per_hbm_channel>(M,
        idx_marker, out_buf_len, vec_buf_len, num_hbm_channels, skip_empty_rows);

    using packed_val_t = CPSRMatrix<float, num_PEs_per_hbm_channel>::packed_val_t;
    using packed_idx_t = CPSRMatrix<float, num_PEs_per_hbm_channel>::packed_idx_t;

    std::vector<packed_val_t> reference_data_col_partition_1_channel_1 =
        {{1, 5}, {2, 6}, {3, 1}, {4, 0}, {1, 0}};
    std::vector<packed_idx_t> reference_indices_col_partition_1_channel_1 =
        {{0, 0}, {1, 2}, {2, idx_marker}, {3, 0}, {idx_marker, 0}};
    std::vector<packed_idx_t> reference_indptr_col_partition_1_channel_1 =
        {{0, 0}, {5, 3}}; // a marker is inserted to the end of each row
    std::vector<packed_val_t> reference_data_col_partition_1_channel_2 =
        {{7, 8}, {1, 1}};
    std::vector<packed_idx_t> reference_indices_col_partition_1_channel_2 =
        {{1, 3}, {idx_marker, idx_marker}};
    std::vector<packed_idx_t> reference_indptr_col_partition_1_channel_2 =
        {{0, 0}, {2, 2}};
    auto reference_data_col_partition_2_channel_1 = reference_data_col_partition_1_channel_1;
    auto reference_indices_col_partition_2_channel_1 = reference_indices_col_partition_1_channel_1;
    auto reference_indptr_col_partition_2_channel_1 = reference_indptr_col_partition_1_channel_1;
    auto reference_data_col_partition_2_channel_2 = reference_data_col_partition_1_channel_2;
    auto reference_indices_col_partition_2_channel_2 = reference_indices_col_partition_1_channel_2;
    auto reference_indptr_col_partition_2_channel_2 = reference_indptr_col_partition_1_channel_2;

    check_packed_vector_equal<packed_val_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_data(0, 0, 0), reference_data_col_partition_1_channel_1);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_indices(0, 0, 0), reference_indices_col_partition_1_channel_1);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_indptr(0, 0, 0), reference_indptr_col_partition_1_channel_1);
    check_packed_vector_equal<packed_val_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_data(0, 0, 1), reference_data_col_partition_1_channel_2);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_indices(0, 0, 1), reference_indices_col_partition_1_channel_2);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_indptr(0, 0, 1), reference_indptr_col_partition_1_channel_2);
    check_packed_vector_equal<packed_val_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_data(0, 1, 0), reference_data_col_partition_2_channel_1);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_indices(0, 1, 0), reference_indices_col_partition_2_channel_1);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_indptr(0, 1, 0), reference_indptr_col_partition_2_channel_1);
    check_packed_vector_equal<packed_val_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_data(0, 1, 1), reference_data_col_partition_2_channel_2);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_indices(0, 1, 1), reference_indices_col_partition_2_channel_2);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_indptr(0, 1, 1), reference_indptr_col_partition_2_channel_2);
}


TEST(DataFormatter, Csr2CpsrRowPartitioning) {
    CSRMatrix<float> M = csr_matrix_1;
    const uint32_t out_buf_len = 2;
    const uint32_t vec_buf_len = 4;
    const uint32_t num_hbm_channels = 1;
    const uint32_t num_PEs_per_hbm_channel = 2;
    unsigned idx_marker = std::numeric_limits<unsigned>::max();
    const bool skip_empty_rows = false;

    CPSRMatrix<float, num_PEs_per_hbm_channel> cpsr_matrix = csr2cpsr<float, num_PEs_per_hbm_channel>(M,
        idx_marker, out_buf_len, vec_buf_len, num_hbm_channels, skip_empty_rows);

    using packed_val_t = CPSRMatrix<float, num_PEs_per_hbm_channel>::packed_val_t;
    using packed_idx_t = CPSRMatrix<float, num_PEs_per_hbm_channel>::packed_idx_t;

    std::vector<packed_val_t> reference_data_row_partition_1 =
        {{1, 5}, {2, 6}, {3, 1}, {4, 0}, {1, 0}};
    std::vector<packed_idx_t> reference_indices_row_partition_1 =
        {{0, 0}, {1, 2}, {2, idx_marker}, {3, 0}, {idx_marker, 0}};
    std::vector<packed_idx_t> reference_indptr_row_partition_1 =
        {{0, 0}, {5, 3}};  // a marker is inserted to the end of each row
    std::vector<packed_val_t> reference_data_row_partition_2 =
        {{7, 8}, {1, 1}};
    std::vector<packed_idx_t> reference_indices_row_partition_2 =
        {{1, 3}, {idx_marker, idx_marker}};
    std::vector<packed_idx_t> reference_indptr_row_partition_2 =
        {{0, 0}, {2, 2}};

    check_packed_vector_equal<packed_val_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_data(0, 0, 0), reference_data_row_partition_1);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_indices(0, 0, 0), reference_indices_row_partition_1);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_indptr(0, 0, 0), reference_indptr_row_partition_1);
    check_packed_vector_equal<packed_val_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_data(1, 0, 0), reference_data_row_partition_2);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_indices(1, 0, 0), reference_indices_row_partition_2);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_indptr(1, 0, 0), reference_indptr_row_partition_2);
}


// csr_matrix_3 is:
//     [[0, 0, 0, 0],
//      [1, 0, 2, 0],
//      [3, 0, 0, 0],
//      [0, 0, 0, 0],
//      [0, 0, 0, 0],
//      [0, 4, 0, 0],
//      [5, 0, 0, 0],
//      [0, 0, 0, 0]]
const static CSRMatrix<float> csr_matrix_3 = {
    .num_rows=8,
    .num_cols=4,
    .adj_data=std::vector<float>{1, 2, 3, 4, 5},
    .adj_indices=std::vector<uint32_t>{0, 2, 0, 1, 0},
    .adj_indptr=std::vector<uint32_t>{0, 0, 2, 3, 3, 3, 4, 5, 5},
};


TEST(DataFormatter, Csr2CpsrRowPartitioningSkipEmptyRows) {
    CSRMatrix<float> M = csr_matrix_3;
    const uint32_t out_buf_len = 8;
    const uint32_t vec_buf_len = 4;
    const uint32_t num_hbm_channels = 1;
    const uint32_t num_PEs_per_hbm_channel = 2;
    unsigned idx_marker = std::numeric_limits<unsigned>::max();
    const bool skip_empty_rows = true;

    CPSRMatrix<float, num_PEs_per_hbm_channel> cpsr_matrix = csr2cpsr<float, num_PEs_per_hbm_channel>(M,
        idx_marker, out_buf_len, vec_buf_len, num_hbm_channels, skip_empty_rows);

    using packed_val_t = CPSRMatrix<float, num_PEs_per_hbm_channel>::packed_val_t;
    using packed_idx_t = CPSRMatrix<float, num_PEs_per_hbm_channel>::packed_idx_t;

    std::vector<packed_val_t> reference_data =
        {{1, 1}, {3, 2}, {2, 2}, {5, 4}, {1, 2}};
    std::vector<packed_idx_t> reference_indices =
        {{idx_marker, 0}, {0, 2}, {idx_marker, idx_marker}, {0, 1}, {idx_marker, idx_marker}};
    std::vector<packed_idx_t> reference_indptr =
        {{0, 0}, {1, 3}, {3, 3}, {3, 5}, {5, 5}};

    check_packed_vector_equal<packed_val_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_data(0, 0, 0), reference_data);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_indices(0, 0, 0), reference_indices);
    check_packed_vector_equal<packed_idx_t, num_PEs_per_hbm_channel>(
        cpsr_matrix.get_packed_indptr(0, 0, 0), reference_indptr);
}


int main(int argc, char ** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#pragma GCC diagnostic pop
