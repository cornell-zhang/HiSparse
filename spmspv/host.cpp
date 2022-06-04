#include "hisparse.h"

#include "data_loader.h"
#include "data_formatter.h"

#include <iostream>
#include <iomanip>
#include <assert.h>

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

template<typename T>
using aligned_vector = std::vector<T, aligned_allocator<T> >;

// assume IDX_T <-> unsigned int is 32bits
using aligned_idx_t = aligned_vector<IDX_T>;
using aligned_dense_vec_t = aligned_vector<VAL_T>;
using aligned_sparse_vec_t = aligned_vector<IDX_VAL_T>;

using aligned_dense_float_vec_t = aligned_vector<float>;
typedef struct {IDX_T index; float val;} IDX_FLOAT_T;
using aligned_sparse_float_vec_t = std::vector<IDX_FLOAT_T>;

using packet_t = struct {IDX_T indices[PACK_SIZE]; VAL_T vals[PACK_SIZE];};
using aligned_packet_t = aligned_vector<packet_t>;

//--------------------------------------------------------------------------------------------------
// reference and verify utils
//--------------------------------------------------------------------------------------------------

void compute_ref(
    spmspv::io::CSCMatrix<float> &mat,
    aligned_sparse_float_vec_t &vector,
    aligned_dense_float_vec_t &ref_result
) {
    // measure dimensions
    unsigned vec_nnz_total = vector[0].index;

    // create result container
    ref_result.resize(mat.num_rows);
    std::fill(ref_result.begin(), ref_result.end(), 0);

    // indices of active columns are stored in vec_idx
    // number of active columns = vec_nnz_total
    // loop over all active columns
    for (unsigned active_col_id = 0; active_col_id < vec_nnz_total; active_col_id++) {

        float nnz_from_vec = vector[active_col_id + 1].val;
        unsigned current_col_id = vector[active_col_id + 1].index;

        // slice out the current column out of the active columns
        unsigned col_start = mat.adj_indptr[current_col_id];
        unsigned col_end = mat.adj_indptr[current_col_id + 1];

        // loop over all nnzs in the current column
        for (unsigned mat_element_id = col_start; mat_element_id < col_end; mat_element_id++) {
            unsigned current_row_id = mat.adj_indices[mat_element_id];
            float nnz_from_mat = mat.adj_data[mat_element_id];
            ref_result[current_row_id] += nnz_from_mat * nnz_from_vec;
        }
    }
}

template<typename data_t>
bool verify(aligned_vector<float> reference_results,
            aligned_vector<data_t> kernel_results) {
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

// convert a sparse vector to dense
template<typename sparse_vec_t, typename dense_vec_t>
void convert_sparse_vec_to_dense_vec(const sparse_vec_t &sparse_vector,
                                            dense_vec_t &dense_vector,
                                         const unsigned range) {
    dense_vector.resize(range);
    std::fill(dense_vector.begin(), dense_vector.end(), 0);
    int nnz = sparse_vector[0].index;
    for (int i = 1; i < nnz + 1; i++) {
        dense_vector[sparse_vector[i].index] = sparse_vector[i].val;
    }
}

// TODO: check

//---------------------------------------------------------------
// test harness utils
//---------------------------------------------------------------

#define CL_CREATE_EXT_PTR(name, data, channel)                  \
cl_mem_ext_ptr_t name;                                          \
name.obj = data;                                                \
name.param = 0;                                                 \
name.flags = channel;

#define CL_BUFFER_RDONLY(context, size, ext, err)               \
cl::Buffer(context,                                             \
CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR, \
size, &ext, &err);

#define CL_BUFFER_WRONLY(context, size, ext, err)               \
cl::Buffer(context,                                             \
CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,\
size, &ext, &err);

#define CL_BUFFER(context, size, ext, err)                      \
cl::Buffer(context,                                             \
CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,\
size, &ext, &err);

#define CHECK_ERR(err)                                          \
    if (err != CL_SUCCESS) {                                    \
      printf("OCL Error at %s:%d, error code is: %d\n",         \
              __FILE__,__LINE__, err);                          \
      exit(EXIT_FAILURE);                                       \
    }

struct cl_runtime {
    cl::Context context;
    cl::CommandQueue command_queue;
    cl::Kernel spmspv;
};

using spmv::io::CSRMatrix;
using spmv::io::load_csr_matrix_from_float_npz;

using spmspv::io::csr2csc;
using spmspv::io::CSCMatrix;
using spmspv::io::FormattedCSCMatrix;
using spmspv::io::ColumnCyclicSplitCSC;
using spmspv::io::formatCSC;
using spmspv::io::csc_matrix_convert_from_float;


//---------------------------------------------------------------
// test harness
//---------------------------------------------------------------

bool spmspv_test_harness (
    cl_runtime &runtime,
    CSCMatrix<float> &csc_matrix_float,
    float vector_sparsity
) {
    //--------------------------------------------------------------------
    // load and format the matrix
    //--------------------------------------------------------------------
    std::cout << "INFO : Test started" << std::endl;
    std::vector<aligned_packet_t> channel_packets(SPMSPV_NUM_HBM_CHANNEL);
    std::vector<aligned_idx_t> channel_indptr(SPMSPV_NUM_HBM_CHANNEL);
    std::vector<aligned_idx_t> channel_partptr(SPMSPV_NUM_HBM_CHANNEL);
    std::vector<uint32_t> num_cols_each_channel(SPMSPV_NUM_HBM_CHANNEL);
    aligned_sparse_vec_t vector;
    aligned_sparse_vec_t result;

    CSCMatrix<VAL_T> csc_matrix = csc_matrix_convert_from_float<VAL_T>(csc_matrix_float);

    std::vector<CSCMatrix<VAL_T>> csc_matrices = ColumnCyclicSplitCSC<VAL_T>(csc_matrix, SPMSPV_NUM_HBM_CHANNEL);
    FormattedCSCMatrix<packet_t> formatted_csc_matrices[SPMSPV_NUM_HBM_CHANNEL];
    for (uint32_t c = 0; c < SPMSPV_NUM_HBM_CHANNEL; c++) {
        formatted_csc_matrices[c] = formatCSC<VAL_T, packet_t>(csc_matrices[c],
                                                               PACK_SIZE,
                                                               SPMSPV_OUT_BUF_LEN);
        channel_packets[c] = formatted_csc_matrices[c].get_formatted_packet();
        channel_indptr[c] = formatted_csc_matrices[c].get_formatted_indptr();
        channel_partptr[c] = formatted_csc_matrices[c].get_formatted_partptr();
        num_cols_each_channel[c] = formatted_csc_matrices[c].num_cols;
    }
    uint32_t num_row_partitions = formatted_csc_matrices[0].num_row_partitions;

    std::cout << "INFO : Matrix loading/preprocessing complete!" << std::endl;

    //--------------------------------------------------------------------
    // generate input vector
    //--------------------------------------------------------------------
    unsigned vector_length = csc_matrix.num_cols;
    unsigned vector_nnz_cnt = (unsigned)floor(vector_length * (1 - vector_sparsity));
    unsigned vector_indices_increment = vector_length / vector_nnz_cnt;

    aligned_sparse_float_vec_t vector_float(vector_nnz_cnt);
    for (size_t i = 0; i < vector_nnz_cnt; i++) {
        vector_float[i].val = (float)(rand() % 10) / 10;
        vector_float[i].index = i * vector_indices_increment;
    }
    IDX_FLOAT_T vector_head;
    vector_head.index = vector_nnz_cnt;
    vector_head.val = 0;
    vector_float.insert(vector_float.begin(), vector_head);
    vector.resize(vector_float.size());
    for (size_t i = 0; i < vector_nnz_cnt + 1; i++) {
        vector[i].index = vector_float[i].index;
        vector[i].val = vector_float[i].val;
    }

    //--------------------------------------------------------------------
    // allocate space for results
    //--------------------------------------------------------------------
    result.resize(csc_matrix.num_rows + 1);
    std::fill(result.begin(), result.end(), (IDX_VAL_T){0, 0});
    std::cout << "INFO : Input/result initialization complete!" << std::endl;

    //--------------------------------------------------------------------
    // allocate memory on FPGA and move data
    //--------------------------------------------------------------------
    cl_int err;

    // Device buffers
    std::vector<cl::Buffer> channel_packets_buf(SPMSPV_NUM_HBM_CHANNEL);
    std::vector<cl::Buffer> channel_indptr_buf(SPMSPV_NUM_HBM_CHANNEL);
    std::vector<cl::Buffer> channel_partptr_buf(SPMSPV_NUM_HBM_CHANNEL);

    // Handle matrix packet, indptr and partptr
    cl_mem_ext_ptr_t channel_packets_ext[SPMSPV_NUM_HBM_CHANNEL];
    cl_mem_ext_ptr_t channel_indptr_ext[SPMSPV_NUM_HBM_CHANNEL];
    cl_mem_ext_ptr_t channel_partptr_ext[SPMSPV_NUM_HBM_CHANNEL];

    for (size_t c = 0; c < SPMSPV_NUM_HBM_CHANNEL; c++) {
        channel_packets_ext[c].obj = channel_packets[c].data();
        channel_packets_ext[c].param = 0;
        channel_packets_ext[c].flags = HBM[c];

        channel_indptr_ext[c].obj = channel_indptr[c].data();
        channel_indptr_ext[c].param = 0;
        channel_indptr_ext[c].flags = HBM[c];

        channel_partptr_ext[c].obj = channel_partptr[c].data();
        channel_partptr_ext[c].param = 0;
        channel_partptr_ext[c].flags = HBM[c];

        size_t channel_packets_size = sizeof(packet_t) * channel_packets[c].size()
                                      + sizeof(unsigned) * channel_indptr[c].size()
                                      + sizeof(unsigned) * channel_partptr[c].size();
        // std::cout << "channel_packets_size: " << channel_packets_size << std::endl;
        if (channel_packets_size >= 256 * 1024 * 1024) {
            std::cout << "The capcity of one HBM channel is 256 MB" << std::endl;
            exit(EXIT_FAILURE);
        }

        channel_packets_buf[c] = CL_BUFFER_RDONLY(
            runtime.context,
            sizeof(packet_t) * channel_packets[c].size(),
            channel_packets_ext[c],
            err
        );
        channel_indptr_buf[c] = CL_BUFFER_RDONLY(
            runtime.context,
            sizeof(IDX_T) * (num_cols_each_channel[c] + 1) * num_row_partitions,
            channel_indptr_ext[c],
            err
        );
        channel_partptr_buf[c] = CL_BUFFER_RDONLY(
            runtime.context,
            sizeof(IDX_T) * (num_row_partitions + 1),
            channel_partptr_ext[c],
            err
        );
        CHECK_ERR(err);
    }

    // Handle vector and result
    CL_CREATE_EXT_PTR(vector_ext, vector.data(), HBM[30]);
    CL_CREATE_EXT_PTR(result_ext, result.data(), HBM[31]);

    size_t vector_size = sizeof(IDX_VAL_T) * vector.size();
    size_t result_size = sizeof(IDX_VAL_T) * (csc_matrix.num_rows + 1);
    cl::Buffer vector_buf
        = CL_BUFFER_RDONLY(runtime.context, vector_size, vector_ext, err);
    cl::Buffer result_buf
        = CL_BUFFER_WRONLY(runtime.context, result_size, result_ext, err);
    CHECK_ERR(err);

    // transfer data
    for (size_t c = 0; c < SPMSPV_NUM_HBM_CHANNEL; c++) {
        OCL_CHECK(err, err = runtime.command_queue.enqueueMigrateMemObjects({
            channel_packets_buf[c],
            channel_indptr_buf[c],
            channel_partptr_buf[c],
            }, 0 /* 0 means from host*/));
    }
    OCL_CHECK(err, err = runtime.command_queue.enqueueMigrateMemObjects(
        {vector_buf}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = runtime.command_queue.finish());
    std::cout << "INFO : Host -> Device data transfer complete!" << std::endl;

    //--------------------------------------------------------------------
    // invoke kernel
    //--------------------------------------------------------------------
    // set kernel arguments that won't change across row iterations
    std::cout << "INFO : Invoking kernel:";
    std::cout << "  row_partitions: " << num_row_partitions << std::endl;

    for (size_t c = 0; c < SPMSPV_NUM_HBM_CHANNEL; c++) {
        OCL_CHECK(err, err = runtime.spmspv.setArg(0 + 3*c, channel_packets_buf[c]));
        OCL_CHECK(err, err = runtime.spmspv.setArg(1 + 3*c, channel_indptr_buf[c]));
        OCL_CHECK(err, err = runtime.spmspv.setArg(2 + 3*c, channel_partptr_buf[c]));
    }

    size_t arg_index_offset = 3*SPMSPV_NUM_HBM_CHANNEL;
    OCL_CHECK(err, err = runtime.spmspv.setArg(arg_index_offset + 0, vector_buf));
    OCL_CHECK(err, err = runtime.spmspv.setArg(arg_index_offset + 1, result_buf));
    OCL_CHECK(err, err = runtime.spmspv.setArg(arg_index_offset + 2, csc_matrix.num_rows));
    OCL_CHECK(err, err = runtime.spmspv.setArg(arg_index_offset + 3, csc_matrix.num_cols));

    OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmspv));
    OCL_CHECK(err, err = runtime.command_queue.finish());
    std::cout << "INFO : SpMSpV Kernel complete!"<< std::endl;

    //--------------------------------------------------------------------
    // compute reference
    //--------------------------------------------------------------------
    aligned_dense_float_vec_t ref_result;
    compute_ref(csc_matrix_float, vector_float, ref_result);
    std::cout << "INFO : Compute reference complete!" << std::endl;

    //--------------------------------------------------------------------
    // verify
    //--------------------------------------------------------------------
    OCL_CHECK(err, err = runtime.command_queue.enqueueMigrateMemObjects(
        {result_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = runtime.command_queue.finish());
    std::cout << "INFO : Device -> Host data transfer complete!" << std::endl;

    aligned_dense_vec_t upk_result;
    convert_sparse_vec_to_dense_vec(result, upk_result, csc_matrix.num_rows);
    std::cout << "INFO : Device -> Host data transfer complete!" << std::endl;
    return verify(ref_result, upk_result);
}

//---------------------------------------------------------------
// test case utils
//---------------------------------------------------------------
CSRMatrix<float> create_dense_CSR (
    unsigned num_rows,
    unsigned num_cols
) {
    CSRMatrix<float> mat_f;
    mat_f.num_rows = num_rows;
    mat_f.num_cols = num_cols;
    mat_f.adj_data.resize(num_rows * num_cols);
    mat_f.adj_indices.resize(num_rows * num_cols);
    mat_f.adj_indptr.resize(num_rows + 1);

    for (auto &x : mat_f.adj_data) {x = 1;}

    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
            mat_f.adj_indices[i*num_cols + j] = j;
        }
    }
    for (size_t i = 0; i < num_rows + 1; i++) {
        mat_f.adj_indptr[i] = num_cols*i;
    }
    return mat_f;
}

CSRMatrix<float> create_uniform_sparse_CSR (
    unsigned num_rows,
    unsigned num_cols,
    unsigned nnz_per_row
) {
    CSRMatrix<float> mat_f;
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
std::string GRAPH_DATASET_DIR = "../datasets/graph/";
std::string NN_DATASET_DIR = "../datasets/pruned_nn/";

bool test_basic(cl_runtime &runtime) {
    std::cout << "------ Running test: on basic dense matrix " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(create_dense_CSR(128, 128));
    for (auto &x : mat_f.adj_data) x = 1.0;
    if (spmspv_test_harness(runtime, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_basic_sparse(cl_runtime &runtime) {
    std::cout << "------ Running test: on basic sparse matrix " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(create_uniform_sparse_CSR(1000, 1024, 10));
    if (spmspv_test_harness(runtime, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_medium_sparse(cl_runtime &runtime) {
    std::cout << "------ Running test: on uniform 10K 10 (100K, 1M) " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(create_uniform_sparse_CSR(10000, 10000, 10));
    for (auto &x : mat_f.adj_data) x = 1.0;
    if (spmspv_test_harness(runtime, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_gplus(cl_runtime &runtime) {
    std::cout << "------ Running test: on google_plus (108K, 13M) " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(
        load_csr_matrix_from_float_npz(GRAPH_DATASET_DIR + "gplus_108K_13M_csr_float32.npz"));
    for (auto &x : mat_f.adj_data) x = 1.0 / mat_f.num_cols;
    if (spmspv_test_harness(runtime, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_ogbl_ppa(cl_runtime &runtime) {
    std::cout << "------ Running test: on ogbl_ppa (576K, 42M) " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(
        load_csr_matrix_from_float_npz(GRAPH_DATASET_DIR + "ogbl_ppa_576K_42M_csr_float32.npz"));
    for (auto &x : mat_f.adj_data) x = 1.0 / mat_f.num_cols;
    if (spmspv_test_harness(runtime, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_transformer_50_t(cl_runtime &runtime) {
    std::cout << "------ Running test: on transformer-50-t" << std::endl;
    CSCMatrix<float> mat_f = csr2csc(
        load_csr_matrix_from_float_npz(NN_DATASET_DIR + "transformer_50_512_33288_csr_float32.npz"));
    for (auto &x : mat_f.adj_data) x = 1.0 / mat_f.num_cols;
    if (spmspv_test_harness(runtime, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_transformer_95_t(cl_runtime &runtime) {
    std::cout << "------ Running test: on transformer-95-t" << std::endl;
    CSCMatrix<float> mat_f = csr2csc(
        load_csr_matrix_from_float_npz(NN_DATASET_DIR + "transformer_95_512_33288_csr_float32.npz"));
    for (auto &x : mat_f.adj_data) x = 1.0 / mat_f.num_cols;
    if (spmspv_test_harness(runtime, mat_f, 0.5)) {
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
    // parse command-line arguments
    if (argc != 3) {
        std::cout << "Usage: " << argv[0]
                  << " <sw_emu/hw_emu/hw> <xclbin>" << std::endl;
        return 0;
    }
    std::string target = argv[1];
    std::string xclbin = argv[2];
    if (target != "sw_emu" && target != "hw_emu" && target != "hw") {
        std::cout << "This host program only support sw_emu, hw_emu and hw!" << std::endl;
        return 1;
    }

    // setup Xilinx openCL runtime
    cl_runtime runtime;
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
        std::cout << "ERROR : Failed to find " << "xilinx_u280_xdma_201920_3" << ", exit!\n";
        exit(EXIT_FAILURE);
    }
    runtime.context = cl::Context(device, NULL, NULL, NULL);
    auto file_buf = xcl::read_binary_file(xclbin);
    cl::Program::Binaries binaries{{file_buf.data(), file_buf.size()}};
    cl::Program program(runtime.context, {device}, binaries, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "ERROR : Failed to program device with xclbin file" << std::endl;
        return 1;
    } else {
        std::cout << "INFO : Successfully programmed device with xclbin file" << std::endl;
    }
    OCL_CHECK(err, runtime.spmspv = cl::Kernel(program, "spmspv", &err));

    OCL_CHECK(err, runtime.command_queue = cl::CommandQueue(
        runtime.context,
        device,
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
        &err));

    // run tests
    bool passed = true;
    passed = passed && test_basic(runtime);
    passed = passed && test_basic_sparse(runtime);
    passed = passed && test_medium_sparse(runtime);
    if (target != "hw_emu") {
        passed = passed && test_gplus(runtime);
        // passed = passed && test_ogbl_ppa(runtime);
        passed = passed && test_transformer_50_t(runtime);
    }
    passed = passed && test_transformer_95_t(runtime);

    std::cout << (passed ? "===== All Test Passed! =====" : "===== Test FAILED! =====") << std::endl;
    return passed ? 0 : 1;
}
