#include "hisparse.h"

#include "data_loader.h"
#include "data_formatter.h"

#include <iostream>
#include <iomanip>
#include <assert.h>
#include <string>

#include "xcl2.hpp"

const unsigned NUM_RUNS = 50;

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

struct benchmark_result {
    double preprocess_time_s;
    double spmv_time_ms;
    double throughput_GBPS;
    double throughput_GOPS;
};

std::ostream& operator<<(std::ostream& os, const benchmark_result &p) {
    os << '{'
        << "Preprocessing: " << p.preprocess_time_s << " s | "
        << "SpMSpV: " << p.spmv_time_ms << " ms | "
        << p.throughput_GBPS << " GBPS | "
        << p.throughput_GOPS << " GOPS }";
    return os;
}

using spmv::io::load_csr_matrix_from_float_npz;

using spmspv::io::csr2csc;
using spmspv::io::CSCMatrix;
using spmspv::io::FormattedCSCMatrix;
using spmspv::io::ColumnCyclicSplitCSC;
using spmspv::io::formatCSC;
using spmspv::io::csc_matrix_convert_from_float;


//---------------------------------------------------------------
// benchmark function
//---------------------------------------------------------------
benchmark_result spmspv_benchmark (
    cl_runtime &runtime,
    CSCMatrix<float> &csc_matrix_float,
    float vector_sparsity
) {
    using namespace std::chrono;
    benchmark_result record;

    //--------------------------------------------------------------------
    // load and format the matrix
    //--------------------------------------------------------------------
    std::cout << "INFO : Test started" << std::endl;
    auto t0 = high_resolution_clock::now();
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

    auto t1 = high_resolution_clock::now();
    record.preprocess_time_s = double(duration_cast<microseconds>(t1 - t0).count()) / 1000000;
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
    for (size_t i = 0; i < vector[0].index + 1; i++) {
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
        if (err != CL_SUCCESS) {
            std::cout << "ERROR : exception catched when trying to create CL buffer of "
                      << channel_packets_size/1024/1024 << " MB on HBM "
                      << c << std::endl;
        }
        CHECK_ERR(err);
    }

    // Handle vector and result
    CL_CREATE_EXT_PTR(vector_ext, vector.data(), HBM[20]);
    CL_CREATE_EXT_PTR(result_ext, result.data(), HBM[21]);

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

    //--------------------------------------------------------------------
    // benchmarking
    //--------------------------------------------------------------------
    double total_time = 0;
    unsigned Nnz = csc_matrix.adj_data.size();
    double Mops = 2.0 * Nnz / 1000 / 1000;
    double gbs = double(Nnz * 2 * 4) / 1024.0 / 1024.0 / 1024.0;
    for (unsigned i = 0; i < NUM_RUNS; i++) {
        auto t0 = high_resolution_clock::now();
        OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmspv));
        OCL_CHECK(err, err = runtime.command_queue.finish());
        auto t1 = high_resolution_clock::now();
        total_time += double(duration_cast<microseconds>(t1 - t0).count()) / 1000;
    }
    std::cout << "INFO : SpMSpV Kernel complete "<< NUM_RUNS << " runs!" << std::endl;

    record.spmv_time_ms = total_time / NUM_RUNS;
    record.throughput_GBPS = gbs / (record.spmv_time_ms / 1000);
    record.throughput_GOPS = Mops / record.spmv_time_ms;

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

    verify(ref_result, upk_result); // TODO: record verification
    std::cout << "INFO: Result verification complete!" << std::endl;

    return record;
}

//---------------------------------------------------------------
// test cases
//---------------------------------------------------------------

#define LOAD_DATASET(dataset) \
  CSCMatrix<float> mat_f = csr2csc(load_csr_matrix_from_float_npz((dataset))); \
  for (auto &x : mat_f.adj_data) x = 1.0 / mat_f.num_cols;

std::map<std::string, std::string> test_cases = {
    { "googleplus",     "graph/gplus_108K_13M_csr_float32.npz" },
    { "ogbl-ppa",       "graph/ogbl_ppa_576K_42M_csr_float32.npz" },
    { "hollywood",      "graph/hollywood_1M_113M_csr_float32.npz" },
    { "pokec",          "graph/pokec_1633K_31M_csr_float32.npz" },
    { "ogbn-products",  "graph/ogbn_products_2M_124M_csr_float32.npz" },
    { "mouse-gene",     "graph/mouse_gene_45K_29M_csr_float32.npz" },
    { "transformer-50", "pruned_nn/transformer_50_512_33288_csr_float32.npz" },
    { "transformer-60", "pruned_nn/transformer_60_512_33288_csr_float32.npz" },
    { "transformer-70", "pruned_nn/transformer_70_512_33288_csr_float32.npz" },
    { "transformer-80", "pruned_nn/transformer_80_512_33288_csr_float32.npz" },
    { "transformer-90", "pruned_nn/transformer_90_512_33288_csr_float32.npz" },
    { "transformer-95", "pruned_nn/transformer_95_512_33288_csr_float32.npz" },
};

//---------------------------------------------------------------
// main
//---------------------------------------------------------------

int main (int argc, char** argv) {
    // parse command-line arguments
    if (argc != 2) {
        std::cout << "Usage: " << argv[0]
                  << " <xclbin>" << std::endl;
        return 0;
    }
    std::string target = "hw";
    std::string xclbin = argv[1];

    // setup Xilinx openCL runtime
    cl_runtime runtime;
    cl_int err;
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

    for (const auto &x : test_cases ) {
      std::cout << "------ Running benchmark on " << x.first << std::endl;
      LOAD_DATASET("../datasets/" + x.second);
      std::cout << spmspv_benchmark(runtime, mat_f, 0.5) << std::endl;
      std::cout << "===== Benchmark Finished =====" << std::endl;
    }

    std::cout << "===== All Benchmark Finished =====" << std::endl;
    return 0;
}
