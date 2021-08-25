#include "common.h"

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


//---------------------------------------------------------------
// datasets to be tested
//---------------------------------------------------------------
const unsigned NUM_RUNS = 20;
// std::string DATASET_PATH = "/work/shared/common/project_build/graphblas/data/sparse_matrix_graph/";
// std::vector<std::string> DATASETS = {
//     "gplus_108K_13M_csr_float32.npz",
//     "ogbl_ppa_576K_42M_csr_float32.npz",
//     "hollywood_1M_113M_csr_float32.npz",
//     "pokec_1633K_31M_csr_float32.npz",
//     "ogbn_products_2M_124M_csr_float32.npz",
//     "orkut_3M_213M_csr_float32.npz",
// };

//---------------------------------------------------------------
// benchmark utils
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
    cl::Kernel spmv_sk0;
    cl::Kernel spmv_sk1;
    cl::Kernel spmv_sk2;
    cl::Kernel vector_loader;
    cl::Kernel result_drain;
};

struct benckmark_result {
    double preprocess_time_s;
    double spmv_time_ms;
    double throughput_GBPS;
    double throughput_GOPS;
};

std::ostream& operator<<(std::ostream& os, const benckmark_result &p) {
    os << '{'
        << "Preprocessing: " << p.preprocess_time_s << " s | "
        << "SpMV: " << p.spmv_time_ms << " ms | "
        << p.throughput_GBPS << " GBPS | "
        << p.throughput_GOPS << " GOPS }";
    return os;
}

//---------------------------------------------------------------
// benchmark function
//---------------------------------------------------------------

benckmark_result spmv_benchmark (
    cl_runtime &runtime,
    spmv::io::CSRMatrix<float> &ext_matrix,
    bool skip_empty_rows
) {
    using namespace spmv::io;
    using namespace std::chrono;

    benckmark_result bmark_res;

    //--------------------------------------------------------------------
    // load and format the matrix
    //--------------------------------------------------------------------
    std::cout << "INFO : Test started" << std::endl;
    auto t0 = high_resolution_clock::now();
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
    using ch_mat_pkt_t = aligned_vector<SPMV_MAT_PKT_T>;
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
    auto t1 = high_resolution_clock::now();
    bmark_res.preprocess_time_s = double(duration_cast<microseconds>(t1 - t0).count()) / 1000000;
    std::cout << "INFO : Matrix loading/preprocessing complete!" << std::endl;
    std::cout << "  row_partitions: " << num_row_partitions << std::endl;
    std::cout << "  col_partitions: " << num_col_partitions << std::endl;

    //--------------------------------------------------------------------
    // generate input vector
    //--------------------------------------------------------------------
    std::vector<float> vector_f(ext_matrix.num_cols);
    std::generate(vector_f.begin(), vector_f.end(), [&](){return float(rand() % 2);});
    aligned_vector<PACKED_VAL_T> vector(mat.num_cols / PACK_SIZE);
    for (size_t i = 0; i < vector.size(); i++) {
        for (size_t k = 0; k < PACK_SIZE; k++) {
            vector[i].data[k] = VAL_T(vector_f[i*PACK_SIZE + k]);
        }
    }

    //--------------------------------------------------------------------
    // allocate space for results
    //--------------------------------------------------------------------
    aligned_vector<PACKED_VAL_T> result(mat.num_rows / PACK_SIZE);
    for (size_t i = 0; i < result.size(); i++) {
        for (size_t k = 0; k < PACK_SIZE; k++) {
            result[i].data[k] = 0;
        }
    }
    std::cout << "INFO : Input/result initialization complete!" << std::endl;

    //--------------------------------------------------------------------
    // allocate memory on FPGA and move data
    //--------------------------------------------------------------------
    cl_int err;

    // handle matrix
    std::vector<cl::Buffer> channel_packets_buf(NUM_HBM_CHANNELS);
    cl_mem_ext_ptr_t channel_packets_ext[NUM_HBM_CHANNELS];
    for (size_t c = 0; c < NUM_HBM_CHANNELS; c++) {
        channel_packets_ext[c].obj = channel_packets[c].data();
        channel_packets_ext[c].param = 0;
        channel_packets_ext[c].flags = HBM[c];
        size_t channel_packets_size = sizeof(SPMV_MAT_PKT_T) * channel_packets[c].size();
        if (channel_packets_size >= 256 * 1024 * 1024) {
            std::cout << "ERROR : Trying to allocate " << channel_packets_size/1024/1024
            << " MB on HBM channel " << c
            << ", but the capcity of one HBM channel is 256 MB." << std::endl;
            exit(EXIT_FAILURE);
        }
        channel_packets_buf[c]
            = CL_BUFFER_RDONLY(runtime.context, channel_packets_size, channel_packets_ext[c], err);
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
    size_t vector_size = sizeof(VAL_T) * mat.num_cols;
    size_t result_size = sizeof(VAL_T) * mat.num_rows;
    cl::Buffer vector_buf
        = CL_BUFFER_RDONLY(runtime.context, vector_size, vector_ext, err);
    cl::Buffer result_buf
        = CL_BUFFER_WRONLY(runtime.context, result_size, result_ext, err);
    CHECK_ERR(err);

    // transfer data
    for (size_t c = 0; c < NUM_HBM_CHANNELS; c++) {
        OCL_CHECK(err, err = runtime.command_queue.enqueueMigrateMemObjects(
        {channel_packets_buf[c]}, 0 /* 0 means from host*/));
    }
    OCL_CHECK(err, err = runtime.command_queue.enqueueMigrateMemObjects(
        {vector_buf}, 0 /* 0 means from host*/));
    runtime.command_queue.finish();
    std::cout << "INFO : Host -> Device data transfer complete!" << std::endl;

    //--------------------------------------------------------------------
    // invoke kernel: warm-up run and verify results
    //--------------------------------------------------------------------
    // set kernel arguments that won't change across row iterations
    std::cout << "INFO : Invoking kernel:" << std::endl;

    for (size_t c = 0; c < SK0_CLUSTER; c++) {
        OCL_CHECK(err, err = runtime.spmv_sk0.setArg(c, channel_packets_buf[c]));
    }
    for (size_t c = 0; c < SK1_CLUSTER; c++) {
        OCL_CHECK(err, err = runtime.spmv_sk1.setArg(c, channel_packets_buf[c + SK0_CLUSTER]));
    }
    for (size_t c = 0; c < SK2_CLUSTER; c++) {
        OCL_CHECK(err, err = runtime.spmv_sk2.setArg(c, channel_packets_buf[c + SK0_CLUSTER + SK1_CLUSTER]));
    }
    OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 4, (unsigned)num_col_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 5, (unsigned)num_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 4, (unsigned)num_col_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 5, (unsigned)num_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 4, (unsigned)num_col_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 5, (unsigned)num_partitions));
    OCL_CHECK(err, err = runtime.vector_loader.setArg(0, vector_buf));
    OCL_CHECK(err, err = runtime.vector_loader.setArg(1, (unsigned)mat.num_cols));
    OCL_CHECK(err, err = runtime.result_drain.setArg(0, result_buf));
    // std::cout << "  non-changing arguments set." << std::endl;

    size_t rows_per_ch_in_last_row_part;
    if (mat.num_rows % LOGICAL_OB_SIZE == 0) {
        rows_per_ch_in_last_row_part = LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
    } else {
        rows_per_ch_in_last_row_part = mat.num_rows % LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
    }

    //--------------------------------------------------------------------
    // benchmarking
    //--------------------------------------------------------------------
    double total_time = 0;
    unsigned Nnz = mat.adj_data.size();
    double Mops = 2 * Nnz / 1000 / 1000;
    double gbs = double(Nnz * 2 * 4) / 1024.0 / 1024.0 / 1024.0;
    for (unsigned i = 0; i < NUM_RUNS; i++) {
        // std::cout << "  Running Run " << i << std::endl;
        auto t0 = high_resolution_clock::now();
        for (size_t row_part_id = 0; row_part_id < num_row_partitions; row_part_id++) {
            unsigned part_len = LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
            if (row_part_id == num_row_partitions - 1) {
                part_len = rows_per_ch_in_last_row_part;
            }
            OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 2, (unsigned)row_part_id));
            OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 3, (unsigned)part_len));
            OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 2, (unsigned)row_part_id));
            OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 3, (unsigned)part_len));
            OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 2, (unsigned)row_part_id));
            OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 3, (unsigned)part_len));
            OCL_CHECK(err, err = runtime.result_drain.setArg(1, (unsigned)row_part_id));
            // std::cout << "    run-specific arguments set." << std::endl;
            OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.vector_loader));
            OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmv_sk0));
            OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmv_sk1));
            OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmv_sk2));
            OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.result_drain));
            // std::cout << "    all kernel enqueued." << std::endl;
            runtime.command_queue.finish();
            // std::cout << "    all kernel finished." << std::endl;
        }
        auto t1 = high_resolution_clock::now();
        total_time += double(duration_cast<microseconds>(t1 - t0).count()) / 1000;

    }
    bmark_res.spmv_time_ms = total_time / NUM_RUNS;
    bmark_res.throughput_GBPS = gbs / (bmark_res.spmv_time_ms / 1000);
    bmark_res.throughput_GOPS = Mops / bmark_res.spmv_time_ms;

    //--------------------------------------------------------------------
    // release host & device memory
    //--------------------------------------------------------------------
    // for (size_t c = 0; c < NUM_HBM_CHANNELS; c++) {
    //     err = clReleaseMemObject(channel_packets_buf[c]());
    //     CHECK_ERR(err);
    // }
    // err = clReleaseMemObject(vector_buf());
    // CHECK_ERR(err);
    // err = clReleaseMemObject(result_buf());
    // CHECK_ERR(err);

    return bmark_res;
}

//---------------------------------------------------------------
// main
//---------------------------------------------------------------

int main (int argc, char** argv) {
    // parse command-line arguments
    if (argc != 3) {
        std::cout << "Usage: " << argv[0]
                  << " <hw-xclbin> <dataset>" << std::endl;
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
    OCL_CHECK(err, runtime.spmv_sk0 = cl::Kernel(program, "spmv_sk0", &err));
    OCL_CHECK(err, runtime.spmv_sk1 = cl::Kernel(program, "spmv_sk1", &err));
    OCL_CHECK(err, runtime.spmv_sk2 = cl::Kernel(program, "spmv_sk2", &err));
    OCL_CHECK(err, runtime.vector_loader = cl::Kernel(program, "spmv_vector_loader", &err));
    OCL_CHECK(err, runtime.result_drain = cl::Kernel(program, "spmv_result_drain", &err));

    OCL_CHECK(err, runtime.command_queue = cl::CommandQueue(
        runtime.context,
        device,
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
        &err));

    // run tests
    std::string dataset = argv[2];

    std::cout << "------ Running benchmark on " << dataset << std::endl;
    spmv::io::CSRMatrix<float> mat_f = spmv::io::load_csr_matrix_from_float_npz(dataset);
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    std::cout << spmv_benchmark(runtime, mat_f, true) << std::endl;

    std::cout << "===== Benchmark Finished =====" << std::endl;
    return 0;
}
