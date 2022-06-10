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

//--------------------------------------------------------------------------------------------------
// reference and verify utils
//--------------------------------------------------------------------------------------------------

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
    aligned_vector<PACKED_VAL_T> &pdv,
    std::vector<VAL_T> &dv
) {
    // auto log = std::ofstream("out_vec.txt");
    // log << "result: " <<  pdv.size() << "\n";
    dv.resize(pdv.size() * PACK_SIZE);
    for (size_t i = 0; i < pdv.size(); i++) {
        for (size_t k = 0; k < PACK_SIZE; k++) {
            dv[i * PACK_SIZE + k] = pdv[i].data[k];
            // assert(pdv[i].data[k] == 0);
            // log << pdv[i].data[k] << " ";
        }
        // log << "\n";
    }

}

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
    cl::Kernel spmv_sk0;
    cl::Kernel spmv_sk1;
    cl::Kernel spmv_sk2;
    cl::Kernel vector_loader;
    cl::Kernel result_drain;
};



//---------------------------------------------------------------
// test harness
//---------------------------------------------------------------

bool spmv_test_harness (
    cl_runtime &runtime,
    spmv::io::CSRMatrix<float> &ext_matrix,
    bool skip_empty_rows
) {
    using namespace spmv::io;

    //--------------------------------------------------------------------
    // load and format the matrix
    //--------------------------------------------------------------------
    std::cout << "INFO : Test started" << std::endl;
    // TODO: check the partition strategy
    printf("before round: %d x %d\n", ext_matrix.num_rows, ext_matrix.num_cols);
    util_round_csr_matrix_dim<float>(ext_matrix, NUM_HBM_CHANNELS * OB_PER_CLUSTER, /*PACK_SIZE*/VB_PER_CLUSTER);
    printf("after round: %d x %d\n", ext_matrix.num_rows, ext_matrix.num_cols);
    CSRMatrix<VAL_T> mat = csr_matrix_convert_from_float<VAL_T>(ext_matrix);

    size_t num_row_partitions = (mat.num_rows + LOGICAL_OB_SIZE - 1) / LOGICAL_OB_SIZE;
    size_t num_col_partitions = (mat.num_cols + LOGICAL_VB_SIZE - 1) / LOGICAL_VB_SIZE;

    auto num_pes = PACK_SIZE * NUM_HBM_CHANNELS;

    std::vector<CSRMatrix<VAL_T>> splitted_csr_mats = RowCyclicSplitCSR(mat, num_pes);
    std::vector<CSRMatrix<VAL_T>> channel_csr_mat_to_concat[NUM_HBM_CHANNELS];
    // HBM channels:    HBM CHANNEL 0           HBM CHANNEL 1        ...
    // sets of PEs : [ [0..PACK_SIZE-1], [PACK_SIZE..2*PACK_SIZE-1], ... ]
    for (size_t c = 0; c < NUM_HBM_CHANNELS; c++) {
        channel_csr_mat_to_concat[c].resize(PACK_SIZE);
        for (size_t k = 0; k < PACK_SIZE; k++) {
            channel_csr_mat_to_concat[c][k] = splitted_csr_mats[c*PACK_SIZE + k];
        }
    }
    std::vector<spmv::io::TileCOO<VAL_T, PACK_SIZE>> tile_coo_mats;
    for (size_t c = 0; c < NUM_HBM_CHANNELS; c++) {
        tile_coo_mats.push_back(spmv::io::TileCOO<VAL_T, PACK_SIZE>(
            c,
            channel_csr_mat_to_concat[c],
            VB_PER_CLUSTER,
            OB_PER_CLUSTER
        ));
    }

    unsigned num_row_tiles = tile_coo_mats[0].num_row_tiles;
    unsigned num_col_tiles = tile_coo_mats[0].num_col_tiles;

    using ch_mat_pkt_t = aligned_vector<SPMV_MAT_PKT_T>;
    std::vector<ch_mat_pkt_t> channel_packets(NUM_HBM_CHANNELS);
    for (size_t c = 0; c < NUM_HBM_CHANNELS; c++) {
        auto &coo = tile_coo_mats[c];
        auto num_stream_data = coo.stream_data[0].size();
        for (size_t s = 0; s < num_stream_data; s++) {
            SPMV_MAT_PKT_T pkt;
            for (size_t k = 0; k < PACK_SIZE; k++) {
                // if (s==0) printf("channel[%ld]: pack[%ld] num_stream = %ld\n", c, k, coo.stream_data[k].size());
                pkt.vals.data[k] = coo.stream_data[k][s].val;
                pkt.indices.data[k] = coo.stream_data[k][s].index;
            }
            channel_packets[c].push_back(pkt);
        }
    }

    std::cout << "INFO : Matrix loading/preprocessing complete!" << std::endl;

    //--------------------------------------------------------------------
    // generate input vector
    //--------------------------------------------------------------------
    std::vector<float> vector_f(ext_matrix.num_cols);
    std::generate(vector_f.begin(), vector_f.end(), [&](){return /*float(rand() % 2)*/ 1.0;});
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
        if (channel_packets_size >= 256 * 1000 * 1000) {
            std::cout << "Error: Trying to allocate " << channel_packets_size/1000/1000
            << " MB on HBM channel " << c << std::endl
            << ", but the capcity of one HBM channel is 256 MB." << std::endl;
            exit(EXIT_FAILURE);
        }
        channel_packets_buf[c]
            = CL_BUFFER_RDONLY(runtime.context, channel_packets_size, channel_packets_ext[c], err);
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
    // invoke kernel
    //--------------------------------------------------------------------
    // set kernel arguments that won't change across row iterations
    std::cout << "INFO : Invoking kernel:" << std::endl;
    std::cout << "  row_partitions: " << num_row_partitions << std::endl;
    std::cout << "  col_partitions: " << num_col_partitions << std::endl;
    std::cout << "  num_row_tiles: " << num_row_tiles << std::endl;
    std::cout << "  num_col_tiles: " << num_col_tiles << std::endl;

    for (size_t c = 0; c < SK0_CLUSTER; c++) {
        OCL_CHECK(err, err = runtime.spmv_sk0.setArg(c, channel_packets_buf[c]));
    }
    for (size_t c = 0; c < SK1_CLUSTER; c++) {
        OCL_CHECK(err, err = runtime.spmv_sk1.setArg(c, channel_packets_buf[c + SK0_CLUSTER]));
    }
    for (size_t c = 0; c < SK2_CLUSTER; c++) {
        OCL_CHECK(err, err = runtime.spmv_sk2.setArg(c, channel_packets_buf[c + SK0_CLUSTER + SK1_CLUSTER]));
    }
    OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 0, (unsigned)num_row_tiles));
    OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 1, (unsigned)num_col_tiles));

    OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 0, (unsigned)num_row_tiles));
    OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 1, (unsigned)num_col_tiles));

    OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 0, (unsigned)num_row_tiles));
    OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 1, (unsigned)num_col_tiles));

    OCL_CHECK(err, err = runtime.vector_loader.setArg(0, vector_buf));
    OCL_CHECK(err, err = runtime.vector_loader.setArg(1, (unsigned)mat.num_cols));
    OCL_CHECK(err, err = runtime.vector_loader.setArg(2, (unsigned)num_row_tiles));

    OCL_CHECK(err, err = runtime.result_drain.setArg(0, result_buf));

    assert(num_row_tiles == num_row_partitions);

    OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.vector_loader));
    OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmv_sk0));
    OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmv_sk1));
    OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmv_sk2));
    OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.result_drain));
    runtime.command_queue.finish();
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
    runtime.command_queue.enqueueMigrateMemObjects({result_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    runtime.command_queue.finish();
    std::cout << "INFO : Device -> Host data transfer complete!" << std::endl;

    std::vector<VAL_T> upk_result;
    unpack_vector(result, upk_result);
    return verify(ref_result, upk_result);
}

//---------------------------------------------------------------
// test case utils
//---------------------------------------------------------------
spmv::io::CSRMatrix<float> create_dense_CSR (
    unsigned num_rows,
    unsigned num_cols
) {
    spmv::io::CSRMatrix<float> mat_f;
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
std::string GRAPH_DATASET_DIR = "../datasets/graph/";
std::string NN_DATASET_DIR = "../datasets/pruned_nn/";

bool test_basic(cl_runtime &runtime) {
    std::cout << "------ Running test: on basic dense matrix " << std::endl;
    spmv::io::CSRMatrix<float> mat_f = create_dense_CSR(128, 128);
    for (auto &x : mat_f.adj_data) {x = 1;}
    if (spmv_test_harness(runtime, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_basic_sparse(cl_runtime &runtime) {
    std::cout << "------ Running test: on basic sparse matrix " << std::endl;
    spmv::io::CSRMatrix<float> mat_f = create_uniform_sparse_CSR(1000, 1024, 10);
    if (spmv_test_harness(runtime, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_medium_sparse(cl_runtime &runtime) {
    std::cout << "------ Running test: on uniform 10K 10 (100K, 1M) " << std::endl;
    spmv::io::CSRMatrix<float> mat_f = create_uniform_sparse_CSR(10000, 10000, 10);
    for (auto &x : mat_f.adj_data) {x = 1;}
    if (spmv_test_harness(runtime, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_gplus(cl_runtime &runtime) {
    std::cout << "------ Running test: on google_plus (108K, 13M) " << std::endl;
    spmv::io::CSRMatrix<float> mat_f =
        spmv::io::load_csr_matrix_from_float_npz(GRAPH_DATASET_DIR + "gplus_108K_13M_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(runtime, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_ogbl_ppa(cl_runtime &runtime) {
    std::cout << "------ Running test: on ogbl_ppa (576K, 42M) " << std::endl;
    spmv::io::CSRMatrix<float> mat_f =
        spmv::io::load_csr_matrix_from_float_npz(GRAPH_DATASET_DIR + "ogbl_ppa_576K_42M_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(runtime, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_transformer_50_t(cl_runtime &runtime) {
    std::cout << "------ Running test: on transformer-50-t" << std::endl;
    spmv::io::CSRMatrix<float> mat_f =
        spmv::io::load_csr_matrix_from_float_npz(NN_DATASET_DIR + "transformer_50_512_33288_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(runtime, mat_f, true)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool test_transformer_95_t(cl_runtime &runtime) {
    std::cout << "------ Running test: on transformer-95-t" << std::endl;
    spmv::io::CSRMatrix<float> mat_f =
        spmv::io::load_csr_matrix_from_float_npz(NN_DATASET_DIR + "transformer_95_512_33288_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(runtime, mat_f, true)) {
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
    bool passed = true;
    // passed = passed && test_basic(runtime);
    // passed = passed && test_basic_sparse(runtime);
    // passed = passed && test_medium_sparse(runtime);
    // if (target != "hw_emu") {
        passed = passed && test_gplus(runtime);
    //     passed = passed && test_ogbl_ppa(runtime);
    //     passed = passed && test_transformer_50_t(runtime);
    // }
    // passed = passed && test_transformer_95_t(runtime);

    std::cout << (passed ? "===== All Test Passed! =====" : "===== Test FAILED! =====") << std::endl;
    return passed ? 0 : 1;
}
