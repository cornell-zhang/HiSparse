#include "hisparse.h"

#include "data_loader.h"
#include "data_formatter.h"

#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <fstream>

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

using spmv::io::load_csr_matrix_from_float_npz;

using spmspv::io::csr2csc;
using spmspv::io::CSCMatrix;
using spmspv::io::FormattedCSCMatrix;
using spmspv::io::ColumnCyclicSplitCSC;
using spmspv::io::formatCSC;
using spmspv::io::csc_matrix_convert_from_float;

//--------------------------------------------------------------------------------------------------
// reference and verify utils
//--------------------------------------------------------------------------------------------------

void compute_ref(
    CSCMatrix<float> &mat,
    aligned_sparse_float_vec_t &vector,
    aligned_dense_float_vec_t &ref_result,
    uint32_t &involved_Nnz
) {
    // measure dimensions
    unsigned vec_nnz_total = vector[0].index;
    involved_Nnz = 0;

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

        // measure the involved Nnz in one SpMSpV run (only measure the matrix)
        involved_Nnz += col_end - col_start;

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
    std::string benchmark_name;
    float spmspv_sparsity;
    double preprocess_time_s;
    double spmspv_time_ms;
    double throughput_GBPS;
    double throughput_GOPS;
    bool verified;
};

template <typename T>
inline std::string fmt_key_val(std::string key, T val) {
    std::stringstream ss;
    ss << "\"" << key << "\": \"" << val << "\"";
    return ss.str();
}

// print the benchmark result in JSON format
std::ostream& operator<<(std::ostream& os, const benchmark_result &p) {
    os << "{ "
       << fmt_key_val("Benchmark", p.benchmark_name) << ", "
       << fmt_key_val("Sparsity", p.spmspv_sparsity) << ", "
       << fmt_key_val("Preprocessing_s", p.preprocess_time_s) << ", "
       << fmt_key_val("SpMSpV_ms", p.spmspv_time_ms) << ", "
       << fmt_key_val("TP_GBPS", p.throughput_GBPS) << ", "
       << fmt_key_val("TP_GOPS", p.throughput_GOPS) << ", "
       << fmt_key_val("verified", (int)p.verified) << " }";
    return os;
}

using namespace std::chrono;

// helper function to generate a dense CSC matrix
CSCMatrix<float> generate_dense_float_csc(unsigned dim) {
    CSCMatrix<float> float_csc;
    float_csc.num_rows = dim;
    float_csc.num_cols = dim;
    unsigned nnz = float_csc.num_rows*float_csc.num_cols;
    float_csc.adj_data.resize(nnz);
    float_csc.adj_indices.resize(nnz);
    float_csc.adj_indptr.resize(float_csc.num_cols + 1);
    std::fill(float_csc.adj_data.begin(), float_csc.adj_data.end(), 1.0/dim);
    float_csc.adj_indptr[0] = 0;
    for (size_t c = 0; c < dim; c++) {
        for (size_t r = 0; r < dim; r++) {
            float_csc.adj_indices[c * dim + r] = r;
        }
        float_csc.adj_indptr[c + 1] = float_csc.adj_indptr[c] + dim;
    }
    return float_csc;
}

class test_harness {
public:
    std::string name;
    std::vector<benchmark_result> benchmark_results;

    test_harness(std::string name, std::string dataset, size_t num_hbm_channels) {
        this->name = name;
        this->num_hbm_channels = num_hbm_channels;

        //--------------------------------------------------------------------
        // load the CSC matrix
        //--------------------------------------------------------------------
        this->csc_matrix_float = csr2csc(load_csr_matrix_from_float_npz(dataset));
        for (auto &x : this->csc_matrix_float.adj_data) {
            x = 1.0 / this->csc_matrix_float.num_cols;
        }
        // this->csc_matrix_float = generate_dense_float_csc(512);
        this->csc_matrix = csc_matrix_convert_from_float<VAL_T>(this->csc_matrix_float);

        //--------------------------------------------------------------------
        // allocate space for results
        //--------------------------------------------------------------------
        this->result.resize(this->csc_matrix.num_rows + 1);
        std::fill(this->result.begin(), this->result.end(), (IDX_VAL_T){0, 0});
    }

    void preprocessing() {
        this->preprocess = std::thread([&]{
            this->format_matrix();
        });
    }

    bool warming_up(cl_runtime &runtime) {
        this->preprocess.join();
        return this->transfer_static_buffer(runtime);
    }

    void set_sparsity_and_run(cl_runtime &runtime, float vector_sparsity) {
        this->run(runtime, vector_sparsity);
    }

private:

    std::thread preprocess;
    double preprocess_time_s;

    CSCMatrix<VAL_T> csc_matrix;
    CSCMatrix<float> csc_matrix_float;

    size_t num_hbm_channels;
    std::vector<aligned_packet_t> channel_packets;
    std::vector<uint32_t> num_cols_each_channel;
    aligned_sparse_vec_t result;
    uint32_t num_row_partitions;

    // Device static buffers
    std::vector<cl::Buffer> channel_packets_buf;
    cl::Buffer result_buf;

    // Handle matrix (packet, indptr and partptr) and result
    std::vector<cl_mem_ext_ptr_t> channel_packets_ext;
    cl_mem_ext_ptr_t result_ext;

    void format_matrix() {
        auto t0 = high_resolution_clock::now();
        this->channel_packets.resize(num_hbm_channels);
        this->num_cols_each_channel.resize(num_hbm_channels);

        std::vector<CSCMatrix<VAL_T>> csc_matrices =
            ColumnCyclicSplitCSC<VAL_T>(this->csc_matrix, num_hbm_channels);
        FormattedCSCMatrix<packet_t> formatted_csc_matrices[num_hbm_channels];
        for (size_t c = 0; c < num_hbm_channels; c++) {
            formatted_csc_matrices[c] = formatCSC<VAL_T, packet_t>(csc_matrices[c],
                                                                   PACK_SIZE,
                                                                   SPMSPV_OUT_BUF_LEN);
            this->channel_packets[c] = formatted_csc_matrices[c].get_fused_matrix<VAL_T>();
            this->num_cols_each_channel[c] = formatted_csc_matrices[c].num_cols;
        }
        this->num_row_partitions = formatted_csc_matrices[0].num_row_partitions;

        auto t1 = high_resolution_clock::now();
        this->preprocess_time_s = double(duration_cast<microseconds>(t1 - t0).count()) / 1000000;
    }

    bool transfer_static_buffer(cl_runtime &runtime) {
        //--------------------------------------------------------------------
        // allocate static memory on FPGA and move data
        //--------------------------------------------------------------------
        cl_int err;

        this->channel_packets_buf.resize(num_hbm_channels);
        this->channel_packets_ext.resize(num_hbm_channels);

        for (size_t c = 0; c < num_hbm_channels; c++) {
            this->channel_packets_ext[c].obj = this->channel_packets[c].data();
            this->channel_packets_ext[c].param = 0;
            #ifdef USE_DDR_CHANNEL
            this->channel_packets_ext[c].flags = DDR[c];
            #else
            this->channel_packets_ext[c].flags = HBM[c];
            #endif

            size_t channel_packets_size = sizeof(packet_t) * this->channel_packets[c].size();
            // std::cout << "channel_packets_size: " << channel_packets_size << std::endl;
            #ifdef USE_DDR_CHANNEL
            if (channel_packets_size / 1024 / 1024 >= 4 * 1024) {
                std::cout << "The maximum size of one buffer is 4GB in XRT, ";
                std::cout << "but trying to allocate " << channel_packets_size/1024/1024;
                std::cout << " MB on DDR " << c << std::endl;
            #else
            if (channel_packets_size / 1024 / 1024 >= 256) {
                std::cout << "The capcity of one HBM channel is 256 MB, ";
                std::cout << "but trying to allocate " << channel_packets_size/1024/1024;
                std::cout << " MB on HBM " << c << std::endl;
            #endif
                this->benchmark_results.push_back((benchmark_result){
                    .benchmark_name = this->name,
                    .spmspv_sparsity = -1,
                    .preprocess_time_s = this->preprocess_time_s,
                    .spmspv_time_ms = -1,
                    .throughput_GBPS = -1,
                    .throughput_GOPS = -1,
                    .verified = false,
                });
                return false;
            }

            this->channel_packets_buf[c] = CL_BUFFER_RDONLY(
                runtime.context,
                sizeof(packet_t) * this->channel_packets[c].size(),
                channel_packets_ext[c],
                err
            );
            if (err != CL_SUCCESS) {
                std::cout << "ERROR : exception catched when trying to create CL buffer of "
                #ifdef USE_DDR_CHANNEL
                          << channel_packets_size/1024/1024 << " MB on DDR "
                #else
                          << channel_packets_size/1024/1024 << " MB on HBM "
                #endif
                          << c << std::endl;
            }
            CHECK_ERR(err);
        }

        // Handle result only. Note: vector (deps on sparsity) is not static data
        result_ext.obj = result.data();
        result_ext.param = 0;
        result_ext.flags = HBM[21];

        size_t result_size = sizeof(IDX_VAL_T) * (this->csc_matrix.num_rows + 1);
        this->result_buf = CL_BUFFER_WRONLY(runtime.context, result_size, result_ext, err);
        CHECK_ERR(err);

        // transfer data
        for (size_t c = 0; c < num_hbm_channels; c++) {
            OCL_CHECK(err, err = runtime.command_queue.enqueueMigrateMemObjects({
                channel_packets_buf[c]
                }, 0 /* 0 means from host*/));
        }
        OCL_CHECK(err, err = runtime.command_queue.finish());
        std::cout << "INFO : Host -> Device data transfer complete!" << std::endl;

        //--------------------------------------------------------------------
        // invoke kernel
        //--------------------------------------------------------------------
        // set kernel arguments that won't change across row iterations
        std::cout << "INFO : Invoking kernel:";
        std::cout << " row_partitions: " << this->num_row_partitions << std::endl;

        for (size_t c = 0; c < num_hbm_channels; c++) {
            OCL_CHECK(err, err = runtime.spmspv.setArg(c, channel_packets_buf[c]));
        }

        size_t arg_index_offset = num_hbm_channels;
        OCL_CHECK(err, err = runtime.spmspv.setArg(arg_index_offset + 1, result_buf));
        OCL_CHECK(err, err = runtime.spmspv.setArg(arg_index_offset + 2, this->csc_matrix.num_rows));
        OCL_CHECK(err, err = runtime.spmspv.setArg(arg_index_offset + 3, this->csc_matrix.num_cols));

        return true;
    }

    void run(cl_runtime &runtime, float vector_sparsity) {
        //--------------------------------------------------------------------
        // handle the vector and send to device
        //--------------------------------------------------------------------
        cl_int err;

        unsigned vector_length = this->csc_matrix.num_cols;
        // unsigned vector_nnz_cnt = vector_length;
        unsigned vector_nnz_cnt = (unsigned)floor(vector_length * (1 - vector_sparsity));
        // unsigned vector_indices_increment = vector_length / vector_nnz_cnt;

        aligned_sparse_float_vec_t vector_float(vector_nnz_cnt);
        std::vector<unsigned> used_indices;
        used_indices.reserve(vector_nnz_cnt);
        for (size_t i = 0; i < vector_nnz_cnt; i++) {
            vector_float[i].val = (float)(rand() % 10) / 10;
            // vector_float[i].index = i * vector_indices_increment;
            while (true) {
                IDX_T x = rand() % this->csc_matrix.num_cols;
                for (size_t j = 0; j < used_indices.size(); j++) {
                    if (x == used_indices[j]) { //! TODO: FIX THIS BUG!
                        break; //! we need to reroll the index, not break
                    }
                }
                used_indices.push_back(x);
                vector_float[i].index = x;
                break;
            }
        }
        IDX_FLOAT_T vector_head;
        vector_head.index = vector_nnz_cnt;
        vector_head.val = 0;
        vector_float.insert(vector_float.begin(), vector_head);

        aligned_sparse_vec_t vector(vector_float.size());
        for (size_t i = 0; i < vector_nnz_cnt + 1; i++) {
            vector[i].index = vector_float[i].index;
            vector[i].val = vector_float[i].val;
        }

        CL_CREATE_EXT_PTR(vector_ext, vector.data(), HBM[20]);
        cl::Buffer vector_buf = CL_BUFFER_RDONLY(
            runtime.context,
            sizeof(IDX_VAL_T) * vector.size(),
            vector_ext,
            err
        );
        CHECK_ERR(err);

        OCL_CHECK(err, err = runtime.command_queue.enqueueMigrateMemObjects(
            {vector_buf}, 0 /* 0 means from host*/));
        OCL_CHECK(err, err = runtime.command_queue.finish());

        size_t arg_index_offset = num_hbm_channels;
        OCL_CHECK(err, err = runtime.spmspv.setArg(arg_index_offset + 0, vector_buf));

        //--------------------------------------------------------------------
        // compute reference
        //--------------------------------------------------------------------
        uint32_t involved_Nnz = 0;
        aligned_dense_float_vec_t ref_result;
        std::thread device_compute([&]{
            compute_ref(this->csc_matrix_float, vector_float, ref_result, involved_Nnz);
            // std::cout << "INFO : Compute reference complete!" << std::endl;
        });

        //--------------------------------------------------------------------
        // benchmarking
        //--------------------------------------------------------------------
        double total_time = 0;
        for (unsigned i = 0; i < NUM_RUNS; i++) {
            auto t0 = high_resolution_clock::now();
            OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmspv));
            OCL_CHECK(err, err = runtime.command_queue.finish());
            auto t1 = high_resolution_clock::now();
            total_time += double(duration_cast<microseconds>(t1 - t0).count()) / 1000;
        }
        std::cout << "INFO : SpMSpV Kernel complete "<< NUM_RUNS << " runs!" << std::endl;

        double average_time = total_time / NUM_RUNS;

        //--------------------------------------------------------------------
        // verify
        //--------------------------------------------------------------------
        OCL_CHECK(err, err = runtime.command_queue.enqueueMigrateMemObjects(
            {result_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
        OCL_CHECK(err, err = runtime.command_queue.finish());
        std::cout << "INFO : Device -> Host data transfer complete!" << std::endl;
        std::cout << "INFO : Result NNZ: " << result[0].index << std::endl;

        std::ofstream res_log("gblus_result.dat");
        unsigned result_nnz = result[0].index;
        res_log << "Result Nnz: " << result_nnz << "\n";
        for (size_t i = 1; i < result_nnz + 1; i++) {
            unsigned idx = result[i].index;
            unsigned pe = idx % PACK_SIZE;
            res_log << idx << ", " << pe << "\n";
        }
        res_log.close();

        aligned_dense_vec_t upk_result;
        convert_sparse_vec_to_dense_vec(result, upk_result, csc_matrix.num_rows);

        device_compute.join();
        bool verified = verify(ref_result, upk_result);
        std::cout << "INFO : Result verification complete!" << std::endl;

        double Mops = 2.0 * involved_Nnz / 1000 / 1000;
        double gbs = double(involved_Nnz * 2 * 4) / 1024.0 / 1024.0 / 1024.0;

        auto rec = (benchmark_result){
            .benchmark_name = this->name,
            .spmspv_sparsity = vector_sparsity,
            .preprocess_time_s = this->preprocess_time_s,
            .spmspv_time_ms = average_time,
            .throughput_GBPS = gbs / (average_time / 1000),
            .throughput_GOPS = Mops / average_time,
            .verified = verified,
        };
        std::cout << rec << std::endl;
        this->benchmark_results.push_back(rec);
    }
};

// std::vector<float> test_sparsity = { 0.50, 0.90, 0.990, 0.995, 0.999, 0.9995, 0.9999 };
std::vector<float> test_sparsity = {0.990};

//---------------------------------------------------------------
// main
//---------------------------------------------------------------
int main (int argc, char** argv) {
    // ./bench <dataset_name> <dataset_path> <hw_xclbin_path> <hw_mem_channels> <log_path>
    std::string name = argv[1], dataset = argv[2], xclbin = argv[3], metric = argv[5];
    size_t num_channels = std::atoi(argv[4]);

    test_harness test(name, dataset, num_channels);
    test.preprocessing(); // process in background thread

    // setenv("XCL_EMULATION_MODE", "sw_emu", true);

    // setup Xilinx openCL runtime
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

    // load different xclbin and create runtime respectively
    cl_int err;
    cl_runtime rt;
    rt.context = cl::Context(device, NULL, NULL, NULL);
    auto binary_buf = xcl::read_binary_file(xclbin);
    cl::Program::Binaries binary{{binary_buf.data(), binary_buf.size()}};
    cl::Program program(rt.context, {device}, binary, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "ERROR : Failed to program device with xclbin file" << std::endl;
        exit(EXIT_FAILURE);
    } else {
        std::cout << "INFO : Successfully programmed device with xclbin file" << std::endl;
    }
    OCL_CHECK(err, rt.spmspv = cl::Kernel(program, "spmspv", &err));
    OCL_CHECK(err, rt.command_queue = cl::CommandQueue(
        rt.context,
        device,
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
        &err
    ));

    // run benchmark with the specific num_hbm_channels
    std::cout << "------ Running benchmark on " << test.name << std::endl;
    if (test.warming_up(rt)) {
        for (const float &sparsity : test_sparsity) {
            test.set_sparsity_and_run(rt, sparsity);
        }
    }
    std::cout << "===== Benchmark Finished =====" << std::endl;

    // output benchmark results
    std::ofstream log(metric);
    size_t len = test.benchmark_results.size();
    for (size_t i = 0; i < len; ++i) {
        log << test.benchmark_results[i];
        log << (i == len - 1 ? "\n" : ", ");
    }
    log.close();

    return 0;
}
