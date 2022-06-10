#ifndef GRAPHLILY_IO_DATA_FORMAT_TILE_COO_H_
#define GRAPHLILY_IO_DATA_FORMAT_TILE_COO_H_

#include "data_loader.h"
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "common.h"

// #define DATA_FORMAT_TILE_COO_DEBUG

#ifdef DATA_FORMAT_TILE_COO_DEBUG
#include <bitset>
#include <fstream>
using bits32 = std::bitset<32>;
std::ofstream mat_log("data_format_matrix.log");
#endif

namespace spmv {
namespace io {


// Data structure for tile-COO matrix.
template<typename data_type, uint32_t pack_size>/*, data_type val_marker,
        uint32_t idx_row_marker, uint32_t idx_col_marker, uint32_t idx_dummy_marker>*/
struct TileCOO {
    /*! \brief The row indices of the sparse matrix */
    uint32_t num_rows;
    uint32_t num_cols;
    uint32_t num_row_tiles;
    uint32_t num_col_tiles;

    using indexed_data_type = struct { uint32_t index; data_type val;};
    std::vector<indexed_data_type> stream_data[pack_size];

    TileCOO(uint32_t channel_id, std::vector<CSRMatrix<data_type>> &csr_mats, uint32_t tile_width,
            uint32_t tile_height) {
        // dimension cannot exceed 2^16 due to the addressable space of half uint
        assert(tile_width <= 65536 && tile_height <= 65536);
        assert(csr_mats.size() == pack_size);

        this->num_rows = 0;
        for (const auto &mat : csr_mats) this->num_rows += mat.num_rows;
        this->num_cols = csr_mats[0].num_cols;

        // assert(this->num_rows % tile_height == 0);
        // assert(this->num_cols % tile_width == 0);
        // this->num_row_tiles = this->num_rows/tile_height;
        // this->num_col_tiles = this->num_cols/tile_width;
        this->num_row_tiles = (this->num_rows + tile_height - 1) / tile_height;
        this->num_col_tiles = (this->num_cols + tile_width - 1) / tile_width;

        std::vector<indexed_data_type> tiles[this->num_row_tiles]
                                            [this->num_col_tiles]
                                            [pack_size];
        // assume pack_size is 3, here is an example of concatenating matrices
        // row of new mat
        // 0: mat 0 row 0
        // 1: mat 1 row 0
        // 2: mat 2 row 0
        // 3: mat 0 row 1
        // 4: mat 1 row 1
        // ...
        for (uint32_t mat_idx = 0; mat_idx < pack_size; mat_idx++) {
            const auto &mat = csr_mats[mat_idx];
            for (uint32_t row_idx = 0; row_idx < mat.num_rows; row_idx++) {
                uint32_t real_row_idx = row_idx*pack_size + mat_idx;
                uint32_t start = mat.adj_indptr[row_idx];
                uint32_t end = mat.adj_indptr[row_idx + 1];
                for (uint32_t i = start; i < end; i++) {
                    uint32_t real_col_idx = mat.adj_indices[i];
                    // insert element to a certain tiles
                    uint32_t row_idx_in_tile = real_row_idx % tile_height;
                    uint32_t col_idx_in_tile = real_col_idx % tile_width;
                    tiles[real_row_idx/tile_height][real_col_idx/tile_width][mat_idx].push_back(
                        (indexed_data_type){
                            .index = ((row_idx_in_tile << 16) | (col_idx_in_tile & 0x0000ffff)),
                            .val = mat.adj_data[i]
                        }
                    );
#ifdef DATA_FORMAT_TILE_COO_DEBUG
                    mat_log << std::bitset<16>(row_idx_in_tile) << " "
                            << std::bitset<16>(col_idx_in_tile & 0x0000ffff) << ": "
                            << mat.adj_data[i] << std::endl;
#endif
                }
            }
        }

        // loop as [0,0] -> [0,1] -> [0,2] ... -> [1,0] -> ... , since tile rows
        // equals to nums of row partition in SpMV, while tile cols <=> nums of
        // cols partitions.
        for (uint32_t stream_idx = 0; stream_idx < pack_size; stream_idx++) {
            auto &current_stream = this->stream_data[stream_idx];
            current_stream.push_back({0, 0});
            for (uint32_t tile_row = 0; tile_row < this->num_row_tiles; tile_row++) {
                for (uint32_t tile_col = 0; tile_col < this->num_col_tiles; tile_col++) {
                    // marker is used to distinguish two adjacent tile columns,
                    // and temporarily reserved space to store the index of the
                    // last element (no marker in the end) of this tile row.
                    current_stream.insert(current_stream.end(),
                        tiles[tile_row][tile_col][stream_idx].begin(),
                        tiles[tile_row][tile_col][stream_idx].end()
                    );
                    indexed_data_type end_marker;
                    end_marker.val = VAL_MARKER;
                    end_marker.index = (tile_col == this->num_col_tiles - 1) ?
                                        IDX_ROW_TILE_MARKER : IDX_COL_TILE_MARKER;
                    current_stream.push_back(end_marker);
                }
            }
            // remove the marker at the end of stream
            current_stream.pop_back();
            // store the last position of stream in the header, i.e. the traverse
            // loop should be `for (i = 0; i <= header.index; i++)`
            current_stream[0].index = current_stream.size() - 1;
        }

        // pad dummy elements to the end of data, in order to keep the same size
        // across all streams
        uint32_t max_stream_size = this->stream_data[0][0].index;
        for (uint32_t sid = 1; sid < pack_size; sid++) {
            max_stream_size = std::max(max_stream_size, this->stream_data[sid][0].index);
        }
        indexed_data_type dummy_marker = {.index = IDX_DUMMY_MARKER, .val = VAL_MARKER};
        for (uint32_t sid = 0; sid < pack_size; sid++) {
            // Note: `k < max_stream_size` cannot be `<=` because the loop bases on offset
            // of two so-called last indices of streams
            for (uint32_t k = this->stream_data[sid][0].index; k < max_stream_size; k++) {
                this->stream_data[sid].push_back(dummy_marker);
            }
            this->stream_data[sid][0].index = max_stream_size;
        }
        // store the number of elements of last row partition in stream[1] header
        this->stream_data[1][0].index = ((this->num_rows % tile_height) == 0) ?
                                        tile_height / pack_size : ((this->num_rows % tile_height) / pack_size);

        // Summary of meta data in the format:
        // Header of stream[0] in a certain HBM channel: the last index of stream
        // data, and all streams in the same channels have the same alignments.
        // Header of stream[1]: the number of elements in the last row partition,
        // be used in result dump of PE.

#ifdef DATA_FORMAT_TILE_COO_DEBUG
        std::ofstream log("data_format_channel_"+std::to_string(channel_id)+".log");
        log << "VAL_MARKER is "          << VAL_MARKER(31,0) << std::endl
            << "IDX_DUMMY_MARKER is "    << bits32(IDX_DUMMY_MARKER) << std::endl
            << "IDX_ROW_TILE_MARKER is " << bits32(IDX_ROW_TILE_MARKER) << std::endl
            << "IDX_COL_TILE_MARKER is " << bits32(IDX_COL_TILE_MARKER) << std::endl;
        for (uint32_t sid = 0; sid < pack_size; sid++) {
            for (size_t i = 0; i <= this->stream_data[sid][0].index; i++) {
                auto idx = bits32(this->stream_data[sid][i].index);
                auto val = this->stream_data[sid][i].val(31,0);
                log << i << ": " << idx << " "<< val << std::endl;
            }
        }
#endif
    }
};

/*!
 * \brief Split a CSR matrix along the row dimension in a cyclic manner.
 *
 * \tparam DataT The data type of non-zero values of the sparse matrix.
 *
 * \param in The input matrix in CSR format.
 * \param factor The split factor.
 *
 * \return A vector of CSR matrices.
 */
template<typename DataT>
std::vector<CSRMatrix<DataT> > RowCyclicSplitCSR(CSRMatrix<DataT> const &in, uint32_t factor) {
    std::vector<uint32_t> nnz_each_row(in.num_rows);
    std::vector<uint32_t> nnz_each_split(factor, 0);
    for (uint32_t i = 0; i < in.num_rows; i++) {
        nnz_each_row[i] = in.adj_indptr[i + 1] - in.adj_indptr[i];
        nnz_each_split[i%factor] += nnz_each_row[i];
    }

    std::vector<CSRMatrix<DataT> > out;
    for (uint32_t f = 0; f < factor; f++) {
        CSRMatrix<DataT> csr_matrix;
        csr_matrix.num_cols = in.num_cols;
        csr_matrix.num_rows = (in.num_rows - 1 - f)/factor + 1;
        csr_matrix.adj_data = std::vector<DataT>(nnz_each_split[f]);
        csr_matrix.adj_indices = std::vector<uint32_t>(nnz_each_split[f]);
        csr_matrix.adj_indptr = std::vector<uint32_t>(csr_matrix.num_rows + 1);
        csr_matrix.adj_indptr[0] = 0;
        for (uint32_t i = f; i < in.num_rows; i+=factor) {
            uint32_t start_in = in.adj_indptr[i];
            uint32_t start_csr_matrix = csr_matrix.adj_indptr[i/factor];
            for (uint32_t j = 0; j < nnz_each_row[i]; j++) {
                csr_matrix.adj_data[start_csr_matrix + j] = in.adj_data[start_in + j];
                csr_matrix.adj_indices[start_csr_matrix + j] = in.adj_indices[start_in + j];
            }
            csr_matrix.adj_indptr[i/factor + 1] = csr_matrix.adj_indptr[i/factor] + nnz_each_row[i];
        }
        out.push_back(csr_matrix);
    }

    return out;
}

}  // namespace io
}  // namespace spmv

#endif
