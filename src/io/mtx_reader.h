//
// Created by ishaan on 12/1/25.
//

#pragma once

#include <mpi.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include "../core/csr_matrix.h"

namespace io {
    struct triplet { // temp struct & comparator for sorting
        int row, col;
        double value;
        bool operator<(const triplet &other) const {
            return row < other.row || (row == other.row && col < other.col);
        }
    };

    inline core::CSRMatrix read_matrix_market(const std::string& filename) {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        long g_rows = 0, g_cols = 0, g_nnz = 0;

        //temp buffer for broadcasting
        std::vector<int> all_rows, all_cols;
        std::vector<double> all_values;

        if (rank == 0) {
            std::ifstream ifs(filename, std::ifstream::in);
            if (!ifs.is_open()) {
                std::cerr << "Could not open file " << filename << std::endl;
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            std::cout << "[IO] Reading " << filename << "..." << '\n';
            std::string line;
            while (std::getline(ifs, line)) {
                if (line.empty() || line[0] == '%') continue;
                break;
            }

            std::istringstream ss(line); // treat each line as an input stream so i can use stream ops on it
            ss >> g_rows >> g_cols >> g_nnz; // read int till ws while incrementing read-head so reads g_rows ws then g_cols so on.....

            std::vector<triplet> triplets;
            triplets.reserve(g_nnz);

            for (long i = 0; i < g_nnz; i++) {
                int r, c;
                double v;
                // .mtx file is 1-based index, convert to 0-based
                ifs >> r >> c >> v;
                triplets.push_back({r - 1, c - 1, v});
            }

            std::sort(triplets.begin(), triplets.end()); // sort using our comparator, have to do this because csr format is sorted i think

            // flatten into separate arrays for MPI_Bcast
            all_rows.resize(g_nnz);
            all_cols.resize(g_nnz);
            all_values.resize(g_nnz);

            for (long i = 0; i < g_nnz; i++) {
                all_rows[i] = triplets[i].row;
                all_cols[i] = triplets[i].col;
                all_values[i] = triplets[i].value;
            }

            std::cout << "[IO] Broadcast started for " << g_nnz << " entries..." << '\n';
        }

        // broadcast global dims to all
        MPI_Bcast(&g_rows, 1, MPI_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&g_cols, 1, MPI_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&g_nnz, 1, MPI_LONG, 0, MPI_COMM_WORLD);

        // for non 0 ranks prepare the buffers
        if (rank != 0) {
            all_rows.resize(g_nnz);
            all_cols.resize(g_nnz);
            all_values.resize(g_nnz);
        }


        MPI_Bcast(all_rows.data(), g_nnz, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(all_cols.data(), g_nnz, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(all_values.data(), g_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        core::CSRMatrix mat;
        mat.global_rows = g_rows;
        mat.global_cols = g_cols;

        core::CSRMatrix::partition(g_rows, rank, size, mat.local_rows, mat.rows_offset);

        long curMat_start = mat.rows_offset;
        long curMat_end = mat.local_rows + mat.rows_offset;
        mat.row_ptr.reserve(mat.local_rows + 1);
        mat.row_ptr.push_back(0);

        long current_global_row = curMat_start;
        for (long i = 0; i < g_nnz; ++i) {
            int r = all_rows[i];

            if (r >= curMat_end) break;

            if (r >= curMat_start) {
                while (current_global_row < r) {
                    mat.row_ptr.push_back(mat.vals.size());
                    current_global_row++;
                }

                // Store the data
                mat.vals.push_back(all_values[i]);
                mat.col_idx.push_back(all_cols[i]);
            }
        }

        while (current_global_row < curMat_end) {
            mat.row_ptr.push_back(mat.vals.size());
            current_global_row++;
        }

        mat.nnz = mat.vals.size();

        return mat;

    }
}// namespace io
