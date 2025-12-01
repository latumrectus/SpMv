//
// Created by ishaan on 12/1/25.
//

#pragma once

#include <vector>
#include <iostream>
//#include <algorithm>

namespace core {
    struct CSRMatrix {
        std::vector<int> row_ptr, col_idx;
        std::vector<double> vals;

        long global_rows = 0, global_cols = 0, local_rows = 0, rows_offset = 0, nnz = 0;

        // Helper: Deterministic 1D row partitioning
        // Calculates the num rows a rank gets and where they start, if number is not evenly divisible, remainder is given to the first "rem" ranks
        static void partition(const long N, const int rank, const int size, long& out_local_rows, long& out_offset) {
            const long base = N/size;
            const long rem = N%size;

            out_local_rows = (rank < rem) ? base + 1 : base;  // if rank is less than remainder give this rank 1 extra row
            out_offset = rank * base + std::min<long>(rank, rem); // offset is rank*base + min(rank, rem)
        }

        void print_info(int rank) const {
            std::cout << "[Rank " << rank << "] "
                      << "Owns " << local_rows << " rows "
                      << "(Global: " << rows_offset << " -> " << rows_offset + local_rows - 1 << ") "
                      << "| Local NNZ: " << vals.size()
                      << std::endl;
        }
    };
}// namespace core