//
// Created by ishaan on 12/25/25.
//
#include "neighbourBatch.h"
#include "../core/csr_matrix.h"

namespace comm {
    void build_batches(const core::CSRMatrix& mat, int my_rank, int world_size) {

        batches.clear();

        for (int col : mat.col_idx) {
            long owner = get_owner(col, mat.global_cols, world_size);
            if (owner == my_rank) {
                continue;
            }
            batches[owner].needed_indices.push_back(col);
        }
        for (auto& kv : batches) {
            NeighborBatch& batch = kv.second;
            std::sort(batch.needed_indices.begin(), batch.needed_indices.end());
            auto last = std::unique(batch.needed_indices.begin(), batch.needed_indices.end());
            batch.needed_indices.erase(last, batch.needed_indices.end());
        }
    }

    long get_owner(long index, long N, int size) {
        long base = N / size;
        long rem = N % size;

        long lucky_rows = rem * (base + 1);

        if (index < lucky_rows) {
            return index/(base+1);
        }

        return rem + (index - lucky_rows)/base;
    }
}