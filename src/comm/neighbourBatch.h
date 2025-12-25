//
// Created by ishaan on 12/25/25.
//

#pragma once
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace core { struct CSRMatrix; }

namespace comm {
    struct NeighborBatch {
        std::vector<long> needed_indices;
        std::vector<double> received_values;
    };


    inline std::unordered_map<int, NeighborBatch> batches;

    //Helper: The reverse of partition to find the rank that owns the specific row of matrix/vector
    long get_owner(long index, long N, int size);
    void build_batches(const core::CSRMatrix& mat, int my_rank, int world_size);

}

