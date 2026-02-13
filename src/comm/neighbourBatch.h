//
// Created by ishaan on 12/25/25.
//

#pragma once
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace core { struct CSRMatrix; }

namespace comm {
    struct CommPlan {
        // Sending Side (Responding to requests)
        std::vector<int> send_counts;   // How many items to send to each rank
        std::vector<int> send_displs;   // Offsets in the send buffer
        std::vector<int> pack_map;      // The specific LOCAL indices to pack into the send buffer
        int total_send = 0;             // Total size of send buffer

        // Receiving Side (My requests)
        std::vector<int> recv_counts;   // How many items I will receive from each rank
        std::vector<int> recv_displs;   // Offsets in the ghost buffer
        int total_recv = 0;             // Total size of ghost buffer
    };

    struct NeighborBatch {
        std::vector<long> needed_indices;
        std::vector<double> received_values;
    };


    inline std::unordered_map<int, NeighborBatch> batches;

    //Helper: The reverse of partition to find the rank that owns the specific row of matrix/vector
    long get_owner(long index, long N, int size);
    void build_batches(const core::CSRMatrix& mat, int my_rank, int world_size);
    CommPlan inspector_exchange(const core::CSRMatrix& mat);
    void renumber_cols(core::CSRMatrix& mat, const CommPlan& plan, const std::unordered_map<int, NeighborBatch> &batches);
}

