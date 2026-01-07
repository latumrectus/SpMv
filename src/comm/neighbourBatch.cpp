//
// Created by ishaan on 12/25/25.
//
#include "neighbourBatch.h"
#include <mpi.h>
#include <numeric>
#include <vector>
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

    void inspector_exchange() {
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        std::vector<int> send_counts(world_size, 0);
        std::vector<int> receive_counts(world_size, 0);

        // important lesson: dont use the operator[] with hashmap because it'll insert by default if the value doesnt exist and in this case that'll ruin the sparsity of the hashmap
        for (const auto& pair : batches) {
            int rank_id = pair.first;
            const NeighborBatch& batch = pair.second;
            send_counts[rank_id] =
                static_cast<int>(batch.needed_indices.size());
        }

        MPI_Alltoall(
            send_counts.data(), 1, MPI_INT,
            receive_counts.data(), 1, MPI_INT,
            MPI_COMM_WORLD
        );

        // Calculate displacements
        std::vector<int> sdispls(world_size);
        std::vector<int> rdispls(world_size);

        // Use exclusive prefix sum
        std::exclusive_scan(send_counts.begin(), send_counts.end(), sdispls.begin(), 0);
        std::exclusive_scan(receive_counts.begin(), receive_counts.end(), rdispls.begin(), 0);

        // Calculate total buffer sizes
        int total_send = sdispls.back() + send_counts.back();
        int total_recv = rdispls.back() + receive_counts.back();

        std::vector<int> send_buffer(total_send);
        std::vector<int> recv_buffer(total_recv);

        // Iterate through the map to copy data into the pre-calculated positions
        for (const auto& pair : batches) {
            int target_rank = pair.first;
            const NeighborBatch& batch = pair.second;

            // where does this rank's data start?
            int start_offset = sdispls[target_rank];

            std::copy(batch.needed_indices.begin(),
                      batch.needed_indices.end(),
                      send_buffer.begin() + start_offset);
        }

        MPI_Alltoallv(
            send_buffer.data(), send_counts.data(), sdispls.data(), MPI_INT,
            recv_buffer.data(), receive_counts.data(), rdispls.data(), MPI_INT,
            MPI_COMM_WORLD
        );
    }
}