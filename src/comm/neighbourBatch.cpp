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

    CommPlan inspector_exchange(const core::CSRMatrix& mat) {
        int world_size, my_rank;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

        CommPlan plan;

        // prep MY request counts
        std::vector request_counts(world_size, 0);
        std::vector incoming_req_counts(world_size, 0);

        // important lesson: dont use the operator[] with hashmap because it'll insert by default
        for (const auto& pair : batches) {
            int rank_id = pair.first;
            const NeighborBatch& batch = pair.second;
            request_counts[rank_id] = static_cast<int>(batch.needed_indices.size());
        }

        MPI_Alltoall(
            request_counts.data(), 1, MPI_INT,
            incoming_req_counts.data(), 1, MPI_INT,
            MPI_COMM_WORLD
        );

        // Calculate displacements
        std::vector<int> sdispls(world_size);
        std::vector<int> rdispls(world_size);

        // Use exclusive prefix sum
        std::exclusive_scan(request_counts.begin(), request_counts.end(), sdispls.begin(), 0);
        std::exclusive_scan(incoming_req_counts.begin(), incoming_req_counts.end(), rdispls.begin(), 0);

        // Calculate total buffer sizes
        int total_requests_out = sdispls.back() + request_counts.back();
        int total_requests_in  = rdispls.back() + incoming_req_counts.back();

        std::vector<int> requests_out_buf(total_requests_out);
        std::vector<int> requests_in_buf(total_requests_in);

        // Iterate through the map to copy data into the pre-calculated positions
        for (const auto& pair : batches) {
            int target_rank = pair.first;
            const NeighborBatch& batch = pair.second;

            // where does this rank's data start?
            int start_offset = sdispls[target_rank];

            std::copy(batch.needed_indices.begin(),
                      batch.needed_indices.end(),
                      requests_out_buf.begin() + start_offset);
        }

        // Exchange the actual Indices
        // requests_in_buf will now contain Global Indices that neighbors need ME to send THEM
        MPI_Alltoallv(
            requests_out_buf.data(), request_counts.data(), sdispls.data(), MPI_INT,
            requests_in_buf.data(), incoming_req_counts.data(), rdispls.data(), MPI_INT,
            MPI_COMM_WORLD
        );


        plan.recv_counts = request_counts;
        plan.recv_displs = sdispls;
        plan.total_recv = total_requests_out;

        plan.send_counts = incoming_req_counts;
        plan.send_displs = rdispls;
        plan.total_send = total_requests_in;

        plan.pack_map.resize(total_requests_in);
        long my_start_row = mat.rows_offset;

        for (int i = 0; i < total_requests_in; i++) {
            long global_idx = requests_in_buf[i];
            plan.pack_map[i] = static_cast<int>(global_idx - my_start_row);
        }

        return plan;
    }
}