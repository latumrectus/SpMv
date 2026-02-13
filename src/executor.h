//
// Created by ishaan on 2/13/26.
//

#pragma once

#include <mpi.h>
#include <vector>
#include <omp.h>
#include <iostream>
#include "core/csr_matrix.h"
#include "comm/neighbourBatch.h"

class SpMvExecutor {

private:
    const core::CSRMatrix& mat;
    const comm::CommPlan& plan;

    // Buffers for communication
    // send_buffer: data we pack from x to send to neighbors
    // recv_buffer: ghost data we receive from neighbors
    std::vector<double> send_buffer;
    std::vector<double> recv_buffer;

    // MPI Request tracking
    std::vector<MPI_Request> requests;

public:
    SpMvExecutor(const core::CSRMatrix& m, const comm::CommPlan& p)
        : mat(m), plan(p) {
        send_buffer.resize(plan.total_send);
        recv_buffer.resize(plan.total_recv);
        requests.reserve(plan.send_counts.size() * 2);
    }
    //Matrix Vector multiplication loop: y = Ax
    void spmv(const std::vector<double>& x, std::vector<double>& y) {

        // sanity check
        if (x.size() != static_cast<size_t>(mat.local_rows)) {
            std::cerr << "Error: Input vector size mismatch!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        requests.clear();
        int world_size = static_cast<int>(plan.send_counts.size());

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < plan.pack_map.size(); ++i) {
            send_buffer[i] = x[plan.pack_map[i]];
        }

        // Post receives
        for (int r = 0; r < world_size; ++r) {
            int count = plan.recv_counts[r];
            if (count > 0) {
                MPI_Request req;
                // Receive directly into the correct offset of recv_buffer
                MPI_Irecv(&recv_buffer[plan.recv_displs[r]], count, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &req);
                requests.push_back(req);
            }
        }

        // Post Sends
        for (int r = 0; r < world_size; ++r) {
            int count = plan.send_counts[r];
            if (count > 0) {
                MPI_Request req;
                // Send from the correct offset of send_buffer
                MPI_Isend(&send_buffer[plan.send_displs[r]], count, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &req);
                requests.push_back(req);
            }
        }

        // Ensure we have all ghost data before computing boundary rows.
        // (Future optimization: Overlap by computing local-only rows here)
        if (!requests.empty()) {
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        }


        // With renumbered columns:
        // col_idx < local_rows  => Value is in 'x'
        // col_idx >= local_rows => Value is in 'recv_buffer' (at index: col_idx - local_rows)

        const int local_rows = mat.local_rows;

        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < local_rows; ++i) {
            double sum = 0.0;
            int row_start = mat.row_ptr[i];
            int row_end   = mat.row_ptr[i+1];

            for (int j = row_start; j < row_end; ++j) {
                int col = mat.col_idx[j];
                double val = mat.vals[j];
                double vec_val;

                if (col < local_rows) {
                    // Local Data
                    vec_val = x[col];
                } else {
                    // Ghost Data
                    vec_val = recv_buffer[col - local_rows];
                }
                sum += val * vec_val;
            }
            y[i] = sum;
        }

    }
};