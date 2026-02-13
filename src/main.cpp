#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include "core/csr_matrix.h"
#include "io/mtx_reader.h"
#include "comm/neighbourBatch.h"
#include "executor.h"

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Who am I?
    MPI_Comm_size(MPI_COMM_WORLD, &size); // How many of us?

    // Parse Command Line Arguments
    // need at least one argument: the matrix file path.
    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <matrix_file.mtx>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Load and Partition Matrix
    if (rank == 0) std::cout << "[Step 1] Reading Matrix..." << std::endl;
    double t_io_start = MPI_Wtime();

    // Rank 0 reads, broadcasts; everyone filters their rows.
    core::CSRMatrix mat = io::read_matrix_market(argv[1]);

    double t_io_end = MPI_Wtime();
    if (rank == 0) std::cout << "IO Time: " << (t_io_end - t_io_start) << "s" << std::endl;

    // Analyze Communication Pattern
    if (rank == 0) std::cout << "[Step 2] Inspector Phase..." << std::endl;
    MPI_Barrier(MPI_COMM_WORLD); // Sync before timing
    double t_insp_start = MPI_Wtime();

    //Build Neighbor Batches (Identify who needs what from whom)
    comm::build_batches(mat, rank, size);

    // Create the CommPlan
    comm::CommPlan plan = comm::inspector_exchange(mat);

    // Renumber Columns
    comm::renumber_cols(mat, plan, comm::batches);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_insp_end = MPI_Wtime();
    if (rank == 0) std::cout << "Inspector Time: " << (t_insp_end - t_insp_start) << "s" << std::endl;

    // Initialize vectors. x = 1.0, y = 0.0
    std::vector<double> x(mat.local_rows, 1.0);
    std::vector<double> y(mat.local_rows, 0.0);

    // Create the Executor with the plan we just built
    SpMvExecutor executor(mat, plan);

    if (rank == 0) std::cout << "[Step 3] Benchmarking SpMV..." << std::endl;

    // Warmup iteration (allocates MPI buffers internally, etc.)
    executor.spmv(x, y);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_exec_start = MPI_Wtime();

    const int ITERATIONS = 1000;
    for (int i = 0; i < ITERATIONS; ++i) {
        executor.spmv(x, y);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_exec_end = MPI_Wtime();

    double total_time = t_exec_end - t_exec_start;
    double avg_time = total_time / ITERATIONS;

    long local_nnz = mat.nnz;
    long global_nnz = 0;
    MPI_Reduce(&local_nnz, &global_nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double gflops = (2.0 * global_nnz * ITERATIONS) / total_time / 1e9;

        std::cout << "=========================================" << std::endl;
        std::cout << " Distributed SpMV Results (" << size << " Ranks)" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << " Matrix       : " << argv[1] << std::endl;
        std::cout << " Global NNZ   : " << global_nnz << std::endl;
        std::cout << " Iterations   : " << ITERATIONS << std::endl;
        std::cout << " Total Time   : " << total_time << " s" << std::endl;
        std::cout << " Avg Time     : " << avg_time * 1000.0 << " ms" << std::endl;
        std::cout << " Performance  : " << gflops << " GFLOP/s" << std::endl;
        std::cout << "=========================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}