#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include "core/csr_matrix.h"
#include "io/mtx_reader.h"

// helper for timing
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char *argv[]) {
    // initialize the MPI Environment
    // always the first thing called.
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Who am I?
    MPI_Comm_size(MPI_COMM_WORLD, &size); // How many of us?

    // Parse Command Line Arguments
    // need at least one argument: the matrix file path.
    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <matrix_file.mtx> [options]" << '\n';
        }
        MPI_Finalize();
        return 1;
    }
    std::string input_file = argv[1];

    // Print Job Metadata (Rank 0 only)
    if (rank == 0) {
        std::cout << "==================================================" << '\n';
        std::cout << " Distributed Sparse Matrix Engine (MPI + CUDA)" << '\n';
        std::cout << "==================================================" << '\n';
        std::cout << " Ranks        : " << size << '\n';
        std::cout << " Input File   : " << input_file << '\n';
        // std::cout << " CUDA Devices : " << // for later
        std::cout << "==================================================" << '\n';
    }

    // Load & Distribute
    // Rank 0 reads -> Broadcasts -> Everyone filters their part
    core::CSRMatrix matrix = io::read_matrix_market(input_file);

    // Each rank will print its stats to check if the math was right
    matrix.print_info(rank);

    MPI_Finalize();
    return 0;
}