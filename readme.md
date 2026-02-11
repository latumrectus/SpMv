
---

# Distributed Sparse Matrix–Vector Multiplication (SpMV) with MPI

## Overview

This project implements a **communication-aware distributed Sparse Matrix–Vector Multiplication (SpMV)** engine in **C++ using MPI**, with a focus on:

* Explicit communication planning
* Compute–communication overlap
* Production-style software engineering

The system follows an **inspector–executor model**:

* A one-time **inspector phase** analyzes the sparse matrix structure and constructs an optimized communication plan.
* A repeated **executor phase** performs SpMV iterations using **nonblocking halo exchange** and **overlapped computation**.

The project targets irregular, real-world sparse matrices and emphasizes **measurable performance improvements** over naïve blocking implementations.

---

## Key Features

* Distributed CSR storage with deterministic 1D row partitioning
* Inspector–executor architecture (static communication pattern)
* Sparse halo exchange using precomputed neighbor batches
* Nonblocking MPI communication (`MPI_Isend`, `MPI_Irecv`)
* Overlapped computation and communication
* Hybrid MPI + OpenMP parallelism
* Instrumentation for detailed performance breakdowns
* Clean modular C++ design with CMake build system

---

## Architecture

```
dist-spmv/
├── core/        # Passive data structures (CSR matrix, partitioning)
├── comm/        # Communication planning and halo exchange
├── src/         # SpMV executor and main driver
├── tests/       # Correctness tests
├── bench/       # Benchmark drivers
├── scripts/     # Experiment and plotting scripts
└── README.md
```

---

## Core Concepts

### Distributed CSR Matrix

Each MPI rank owns a contiguous block of rows of the global matrix in **Compressed Sparse Row (CSR)** format.

```
Global matrix A:
  rows [offset, offset + local_rows)
```

Column indices remain **global**, enabling flexible communication planning.

---

## Inspector Phase (Setup)

The inspector phase runs **once per matrix** and determines all communication required during SpMV.

### 1. Neighbor Discovery

For each local nonzero `(i, j)`:

* Determine which MPI rank owns column `j`
* If non-local, record `j` in a **neighbor batch** for that rank

```cpp
struct NeighborBatch {
    std::vector<int> needed_indices;   // Global column indices required
    std::vector<double> received_values; // Buffer for halo values
};
```

Only neighbors that are actually needed are stored (sparse communication graph).

---

### 2. Metadata Exchange

Each rank initially knows:

> “Which values I need from each neighbor.”

A one-time **metadata exchange** makes this symmetric:

1. Exchange request counts using `MPI_Alltoall`
2. Exchange requested indices using `MPI_Alltoallv`
3. Build **send batches**:

    * For each neighbor, determine which local vector entries must be sent

After this step, every rank knows:

* What it must **send**
* What it will **receive**
* Exact message sizes and neighbor relationships

This completes the inspector phase.

---

## Executor Phase (SpMV Loop)

Each SpMV iteration performs:

1. Pack send buffers from local vector `x`
2. Post nonblocking receives for halo values
3. Post nonblocking sends to neighbors
4. Compute rows using only local data
5. Wait for halo completion
6. Compute rows dependent on halo data

This enables **true overlap of communication and computation**.

```
Communication  |=====>|
Computation    |==LOCAL==|==REMOTE==|
```

---

## Parallelism Model

* **MPI**: Distributed-memory parallelism across ranks
* **OpenMP**: Thread-level parallelism within each rank
* Thread-safe halo buffers and deterministic execution

---

## Performance Instrumentation

Each iteration records:

* Local computation time
* Communication time
* Communication wait time
* Total iteration time

Metrics are logged per rank to CSV files for post-analysis.

---

## Benchmarks

### Datasets

* SuiteSparse matrices (e.g., `web-Google`, `roadNet-CA`)
* Synthetic sparse matrices (optional)

### Experiments

* **Strong scaling**: Fixed problem size, increasing MPI ranks
* **Weak scaling**: Fixed rows per rank
* **Overlap analysis**: Blocking vs. nonblocking halo exchange

**Target result:**

> ≥ 15–25% speedup from overlapping communication on irregular matrices

---

## Correctness

* Sequential SpMV reference implementation
* Distributed SpMV validated for `P = 1, 2, 4, ...`
* Infinity-norm error check:

```
||y_dist − y_ref||∞ < 1e-12
```

Results are invariant to the number of MPI ranks.

---

## Build Instructions

### Requirements

* C++17 compiler
* MPI (OpenMPI or MPICH)
* Optional: OpenMP

### Build

```bash
mkdir build && cd build
cmake .. -DENABLE_OPENMP=ON -DCMAKE_BUILD_TYPE=Release
make -j
```

### Run

```bash
mpirun -np 4 ./spmv matrix.mtx
```

---

## Why This Project Matters

This project demonstrates:

* Practical MPI programming beyond collectives
* Communication-aware algorithm design
* Inspector–executor optimization patterns
* Performance measurement and reasoning
* Production-quality C++ system design

It closely mirrors techniques used in real HPC libraries such as **PETSc** and **Trilinos**, while remaining self-contained and readable.

---

## Future Extensions

* 2D block decomposition
* METIS graph partitioning
* MPI persistent communication
* GPU offload (CUDA / cuSPARSE)
* Parallel I/O (MPI-IO)

---

## Author

**Ishaan Bagai**
Distributed systems • Parallel programming • Performance engineering

---
