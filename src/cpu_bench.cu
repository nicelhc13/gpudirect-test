#include <iostream>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <omp.h>
#include "mpi.h"

#define NODES      2

#ifdef ENABLE_MULTI_MSG
#define MAX_COMM   3000
#else
#define MAX_COMM   1
#endif

#ifdef ENABLE_HEAVY
#define MSG_SIZE   5500000 //! 22MB
#else
#define MSG_SIZE   100     //! 25xINT
#endif

#define BLOCKING_COMM_MODE 0
#define NONBLOCKING_COMM_MODE 1
#define NONBLOCKING_SYNC_COMM_MODE 2

/**
 * Verify both gpu and cpu buffers.
 * Each element is initialized with the corresponding index.
 */
__global__ void verifyGPUBuffers(int *send_buffer,
                                 int *reduce) {
  for (int i = 0; i < MSG_SIZE; i++) {
    if (send_buffer[i] != i) {
      printf("%d is failed to verified; %d\n", i, send_buffer[i]);
      *reduce += 1;
      break;
    }
  }
  if (*reduce > 0) {
    printf("verifying failed\n");
  }
}

void verifyRecvedBuffers(int *send_buffer,
                         int *reduce) {
  for (int i = 0; i < MSG_SIZE; i++) {
    if (send_buffer[i] != i) {
      printf("%d is failed to verified; %d\n", i, send_buffer[i]);
      *reduce += 1;
      break;
    }
  }
  if (*reduce > 0) {
    printf("verifying failed\n");
  }
}

//! Initialize buffer. Each element is initialized with its index.
void initializeBuffers(int *send_buffer) {
  for (int i = 0; i < MSG_SIZE; i++) {
    send_buffer[i] = i;
  }
}

//! Print buffer.
void printBuffer(int *buffer) {
  for (int i = 0; i < MSG_SIZE; i++) {
    printf("\tbuffer[%d] = %d ", i, buffer[i]);
  }
  printf("\n");
}

void run(int *buffer, int rank) {
  if (rank == 0) { ///< Rank0 node.
    printf("RANK 0: Initialize msg..\n");
    initializeBuffers(buffer);
    printf("RANK 0: Send msg..\n");
#ifdef PRINT_BUFFER
    printf("RANK 0: Printing send-msg %d-th\n", i);
    printBuffer(buffer);
#endif
    //! Send the msg one by one.
    for (int neigh = 1; neigh < NODES; neigh++) {
#ifdef PARALLEL_MSG_MODE
      #pragma omp parallel for
#endif
      for (int i = 0; i < MAX_COMM; i++) {
        printf("RANK 0: Sending msg %d-th\n", i);
#if (COMM_MODE != BLOCKING_COMM_MODE)
        MPI_Request req;
        MPI_Status stat;
#endif
#if (COMM_MODE == BLOCKING_COMM_MODE)
        std::cout << "Blocking Mode sending..\n";
        MPI_Send(buffer, MSG_SIZE, MPI_INT,
                 neigh, 0, MPI_COMM_WORLD);
#elif (COMM_MODE == NONBLOCKING_COMM_MODE)
        std::cout << "Non-Blocking Mode sending..\n";
        MPI_Isend(buffer, MSG_SIZE, MPI_INT,
                  neigh, 0, MPI_COMM_WORLD, &req);
#elif (COMM_MODE == NONBLOCKING_SYNC_COMM_MODE)
        std::cout << "Non-Blocking synchronous Mode sending..\n";
        MPI_Issend(buffer, MSG_SIZE, MPI_INT,
                   neigh, 0, MPI_COMM_WORLD, &req);
#endif
        printf("RANK 0: Sending msg %d-th to %d: done\n", i, neigh);
#if (COMM_MODE != BLOCKING_COMM_MODE)
        MPI_Wait(&req, &stat); 
#endif
      }
    }
    printf("RANK 0: Rank 0 is done\n");
  } else { ///< Not rank0 nodes.
    //! Initialize receiver-side buffers.
    memset(buffer, 0, sizeof(int)*MSG_SIZE);
#ifdef PRINT_BUFFER
    printf("RANK %d: Print recv buffer before recving\n", rank);
    printBuffer(buffer);
#endif
    int reduce;
    //! Receive the msg one by one.
    for (int i = 0; i < MAX_COMM; i++) {
      reduce = 0;
      printf("RANK %d: Tries to recv %d-th msg (size: %d)\n", rank, i, MSG_SIZE);
#if COMM_MODE == BLOCKING_COMM_MODE
      MPI_Recv(buffer, MSG_SIZE, MPI_INT, 0,
               0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#else
      MPI_Request req;
      MPI_Status stat;
      MPI_Irecv(buffer, MSG_SIZE, MPI_INT, 0,
                0, MPI_COMM_WORLD, &req); 
      MPI_Wait(&req, &stat);
#endif

#ifdef PRINT_BUFFER
      printf("RANK %d: Print recv %d-th msg\n", rank, i);
      printBuffer(buffer);
#endif
      printf("Starts to verifying.. %d-th msg\n", i);
      verifyRecvedBuffers(buffer, &reduce); 
      printf("Verified done.. %d-th msg\n", i);
    }
    printf("RANK %d: Received msg\n", rank);

    //! Copy to GPU.
    int* gpu_buffer;
    cudaMalloc((void **)&gpu_buffer, sizeof(int)*MSG_SIZE);
    cudaMemcpy(gpu_buffer, buffer, sizeof(int)*MSG_SIZE, cudaMemcpyHostToDevice);
    printf("RANK %d: Verify the copied data from cpu to gpu..\n", rank);
    reduce = 0;
    verifyGPUBuffers<<<1,1>>>(gpu_buffer, &reduce);
    printf("RANK %d: All jobs are done\n", rank);
    cudaDeviceSynchronize();
  }
}

int main(int argc, char** argv) {
  int rank, p;
  int *buffer;
  int supportProvided;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &supportProvided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  printf("Welcome to MPI world. %d out of %d processors\n",
         rank, p);
  printf("Number of nodes: %d, Number of msgs: %d,"
         "Msg size: %d\n", NODES, MAX_COMM, MSG_SIZE);
#ifdef PARALLEL_MSG_MODE
  omp_set_num_threads(56);
  printf("Parallel mode is enabled. Used number of threads is 56\n");
#endif

  //! Initialize buffers.
  buffer = (int *) malloc(MSG_SIZE*sizeof(int));

  run(buffer, rank);

  MPI_Finalize();
  free(buffer);
  return 0;
}
