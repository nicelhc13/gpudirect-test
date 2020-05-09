#include <iostream>
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

__global__ void initializeBuffers(int *send_buffer) {
  for (int i = 0; i < MSG_SIZE; i++) {
    send_buffer[i] = i;
  }
}

__global__ void printBuffer(int *buffer) {
  for (int i = 0; i < MSG_SIZE; i++) {
    printf("\tbuffer[%d] = %d ", i, buffer[i]);
  }
  printf("\n");
}

__global__ void verifyRecvedBuffers(int *send_buffer,
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

int main(int argc, char** argv) {
  int rank, p;
  int *buffer;
  int supportProvided;

  cudaSetDevice(0);
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
  cudaMalloc((void **) &buffer, MSG_SIZE*sizeof(int));

  if (rank == 0) { ///< Rank0 node.
    printf("RANK 0: Initialize msg..\n");
    initializeBuffers<<<1, 1>>>(buffer);
    printf("RANK 0: Send msg..\n");
#ifdef PRINT_BUFFER
    printf("RANK 0: Printing send-msg\n");
    printBuffer<<<1, 1>>>(buffer);
#endif
    //! Send the msg one by one.
    for (int neigh = 1; neigh < NODES; neigh++) {
#ifdef PARALLEL_MSG_MODE
      #pragma omp parallel for
#endif
      for (int i = 0; i < MAX_COMM; i++) {
        printf("RANK 0: Sending msg %d-th\n", i);
        MPI_Send(buffer, MSG_SIZE, MPI_INT,
                 neigh, 0, MPI_COMM_WORLD);
        printf("RANK 0: Sending msg %d-th to %d: done\n", i, neigh);
      }
    }
    printf("RANK 0: Rank 0 is done\n");
  } else { ///< Not rank0 nodes.
    //! Initialize receiver-side buffers.
    cudaMemset(buffer, 0, sizeof(int)*MSG_SIZE);
#ifdef PRINT_BUFFER
    printf("RANK %d: Print recv buffer before recving\n", rank, i);
    printBuffer<<<1, 1>>>(buffer);
#endif
    int *reduce;
    cudaMalloc(&reduce, sizeof(int));
    //! Receive the msg one by one.
    for (int i = 0; i < MAX_COMM; i++) {
      cudaMemset(reduce, 0, sizeof(int));
      printf("RANK %d: Tries to recv %d-th msg (size: %d)\n", rank, i, MSG_SIZE);
      MPI_Recv(buffer, MSG_SIZE, MPI_INT, 0,
               0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#ifdef PRINT_BUFFER
      printf("RANK %d: Print recv %d-th msg\n", rank, i);
      printBuffer<<<1, 1>>>(buffer);
#endif
      printf("Starts to verifying.. %d-th msg\n", i);
      verifyRecvedBuffers<<<1, 1>>>(buffer, reduce);
      printf("Verified done.. %d-th msg\n", i);
    }
    printf("RANK %d: Received msg\n", rank);
  }

  MPI_Finalize();
  cudaFree(buffer);
  return 0;
}
