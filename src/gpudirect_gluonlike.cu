#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <omp.h>
#include <thread>
#include "mpi.h"
#include "timer.h"

#define BUFFER_SIZE 55000000
#define CPU_BUFFER_MODE

int rank, num_hosts;
bool run_probing;

#define GPU_ERR_CHK(ans) { GpuAssert((ans), __FILE__, __LINE__); }
inline void GpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void check_gpu(int rank) {
  printf("[%d] GPU is set\n", rank);
}

/* Construct a buffer depending on a rank */
__global__ void construct_own_data(uint64_t* buffer, int rank) {
  for (size_t i = 0; i < BUFFER_SIZE; i++) {
    buffer[rank*BUFFER_SIZE + i] = rank*BUFFER_SIZE + i; 
  }
}

/* Verify a buffer depending on a rank */
__global__ void verify_receive_buffer(uint64_t* buffer, uint32_t sender, int rank) {
  size_t reduce = 0;

  printf("[%d] Verify recv buffer starts\n", rank);
  for (size_t i = 0; i < BUFFER_SIZE; i++) {
#if 0
    if (rank == 0) {
    printf("%lld\n", i);
    printf("%lld\n", sender * BUFFER_SIZE + i);
    printf("buffer[%lld]: %lld\n", i, uint64_t(buffer[i]));
    }
#endif
    if (buffer[i] != (sender*BUFFER_SIZE + i)) {
      reduce++;
    }
  }

  if (reduce > 0) {
    printf("[%d --> %d] verifying failed.\n", sender, rank);
  } else {
    printf("[%d --> %d] verifying succeeded.\n", sender, rank);
  }
}

__global__ void verify_final_data(uint64_t* buffer, int rank, int num_hosts) {
  size_t reduce = 0; 
  printf("[%d] Verify final data\n", rank);
  for (size_t i = 0; i < num_hosts * BUFFER_SIZE; i++) {
#if 0
    if (rank == 0) 
    printf("buffer[%lld]: %lld\n", i, uint64_t(buffer[i]));
#endif

    if (buffer[i] != i) { reduce ++; } 
  }

  if (reduce > 0) {
    printf("[%d] verifying failed.\n", rank);
  } else {
    printf("[%d] verifying succeeded.\n", rank);
  }
}

void CopyReceivedData(uint64_t* received_buf, uint64_t* gpu_data, uint32_t from,
                      size_t received_buf_size) {
  GPU_ERR_CHK(cudaMemcpy(&gpu_data[from*BUFFER_SIZE], received_buf,
                         received_buf_size, cudaMemcpyDeviceToDevice)); 
}

void PostReceive(std::vector<uint64_t*>& recv_buffer,
                 std::vector<MPI_Request>& request) {
  for (uint32_t h = 1; h < num_hosts; ++h) {
    uint32_t from = (rank + h) % num_hosts; 

#ifdef CPU_BUFFER_MODE
    MPI_Irecv(recv_buffer[from],
              BUFFER_SIZE * sizeof(uint64_t), MPI_BYTE, from, 32767,
              MPI_COMM_WORLD, &request[from]);
#else
    MPI_Irecv(recv_buffer[from], BUFFER_SIZE * sizeof(uint64_t),
              MPI_BYTE, from, 32767,
              MPI_COMM_WORLD, &request[from]);
#endif
  }
}

void Send(uint64_t* data, std::vector<uint64_t*>& send_buffer) {
  /* Copy the current host part from the data to the send buffer */
  GPU_ERR_CHK(cudaMemcpy(send_buffer[rank],
              &data[rank * BUFFER_SIZE],
              sizeof(uint64_t) * BUFFER_SIZE,
              cudaMemcpyDeviceToDevice));
#ifdef CPU_BUFFER_MODE
  uint64_t* cpu_send_buffer =
          (uint64_t *) malloc(sizeof(uint64_t) * BUFFER_SIZE);
  GPU_ERR_CHK(cudaMemcpy(cpu_send_buffer,
              send_buffer[rank],
              sizeof(uint64_t) * BUFFER_SIZE,
              cudaMemcpyDeviceToHost));
#endif

  for (uint32_t h = 1; h < num_hosts; ++h) {
    uint32_t to = (rank + h) % num_hosts; 

#ifdef CPU_BUFFER_MODE
    MPI_Send(cpu_send_buffer, BUFFER_SIZE * sizeof(uint64_t),
             MPI_BYTE, to, 32767, MPI_COMM_WORLD); 
#else
    MPI_Send(send_buffer[rank], BUFFER_SIZE * sizeof(uint64_t),
             MPI_BYTE, to, 32767, MPI_COMM_WORLD); 
#endif
  }
}

#ifdef CPU_BUFFER_MODE
void Receive(std::vector<uint64_t*>& cpu_recv_buffer,
             std::vector<uint64_t*>& recv_buffer,
             std::vector<MPI_Request>& request,
             uint64_t* gpu_data) {
#else
void Receive(std::vector<uint64_t*>& recv_buffer,
             std::vector<MPI_Request>& request,
             uint64_t* gpu_data) {
#endif
  for (uint32_t h = 1; h < num_hosts; ++h) {
    uint32_t from = (rank + h) % num_hosts; 
    MPI_Status status;
    MPI_Wait(&request[from], &status);
    int size{0};
    MPI_Get_count(&status, MPI_BYTE, &size);

#ifdef CPU_BUFFER_MODE
    GPU_ERR_CHK(cudaMemcpy(recv_buffer[from],
                  cpu_recv_buffer[from],
                  sizeof(uint64_t) * BUFFER_SIZE,
                  cudaMemcpyHostToDevice));
#endif
    verify_receive_buffer<<<1, 1>>>(recv_buffer[from],
                                    from, rank);
    cudaDeviceSynchronize();
    CopyReceivedData(recv_buffer[from], gpu_data, from, size);
  }
}

void StartCommunication() {
  std::vector<uint64_t*> recv_buffer, send_buffer;
#ifdef CPU_BUFFER_MODE
  std::vector<uint64_t*> cpu_recv_buffer;
#endif
  uint64_t* gpu_data; 
  std::vector<MPI_Request> request;

  recv_buffer.resize(num_hosts);
  send_buffer.resize(num_hosts);
#ifdef CPU_BUFFER_MODE
  cpu_recv_buffer.resize(num_hosts);
#endif

  request.resize(num_hosts, MPI_REQUEST_NULL);

  for (uint32_t h = 0; h < num_hosts; h++) {
    uint64_t** hth_recv_buffer = &recv_buffer[h];
    uint64_t** hth_send_buffer = &send_buffer[h];
    GPU_ERR_CHK(cudaMalloc(hth_recv_buffer,
               sizeof(uint64_t) * BUFFER_SIZE));
    GPU_ERR_CHK(cudaMalloc(hth_send_buffer,
               sizeof(uint64_t) * BUFFER_SIZE));
#ifdef CPU_BUFFER_MODE
    cpu_recv_buffer[h] =
      (uint64_t *) malloc(sizeof(uint64_t) * BUFFER_SIZE);
#endif
  }
  GPU_ERR_CHK(cudaMalloc(&gpu_data,
             sizeof(uint64_t) * BUFFER_SIZE * num_hosts));


  printf("rank %d starts\n", rank);
#ifdef CPU_BUFFER_MODE
  PostReceive(cpu_recv_buffer, request);
  construct_own_data<<<1, 1>>>(gpu_data, rank);
  Send(gpu_data, send_buffer);
  Receive(cpu_recv_buffer, recv_buffer, request, gpu_data);
#else
  PostReceive(recv_buffer, request);
  construct_own_data<<<1, 1>>>(gpu_data, rank);
  Send(gpu_data, send_buffer);
  Receive(recv_buffer, request, gpu_data);
#endif
  verify_final_data<<<1, 1>>>(gpu_data, rank, num_hosts); 
  cudaDeviceSynchronize();
  printf("rank %d done\n", rank);
}

void RunBackground() {
  while (true) {
    MPI_Status status;
    int flag{0};

    int rv = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                        &flag, &status);
    if (flag) { if (status.MPI_TAG != 32767) { } }

    if (!run_probing) { break; }
  }  
}

int main(int argc, char **argv) {
  int support_provided;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &support_provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_hosts);

  int count{1};
  cudaGetDeviceCount(&count);
  int gpu_id = rank % count;
  cudaSetDevice(gpu_id);

  check_gpu<<<1, 1>>>(rank);
  cudaDeviceSynchronize();

  if (rank == 0) {
    std::cout << "  [INFO]\n";
    std::cout << "\t * # of hosts: " << num_hosts << "\n";

    for (int i = 0; i < count; i++) {
      std::cout << "\t " << i << " sets GPU" << i % count << " \n";
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, gpu_id);
    std::cout << "\t CUDA Capability: " << deviceProp.major << "."
              << deviceProp.minor << "\n";

    std::cout << std::endl;
  }
  run_probing = true;
  std::thread probe_thread(RunBackground);

  StartCommunication();

  run_probing = false;
  probe_thread.join();
  
  MPI_Finalize();
  return 0;
}
