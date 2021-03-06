cmake_minimum_required(VERSION 3.13)

project(GpudirectDebug)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

include(GNUInstallDirs)
set(ENABLE_HEAVY OFF CACHE BOOL "Enable heavy test")
set(ENABLE_MULTI_MSG OFF CACHE BOOL "Enable multiple sendings (x10)")
set(ENABLE_PARALLLEL OFF CACHE BOOL "Enable parallel communications")
set(USE_ICOMM OFF CACHE BOOL "Enable non-blocking MPI communication")
set(USE_ISCOMM OFF CACHE BOOL "Enable non-blocking synchronous MPI sending")
set(CUDA_CAPABILITY "6.0" CACHE BOOL "Cuda capability: default 6.0")
set(COMM_MODE "NB" CACHE BOOL "MPI communication mode: default B (Blocking),
                  possible optiosn - NB (Non-blocking) NBS (Non-blocking synchronous)")
set(CPU_BUFFER OFF CACHE BOOL "Enable cpu-buffer")
set(ENABLE_SET_DEVICE OFF CACHE BOOL "Set cuda device manually")

file(GLOB SRC
    "src/*.h"
    "src/*.cpp"
    "src/*.cu"
    )

add_executable(gpudirect_test src/gpudirect_bench.cu)
add_executable(gpudirect_test_cpu src/gpudirect_bench.cu)
target_compile_definitions(gpudirect_test_cpu PRIVATE _SET_DEVICE_MODE_)
add_executable(gpudirect_gluonlike src/gpudirect_gluonlike.cu)

add_executable(cpu_test src/cpu_bench.cu)
#add_executable(cpu_test src/gpudirect_bench.cu)

find_package(OpenMP)
if(OPENMP_FOUND)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>")
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:${OpenMP_CXX_FLAGS}>")
  target_link_libraries(gpudirect_test PUBLIC OpenMP::OpenMP_CXX)
  target_link_libraries(gpudirect_test_cpu PUBLIC OpenMP::OpenMP_CXX)
  target_link_libraries(gpudirect_gluonlike PUBLIC OpenMP::OpenMP_CXX)
  target_link_libraries(cpu_test PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(CUDA REQUIRED)
enable_language(CUDA)
string(REPLACE "." "" GENCODES ${CUDA_CAPABILITY})
string(REPLACE "," ";" GENCODES ${GENCODES})
foreach(GENCODE ${GENCODES})
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${GENCODE},code=sm_${GENCODE}>")
endforeach()

target_include_directories(gpudirect_test
                           PUBLIC
                           ${CUDA_INCLUDE_DIRS}
)

target_include_directories(gpudirect_gluonlike
                           PUBLIC
                           ${CUDA_INCLUDE_DIRS}
)

target_include_directories(gpudirect_test_cpu
                           PUBLIC
                           ${CUDA_INCLUDE_DIRS}
)

target_include_directories(cpu_test
                           PUBLIC
                           ${CUDA_INCLUDE_DIRS}
)

find_package(MPI REQUIRED)
target_include_directories(gpudirect_test PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_compile_options(gpudirect_test PUBLIC ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(gpudirect_test PUBLIC ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})

target_include_directories(gpudirect_test_cpu PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_compile_options(gpudirect_test_cpu PUBLIC ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(gpudirect_test_cpu PUBLIC ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})

target_include_directories(gpudirect_gluonlike PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_compile_options(gpudirect_gluonlike PUBLIC ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(gpudirect_gluonlike PUBLIC ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})

target_include_directories(cpu_test PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_compile_options(cpu_test PUBLIC ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(cpu_test PUBLIC ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})

if (ENABLE_HEAVY)
  add_compile_definitions(ENABLE_HEAVY)
endif()

if (ENABLE_MULTI_MSG)
  add_compile_definitions(ENABLE_MULTI_MSG)
endif()

if (ENABLE_PARALLLEL)
  add_compile_definitions(PARALLEL_MSG_MODE)
endif()

if (CPU_BUFFER)
  add_compile_definitions(_CPU_BUFFER_MODE_)
endif()

if (ENABLE_SET_DEVICE)
  add_compile_definitions(_SET_DEVICE_MODE_)
endif()

if (COMM_MODE STREQUAL "B")
  message("Blocking mode ..")
  add_definitions(-DCOMM_MODE=0)
elseif(COMM_MODE STREQUAL "NB")
  message("Non-blocking mode ..")
  add_definitions(-DCOMM_MODE=1)
else(COMM_MODE STREQUAL "NBS")
  message("Non-blocking synchronous mode ..")
  add_definitions(-DCOMM_MODE=2)
endif()


remove_definitions(-fexceptions)
