cmake_minimum_required(VERSION 3.13)

project(GpudirectDebug)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

include(GNUInstallDirs)
set(ENABLE_HEAVY OFF CACHE BOOL "Enable heavy test")
set(ENABLE_MULTI_MSG OFF CACHE BOOL "Enable multiple sendings (x10)")
set(USE_ICOMM OFF CACHE BOOL "Enable non-blocking MPI communication")
set(USE_ISCOMM OFF CACHE BOOL "Enable non-blocking synchronous MPI sending")
set(CUDA_CAPABILITY "6.0" CACHE BOOL "Cuda capability: default 6.0")

find_package(OpenMP)
if(OPENMP_FOUND)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>")
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:${OpenMP_CXX_FLAGS}>")
endif()

find_package(CUDA REQUIRED)
enable_language(CUDA)
string(REPLACE "." "" GENCODES ${CUDA_CAPABILITY})
string(REPLACE "," ";" GENCODES ${GENCODES})
foreach(GENCODE ${GENCODES})
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${GENCODE},code=sm_${GENCODE}>")
endforeach()

file(GLOB SRC
    "src/*.h"
    "src/*.cpp"
    "src/*.cu"
    )
#add_executable(gpudirect_test ${SRC})
add_executable(gpudirect_test src/gpudirect_bench.cu)
add_executable(cpu_test src/cpu_bench.cpp)
target_include_directories(gpudirect_test
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

target_include_directories(cpu_test PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_compile_options(cpu_test PUBLIC ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(cpu_test PUBLIC ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})

if (ENABLE_HEAVY)
  add_compile_definitions(ENABLE_HEAVY)
endif()

if (ENABLE_MULTI_MSG)
  add_compile_definitions(ENABLE_MULTI_MSG)
endif()

remove_definitions(-fexceptions)
