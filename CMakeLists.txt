# cmake_minimum_required(VERSION <specify CMake version here>)
project(mpi_task_2 C)

set(CMAKE_C_STANDARD 99)

set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

find_package(MPI REQUIRED)

include_directories(SYSTEM "C:/Program Files (x86)/Microsoft SDKs/MPI/Include" "C:/Program Files (x86)/Microsoft SDKs/MPI/Include/x64")
link_directories("C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")

add_executable(mpi_task_2 main.c methods.c)

target_link_libraries(mpi_task_2 "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")