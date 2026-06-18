cmake_minimum_required(VERSION 3.14)
project(thread_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release 模式，不然测出来的数据没参考意义
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Windows 上需要手动链接线程库，Linux 用 pthreads
if(WIN32)
    set(THREADS_PREFER_PTHREAD_FLAG OFF)
else()
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()
find_package(Threads REQUIRED)

add_executable(thread_benchmark benchmark.cpp)
target_link_libraries(thread_benchmark PRIVATE Threads::Threads)

# MSVC 开 O2，GCC/Clang 开 O3
if(MSVC)
    target_compile_options(thread_benchmark PRIVATE /O2 /W3)
else()
    target_compile_options(thread_benchmark PRIVATE -O3 -Wall -Wextra)
endif()

# 把可执行文件放到 build/ 目录，方便找
set_target_properties(thread_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Output: ${CMAKE_BINARY_DIR}/thread_benchmark")
