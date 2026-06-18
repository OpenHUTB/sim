cmake_minimum_required(VERSION 3.14)
project(thread_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release 模式默认优化
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(thread_benchmark benchmark.cpp)

# Windows 下链接线程库
if(WIN32)
    target_compile_options(thread_benchmark PRIVATE /W3)
else()
    target_link_libraries(thread_benchmark PRIVATE pthread)
    target_compile_options(thread_benchmark PRIVATE -Wall -O2)
endif()
