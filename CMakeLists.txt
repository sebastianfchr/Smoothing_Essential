cmake_minimum_required(VERSION 3.28)

project(function_smoothing VERSION 2.0)
set(CMAKE_CXX_STANDARD 20 REQUIRED)


set(CMAKE_CXX_FLAGS "-g")
include(FetchContent)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(src) # stapelib, included from header-only in /inc

# enable_testing is not recurvise. Call it in relevant subfolders, too!
enable_testing()
add_subdirectory(pybound)


add_executable(function1 function1.cpp)
target_link_libraries(function1 PUBLIC stapelib)
add_executable(function2 function2.cpp)
target_link_libraries(function2 PUBLIC stapelib)

