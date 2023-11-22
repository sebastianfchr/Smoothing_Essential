cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 14) # is this a flag?
set(CMAKE_C_COMPILER gcc)
include(FetchContent)


add_compile_options(-g)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(Smoothing VERSION 1.0)
find_package (Python3 COMPONENTS Development REQUIRED)


# ====================
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")



find_package(dco_cpp_v3 REQUIRED) 

# 1) compile the smoothing-part
add_subdirectory(overload_smooth) # OverSmoothLibrary (library) in here


# Uses dco!
list(APPEND EXTRA_LIBS OverSmoothLibrary) # EXTRA_LIBS used by pybind_integrated
add_subdirectory(pybind_integrated)

# make tf-integration optional
# (otherwise, we'd have to require a tensorflow installation)
set(INTEGRATE_TF 0)
if(INTEGRATE_TF)
    add_subdirectory(tf_integrated)
endif()

# (absolutely necessary for pyBind?)
find_package(Python REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
