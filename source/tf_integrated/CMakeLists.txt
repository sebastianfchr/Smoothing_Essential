
cmake_minimum_required(VERSION 3.10)

# set(CMAKE_C_COMPILER "gcc")
# set(CMAKE_CXX_COMPILER "g++")


project(CustomFuncsCpu VERSION 1.0)


# CustomFuncsCpu consists of these tf-functions
add_library(CustomFuncsCpu SHARED functions/cb2.cc functions/spiral.cc functions/crescent.cc functions/ql.cc functions/set_global_smoothing_factor.cc functions/step.cc) 
# add_library(CustomFuncsCpu SHARED functions/cb2.cc functions/spiral.cc functions/crescent.cc functions/ql.cc functions/set_global_smoothing_factor.cc) 



execute_process(COMMAND python3 -m site --user-site OUTPUT_VARIABLE SITE_PATH_RAW OUTPUT_STRIP_TRAILING_WHITESPACE)
# !!!!!!!!!!!! THIS FINDS TF IN CASE TENSORFLOW IS INSTALLED WITH --user FLAG !!!!!!!!!
# TODO: Generally, one should account for the case where it's installed without --user (must be represented by similar PYTHON COMMANDS!)
set(SITE_PATH ${SITE_PATH_RAW})

message(${SITE_PATH})
find_path(TensorFlow_INCLUDE_DIR HINTS "${SITE_PATH}/tensorflow/include" NAMES "Eigen" "external" "google" "unsupported" REQUIRED)
find_library(TensorFlow_LIBRARY HINTS "${SITE_PATH}/tensorflow" NAMES "libtensorflow_framework.so.2" REQUIRED)


# tf compile options
set(CPU_COMPILE_OPTIONS -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -g)
target_compile_options(CustomFuncsCpu PUBLIC ${CPU_COMPILE_OPTIONS})


target_link_libraries(CustomFuncsCpu PUBLIC 
    ${TensorFlow_LIBRARY}
    ${DCO_CPP_LIBRARY}
    OverSmoothLibrary # This is built beforehand, (outer level of project) and used in the tf-integrated functions
)
target_include_directories(CustomFuncsCpu PUBLIC
    "${TensorFlow_INCLUDE_DIR}"
    "${PROJECT_BINARY_DIR}"
    ${DCO_CPP_INCLUDE_DIR}
)

message("=== tf-integration compiled with options: ==")
message("${CPU_COMPILE_OPTIONS}")

message("=== tf-include and library: ==")
message(${TensorFlow_INCLUDE_DIR})
message(${TensorFlow_LIBRARY})
