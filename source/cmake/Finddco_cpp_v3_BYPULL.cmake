# **************************************************************************** #
# This file is part of the rsqtoa build system. Search / Fetch dco/c++.
#
# Version: 0.1.0
# Author:  Simon Maertens, STCE (info@stce.rwth-aachen.de)
# **************************************************************************** #

include(FetchContent)

# Fetch dco/c++ v3 develop (without submodules)
message(STATUS "dco/c++ v3: Fetching develop HEAD of upstream ...")


FetchContent_Declare(
  dco_cpp_v3
  GIT_REPOSITORY git@gitlab.stce.rwth-aachen.de:stce/dco_cpp_dev.git
  GIT_TAG        912954de3ebafe6e55641f31d1a198fb542e9131
  GIT_SHALLOW    ON
  #GIT_SUBMODULES "${submodules_arg}"
  )

if(NOT dco_cpp_v3_POPULATED)
  FetchContent_Populate(dco_cpp_v3)
endif()

execute_process(COMMAND git rev-parse HEAD
                WORKING_DIRECTORY ${dco_cpp_v3_SOURCE_DIR}
                OUTPUT_VARIABLE dco_cpp_v3_sha)
string(REPLACE "\n" "" dco_cpp_v3_sha ${dco_cpp_v3_sha})
message(STATUS "dco/c++ v3: Successfully fetched develop "
              "HEAD at SHA ${dco_cpp_v3_sha}")


# Set dco/c++ base dir. Never used!
set(DCO_CPP_BASE_DIR ${dco_cpp_v3_SOURCE_DIR}
    CACHE PATH "installation path of dco/c++ (fetched)")

set(DCO_CPP_INCLUDE_DIR ${dco_cpp_v3_SOURCE_DIR}/src
    CACHE PATH "include path of dco/c++ (fetched)")

