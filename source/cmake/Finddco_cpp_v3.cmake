
# new timestamp policy: download timestamp
cmake_policy(SET CMP0135 NEW)

# downloading version
# Downloads dco/c++, unpack, and link!
FetchContent_Declare(
  dco_cpp_v3
  URL https://www.nag.com/downloads/impl/dcl6i37ngl_v371.tgz	
)

# we need Python-dev to integrate the compiled modules!
# TODO: FURTHER UP! DOESN'T BELONG HERE!
find_package (Python3 COMPONENTS Interpreter Development)

FetchContent_GetProperties(dco_cpp_v3)
message("dco populated? ${dco_cpp_v3_POPULATED}")
if (NOT dco_cpp_v3_POPULATED) # populates dco_cpp_v3_SOURCE_DIR
  FetchContent_Populate(dco_cpp_v3)
  message("downloaded to ${dco_cpp_v3_SOURCE_DIR}")
endif()



# unzip tarball. Does this work on windows as well?
execute_process (
    COMMAND tar -xzf "dcl6i37ngl_v371.tar.gz" 
    WORKING_DIRECTORY ${dco_cpp_v3_SOURCE_DIR}
    OUTPUT_VARIABLE outVar
)

set(DCO_CPP_BASE_DIR "${dco_cpp_v3_SOURCE_DIR}/dcl6i37ngl_v371")
# set(DCO_CPP_SOURCE_DIR "${DCO_CPP_BASE_DIR}/dcl6i37ngl_v371")
set(DCO_CPP_INCLUDE_DIR "${DCO_CPP_BASE_DIR}/include")
set(DCO_CPP_LIB "${DCO_CPP_BASE_DIR}/lib/libdcoc.a")


add_library(DCO_CPP_LIBRARY STATIC IMPORTED GLOBAL)
set_target_properties(DCO_CPP_LIBRARY PROPERTIES IMPORTED_LOCATION "${DCO_CPP_LIB_PATH}")

message("dco-library: ${DCO_CPP_LIB_PATH}")
message("dco-include: ${DCO_CPP_INCLUDE_DIR}")



