# asd
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        #80dc998efced8ceb2be59756668a7e90e8bef917 # release-1.10.0
)

find_package(Python3 COMPONENTS Interpreter Development) # pybind needs to be sure of this...

FetchContent_GetProperties(pybind11)
message("pybind populated? ${pybind11_POPULATED}")
# recurse into the directory to build it (if not built yet)

if(NOT pybind11_POPULATED)
  message("fetching pybind")
  # Fetch the content using previously declared details
  FetchContent_Populate(pybind11)  
  # pybind is header only, so the SOURCE_DIR is enough, no linking!
  # why does this need a binary dir tho?
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR}) # calls the CMakeLists.txt inside these
else()
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR}) # calls the CMakeLists.txt inside these
  message("Pybind already fetched")
endif()

message("pybind source dir? ${pybind11_SOURCE_DIR}")
message("pybind binary dir? ${pybind11_BINARY_DIR}")
message("pybind include dir? ${pybind11_INCLUDE_DIR}")
#message("pybind lib ? ${pybind11_LIBRARY_DIR}")

