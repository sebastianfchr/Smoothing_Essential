find_package (Python3 COMPONENTS Development REQUIRED)
find_package(pybind11 MODULE REQUIRED)


# add_subdirectory(funcs)

# # pybind has a modified add
pybind11_add_module(smfuncs pybinder.cpp) 
target_link_libraries(smfuncs PUBLIC stapelib)

# this binary will be compiled in a folder within build
# placing an __init__.py in this folder-structure let python load it from outside
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/__init__.py )
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/../__init__.py)