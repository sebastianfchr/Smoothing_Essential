message(${CMAKE_MODULE_PATH})
message("===========================")


find_package(pybind11 MODULE REQUIRED)

# 2) compile pybinder, which has knowledge of OverSmoothLibrary
pybind11_add_module(Smoothing pybinder.cpp) 
target_link_libraries(Smoothing PUBLIC ${EXTRA_LIBS}) # "central" extra libs, which contain OverSmoothLibrary