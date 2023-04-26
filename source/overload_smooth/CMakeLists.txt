add_library(OverSmoothLibrary branchTape.cpp stype.cpp styperesult.cpp )

# for the online-case, all we have to do is:
# target_include_directories(OverSmoothLibrary PUBLIC ${DCO_CPP_INCLUDE_DIR})

# ====================== manual case ! ===============================

target_include_directories(OverSmoothLibrary PUBLIC ${DCO_CPP_INCLUDE_DIR})
target_link_libraries(OverSmoothLibrary INTERFACE ${DCO_CPP_LIB})



write_file("__init__.py" "")