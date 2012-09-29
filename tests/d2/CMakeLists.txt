include(CheckTypeSize)
check_type_size(void* ptr_size)
if(${ptr_size} MATCHES "^4$")
    set(model 32)
elseif(${ptr_size} MATCHES "^8$")
    set(model 64)
endif()

set(DMD_TEST_MODEL "${model}" CACHE STRING
    "The model argument to use for compiling the D2 testsuite. Defaults to system bitness.")

get_property(ldmd_path TARGET ldmd2 PROPERTY LOCATION)

add_test(NAME dmd-testsuite WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/d2/dmd-testsuite
    COMMAND make RESULTS_DIR=${CMAKE_BINARY_DIR}/dmd_test_results DMD=${ldmd_path} MODEL=${DMD_TEST_MODEL} quick)
