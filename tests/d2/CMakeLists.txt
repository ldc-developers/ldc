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

# Build test suite in both debug and release modes. The DFLAGS environment
# variable read by LDMD is used because the DMD testsuite build system
# provides no way to run the test cases with a given set of flags without
# trying all combinations of them.

add_test(NAME dmd-testsuite-debug
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/d2/dmd-testsuite
    COMMAND make RESULTS_DIR=${CMAKE_BINARY_DIR}/dmd-testsuite-debug DMD=${ldmd_path} DFLAGS="-gc" MODEL=${DMD_TEST_MODEL} quick
)

add_test(NAME dmd-testsuite-release
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/d2/dmd-testsuite
    COMMAND make RESULTS_DIR=${CMAKE_BINARY_DIR}/dmd-testsuite-release DMD=${ldmd_path} DFLAGS=-O3\ -release MODEL=${DMD_TEST_MODEL} quick
)
