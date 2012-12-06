include(CheckTypeSize)
check_type_size(void* ptr_size)
if(${ptr_size} MATCHES "^4$")
    set(models 32)
elseif(${ptr_size} MATCHES "^8$")
    set(models 64)
endif()

if(MULTILIB AND models EQUAL 64)
    list(APPEND models 32)
endif()

get_property(ldmd_path TARGET ldmd2 PROPERTY LOCATION)

foreach(model ${models})
    # Build test suite in both debug and release modes. The DFLAGS environment
    # variable read by LDMD is used because the DMD testsuite build system
    # provides no way to run the test cases with a given set of flags without
    # trying all combinations of them.

    add_test(NAME dmd-testsuite_debug_${model}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/d2/dmd-testsuite
        COMMAND make RESULTS_DIR=${CMAKE_BINARY_DIR}/dmd-testsuite_debug_${model} DMD=${ldmd_path} DFLAGS=-gc MODEL=${model} quick
    )

    # Would like to specify the "-release" flag here, but some of the tests (e.g.
    # 'testdstress') depend on contracts and invariants being active. Need a solution
    # integrated with d_do_test.
    add_test(NAME dmd-testsuite_release_${model}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/d2/dmd-testsuite
        COMMAND make RESULTS_DIR=${CMAKE_BINARY_DIR}/dmd-testsuite_release_${model} DMD=${ldmd_path} DFLAGS=-O3 MODEL=${model} quick
    )
endforeach()
