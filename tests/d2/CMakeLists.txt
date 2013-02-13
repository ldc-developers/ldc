include(CheckTypeSize)
check_type_size(void* ptr_size)
if(${ptr_size} MATCHES "^4$")
    set(host_model 32)
elseif(${ptr_size} MATCHES "^8$")
    set(host_model 64)
endif()

get_property(ldmd_path TARGET ldmd2 PROPERTY LOCATION)

function(add_testsuite config_name dflags model)
    set(name dmd-testsuite_${config_name})
    set(outdir ${CMAKE_BINARY_DIR}/${name})

    add_test(NAME ${name}_clean
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${outdir})

    # The DFLAGS environment variable read by LDMD is used because the DMD
    # testsuite build system provides no way to run the test cases with a
    # given set of flags without trying all combinations of them.
    add_test(NAME ${name}
        COMMAND make -C ${PROJECT_SOURCE_DIR}/tests/d2/dmd-testsuite RESULTS_DIR=${outdir} DMD=${ldmd_path} DFLAGS=${dflags} MODEL=${model} quick
    )
    set_tests_properties(${name} PROPERTIES DEPENDS ${name}_clean)
endfunction()

# Would like to specify the "-release" flag for relase builds, but some of the
# tests (e.g. 'testdstress') depend on contracts and invariants being active.
# Need a solution integrated with d_do_test.
add_testsuite(debug -gc ${host_model})
add_testsuite(release -O3 ${host_model})

if(MULTILIB AND host_model EQUAL 64)
    # Also test in 32 bit mode on x86_64 multilib builds.
    add_testsuite(debug_32 -gc 32)
    add_testsuite(release_32 -O3 32)
endif()
