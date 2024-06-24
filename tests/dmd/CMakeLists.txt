if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(host_model 32)
else()
    set(host_model 64)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(gdb_result_code)
    execute_process(COMMAND gdb --version
                    RESULT_VARIABLE gdb_result_code)
    if(gdb_result_code)
        message(STATUS "GDB not detected, will exclude GDB tests from test suite")
        set(gdb_flags "OFF")
    else()
        execute_process(COMMAND gdb --version
                        COMMAND head -n 1
                        RESULT_VARIABLE gdb_result_code
                        OUTPUT_VARIABLE GDB_VERSION
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REGEX REPLACE "[^0-9]*([0-9]+[0-9.]*).*" "\\1" GDB_VERSION "${GDB_VERSION}")
        message(STATUS "GDB ${GDB_VERSION} detected")
        set(gdb_flags "ON")
    endif()
else()
    set(gdb_flags "OFF")
endif()

set(DMD_TESTSUITE_SRC_DIR ${PROJECT_SOURCE_DIR}/tests/dmd)
set(RUN_DMD_TESTSUITE_EXE_FULL ${CMAKE_CURRENT_BINARY_DIR}/run-dmd-testsuite${CMAKE_EXECUTABLE_SUFFIX})
add_custom_command(
    OUTPUT  ${RUN_DMD_TESTSUITE_EXE_FULL}
    COMMAND ${LDC_EXE_FULL} -of=${RUN_DMD_TESTSUITE_EXE_FULL}
            ${DMD_TESTSUITE_SRC_DIR}/run.d ${DMD_TESTSUITE_SRC_DIR}/tools/paths.d
    DEPENDS ${DMD_TESTSUITE_SRC_DIR}/run.d ${DMD_TESTSUITE_SRC_DIR}/tools/paths.d
            # actually, LDC, druntime and Phobos too, but library target names vary and
            # potentially recompiling outdated compiler and/or libs isn't always desirable
)
add_custom_target(build-run-dmd-testsuite DEPENDS ${RUN_DMD_TESTSUITE_EXE_FULL})
add_test(build-run-dmd-testsuite "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target build-run-dmd-testsuite)

function(add_testsuite config_suffix required_flags gdbflags model make_rule)
    set(name dmd-testsuite${config_suffix})
    set(outdir ${CMAKE_BINARY_DIR}/${name})
    # The DMD testsuite assumes a relative RESULTS_DIR path.
    file(RELATIVE_PATH resultsdir ${DMD_TESTSUITE_SRC_DIR} ${outdir})

    add_test(NAME clean-${name}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${outdir})

    set(dflags "-conf=${PROJECT_BINARY_DIR}/bin/${LDC_EXE}.conf")
    add_test(NAME ${name}
        COMMAND ${RUN_DMD_TESTSUITE_EXE_FULL} RESULTS_DIR=${resultsdir}
                     REQUIRED_ARGS=${required_flags} DFLAGS=${dflags}
                     DMD=${LDMD_EXE_FULL} MODEL=${model} GDB_FLAGS=${gdbflags}
                     ARGS= BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} ${make_rule}
        WORKING_DIRECTORY ${DMD_TESTSUITE_SRC_DIR}
    )
    set_tests_properties(${name} PROPERTIES DEPENDS "build-run-dmd-testsuite;clean-${name}")
endfunction()

string(REGEX REPLACE "[^0-9]*([0-9]+[0-9.]*).*" "\\1" GDB_VERSION "${GDB_VERSION}")
# Would like to specify the "-release" flag for release builds, but some of the
# tests (e.g. 'testdstress') depend on contracts and invariants being active.
# Need a solution integrated with d_do_test.
add_testsuite("-debug" "-g -link-defaultlib-debug" "${gdb_flags}" "${host_model}" "all")
add_testsuite("" "-O -linkonce-templates" "OFF" "${host_model}" "runnable")

if(MULTILIB AND host_model EQUAL 64)
    # Also test in 32 bit mode on x86_64 multilib builds.
    add_testsuite("-debug_32" "-g -link-defaultlib-debug" "${gdb_flags}" "32" "all")
    add_testsuite("_32" "-O -linkonce-templates" "OFF" "32" "runnable")
endif()
