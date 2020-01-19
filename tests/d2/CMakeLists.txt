include(CheckTypeSize)
check_type_size(void* ptr_size)
if(${ptr_size} MATCHES "^4$")
    set(host_model 32)
elseif(${ptr_size} MATCHES "^8$")
    set(host_model 64)
endif()

set(gdb_dflags "")
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    execute_process(COMMAND gdb --version
                    COMMAND head -n 1
                    OUTPUT_VARIABLE GDB_VERSION
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX REPLACE "[^0-9]*([0-9]+[0-9.]*).*" "\\1" GDB_VERSION "${GDB_VERSION}")
    message(STATUS "GDB ${GDB_VERSION} detected")
    if(GDB_VERSION VERSION_LESS "7.6.1")
        set(gdb_flags "NOTLS")
    else()
        set(gdb_flags "ON")
    endif()
    if(GDB_VERSION VERSION_LESS "7.8")
        set(gdb_dflags "-dwarf-version=2")
    endif()
else()
    set(gdb_flags "OFF")
endif()

function(add_testsuite config_suffix required_flags gdbflags model make_rule)
    set(name dmd-testsuite${config_suffix})
    set(outdir ${CMAKE_BINARY_DIR}/${name})
    set(workingdir ${PROJECT_SOURCE_DIR}/tests/d2/dmd-testsuite)
    # The DMD testsuite assumes a relative RESULTS_DIR path.
    file(RELATIVE_PATH resultsdir ${workingdir} ${outdir})

    add_test(NAME clean-${name}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${outdir})

    set(dflags "-conf=${PROJECT_BINARY_DIR}/bin/${LDC_EXE}.conf ${gdb_dflags}")
    add_test(NAME ${name}
        COMMAND ${GNU_MAKE_BIN} -k -C ${workingdir} RESULTS_DIR=${resultsdir}
                     REQUIRED_ARGS=${required_flags} DFLAGS=${dflags}
                     DMD=${LDMD_EXE_FULL} MODEL=${model} GDB_FLAGS=${gdbflags}
                     ARGS= ${make_rule}
    )
    set_tests_properties(${name} PROPERTIES DEPENDS clean-${name})
endfunction()

string(REGEX REPLACE "[^0-9]*([0-9]+[0-9.]*).*" "\\1" GDB_VERSION "${GDB_VERSION}")
# Would like to specify the "-release" flag for release builds, but some of the
# tests (e.g. 'testdstress') depend on contracts and invariants being active.
# Need a solution integrated with d_do_test.
add_testsuite("-debug" "-g -link-defaultlib-debug" "${gdb_flags}" "${host_model}" "start_all_tests")
add_testsuite("" "-O" "OFF" "${host_model}" "start_runnable_tests")

if(MULTILIB AND host_model EQUAL 64)
    # Also test in 32 bit mode on x86_64 multilib builds.
    add_testsuite("-debug_32" "-g -link-defaultlib-debug" "${gdb_flags}" "32" "start_all_tests")
    add_testsuite("_32" "-O" "OFF" "32" "start_runnable_tests")
endif()
