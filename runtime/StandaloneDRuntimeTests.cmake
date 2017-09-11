macro(get_subdirs result dir)
    file(GLOB children RELATIVE ${dir} ${dir}/*)
    set(subdir_list "")
    foreach(child ${children})
        if(IS_DIRECTORY ${dir}/${child})
            list(APPEND subdir_list ${child})
        endif()
    endforeach()
    set(${result} ${subdir_list})
endmacro()

if(MULTILIB AND "${TARGET_SYSTEM}" MATCHES "APPLE")
    # KLUDGE: The library target is a custom command for multilib builds (lipo),
    # so cannot use TARGET_FILE directly. Should stash away that name instead.
    set(druntime_path "${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/libdruntime-ldc.a")
    set(shared_druntime_path "${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/libdruntime-ldc${SHARED_LIB_SUFFIX}.dylib")
else()
    set(druntime_path "$<TARGET_FILE:druntime-ldc>")
    set(shared_druntime_path "$<TARGET_FILE:druntime-ldc${SHARED_LIB_SUFFIX}>")
endif()

get_subdirs(testnames ${PROJECT_SOURCE_DIR}/druntime/test)

# 'shared' is a special case
if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
    set(outdir ${PROJECT_BINARY_DIR}/druntime-test-shared)
    add_test(NAME clean-druntime-test-shared
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${outdir}
    )
    if("${TARGET_SYSTEM}" MATCHES "FreeBSD")
        set(linkflags "")
    else()
        set(linkflags "LINKDL=-L-ldl")
    endif()
    add_test(NAME druntime-test-shared
        COMMAND make -C ${PROJECT_SOURCE_DIR}/druntime/test/shared
            ROOT=${outdir} DMD=${LDMD_EXE_FULL}
            MODEL=default DRUNTIMESO=${shared_druntime_path}
            CFLAGS=-Wall\ -Wl,-rpath,${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX} ${linkflags}
    )
    set_tests_properties(druntime-test-shared
        PROPERTIES DEPENDS clean-druntime-test-shared
    )
endif()
list(REMOVE_ITEM testnames shared)

foreach(name ${testnames})
    set(outdir ${PROJECT_BINARY_DIR}/druntime-test-${name})
    add_test(NAME clean-druntime-test-${name}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${outdir}
    )
    add_test(NAME druntime-test-${name}
        COMMAND make -C ${PROJECT_SOURCE_DIR}/druntime/test/${name}
            ROOT=${outdir} DMD=${LDMD_EXE_FULL}
            MODEL=default DRUNTIME=${druntime_path}
    )
    set_tests_properties(druntime-test-${name}
        PROPERTIES DEPENDS clean-druntime-test-${name}
    )
endforeach()
