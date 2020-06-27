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
    set(shared_druntime_path "${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/libdruntime-ldc${SHARED_LIB_SUFFIX}.dylib")
    if(${BUILD_SHARED_LIBS} STREQUAL "ON")
        set(druntime_path ${shared_druntime_path})
    else()
        set(druntime_path "${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/libdruntime-ldc.a")
    endif()
else()
    set(shared_druntime_path "$<TARGET_FILE:druntime-ldc${SHARED_LIB_SUFFIX}>")
    if(${BUILD_SHARED_LIBS} STREQUAL "ON")
        set(druntime_path ${shared_druntime_path})
    else()
        set(druntime_path "$<TARGET_FILE:druntime-ldc>")
    endif()
endif()

if("${TARGET_SYSTEM}" MATCHES "MSVC")
    set(cflags_base "CFLAGS_BASE=")
else()
    set(cflags_base "CFLAGS_BASE=-Wall -Wl,-rpath,${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}")
endif()

if("${TARGET_SYSTEM}" MATCHES "FreeBSD|DragonFly")
    set(linkdl "")
else()
    set(linkdl "LINKDL=-L-ldl")
endif()

get_subdirs(testnames ${PROJECT_SOURCE_DIR}/druntime/test)
if(${BUILD_SHARED_LIBS} STREQUAL "OFF")
    list(REMOVE_ITEM testnames shared)
elseif(${BUILD_SHARED_LIBS} STREQUAL "ON")
    # gc: replaces druntime modules at link-time and so requires a static druntime
    list(REMOVE_ITEM testnames cycles gc)
endif()
list(REMOVE_ITEM testnames uuid) # MSVC only, custom Makefile (win64.mak)

foreach(name ${testnames})
    set(outdir ${PROJECT_BINARY_DIR}/druntime-test-${name})
    add_test(NAME clean-druntime-test-${name}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${outdir}
    )
    add_test(NAME druntime-test-${name}
        COMMAND ${GNU_MAKE_BIN} -C ${PROJECT_SOURCE_DIR}/druntime/test/${name}
            ROOT=${outdir} DMD=${LDMD_EXE_FULL} MODEL=default
            DRUNTIME=${druntime_path} DRUNTIMESO=${shared_druntime_path}
            ${cflags_base} ${linkdl}
    )
    set_tests_properties(druntime-test-${name}
        PROPERTIES DEPENDS clean-druntime-test-${name}
    )
endforeach()
