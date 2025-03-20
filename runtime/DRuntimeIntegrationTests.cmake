enable_language(CXX) # for CMAKE_CXX_COMPILER

# Try to find GNU make, use specific version first (BSD) and fall back to default 'make' (Linux)
find_program(GNU_MAKE_BIN NAMES gmake gnumake make)
if(NOT GNU_MAKE_BIN)
    message(WARNING "GNU make could not be found. Please install gmake/gnumake/make using your (platform) package installer to enable the druntime integration tests.")
    return()
endif()

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

string(REPLACE ";" " " dflags_base "${D_EXTRA_FLAGS}")
string(REPLACE ";" " " cflags_base "${RT_CFLAGS}")

set(linkdl "")
if("${TARGET_SYSTEM}" MATCHES "Linux")
    set(linkdl "LINKDL=-L-ldl")
endif()

get_subdirs(testnames ${PROJECT_SOURCE_DIR}/druntime/test)
if(${BUILD_SHARED_LIBS} STREQUAL "OFF")
    list(REMOVE_ITEM testnames shared)
elseif(${BUILD_SHARED_LIBS} STREQUAL "ON")
    # gc: replaces druntime modules at link-time and so requires a static druntime
    list(REMOVE_ITEM testnames cycles gc)
endif()
if("${TARGET_SYSTEM}" MATCHES "Windows")
    list(REMOVE_ITEM testnames valgrind)
else()
    list(REMOVE_ITEM testnames uuid)
endif()

if(TARGET_SYSTEM MATCHES "musl")
    set(musl "IS_MUSL=1")
endif()

foreach(name ${testnames})
    foreach(build debug release)
        set(druntime_path_build ${druntime_path})
        set(shared_druntime_path_build ${shared_druntime_path})
        if(${build} STREQUAL "debug")
            string(REPLACE "druntime-ldc" "druntime-ldc-debug" druntime_path_build ${druntime_path_build})
            string(REPLACE "druntime-ldc" "druntime-ldc-debug" shared_druntime_path_build ${shared_druntime_path_build})
        endif()

        set(fullname druntime-test-${name}-${build})
        set(outdir ${PROJECT_BINARY_DIR}/${fullname})
        add_test(NAME clean-${fullname}
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${outdir}
        )
        add_test(NAME ${fullname}
            COMMAND ${GNU_MAKE_BIN} -C ${PROJECT_SOURCE_DIR}/druntime/test/${name}
                ROOT=${outdir} DMD=${LDMD_EXE_FULL} BUILD=${build} SHARED=1
                CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
                DRUNTIME=${druntime_path_build} DRUNTIMESO=${shared_druntime_path_build}
                CFLAGS_BASE=${cflags_base} DFLAGS_BASE=${dflags_base} ${linkdl}
		IN_LDC=1 ${musl}
        )
        set_tests_properties(${fullname} PROPERTIES DEPENDS clean-${fullname})
    endforeach()
endforeach()
