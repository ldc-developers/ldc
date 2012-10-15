project(runtime)

cmake_minimum_required(VERSION 2.6)

#
# Main configuration.
#

include(CheckTypeSize)
check_type_size(void*  ptr_size)
if(${ptr_size} MATCHES "^8$") ## if it's 64-bit OS
    set(MULTILIB_SUFFIX 32)
else()
    set(MULTILIB_SUFFIX 64)
endif()

set(DMDFE_MINOR_VERSION   0)
set(DMDFE_PATCH_VERSION   60)
set(DMDFE_VERSION         ${D_VERSION}.${DMDFE_MINOR_VERSION}.${DMDFE_PATCH_VERSION})

set(MULTILIB              OFF                                                           CACHE BOOL    "Build both 64-bit and 32-bit libraries")
set(BUILD_BC_LIBS         OFF                                                           CACHE BOOL    "Build the runtime as bytecode libraries")
set(BUILD_SINGLE_LIB      ON                                                            CACHE BOOL    "Build single runtime library")
set(LIB_SUFFIX            ""                                                            CACHE STRING  "Takes an empty string or 64. Directory where lib will be installed: lib or lib64")
set(INCLUDE_INSTALL_DIR   ${CMAKE_INSTALL_PREFIX}/include/d                             CACHE PATH    "Directory where will be put header files")
set(BUILD_SHARED_LIBS     OFF                                                           CACHE BOOL    "Build as shared library or as static library")
set(CMAKE_INSTALL_LIBDIR  ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}                      CACHE PATH    "Directory where lib will be installed")
set(D_FLAGS -g -w -d CACHE STRING "runtime build flags, separated by ;")

if(BUILD_SHARED_LIBS)
    list(APPEND D_FLAGS -relocation-model=pic)
    if(APPLE)
         # We need to explicitly specify that __Dmain should be resolved at
         # runtime with the default OS X tool chain.
         list(APPEND LD_FLAGS -Wl,-U,__Dmain)
    endif()
    set(D_LIBRARY_TYPE SHARED)
else(BUILD_SHARED_LIBS)
    set(D_LIBRARY_TYPE STATIC)
    set(CXX_COMPILE_FLAGS " ")
endif(BUILD_SHARED_LIBS)

get_directory_property(PROJECT_PARENT_DIR DIRECTORY ${PROJECT_SOURCE_DIR} PARENT_DIRECTORY)
set(RUNTIME_DIR ${PROJECT_SOURCE_DIR}/druntime CACHE PATH "runtime source dir")

if(D_VERSION EQUAL 1)
    set(RUNTIME_AIO tango)
    configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}_install.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}_install.conf)
    configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}.rebuild.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}_install.rebuild.conf)

    message(STATUS "Note: Tango is no longer included in D1 builds, please compile and install it separately using its own build infrastructure (bob).")
    return()
endif()

#
# Gather source files.
#

set(PHOBOS2_DIR ${PROJECT_SOURCE_DIR}/phobos CACHE PATH "phobos2 source dir")
set(RUNTIME_CC druntime-core)
set(RUNTIME_GC druntime-gc-basic)
set(RUNTIME_DC druntime-rt-ldc)
set(RUNTIME_AIO druntime-ldc)
set(RUNTIME_DC_DIR ${RUNTIME_DIR}/src/rt)
set(RUNTIME_GC_DIR ${RUNTIME_DIR}/src/gc)
set(RUNTIME_INCLUDE ${RUNTIME_DIR}/src)
file(GLOB CORE_D ${RUNTIME_DIR}/src/core/*.d )
file(GLOB CORE_D_SYNC ${RUNTIME_DIR}/src/core/sync/*.d )
file(GLOB CORE_D_STDC ${RUNTIME_DIR}/src/core/stdc/*.d )
file(GLOB_RECURSE GC_D ${RUNTIME_GC_DIR}/*.d)
file(GLOB_RECURSE DCRT_D ${RUNTIME_DC_DIR}/*.d)
file(GLOB_RECURSE LDC_D ${RUNTIME_DIR}/src/ldc/*.d)
list(REMOVE_ITEM DCRT_D
    ${RUNTIME_DC_DIR}/alloca.d
    ${RUNTIME_DC_DIR}/critical_.d
    ${RUNTIME_DC_DIR}/deh.d
    ${RUNTIME_DC_DIR}/deh2.d
    ${RUNTIME_DC_DIR}/llmath.d
    ${RUNTIME_DC_DIR}/memory_osx.d
    ${RUNTIME_DC_DIR}/qsort2.d
    ${RUNTIME_DC_DIR}/trace.d
)
file(GLOB DCRT_C ${RUNTIME_DC_DIR}/*.c)
list(REMOVE_ITEM DCRT_C ${RUNTIME_DC_DIR}/deh.c ${RUNTIME_DC_DIR}/dylib_fixes.c)
if(UNIX)
    file(GLOB_RECURSE CORE_D_SYS ${RUNTIME_DIR}/src/core/sys/posix/*.d)
    if(APPLE)
        file(GLOB_RECURSE CORE_D_SYS_OSX ${RUNTIME_DIR}/src/core/sys/osx/*.d)
        list(APPEND CORE_D_SYS ${CORE_D_SYS_OSX})
    endif()
    list(REMOVE_ITEM LDC_D ${RUNTIME_DIR}/src/ldc/eh2.d)
    list(REMOVE_ITEM DCRT_C ${RUNTIME_DC_DIR}/msvc.c)
elseif(WIN32)
    file(GLOB_RECURSE CORE_D_SYS ${RUNTIME_DIR}/src/core/sys/windows/*.d)
    list(REMOVE_ITEM LDC_D ${RUNTIME_DIR}/src/ldc/eh.d)
    list(REMOVE_ITEM DCRT_C ${RUNTIME_DC_DIR}/monitor.c)
endif()
list(APPEND CORE_D ${CORE_D_SYNC} ${CORE_D_SYS} ${CORE_D_STDC} )
list(APPEND GENERATE_DI ${CORE_D})
list(APPEND CORE_D ${LDC_D} ${RUNTIME_DIR}/src/object_.d)
file(GLOB CORE_C ${RUNTIME_DIR}/src/core/stdc/*.c)

if(PHOBOS2_DIR)
    if(BUILD_SHARED_LIBS)
        # std.net.curl depends on libcurl – when building a shared library, we
        # need to take care of that.
        find_package(CURL REQUIRED)
    endif()

    file(GLOB PHOBOS2_D ${PHOBOS2_DIR}/std/*.d)
    file(GLOB PHOBOS2_D_NET ${PHOBOS2_DIR}/std/net/*.d)
    file(GLOB_RECURSE PHOBOS2_D_INTERNAL ${PHOBOS2_DIR}/std/internal/*.d)
    file(GLOB PHOBOS2_D_C ${PHOBOS2_DIR}/std/c/*.d)
    file(GLOB PHOBOS2_ETC ${PHOBOS2_DIR}/etc/c/*.d)
    if(APPLE)
        file(GLOB PHOBOS2_D_C_SYS ${PHOBOS2_DIR}/std/c/osx/*.d)
    elseif(UNIX)
        # Install Linux headers on all non-Apple *nixes - not correct, but
        # shouldn't cause any harm either.
        file(GLOB PHOBOS2_D_C_SYS ${PHOBOS2_DIR}/std/c/linux/*.d)
    elseif(WIN32)
        file(GLOB PHOBOS2_D_C_SYS ${PHOBOS2_DIR}/std/c/windows/*.d)
    endif()
    file(GLOB ZLIB_C ${PHOBOS2_DIR}/etc/c/zlib/*.c)
    list(REMOVE_ITEM ZLIB_C
        ${PHOBOS2_DIR}/etc/c/zlib/minigzip.c
        ${PHOBOS2_DIR}/etc/c/zlib/example.c
        ${PHOBOS2_DIR}/etc/c/zlib/gzio.c
    )
    if(WIN32)
        file(GLOB PHOBOS2_D_WIN ${PHOBOS2_DIR}/std/windows/*.d)
    endif(WIN32)
    list(APPEND PHOBOS2_D
            ${PHOBOS2_D_NET}
            ${PHOBOS2_D_INTERNAL}
            ${PHOBOS2_D_WIN}
            ${PHOBOS2_D_C}
            ${PHOBOS2_D_C_SYS}
            ${PHOBOS2_ETC}
            ${PHOBOS2_DIR}/crc32.d
    )
    list(REMOVE_ITEM PHOBOS2_D
            ${PHOBOS2_DIR}/std/intrinsic.d
    )
    set(CONFIG_NAME ${LDC_EXE}_phobos)
else(PHOBOS2_DIR)
    set(CONFIG_NAME ${LDC_EXE})
endif(PHOBOS2_DIR)

# should only be necessary if run independently from ldc cmake project
if(NOT LDC_LOC)
    if(NOT LDC_EXE)
        set(LDC_EXE ldc2)
    endif(NOT LDC_EXE)

    find_program(LDC_LOC ${LDC_EXE} ${PROJECT_BINARY_DIR}/../bin DOC "path to ldc binary")
    if(NOT LDC_LOC)
        message(SEND_ERROR "ldc not found")
    endif(NOT LDC_LOC)
    set(LDC_EXE_NAME ${LDC_EXE})
endif(NOT LDC_LOC)

#
# Create configuration files.
#

if(MULTILIB)
    set(MULTILIB_ADDITIONAL_PATH         "\n        \"-L-L${CMAKE_BINARY_DIR}/lib${MULTILIB_SUFFIX}\",")
    set(MULTILIB_ADDITIONAL_INSTALL_PATH "\n        \"-L-L${CMAKE_INSTALL_PREFIX}/lib${MULTILIB_SUFFIX}\",")
endif(MULTILIB)

configure_file(${PROJECT_PARENT_DIR}/${CONFIG_NAME}.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}.conf)
# Prepare the config files for installation in bin.
configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}_install.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}_install.conf)
configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}.rebuild.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}_install.rebuild.conf)

#
# Macros.
#

macro(dc INPUT_D OUTLIST_O OUTLIST_BC MOREFLAGS PATH SUFFIX)
    if ("${PATH}" STREQUAL "")
        file(RELATIVE_PATH output ${RUNTIME_DIR} ${INPUT_D})
    else ("${PATH}" STREQUAL "")
        file(RELATIVE_PATH output ${PATH} ${INPUT_D})
    endif ("${PATH}" STREQUAL "")
    get_filename_component(name ${output} NAME_WE)
    get_filename_component(path ${output} PATH)
    if ("${path}" STREQUAL "")
        set(output ${name})
    else ("${path}" STREQUAL "")
        set(output ${path}/${name})
    endif ("${path}" STREQUAL "")
    set(OUTPUT_O ${PROJECT_BINARY_DIR}/${output}${SUFFIX}${CMAKE_C_OUTPUT_EXTENSION})
    set(OUTPUT_BC ${PROJECT_BINARY_DIR}/${output}${SUFFIX}.bc)
    list(APPEND ${OUTLIST_O} ${OUTPUT_O})
    if(BUILD_BC_LIBS)
        list(APPEND ${OUTLIST_BC} ${OUTPUT_BC})
    endif()

    list(FIND GENERATE_DI "${INPUT_D}" INDEX)
    set(DI_CMD "")
    if(NOT INDEX EQUAL -1)
        string(REGEX REPLACE "src/ldc" "src/core" output ${output})
        string(REGEX REPLACE "^src/" "" di_output ${output})

        # If a hand-written .di file exists along the source in src/, just copy
        # it instead of running it through the compiler.
        if(EXISTS "${INPUT_D}i")
            configure_file("${INPUT_D}i" ${CMAKE_BINARY_DIR}/import/${di_output}.di COPYONLY)
        else()
            set(DI_CMD -Hf=${CMAKE_BINARY_DIR}/import/${di_output}.di)
        endif()

        list(REMOVE_AT GENERATE_DI ${INDEX})
    endif()

    # Compile
    if(BUILD_BC_LIBS)
        set(OUTPUT ${OUTPUT_O} ${OUTPUT_BC})
        set(DC_FLAGS --output-o --output-bc)
    else()
        set(OUTPUT ${OUTPUT_O})
        set(DC_FLAGS --output-o)
    endif()
    add_custom_command(
        OUTPUT
            ${OUTPUT}
        COMMAND ${LDC_LOC} ${DC_FLAGS} -c -I${RUNTIME_INCLUDE} -I${RUNTIME_GC_DIR} ${INPUT_D} -of${OUTPUT_O} ${DI_CMD} ${D_FLAGS} ${MOREFLAGS}
        WORKING_DIRECTORY ${PROJECT_PARENT_DIR}
        DEPENDS ${LDC_LOC}
            ${INPUT_D}
            ${LDC_IMPORTS}
            ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}.conf
    )
endmacro(dc)

macro(build_runtime d_flags c_flags ld_flags lib_suffix path_suffix)
    set(output_path ${CMAKE_BINARY_DIR}/lib${path_suffix})

    set(target_suffix "")
    if(NOT "${lib_suffix}" STREQUAL "")
        set(target_suffix "_${lib_suffix}")
    endif(NOT "${lib_suffix}" STREQUAL "")
    if(NOT "${path_suffix}" STREQUAL "")
        set(target_suffix "${target_suffix}_${path_suffix}")
    endif(NOT "${path_suffix}" STREQUAL "")

    set(CORE_O "")
    set(CORE_BC "")
    foreach(f ${CORE_D})
        dc(${f} CORE_O CORE_BC "${d_flags};-disable-invariants" "" "${target_suffix}")
    endforeach(f)

    set(GC_O "")
    set(GC_BC "")
    foreach(f ${GC_D})
        dc(${f} GC_O GC_BC "${d_flags};-disable-invariants" "" "${target_suffix}")
    endforeach(f)

    set(DCRT_O "")
    set(DCRT_BC "")
    foreach(f ${DCRT_D})
        dc(${f} DCRT_O DCRT_BC "${d_flags}" "" "${target_suffix}")
    endforeach(f)

    if(EXISTS ${RUNTIME_DIR})
        set(GCCBUILTINS "${PROJECT_BINARY_DIR}/gccbuiltins_x86.di")
        add_custom_command(
            OUTPUT ${GCCBUILTINS}
            COMMAND ${GEN_GCCBUILTINS_LOC} ${GCCBUILTINS} "x86"
            DEPENDS ${GEN_GCCBUILTINS_LOC}
        )
        if(BUILD_SINGLE_LIB)
            add_library(${RUNTIME_AIO}${target_suffix}
                        ${D_LIBRARY_TYPE}
                        ${CORE_O}
                        ${CORE_C}
                        ${GC_O}
                        ${DCRT_O}
                        ${DCRT_C}
                        ${GCCBUILTINS}
            )
            set(LIBS ${RUNTIME_AIO}${target_suffix})
            set_target_properties(${RUNTIME_AIO}${target_suffix} PROPERTIES OUTPUT_NAME ${RUNTIME_AIO}${lib_suffix})
        else(BUILD_SINGLE_LIB)
            add_library(${RUNTIME_CC}${target_suffix} ${D_LIBRARY_TYPE} ${CORE_O} ${CORE_C} ${GCCBUILTINS})
            add_library(${RUNTIME_GC}${target_suffix} ${D_LIBRARY_TYPE} ${GC_O})
            add_library(${RUNTIME_DC}${target_suffix} ${D_LIBRARY_TYPE} ${DCRT_O} ${DCRT_C})
            set_target_properties(${RUNTIME_CC}${target_suffix} PROPERTIES OUTPUT_NAME ${RUNTIME_CC}${lib_suffix})
            set_target_properties(${RUNTIME_GC}${target_suffix} PROPERTIES OUTPUT_NAME ${RUNTIME_GC}${lib_suffix})
            set_target_properties(${RUNTIME_DC}${target_suffix} PROPERTIES OUTPUT_NAME ${RUNTIME_DC}${lib_suffix})
            set(LIBS
                ${RUNTIME_CC}${lib_suffix}
                ${RUNTIME_GC}${lib_suffix}
                ${RUNTIME_DC}${lib_suffix}
            )
        endif(BUILD_SINGLE_LIB)
    endif()

    if(BUILD_BC_LIBS)
        find_program(LLVM_AR_EXE llvm-ar ${LLVM_INSTDIR}/bin DOC "path to llvm-ar tool")
        if(NOT LLVM_AR_EXE)
            message(SEND_ERROR "llvm-ar not found")
        endif(NOT LLVM_AR_EXE)

        add_library(${RUNTIME_CC}-c ${CORE_C})
        add_library(${RUNTIME_DC}-c ${DCRT_C})
        list(APPEND LIBS
            ${RUNTIME_CC}-c
            ${RUNTIME_DC}-c
        )
        add_custom_command(
            OUTPUT bclibs
            COMMAND ${LLVM_AR_EXE} rs lib${RUNTIME_CC}-bc.a ${CORE_BC}
            COMMAND ${LLVM_AR_EXE} rs lib${RUNTIME_GC}-bc.a ${GC_BC}
            # cannot parse genobj.bc if built with -g
            # COMMAND ${LLVM_AR_EXE} rs lib${RUNTIME_DC}-bc.a ${DCRT_BC}
            WORKING_DIRECTORY ${output_path}
            DEPENDS
                ${CORE_BC}
                ${GC_BC}
                ${DCRT_BC}
                ${LDC_IMPORTS}
        )
        set(BCLIBS bclibs)
    endif(BUILD_BC_LIBS)

    set_target_properties(
        ${LIBS} PROPERTIES
        VERSION                     ${DMDFE_VERSION}
        SOVERSION                   ${DMDFE_PATCH_VERSION}
        LINKER_LANGUAGE             C
        ARCHIVE_OUTPUT_DIRECTORY    ${output_path}
        LIBRARY_OUTPUT_DIRECTORY    ${output_path}
        RUNTIME_OUTPUT_DIRECTORY    ${output_path}
        COMPILE_FLAGS               "${c_flags}"
        LINK_FLAGS                  "${ld_flags}"
    )
    install(TARGETS ${LIBS} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${path_suffix})

    # BCLIBS is empty if BUILD_BC_LIBS is not selected
    add_custom_target(runtime${target_suffix} DEPENDS ${LIBS} ${BCLIBS})

    if(PHOBOS2_DIR)
        set(PHOBOS2_O "")
        set(PHOBOS2_BC "")
        foreach(f ${PHOBOS2_D})
             dc(${f} PHOBOS2_O PHOBOS2_BC "${d_flags};-I${PHOBOS2_DIR}" ${PHOBOS2_DIR} "${target_suffix}")
        endforeach(f)

        add_library(phobos-ldc${target_suffix} ${D_LIBRARY_TYPE}
            ${ZLIB_C}
            ${PHOBOS2_O}
            ${CORE_O}
            ${CORE_C}
            ${GC_O}
            ${DCRT_O}
            ${DCRT_C}
        )
        add_dependencies(phobos-ldc${target_suffix} runtime)
        set_target_properties(
            phobos-ldc${target_suffix} PROPERTIES
            VERSION                     ${DMDFE_VERSION}
            SOVERSION                   ${DMDFE_PATCH_VERSION}
            OUTPUT_NAME                 phobos-ldc${lib_suffix}
            LINKER_LANGUAGE             C
            ARCHIVE_OUTPUT_DIRECTORY    ${output_path}
            LIBRARY_OUTPUT_DIRECTORY    ${output_path}
            RUNTIME_OUTPUT_DIRECTORY    ${output_path}
            COMPILE_FLAGS               "${c_flags}"
            LINK_FLAGS                  "${ld_flags}"
        )
        # Phobos now uses curl
        if(BUILD_SHARED_LIBS)
            target_link_libraries(phobos-ldc${target_suffix} "curl")
        endif()
        install(TARGETS phobos-ldc${target_suffix} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${path_suffix})
        add_dependencies(phobos2 DEPENDS phobos-ldc${target_suffix})
    endif(PHOBOS2_DIR)
endmacro(build_runtime d_flags c_flags ld_flags lib_suffix path_suffix)

#
# Set up build targets.
#

if(PHOBOS2_DIR)
    add_custom_target(phobos2)
endif(PHOBOS2_DIR)
build_runtime("" "" "${LD_FLAGS}" "" "${LIB_SUFFIX}")
set(GENERATE_DI "")
if(MULTILIB)
    build_runtime("-m${MULTILIB_SUFFIX}" "-m${MULTILIB_SUFFIX}" "-m${MULTILIB_SUFFIX} ${LD_FLAGS}" "" "${MULTILIB_SUFFIX}")
endif(MULTILIB)

#
# Install target.
#

install(DIRECTORY     ${CMAKE_BINARY_DIR}/import/core                  DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.di")
if(PHOBOS2_DIR)
    install(DIRECTORY ${PHOBOS2_DIR}/std                               DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.d")
    install(DIRECTORY ${PHOBOS2_DIR}/etc                               DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.d")
    install(FILES     ${PHOBOS2_DIR}/crc32.d                           DESTINATION ${INCLUDE_INSTALL_DIR})
endif(PHOBOS2_DIR)
install(FILES         ${RUNTIME_DIR}/src/object.di                     DESTINATION ${INCLUDE_INSTALL_DIR}/ldc)
install(DIRECTORY     ${RUNTIME_DIR}/import/ldc                        DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.di")
install(FILES         ${GCCBUILTINS}                                   DESTINATION ${INCLUDE_INSTALL_DIR}/ldc)


#
# Test targets.
#
function(add_tests module_files)
    foreach(file ${module_files})
        string(REPLACE ${PROJECT_SOURCE_DIR}/ "" stripped ${file})
        string(REPLACE ".d" "" stripped ${stripped})
        string(REPLACE "/" "_" testroot ${stripped})

        function(testcase name flags)
            # -singleobj to avoid output file clashes when test are run in parallel.
            add_test(NAME ${testroot}_${name}_build
                COMMAND ${LDC_LOC}
                    -of${PROJECT_BINARY_DIR}/${testroot}_${name}
                    -unittest -d -w -singleobj ${flags}
                    ${file} ${PROJECT_SOURCE_DIR}/emptymain.d
            )
            add_test(NAME ${testroot}_${name}_run COMMAND ${PROJECT_BINARY_DIR}/${testroot}_${name})
            set_tests_properties(${testroot}_${name}_run PROPERTIES DEPENDS ${testroot}_${name}_build)
        endfunction()

        testcase(debug -g -debug)
        testcase(release -O -release)
    endforeach()
endfunction()

add_tests("${CORE_D}")
add_tests("${DCRT_D}")
add_tests("${GC_D}")
if(PHOBOS2_DIR)
    add_tests("${PHOBOS2_D}")
endif()
