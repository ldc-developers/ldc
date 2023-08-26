project(runtime C)

cmake_minimum_required(VERSION 3.4.3)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

include(CheckIncludeFile)

# Verify required variables if this CMake project is NOT embedded in the LDC CMake project.
if(NOT LDC_EXE)
    if(NOT LDC_EXE_FULL)
        message(FATAL_ERROR "Please define the path to the LDC executable via -DLDC_EXE_FULL=...")
    endif()
    if(NOT DEFINED DMDFE_MINOR_VERSION OR NOT DEFINED DMDFE_PATCH_VERSION)
        message(FATAL_ERROR "Please define the CMake variables DMDFE_MINOR_VERSION and DMDFE_PATCH_VERSION.")
    endif()

    # Helper function
    function(append value)
        foreach(variable ${ARGN})
            if(${variable} STREQUAL "")
                set(${variable} "${value}" PARENT_SCOPE)
            else()
                set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
            endif()
        endforeach(variable)
    endfunction()
endif()

#
# Main configuration.
#

set(MULTILIB              OFF                                 CACHE BOOL   "Build both 32/64 bit runtime libraries")
set(BUILD_LTO_LIBS        OFF                                 CACHE BOOL   "Also build the runtime as LLVM bitcode libraries for LTO")
set(INCLUDE_INSTALL_DIR   ${CMAKE_INSTALL_PREFIX}/include/d   CACHE PATH   "Path to install D modules to")
set(BUILD_SHARED_LIBS     AUTO                                CACHE STRING "Whether to build the runtime as a shared library (ON|OFF|BOTH)")
set(D_FLAGS               -w;-de;-preview=dip1000;-preview=dtorfields;-preview=fieldwise CACHE STRING "Runtime D compiler flags, separated by ';'")
set(D_EXTRA_FLAGS         ""                                  CACHE STRING "Runtime extra D compiler flags, separated by ';'")
set(D_FLAGS_DEBUG         -g;-link-defaultlib-debug;-d-debug  CACHE STRING "Runtime D compiler flags (debug libraries), separated by ';'")
set(D_FLAGS_RELEASE       -O3;-release;-femit-local-var-lifetime  CACHE STRING "Runtime D compiler flags (release libraries), separated by ';'")
set(COMPILE_ALL_D_FILES_AT_ONCE ON                            CACHE BOOL   "Compile all D files for the runtime libs in a single command line instead of separately. Disabling this is useful for many CPU cores and/or iterative development.")
set(RT_ARCHIVE_WITH_LDC   ON                                  CACHE STRING "Whether to archive the static runtime libs via LDC instead of CMake archiver")
set(RT_CFLAGS             ""                                  CACHE STRING "Runtime extra C compiler flags, separated by ' '")
set(LD_FLAGS              ""                                  CACHE STRING "Runtime extra C linker flags, separated by ' '")
set(C_SYSTEM_LIBS         AUTO                                CACHE STRING "C system libraries for linking shared libraries and test runners, separated by ';'")
set(TARGET_SYSTEM         AUTO                                CACHE STRING "Target OS/toolchain for cross-compilation (e.g., 'Linux;UNIX', 'Darwin;APPLE;UNIX', 'Windows;MSVC')")
set(RT_SUPPORT_SANITIZERS OFF                                 CACHE BOOL   "Build runtime libraries with sanitizer support (e.g. for AddressSanitizer)")

set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})

set(DRUNTIME_EXTRA_FLAGS          )
set(DRUNTIME_EXTRA_UNITTEST_FLAGS -d-version=CoreUnittest -unittest -checkaction=context -linkonce-templates)
set(PHOBOS2_EXTRA_FLAGS           )
set(PHOBOS2_EXTRA_UNITTEST_FLAGS  -d-version=StdUnittest -unittest -linkonce-templates)

# Shadow the D_FLAGS cache variable by a regular variable containing all base D flags
set(D_FLAGS ${D_FLAGS} ${D_EXTRA_FLAGS})
if (RT_SUPPORT_SANITIZERS)
    message(STATUS "Building runtime libraries with sanitizer support")
    list(APPEND D_FLAGS -d-version=SupportSanitizers)
endif()

# Auto-detect TARGET_SYSTEM from host
if("${TARGET_SYSTEM}" STREQUAL "AUTO")
    set(TARGET_SYSTEM ${CMAKE_SYSTEM_NAME})
    # Additionally add MSVC, APPLE and/or UNIX
    if(MSVC)
        list(APPEND TARGET_SYSTEM "MSVC")
    endif()
    if(APPLE)
        list(APPEND TARGET_SYSTEM "APPLE")
    endif()
    if(UNIX)
        list(APPEND TARGET_SYSTEM "UNIX")
    endif()
endif()

if("${TARGET_SYSTEM}" MATCHES "UNIX")
    ENABLE_LANGUAGE(ASM)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(HOST_BITNESS 64)
    set(MULTILIB_SUFFIX 32)
else()
    set(HOST_BITNESS 32)
    set(MULTILIB_SUFFIX 64)
endif()

set(SHARED_LIBS_SUPPORTED OFF)
if("${TARGET_SYSTEM}" MATCHES "Windows|Linux|FreeBSD|DragonFly|APPLE"
   AND (NOT "${TARGET_SYSTEM}" MATCHES "Android"))
    set(SHARED_LIBS_SUPPORTED ON)
endif()

if(${BUILD_SHARED_LIBS} STREQUAL "AUTO")
    if(SHARED_LIBS_SUPPORTED)
        set(BUILD_SHARED_LIBS BOTH)
    else()
        set(BUILD_SHARED_LIBS OFF)
    endif()
endif()
if(LDC_EXE)
    set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} PARENT_SCOPE) # e.g., for tests/d2/CMakeLists.txt and forwarding to dmd-testsuite
endif()

set(SHARED_LIB_SUFFIX "")
if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
    if(NOT SHARED_LIBS_SUPPORTED)
        message(FATAL_ERROR "Shared libraries (BUILD_SHARED_LIBS) are only supported on Windows, Linux, macOS, FreeBSD and DragonFly for the time being.")
    endif()
    set(SHARED_LIB_SUFFIX "-shared")
endif()

# Auto-detect C system libraries
if("${C_SYSTEM_LIBS}" STREQUAL "AUTO")
    if("${TARGET_SYSTEM}" MATCHES "MSVC")
        set(C_SYSTEM_LIBS        libcmt libvcruntime)
        set(C_SYSTEM_LIBS_SHARED msvcrt vcruntime oldnames legacy_stdio_definitions)
    else()
        if("${TARGET_SYSTEM}" MATCHES "Android")
            set(C_SYSTEM_LIBS m c)
        elseif("${TARGET_SYSTEM}" MATCHES "Linux")
            set(C_SYSTEM_LIBS m pthread rt dl)
        else()
            set(C_SYSTEM_LIBS m pthread)
        endif()
        set(C_SYSTEM_LIBS_SHARED ${C_SYSTEM_LIBS})
    endif()
else()
    set(C_SYSTEM_LIBS_SHARED ${C_SYSTEM_LIBS})
endif()

message(STATUS "-- LDC runtime configuration:")
message(STATUS "--  - Building 32/64-bit libraries (MULTILIB): ${MULTILIB}")
message(STATUS "--  - Building shared libraries (BUILD_SHARED_LIBS): ${BUILD_SHARED_LIBS}")
message(STATUS "--  - Building LTO libraries (BUILD_LTO_LIBS): ${BUILD_LTO_LIBS}")

get_directory_property(PROJECT_PARENT_DIR DIRECTORY ${PROJECT_SOURCE_DIR} PARENT_DIRECTORY)
set(RUNTIME_DIR ${PROJECT_SOURCE_DIR}/druntime CACHE PATH "druntime root directory")
set(PHOBOS2_DIR ${PROJECT_SOURCE_DIR}/phobos CACHE PATH "Phobos root directory")
set(JITRT_DIR ${PROJECT_SOURCE_DIR}/jit-rt CACHE PATH "jit runtime root directory")

#
# Gather source files.
#

# druntime D parts
file(GLOB_RECURSE DRUNTIME_D ${RUNTIME_DIR}/src/*.d)
list(REMOVE_ITEM DRUNTIME_D ${RUNTIME_DIR}/src/test_runner.d)
# remove unsupported etc/linux/memoryerror.d (see issue #1915)
list(REMOVE_ITEM DRUNTIME_D ${RUNTIME_DIR}/src/etc/linux/memoryerror.d)
# remove some modules in rt/
list(REMOVE_ITEM DRUNTIME_D
    ${RUNTIME_DIR}/src/rt/alloca.d
    ${RUNTIME_DIR}/src/rt/cmath2.d
    ${RUNTIME_DIR}/src/rt/deh_win32.d
    ${RUNTIME_DIR}/src/rt/llmath.d
    ${RUNTIME_DIR}/src/rt/memset.d
    ${RUNTIME_DIR}/src/rt/sections_osx_x86.d
    ${RUNTIME_DIR}/src/rt/sections_osx_x86_64.d
    ${RUNTIME_DIR}/src/rt/sections_solaris.d
    ${RUNTIME_DIR}/src/rt/sections_win32.d
)
# only include core/sys/ modules matching the platform
file(GLOB_RECURSE DRUNTIME_D_BIONIC  ${RUNTIME_DIR}/src/core/sys/bionic/*.d)
file(GLOB_RECURSE DRUNTIME_D_DARWIN  ${RUNTIME_DIR}/src/core/sys/darwin/*.d)
file(GLOB_RECURSE DRUNTIME_D_DRAGONFLYBSD ${RUNTIME_DIR}/src/core/sys/dragonflybsd/*.d)
file(GLOB_RECURSE DRUNTIME_D_FREEBSD ${RUNTIME_DIR}/src/core/sys/freebsd/*.d)
file(GLOB_RECURSE DRUNTIME_D_LINUX   ${RUNTIME_DIR}/src/core/sys/linux/*.d)
file(GLOB_RECURSE DRUNTIME_D_NETBSD  ${RUNTIME_DIR}/src/core/sys/netbsd/*.d)
file(GLOB_RECURSE DRUNTIME_D_OPENBSD ${RUNTIME_DIR}/src/core/sys/openbsd/*.d)
file(GLOB_RECURSE DRUNTIME_D_POSIX   ${RUNTIME_DIR}/src/core/sys/posix/*.d)
file(GLOB_RECURSE DRUNTIME_D_SOLARIS ${RUNTIME_DIR}/src/core/sys/solaris/*.d)
file(GLOB_RECURSE DRUNTIME_D_WINDOWS ${RUNTIME_DIR}/src/core/sys/windows/*.d)
list(REMOVE_ITEM DRUNTIME_D
    ${DRUNTIME_D_BIONIC}  ${DRUNTIME_D_DARWIN}  ${DRUNTIME_D_DRAGONFLYBSD}
    ${DRUNTIME_D_FREEBSD} ${DRUNTIME_D_LINUX}   ${DRUNTIME_D_NETBSD}
    ${DRUNTIME_D_OPENBSD} ${DRUNTIME_D_POSIX}   ${DRUNTIME_D_SOLARIS}
    ${DRUNTIME_D_WINDOWS}
)
if("${TARGET_SYSTEM}" MATCHES "Windows")
    list(APPEND DRUNTIME_D ${DRUNTIME_D_WINDOWS})
elseif("${TARGET_SYSTEM}" MATCHES "UNIX")
    list(APPEND DRUNTIME_D ${DRUNTIME_D_POSIX})
    if("${TARGET_SYSTEM}" MATCHES "APPLE")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_DARWIN})
    elseif("${TARGET_SYSTEM}" MATCHES "DragonFly")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_DRAGONFLYBSD})
    elseif("${TARGET_SYSTEM}" MATCHES "FreeBSD")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_FREEBSD})
    elseif("${TARGET_SYSTEM}" MATCHES "Linux")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_LINUX})
        list(APPEND DRUNTIME_D ${DRUNTIME_D_BIONIC})
    elseif("${TARGET_SYSTEM}" MATCHES "NetBSD")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_NETBSD})
    elseif("${TARGET_SYSTEM}" MATCHES "OpenBSD")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_OPENBSD})
    elseif("${TARGET_SYSTEM}" MATCHES "SunOS")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_SOLARIS})
    endif()
endif()

# druntime C parts
file(GLOB_RECURSE DRUNTIME_C ${RUNTIME_DIR}/src/*.c)
list(REMOVE_ITEM DRUNTIME_C ${RUNTIME_DIR}/src/rt/dylib_fixes.c)
# remove unsupported valgrind.c on Windows
if("${TARGET_SYSTEM}" MATCHES "Windows")
    list(REMOVE_ITEM DRUNTIME_C ${RUNTIME_DIR}/src/etc/valgrind/valgrind.c)
endif()

# druntime ASM parts
set(DRUNTIME_ASM)
if("${TARGET_SYSTEM}" MATCHES "UNIX")
    list(APPEND DRUNTIME_ASM ${RUNTIME_DIR}/src/core/threadasm.S ${RUNTIME_DIR}/src/ldc/eh_asm.S)
endif()

if(PHOBOS2_DIR)
    # Phobos D parts
    file(GLOB_RECURSE PHOBOS2_D_ETC ${PHOBOS2_DIR}/etc/*.d)
    file(GLOB_RECURSE PHOBOS2_D_STD ${PHOBOS2_DIR}/std/*.d)
    set(PHOBOS2_D ${PHOBOS2_D_ETC} ${PHOBOS2_D_STD})
    # only include std/windows/ modules on Windows
    if(NOT "${TARGET_SYSTEM}" MATCHES "Windows")
        file(GLOB_RECURSE PHOBOS2_D_WINDOWS ${PHOBOS2_DIR}/std/windows/*.d)
        list(REMOVE_ITEM PHOBOS2_D ${PHOBOS2_D_WINDOWS})
    endif()

    # Phobos C parts
    file(GLOB_RECURSE PHOBOS2_C ${PHOBOS2_DIR}/etc/*.c)
    # remove zlib test modules
    list(REMOVE_ITEM PHOBOS2_C
        ${PHOBOS2_DIR}/etc/c/zlib/test/example.c
        ${PHOBOS2_DIR}/etc/c/zlib/test/infcover.c
        ${PHOBOS2_DIR}/etc/c/zlib/test/minigzip.c
    )
endif()

#
# Create configuration files.
#

# Default -rpath linker option when linking against shared libraries.
if(SHARED_LIBS_SUPPORTED)
    set(SHARED_LIBS_RPATH         "${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}")
    set(SHARED_LIBS_INSTALL_RPATH "${CMAKE_INSTALL_LIBDIR}")
endif()

# Only have either shared or static libs?
# Then explicitly default to linking against them via default LDC switch.
if(${BUILD_SHARED_LIBS} STREQUAL "ON")
    set(ADDITIONAL_DEFAULT_LDC_SWITCHES "${ADDITIONAL_DEFAULT_LDC_SWITCHES}\n        \"-link-defaultlib-shared\",")
elseif(${BUILD_SHARED_LIBS} STREQUAL "OFF")
    set(ADDITIONAL_DEFAULT_LDC_SWITCHES "${ADDITIONAL_DEFAULT_LDC_SWITCHES}\n        \"-link-defaultlib-shared=false\",")
endif()

# LLVM 16: Disable function specializations by default.
# They cause miscompiles of e.g. the frontend for some targets (macOS x86_64 and Windows x64).
if(LDC_LLVM_VER GREATER 1599 AND LDC_LLVM_VER LESS 1700)
    set(ADDITIONAL_DEFAULT_LDC_SWITCHES "${ADDITIONAL_DEFAULT_LDC_SWITCHES}\n        \"-func-specialization-size-threshold=1000000000\",")
endif()

# Default wasm stack is only 64kb, this is rather small, let's bump it to 1mb
set(WASM_DEFAULT_LDC_SWITCHES "${WASM_DEFAULT_LDC_SWITCHES}\n        \"-L-z\", \"-Lstack-size=1048576\",")
# Protect from stack overflow overwriting global memory
set(WASM_DEFAULT_LDC_SWITCHES "${WASM_DEFAULT_LDC_SWITCHES}\n        \"-L--stack-first\",")
if(LDC_WITH_LLD)
    set(WASM_DEFAULT_LDC_SWITCHES "${WASM_DEFAULT_LDC_SWITCHES}\n        \"-link-internally\",")
endif()
# LLD 8+ requires (new) `--export-dynamic` for WebAssembly (https://github.com/ldc-developers/ldc/issues/3023).
set(WASM_DEFAULT_LDC_SWITCHES "${WASM_DEFAULT_LDC_SWITCHES}\n        \"-L--export-dynamic\",")

# Directory filled with auto-generated import files
set(LDC_BUILD_IMPORT_DIR "${PROJECT_BINARY_DIR}/import")

# Only generate the config files if this CMake project is embedded in the LDC CMake project.
if(LDC_EXE)
    if(PHOBOS2_DIR)
        set(CONFIG_NAME ${LDC_EXE}_phobos)
    else()
        set(CONFIG_NAME ${LDC_EXE})
    endif()
    set(conf_path         ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}.conf)
    set(install_conf_path ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}_install.conf)
    configure_file(${PROJECT_PARENT_DIR}/${CONFIG_NAME}.conf.in ${conf_path} @ONLY)
    configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}_install.conf.in ${install_conf_path} @ONLY)

    # macOS has fat libraries; otherwise, append a separate config file section for the
    # multilib target and override the lib directory.
    if(MULTILIB AND NOT "${TARGET_SYSTEM}" MATCHES "APPLE")
        # Rename the just configured host-only *.conf files to *.conf.in during CMake invocation.
        file(RENAME ${conf_path} ${conf_path}.in)
        file(RENAME ${install_conf_path} ${install_conf_path}.in)

        set(MULTILIB_DIR         "${CMAKE_BINARY_DIR}/lib${MULTILIB_SUFFIX}")
        set(MULTILIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib${MULTILIB_SUFFIX}")

        # The helper script determines the multilib triple by parsing the
        # `ldc2 -m32 -v` output. This means we depend on a built ldc2
        # executable; sadly, we cannot add post-build commands to that
        # top-level target directly, so use a custom target (built as part
        # of `all`) to finalize the 2 config files.
        add_custom_command(OUTPUT ${conf_path} VERBATIM
            COMMAND ${PROJECT_PARENT_DIR}/tools/add-multilib-section.sh
                    ${LDC_EXE_FULL}
                    ${MULTILIB_SUFFIX}
                    ${MULTILIB_DIR}
                    "$<$<BOOL:${SHARED_LIBS_SUPPORTED}>:${MULTILIB_DIR}>"
                    ${conf_path}.in
                    ${conf_path}
            DEPENDS ${LDC_EXE} ${LDC_EXE_FULL}
        )
        add_custom_command(OUTPUT ${install_conf_path} VERBATIM
            COMMAND ${PROJECT_PARENT_DIR}/tools/add-multilib-section.sh
                    ${LDC_EXE_FULL}
                    ${MULTILIB_SUFFIX}
                    ${MULTILIB_INSTALL_DIR}
                    "$<$<BOOL:${SHARED_LIBS_SUPPORTED}>:${MULTILIB_INSTALL_DIR}>"
                    ${install_conf_path}.in
                    ${install_conf_path}
            DEPENDS ${LDC_EXE} ${LDC_EXE_FULL}
        )
        add_custom_target(add-multilib-section ALL
            DEPENDS ${conf_path} ${install_conf_path}
        )
    endif()
endif()

#
# druntime/Phobos compilation helpers.
#

set(GCCBUILTINS "")
if(TARGET gen_gccbuiltins)
  file(MAKE_DIRECTORY "${LDC_BUILD_IMPORT_DIR}/ldc")

  function(gen_gccbuiltins name)
    set(module "${LDC_BUILD_IMPORT_DIR}/ldc/gccbuiltins_${name}.di")
    if (GCCBUILTINS STREQUAL "")
      set(GCCBUILTINS "${module}" PARENT_SCOPE)
    else()
      set(GCCBUILTINS "${GCCBUILTINS};${module}" PARENT_SCOPE)
    endif()
    add_custom_command(
        OUTPUT ${module}
        COMMAND gen_gccbuiltins ${module} "${name}"
        DEPENDS gen_gccbuiltins
    )
  endfunction()

  set(target_arch "AArch64;AMDGPU;ARM;Mips;RISCV;NVPTX;PowerPC;SystemZ;X86")
  set(target_name "aarch64;amdgcn;arm;mips;riscv;nvvm;ppc;s390;x86")

  foreach(target ${LLVM_TARGETS_TO_BUILD})
    list(FIND target_arch ${target} idx)
    if(idx GREATER -1)
      list(GET target_name ${idx} name)
      gen_gccbuiltins(${name})
    endif()
  endforeach()
endif()

# Always build zlib and other C parts of the runtime in release mode, regardless
# of what the user chose for LDC itself. Also add other C_FLAGS here.
# 1) Set up CMAKE_C_FLAGS_RELEASE
if("${TARGET_SYSTEM}" MATCHES "MSVC")
    # Omit all references to the default C runtime libs.
    # We want to be able to link against all 4 C runtime variants (libcmt[d] / msvcrt[d]).
    string(REGEX REPLACE "(^| )[/-]M[TD]d?( |$)" "\\2" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    append("/MT /Zl" CMAKE_C_FLAGS_RELEASE)
    # Disable buffer overflow checks to prevent dependencies on special C symbols
    # not available in the MinGW-w64 libs.
    append("/GS-" CMAKE_C_FLAGS_RELEASE)
    # warning C4100: unreferenced formal parameter
    # warning C4127: conditional expression is constant
    # warning C4131: uses old-style declarator
    # warning C4206: nonstandard extension used: translation unit is empty
    # warnings C42{44,67}: conversion from '...' to '...', possible loss of data
    # warning C4245: conversion from '...' to '...', signed/unsigned mismatch
    # warning C4996: zlib uses 'deprecated' POSIX names
    append("/wd4100 /wd4127 /wd4131 /wd4206 /wd4244 /wd4245 /wd4267 /wd4996" CMAKE_C_FLAGS_RELEASE)
endif()
CHECK_INCLUDE_FILE(unistd.h HAVE_UNISTD_H)
if (HAVE_UNISTD_H)
    # Needed for zlib
    append("-DHAVE_UNISTD_H" CMAKE_C_FLAGS)
endif()
# 2) Set all other CMAKE_C_FLAGS variants to CMAKE_C_FLAGS_RELEASE
set(variables
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_MINSIZEREL
    CMAKE_C_FLAGS_RELWITHDEBINFO
)
foreach(variable ${variables})
    set(${variable} "${CMAKE_C_FLAGS_RELEASE}")
endforeach()

# Compiles the given D modules to object files, and if enabled, bitcode files.
# The paths of the output files are appended to outlist_o and outlist_bc, respectively.
macro(dc src_files src_basedir d_flags output_basedir emit_bc all_at_once single_obj_name outlist_o outlist_bc)
    set(dc_flags -c --output-o ${d_flags})
    if(${emit_bc})
        list(APPEND dc_flags -flto=thin --output-bc)
    endif()

    set(dc_deps ${LDC_EXE} ${LDC_EXE_FULL} ${GCCBUILTINS})
    if(TARGET add-multilib-section)
        # Make sure the config files are available before invoking LDC.
        set(dc_deps ${dc_deps} add-multilib-section)
    endif()

    set(relative_src_files "")
    set(new_o "")
    set(new_bc "")

    foreach(f ${src_files})
        file(RELATIVE_PATH relative_src_file ${src_basedir} ${f})
        get_filename_component(name ${relative_src_file} NAME_WE)
        get_filename_component(path ${relative_src_file} PATH)
        if("${path}" STREQUAL "")
            set(output_root ${name})
        else()
            set(output_root ${path}/${name})
        endif()

        set(output_o  ${output_basedir}/${output_root}${CMAKE_C_OUTPUT_EXTENSION})
        set(output_bc ${output_basedir}/${output_root}.bc${CMAKE_C_OUTPUT_EXTENSION})
        list(APPEND new_o ${output_o})
        if(${emit_bc})
            list(APPEND new_bc ${output_bc})
        endif()

        if(${all_at_once})
            list(APPEND relative_src_files ${relative_src_file})
        else()
            set(outfiles ${output_o})
            set(renameCommand "")
            if(${emit_bc})
                # rename generated *.bc to *.bc.o[bj]
                set(renameCommand ${CMAKE_COMMAND} -E rename ${output_basedir}/${output_root}.bc ${output_bc})
                list(APPEND outfiles ${output_bc})
            endif()

            add_custom_command(
                OUTPUT  ${outfiles}
                COMMAND ${LDC_EXE_FULL} ${dc_flags} -of=${output_o} ${relative_src_file}
                COMMAND ${renameCommand}
                WORKING_DIRECTORY ${src_basedir}
                DEPENDS ${f} ${dc_deps}
            )
        endif()
    endforeach()

    if(${all_at_once})
        if("${single_obj_name}" STREQUAL "")
            list(APPEND dc_flags -od=${output_basedir} -op)
        else()
            set(new_o ${output_basedir}/${single_obj_name}${CMAKE_C_OUTPUT_EXTENSION})
            if(${emit_bc})
                set(new_bc ${output_basedir}/${single_obj_name}.bc${CMAKE_C_OUTPUT_EXTENSION})
            endif()
            list(APPEND dc_flags -of=${new_o})
        endif()

        # Use a response file on Windows, in order not to exceed the max command-line length.
        if(WIN32)
            set(first_obj "")
            list(GET new_o 0 first_obj)
            string(REPLACE ";" " " relative_src_files "${relative_src_files}")
            file(WRITE ${first_obj}.rsp ${relative_src_files})
            set(relative_src_files "@${first_obj}.rsp")
        endif()

        add_custom_command(
            OUTPUT  ${new_o} ${new_bc}
            COMMAND ${LDC_EXE_FULL} ${dc_flags} ${relative_src_files}
            WORKING_DIRECTORY ${src_basedir}
            DEPENDS ${src_files} ${dc_deps}
        )

        # append a rename command for each generated .bc
        foreach(dst_bc ${new_bc})
            string(REGEX REPLACE "${CMAKE_C_OUTPUT_EXTENSION}$" "" src_bc "${dst_bc}")
            add_custom_command(
                OUTPUT ${new_o} ${new_bc} APPEND
                COMMAND ${CMAKE_COMMAND} -E rename ${src_bc} ${dst_bc}
            )
        endforeach()
    endif()

    list(APPEND ${outlist_o} ${new_o})
    list(APPEND ${outlist_bc} ${new_bc})
endmacro()

# Sets target_suffix to a purely cosmetical suffix for the CMake target names
# from the given suffixes on the library name and the target path. We could use
# any string that resolves the ambiguities between the different variants.
macro(get_target_suffix lib_suffix path_suffix target_suffix)
    set(${target_suffix} "")
    if(NOT "${lib_suffix}" STREQUAL "")
        set(${target_suffix} "${lib_suffix}")
    endif()

    # If LIB_SUFFIX is set there is always a suffix; leave it off for simplicity.
    if(NOT "${path_suffix}" STREQUAL "" AND NOT "${path_suffix}" STREQUAL "${LIB_SUFFIX}")
        set(${target_suffix} "${${target_suffix}}_${path_suffix}")
    endif()
endmacro()

# Sets up the target(s) for building the druntime D object files, appending the
# names of the (bitcode) files to link into the library to outlist_o (outlist_bc).
macro(compile_druntime d_flags lib_suffix path_suffix emit_bc all_at_once single_obj outlist_o outlist_bc)
    get_target_suffix("${lib_suffix}" "${path_suffix}" target_suffix)
    if(${single_obj})
        set(single_obj_name druntime)
    else()
        set(single_obj_name "")
    endif()
    dc("${DRUNTIME_D}"
       "${RUNTIME_DIR}/src"
       "-conf=;${d_flags};${DRUNTIME_EXTRA_FLAGS};-I${RUNTIME_DIR}/src"
       "${PROJECT_BINARY_DIR}/objects${target_suffix}"
       "${emit_bc}"
       "${all_at_once}"
       "${single_obj_name}"
       ${outlist_o}
       ${outlist_bc}
    )
endmacro()

# Sets up the targets for building the Phobos D object files, appending the
# names of the (bitcode) files to link into the library to outlist_o (outlist_bc).
macro(compile_phobos2 d_flags lib_suffix path_suffix emit_bc all_at_once single_obj outlist_o outlist_bc)
    get_target_suffix("${lib_suffix}" "${path_suffix}" target_suffix)
    if(${single_obj})
        set(single_obj_name phobos2)
    else()
        set(single_obj_name "")
    endif()
    dc("${PHOBOS2_D}"
       "${PHOBOS2_DIR}"
       "-conf=;${d_flags};${PHOBOS2_EXTRA_FLAGS};-I${RUNTIME_DIR}/src;-I${PHOBOS2_DIR}"
       "${PROJECT_BINARY_DIR}/objects${target_suffix}"
       "${emit_bc}"
       "${all_at_once}"
       "${single_obj_name}"
       ${outlist_o}
       ${outlist_bc}
    )
endmacro()

# Define an optional 'D' CMake linker language for the static runtime libs,
# using LDC as archiver. LDC handles (cross-)archiving internally via LLVM
# and supports LTO objects.
set(CMAKE_D_CREATE_STATIC_LIBRARY "${LDC_EXE_FULL} -lib -of=<TARGET> <OBJECTS>")
foreach(f ${D_FLAGS})
    append("\"${f}\"" CMAKE_D_CREATE_STATIC_LIBRARY)
endforeach()

# Sets the common properties for all library targets.
function(set_common_library_properties target name output_dir c_flags ld_flags is_shared)
    # Windows: enforce .pdb generation for debug DLLs
    set(full_ldflags "${ld_flags}")
    if(is_shared AND "${name}" MATCHES "-debug" AND "${TARGET_SYSTEM}" MATCHES "MSVC")
        append("/DEBUG" full_ldflags)
    endif()

    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME                 ${name}
        ARCHIVE_OUTPUT_DIRECTORY    ${output_dir}
        LIBRARY_OUTPUT_DIRECTORY    ${output_dir}
        RUNTIME_OUTPUT_DIRECTORY    ${output_dir}
        COMPILE_DEFINITIONS         "LDC=1"
        COMPILE_FLAGS               "${c_flags}"
        LINK_FLAGS                  "${full_ldflags}"
        VERSION                     ${DMDFE_MINOR_VERSION}.${DMDFE_PATCH_VERSION}
        SOVERSION                   ${DMDFE_MINOR_VERSION}
        LINKER_LANGUAGE             C
        MACOSX_RPATH                ON
    )
    if(RT_ARCHIVE_WITH_LDC AND NOT is_shared)
        set_target_properties(${target} PROPERTIES LINKER_LANGUAGE D)
    endif()

    # ldc2 defaults to position-independent code on Linux to match the implicit
    # linker default on Ubuntu 16.10 and above. As we might be building on an
    # older system (e.g. binary packages), we need to make sure the C parts are
    # built as PIC as well.
    if("${TARGET_SYSTEM}" MATCHES "Linux")
        set_target_properties(${target} PROPERTIES
            POSITION_INDEPENDENT_CODE ON)
    endif()
endfunction()

# Builds a copy of druntime/Phobos from the specified object (bitcode) files.
# The names of the added library targets are appended to outlist_targets.
macro(build_runtime_libs druntime_o druntime_bc phobos2_o phobos2_bc c_flags ld_flags
                         lib_suffix path_suffix is_shared emit_bc outlist_targets)
    set(output_path ${CMAKE_BINARY_DIR}/lib${path_suffix})

    set(library_type STATIC)
    if("${is_shared}" STREQUAL "ON")
        set(library_type SHARED)
    endif()

    get_target_suffix("${lib_suffix}" "${path_suffix}" target_suffix)
    add_library(druntime-ldc${target_suffix} ${library_type}
        ${druntime_o} ${DRUNTIME_C} ${DRUNTIME_ASM})
    set_common_library_properties(druntime-ldc${target_suffix}
        druntime-ldc${lib_suffix} ${output_path}
        "${c_flags}" "${ld_flags}" ${is_shared}
    )

    # When building a shared library, we need to link in all the default
    # libraries otherwise implicitly added by LDC to make it loadable from
    # C executables.
    if("${is_shared}" STREQUAL "ON")
        target_link_libraries(druntime-ldc${target_suffix} ${C_SYSTEM_LIBS_SHARED})
    endif()

    list(APPEND ${outlist_targets} druntime-ldc${target_suffix})

    if(PHOBOS2_DIR)
        add_library(phobos2-ldc${target_suffix} ${library_type}
            ${phobos2_o} ${PHOBOS2_C})
        set_common_library_properties(phobos2-ldc${target_suffix}
            phobos2-ldc${lib_suffix} ${output_path}
            "${c_flags}" "${ld_flags}" ${is_shared}
        )

        if("${is_shared}" STREQUAL "ON")
            # Make sure to link shared unittest-Phobos against shared NON-unittest-druntime
            # in order to exclude the druntime tests for the Phobos test runner.
            string(REPLACE "-unittest" "" target_suffix_without_unittest "${target_suffix}")
            target_link_libraries(phobos2-ldc${target_suffix}
                druntime-ldc${target_suffix_without_unittest} ${C_SYSTEM_LIBS_SHARED})

            # Windows: export zlib symbols from Phobos DLL
            if("${TARGET_SYSTEM}" MATCHES "Windows")
                target_compile_definitions(phobos2-ldc${target_suffix} PRIVATE "ZLIB_DLL=1")
            endif()
        endif()

        list(APPEND ${outlist_targets} phobos2-ldc${target_suffix})
    endif()

    if("${emit_bc}")
        add_library(druntime-ldc-lto${target_suffix} STATIC
            ${druntime_bc} ${DRUNTIME_C} ${DRUNTIME_ASM})
        set_common_library_properties(druntime-ldc-lto${target_suffix}
            druntime-ldc-lto${lib_suffix} ${output_path}
            "${c_flags}" "${ld_flags}" OFF
        )

        add_library(phobos2-ldc-lto${target_suffix} STATIC
            ${phobos2_bc} ${PHOBOS2_C})
        set_common_library_properties(phobos2-ldc-lto${target_suffix}
            phobos2-ldc-lto${lib_suffix} ${output_path}
            "${c_flags}" "${ld_flags}" OFF
        )
    endif()
endmacro()

# Builds a static and/or shared copy of druntime/Phobos.
macro(build_runtime_variant d_flags c_flags ld_flags lib_suffix path_suffix emit_bc all_d_files_at_once outlist_targets)
    set(is_unittest OFF)
    if("${lib_suffix}" MATCHES "-unittest")
        set(is_unittest ON)
    endif()

    set(phobos2_d_flags ${d_flags})
    if(is_unittest)
        list(APPEND phobos2_d_flags ${PHOBOS2_EXTRA_UNITTEST_FLAGS})
    endif()

    # Posix: when compiling both static and shared Phobos *and* compiling the modules separately
    # (by default, only for the unittests), only compile once to save build time.
    set(phobos2_common "")
    if((NOT "${TARGET_SYSTEM}" MATCHES "Windows") AND BUILD_SHARED_LIBS STREQUAL "BOTH"
       AND NOT ${all_d_files_at_once})
        set(phobos2_o "")
        set(phobos2_bc "")
        compile_phobos2("${phobos2_d_flags};-relocation-model=pic;-fvisibility=public;-dllimport=all"
                        "${lib_suffix}${SHARED_LIB_SUFFIX}" "${path_suffix}"
                        "${emit_bc}" "${all_d_files_at_once}" "OFF" phobos2_o phobos2_bc)

        # Use a dummy custom target ('phobos2-ldc…-common') depending solely on the Phobos D objects
        # (custom commands) as dependency for static+shared Phobos libs.
        # Otherwise building both libs in parallel may result in conflicting Phobos module compilations
        # (at least with the make generator), a known CMake issue.
        set(phobos2_common phobos2-ldc${target_suffix}-common)
        add_custom_target(${phobos2_common} DEPENDS ${phobos2_o} ${phobos2_bc})
    endif()

    set(druntime_d_flags ${d_flags})
    if(is_unittest)
        list(APPEND druntime_d_flags ${DRUNTIME_EXTRA_UNITTEST_FLAGS})
    endif()

    # static druntime/Phobos
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "ON")
        set(druntime_o "")
        set(druntime_bc "")
        compile_druntime("${druntime_d_flags}" "${lib_suffix}" "${path_suffix}"
                         "${emit_bc}" "${all_d_files_at_once}" "OFF" druntime_o druntime_bc)

        # compile Phobos (with hidden visibility on Windows)
        if(phobos2_common STREQUAL "")
            set(phobos2_o "")
            set(phobos2_bc "")
            compile_phobos2("${phobos2_d_flags}" "${lib_suffix}" "${path_suffix}"
                            "${emit_bc}" "${all_d_files_at_once}" "OFF" phobos2_o phobos2_bc)
        endif()

        build_runtime_libs("${druntime_o}" "${druntime_bc}" "${phobos2_o}" "${phobos2_bc}"
                           "${c_flags}" "${ld_flags}" "${lib_suffix}" "${path_suffix}"
                           "OFF" "${emit_bc}" ${outlist_targets})

        if(NOT phobos2_common STREQUAL "")
            add_dependencies(phobos2-ldc${target_suffix} ${phobos2_common})
        endif()
    endif()
    # shared druntime/Phobos
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
        # compile druntime with public visibility and extra `version (Shared)`
        set(druntime_o "")
        set(druntime_bc "")
        compile_druntime("${druntime_d_flags};-relocation-model=pic;-fvisibility=public;-dllimport=none;-d-version=Shared"
                         "${lib_suffix}${SHARED_LIB_SUFFIX}" "${path_suffix}"
                         "OFF" "${all_d_files_at_once}" "OFF" druntime_o druntime_bc)

        set(rt_dso_o "${PROJECT_BINARY_DIR}/objects${target_suffix}/rt/dso${CMAKE_C_OUTPUT_EXTENSION}")

        # compile Phobos with public visibility (and preferably to a single object file)
        if(phobos2_common STREQUAL "")
            set(phobos2_o "")
            set(phobos2_bc "")
            compile_phobos2("${phobos2_d_flags};-relocation-model=pic;-fvisibility=public;-dllimport=all"
                            "${lib_suffix}${SHARED_LIB_SUFFIX}" "${path_suffix}"
                            "OFF" "${all_d_files_at_once}" "${all_d_files_at_once}" phobos2_o phobos2_bc)
        endif()

        # also link-in special rt.dso druntime module
        list(APPEND phobos2_o ${rt_dso_o})

        build_runtime_libs("${druntime_o}" "${druntime_bc}" "${phobos2_o}" "${phobos2_bc}"
                           "${c_flags}" "${ld_flags}" "${lib_suffix}${SHARED_LIB_SUFFIX}" "${path_suffix}"
                           "ON" "OFF" ${outlist_targets})

        if(NOT phobos2_common STREQUAL "")
            add_dependencies(phobos2-ldc${target_suffix} ${phobos2_common})
        endif()

        if("${lib_suffix}" STREQUAL "") # non-debug & non-unittest
            # copy rt.dso to `ldc_rt.dso.o[bj]` in lib dir
            add_custom_command(
                TARGET druntime-ldc${target_suffix}
                POST_BUILD COMMAND
                ${CMAKE_COMMAND} -E copy ${rt_dso_o} ${CMAKE_BINARY_DIR}/lib${path_suffix}/ldc_rt.dso${CMAKE_C_OUTPUT_EXTENSION}
            )
            # and install it too
            install(
                FILES ${rt_dso_o}
                DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${path_suffix}
                RENAME ldc_rt.dso${CMAKE_C_OUTPUT_EXTENSION}
            )
        endif()
    endif()
endmacro()

# Builds static and/or shared pairs of debug+release copies of druntime/Phobos.
macro(build_runtime_variants d_flags c_flags ld_flags path_suffix outlist_targets)
    # static and/or shared release druntime/Phobos
    build_runtime_variant(
        "${d_flags};${D_FLAGS};${D_FLAGS_RELEASE}"
        "${c_flags}"
        "${ld_flags}"
        ""
        "${path_suffix}"
        "${BUILD_LTO_LIBS}"
        "${COMPILE_ALL_D_FILES_AT_ONCE}"
        ${outlist_targets}
    )
    # static and/or shared debug druntime/Phobos
    build_runtime_variant(
        "${d_flags};${D_FLAGS};${D_FLAGS_DEBUG}"
        "${c_flags}"
        "${ld_flags}"
        "-debug"
        "${path_suffix}"
        "OFF" # no additional bitcode/LTO objects
        "${COMPILE_ALL_D_FILES_AT_ONCE}"
        ${outlist_targets}
    )
endmacro()

# Setup the build of jit runtime
include(jit-rt/DefineBuildJitRT.cmake)

#
# Set up build and install targets
#

set(libs_to_install)
# build host versions
build_runtime_variants("" "${RT_CFLAGS}" "${LD_FLAGS}" "${LIB_SUFFIX}" libs_to_install)
if(MULTILIB)
    # build multi versions
    build_runtime_variants("-m${MULTILIB_SUFFIX}" "-m${MULTILIB_SUFFIX} ${RT_CFLAGS}" "-m${MULTILIB_SUFFIX} ${LD_FLAGS}" "${MULTILIB_SUFFIX}" libs_to_install)
endif()

# Only build the host version of the jit runtime due to LLVM dependency.
build_jit_runtime("${D_FLAGS};${D_FLAGS_RELEASE}" "${RT_CFLAGS}" "${LD_FLAGS}" "${LIB_SUFFIX}" libs_to_install)

# Add the (static- and release-only) bitcode libraries.
if(BUILD_LTO_LIBS AND (NOT ${BUILD_SHARED_LIBS} STREQUAL "ON"))
    list(APPEND libs_to_install druntime-ldc-lto phobos2-ldc-lto)
    if(MULTILIB)
        list(APPEND libs_to_install druntime-ldc-lto_${MULTILIB_SUFFIX} phobos2-ldc-lto_${MULTILIB_SUFFIX})
    endif()
endif()

foreach(libname ${libs_to_install})
    set(target_type)
    get_target_property(target_type ${libname} TYPE)

    set(destination_dir ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
    if(${libname} MATCHES "_${MULTILIB_SUFFIX}$")
        set(destination_dir ${CMAKE_INSTALL_PREFIX}/lib${MULTILIB_SUFFIX})
    endif()

    # On Mac, manually copy static libraries to prevent ranlib from
    # potentially corrupting the file during installation.
    # See https://bugs.llvm.org/show_bug.cgi?id=34808.
    if("${TARGET_SYSTEM}" MATCHES "APPLE" AND "${target_type}" STREQUAL "STATIC_LIBRARY")
        install(FILES       $<TARGET_FILE:${libname}>
                DESTINATION ${destination_dir})
    # Windows DLLs:
    elseif("${TARGET_SYSTEM}" MATCHES "MSVC" AND "${target_type}" STREQUAL "SHARED_LIBRARY")
        # import .lib in regular lib dir
        install(FILES       $<TARGET_LINKER_FILE:${libname}>
                DESTINATION ${destination_dir})
        # .dll in bin dir
        install(FILES       $<TARGET_FILE:${libname}>
                DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
        # .pdb of debug DLLs in bin dir
        if(${libname} MATCHES "-debug")
            install(FILES       $<TARGET_PDB_FILE:${libname}>
                    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
        endif()
    else()
        install(TARGETS     ${libname}
                DESTINATION ${destination_dir})
    endif()
endforeach()

set(DRUNTIME_PACKAGES core etc ldc)

install(FILES ${RUNTIME_DIR}/src/object.d ${RUNTIME_DIR}/src/__builtins.di ${RUNTIME_DIR}/src/importc.h DESTINATION ${INCLUDE_INSTALL_DIR})
foreach(p ${DRUNTIME_PACKAGES})
    install(DIRECTORY ${RUNTIME_DIR}/src/${p} DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.d")
    install(DIRECTORY ${RUNTIME_DIR}/src/${p} DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.di")
endforeach()
if(PHOBOS2_DIR)
    install(DIRECTORY ${PHOBOS2_DIR}/std DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.d")
    install(DIRECTORY ${PHOBOS2_DIR}/etc DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.d")
endif()
install(FILES ${GCCBUILTINS} DESTINATION ${INCLUDE_INSTALL_DIR}/ldc)


#
# Test targets.
#

# Build the "test runner" executables containing the druntime and Phobos unit
# tests. They are invoked with the modules to test later.
# We just build another copy of the two libraries with -unittest enabled and
# link the test runners against those. Some linker command-line magic is
# required to make sure all objects are pulled in.

macro(set_testrunner_linkflags libname path_suffix is_shared output_flags)
    # (Re)set variable 'tested_lib_path', initialize to directory containing the tested lib.
    # It'll contain the full path to the linked-in library when 'returning' to calling build_test_runners().
    set(tested_lib_path "${CMAKE_BINARY_DIR}/lib${path_suffix}")

    if("${TARGET_SYSTEM}" MATCHES "MSVC")
        set(tested_lib_path "${tested_lib_path}/${libname}.lib")
        if("${is_shared}" STREQUAL "ON")
            set(${output_flags} "${tested_lib_path}")
        else()
            # the MS linker supports /WHOLEARCHIVE since VS 2015 Update 2
            # we need to use `-WHOLEARCHIVE` (leading dash) to make CMake recognize it as a linker *flag*
            set(${output_flags} "-WHOLEARCHIVE:${tested_lib_path}")
        endif()
    else()
        if("${is_shared}" STREQUAL "ON")
            set(${output_flags} "-Wl,-rpath,${tested_lib_path}")
            if("${TARGET_SYSTEM}" MATCHES "APPLE")
                set(tested_lib_path "${tested_lib_path}/lib${libname}.dylib")
                append("${tested_lib_path}" ${output_flags})
            else()
                set(tested_lib_path "${tested_lib_path}/lib${libname}.so")
                append("-Wl,--no-as-needed,${tested_lib_path},--as-needed" ${output_flags})
            endif()
        else()
            set(tested_lib_path "${tested_lib_path}/lib${libname}.a")
            if("${TARGET_SYSTEM}" MATCHES "APPLE")
                set(${output_flags} "-Wl,-force_load,${tested_lib_path}")
            else()
                set(${output_flags} "-Wl,--whole-archive,${tested_lib_path},--no-whole-archive")
            endif()
        endif()
    endif()
endmacro()

# Sets `test_runner_o`.
macro(compile_testrunner d_flags is_shared target_suffix extra_dir_suffix)
    set(test_runner_o  "")
    set(test_runner_bc "")
    dc("${RUNTIME_DIR}/src/test_runner.d"
       "${RUNTIME_DIR}/src"
       "${d_flags}"
       "${PROJECT_BINARY_DIR}/objects-unittest${target_suffix}${extra_dir_suffix}"
       "OFF" # emit_bc
       "ON"  # all_at_once
       "test_runner" # single_obj_name
       test_runner_o
       test_runner_bc
    )

    if("${is_shared}" STREQUAL "ON")
        # also link-in special rt.dso druntime module
        list(APPEND test_runner_o ${PROJECT_BINARY_DIR}/objects-unittest${target_suffix}/rt/dso${CMAKE_C_OUTPUT_EXTENSION})
    endif()
endmacro()

set(_GLOBAL_TESTRUNNERS "" CACHE INTERNAL "List of all test runner build targets")

# Generates targets for a pair of druntime/Phobos test runners.
# The build targets are also appended to the _GLOBAL_TESTRUNNERS list.
function(build_test_runners name_suffix path_suffix d_flags linkflags is_shared)
    set(target_suffix)
    get_target_suffix("${name_suffix}" "${path_suffix}" target_suffix)

    set(c_libs "")
    set(full_dflags ${d_flags} -linkonce-templates)
    if("${is_shared}" STREQUAL "OFF")
        set(c_libs ${C_SYSTEM_LIBS})
    elseif("${TARGET_SYSTEM}" MATCHES "MSVC")
        set(c_libs ${C_SYSTEM_LIBS_SHARED})
        set(full_dflags ${full_dflags} -link-defaultlib-shared) # to dllimport data
    endif()

    compile_testrunner("${full_dflags}" "${is_shared}" "${target_suffix}" "")
    add_custom_target(test_runner${target_suffix} DEPENDS ${test_runner_o})

    set(druntime_name druntime-test-runner${target_suffix})
    add_executable(${druntime_name} EXCLUDE_FROM_ALL ${test_runner_o})
    add_dependencies(${druntime_name} test_runner${target_suffix} druntime-ldc-unittest${target_suffix})
    set_testrunner_linkflags("druntime-ldc-unittest${name_suffix}" "${path_suffix}" "${is_shared}" runner_linkflags)
    target_link_libraries(${druntime_name} ${runner_linkflags} ${c_libs})
    set_target_properties(${druntime_name} PROPERTIES
        LINKER_LANGUAGE C
        LINK_FLAGS      ${linkflags}
        LINK_DEPENDS    ${tested_lib_path}
    )
    add_test(build-${druntime_name} "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ${druntime_name})
    set(_GLOBAL_TESTRUNNERS "${_GLOBAL_TESTRUNNERS};${druntime_name}" CACHE INTERNAL "")

    if(PHOBOS2_DIR)
        # Windows: the shared Phobos test runners need to be recompiled to reference a Phobos symbol,
        #          so that the Phobos unittest DLLs are actually loaded
        set(phobos_test_runner_target test_runner${target_suffix})
        if("${is_shared}" STREQUAL "ON" AND "${TARGET_SYSTEM}" MATCHES "Windows")
            compile_testrunner("${full_dflags};-d-version=SharedPhobos" "${is_shared}" "${target_suffix}" "-phobos")
            set(phobos_test_runner_target test_runner_phobos_${target_suffix})
            add_custom_target(${phobos_test_runner_target} DEPENDS ${test_runner_o})
        endif()

        set(phobos_name phobos2-test-runner${target_suffix})
        add_executable(${phobos_name} EXCLUDE_FROM_ALL ${test_runner_o})
        add_dependencies(${phobos_name} ${phobos_test_runner_target} phobos2-ldc-unittest${target_suffix})
        set_testrunner_linkflags("phobos2-ldc-unittest${name_suffix}" "${path_suffix}" "${is_shared}" runner_linkflags)
        target_link_libraries(${phobos_name} ${runner_linkflags} druntime-ldc${target_suffix} ${c_libs})
        set_target_properties(${phobos_name} PROPERTIES
            LINKER_LANGUAGE C
            LINK_FLAGS      ${linkflags}
            LINK_DEPENDS    ${tested_lib_path}
        )
        add_test(build-${phobos_name} "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ${phobos_name})
        set(_GLOBAL_TESTRUNNERS "${_GLOBAL_TESTRUNNERS};${phobos_name}" CACHE INTERNAL "")
    endif()
endfunction()

# Generates targets for static and/or shared pairs of druntime/Phobos test runners.
function(build_test_runner_variant name_suffix path_suffix d_flags c_flags)
    set(full_d_flags ${D_FLAGS} ${d_flags})
    set(full_c_flags "${RT_CFLAGS} ${c_flags}")
    set(linkflags "${LD_FLAGS} ${c_flags}")

    # static and/or shared pair(s) of druntime/Phobos unittest libs
    set(unittest_libs "")
    build_runtime_variant(
        "${full_d_flags}"
        "${full_c_flags}"
        "${linkflags}"
        "-unittest${name_suffix}"
        "${path_suffix}"
        "OFF" # no additional bitcode/LTO objects
        "OFF" # compile separately
        unittest_libs
    )

    # Only build the unittest libraries when running the tests.
    set_target_properties(${unittest_libs} PROPERTIES EXCLUDE_FROM_ALL ON EXCLUDE_FROM_DEFAULT_BUILD ON)

    # static druntime/Phobos test runners
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "ON")
        build_test_runners("${name_suffix}" "${path_suffix}" "${full_d_flags}" "${linkflags}" "OFF")
    endif()
    # shared druntime/Phobos test runners
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
        build_test_runners("${name_suffix}${SHARED_LIB_SUFFIX}" "${path_suffix}" "${full_d_flags}" "${linkflags}" "ON")
    endif()
endfunction()

# Generates targets for static and/or shared debug+release druntime/Phobos test runners.
function(build_test_runner_variants path_suffix d_flags c_flags)
    build_test_runner_variant("" "${path_suffix}" "${D_FLAGS_RELEASE};${d_flags}" "${c_flags}")
    build_test_runner_variant("-debug" "${path_suffix}" "${D_FLAGS_DEBUG};${d_flags}" "${c_flags}")
endfunction()

set(TESTLIB_SUFFIX "${LIB_SUFFIX}")
if(MULTILIB AND "${TARGET_SYSTEM}" MATCHES "APPLE")
    # match `hostsuffix` used for building the normal libs
    set(TESTLIB_SUFFIX "${LIB_SUFFIX}${HOST_BITNESS}")
endif()

# Now generate all test runner targets.
build_test_runner_variants("${TESTLIB_SUFFIX}" "" "")
if(MULTILIB AND ${HOST_BITNESS} EQUAL 64)
    build_test_runner_variants("${MULTILIB_SUFFIX}" "-m32" "-m32")
endif()

# Let's add a meta build target for all test runners.
add_custom_target(all-test-runners DEPENDS ${_GLOBAL_TESTRUNNERS})

# Add the druntime/Phobos test runner invocations for all the different modules.

macro(file_to_module_name file_name out_module_name)
    string(REPLACE ${PROJECT_SOURCE_DIR}/ "" stripped ${file_name})
    string(REPLACE "druntime/src/" "" stripped ${stripped})
    string(REPLACE "phobos/" "" stripped ${stripped})
    string(REPLACE ".d" "" stripped ${stripped})
    string(REPLACE "/" "." module ${stripped})

    # The logical module name for package.d files doesn't include the
    # trailing .package part.
    string(REPLACE ".package" "" module ${module})

    # rt.invariant doesn't have a module declaration, presumably because
    # invariant is a keyword.
    string(REPLACE "rt.invariant" "invariant" ${out_module_name} ${module})
endmacro()

function(add_tests d_files runner target_suffix)
    foreach(file ${d_files})
        # skip rt.dso (no ModuleInfo)
        if(NOT "${file}" MATCHES "druntime/src/rt/dso.d")
            file_to_module_name(${file} module)
            add_test(NAME "${module}${target_suffix}"
                WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
                COMMAND ${runner}-test-runner${target_suffix} ${module}
            )
            set_tests_properties("${module}${target_suffix}" PROPERTIES
                DEPENDS build-${runner}-test-runner${target_suffix}
            )
        endif()
    endforeach()
endfunction()
function(add_runtime_tests name_suffix path_suffix)
    get_target_suffix("${name_suffix}" "${path_suffix}" target_suffix)
    add_tests("${DRUNTIME_D}" "druntime" "${target_suffix}")
    if(PHOBOS2_DIR)
        add_tests("${PHOBOS2_D}" "phobos2" "${target_suffix}")
    endif()
endfunction()

function(add_runtime_tests_variants path_suffix)
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "ON")
        add_runtime_tests("" "${path_suffix}")
        add_runtime_tests("-debug" "${path_suffix}")
    endif()
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
        add_runtime_tests("${SHARED_LIB_SUFFIX}" "${path_suffix}")
        add_runtime_tests("-debug${SHARED_LIB_SUFFIX}" "${path_suffix}")
    endif()
endfunction()

add_runtime_tests_variants("${TESTLIB_SUFFIX}")
if(MULTILIB AND ${HOST_BITNESS} EQUAL 64)
    add_runtime_tests_variants("${MULTILIB_SUFFIX}")
endif()

include(DRuntimeIntegrationTests)
