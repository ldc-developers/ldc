project(runtime C)

cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

include(CheckIncludeFile)

# Verify required variables if this CMake project is NOT embedded in the LDC CMake project.
if(NOT LDC_EXE)
    if(NOT LDC_EXE_FULL)
        message(FATAL_ERROR "Please define the path to the LDC executable via -DLDC_EXE_FULL=...")
    endif()
    if(NOT DEFINED D_VERSION OR NOT DEFINED DMDFE_MINOR_VERSION OR NOT DEFINED DMDFE_PATCH_VERSION)
        message(FATAL_ERROR "Please define the CMake variables D_VERSION, DMDFE_MINOR_VERSION and DMDFE_PATCH_VERSION.")
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

set(DMDFE_VERSION         ${D_VERSION}.${DMDFE_MINOR_VERSION}.${DMDFE_PATCH_VERSION})

set(MULTILIB              OFF                                 CACHE BOOL   "Build both 32/64 bit runtime libraries")
set(BUILD_LTO_LIBS        OFF                                 CACHE BOOL   "Also build the runtime as LLVM bitcode libraries for LTO")
set(INCLUDE_INSTALL_DIR   ${CMAKE_INSTALL_PREFIX}/include/d   CACHE PATH   "Path to install D modules to")
set(BUILD_SHARED_LIBS     AUTO                                CACHE STRING "Whether to build the runtime as a shared library (ON|OFF|BOTH)")
set(D_FLAGS               -w;-de;-dip1000                     CACHE STRING "Runtime D compiler flags, separated by ';'")
set(D_FLAGS_DEBUG         -g;-link-defaultlib-debug           CACHE STRING "Runtime D compiler flags (debug libraries), separated by ';'")
set(D_FLAGS_RELEASE       -O3;-release                        CACHE STRING "Runtime D compiler flags (release libraries), separated by ';'")
set(COMPILE_ALL_D_FILES_AT_ONCE ON                            CACHE BOOL   "Compile all D files for the runtime libs in a single command line instead of separately. Disabling this is useful for many CPU cores and/or iterative development.")
set(RT_ARCHIVE_WITH_LDC   ON                                  CACHE STRING "Whether to archive the static runtime libs via LDC instead of CMake archiver")
set(RT_CFLAGS             ""                                  CACHE STRING "Runtime extra C compiler flags, separated by ' '")
set(LD_FLAGS              ""                                  CACHE STRING "Runtime extra C linker flags, separated by ' '")
set(C_SYSTEM_LIBS         AUTO                                CACHE STRING "C system libraries for linking shared libraries and test runners, separated by ';'")
set(TARGET_SYSTEM         AUTO                                CACHE STRING "Target OS/toolchain for cross-compilation (e.g., 'Linux;UNIX', 'Darwin;APPLE;UNIX', 'Windows;MSVC')")
set(LDC_TARGET_PRESET     ""                                  CACHE STRING "Use a preset target configuration for cross-compilation, by passing format OS-arch (e.g., 'Linux-arm', 'Mac-x64', 'Windows-aarch64', 'Android-x86')")
set(RT_SUPPORT_SANITIZERS OFF                                 CACHE BOOL   "Build runtime libraries with sanitizer support (e.g. for AddressSanitizer)")

set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})

if (RT_SUPPORT_SANITIZERS)
    message(STATUS "Building runtime libraries with sanitizer support")
    set(D_FLAGS "${D_FLAGS};-d-version=SupportSanitizers")
endif()

# Find gnu make, use specific version first (BSD) and fall back to default 'make' (LINUX)
find_program(GNU_MAKE_BIN NAMES gmake gnumake make)
if ("${GNU_MAKE_BIN}" STREQUAL "")
    message(FATAL_ERROR "GNU make could not be found. Please install gmake/gnumake/make using your (platform) package installer")
endif()

# Use LDC_TARGET_PRESET to configure target runtime
include(PresetRuntimeConfiguration)

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
elseif("${TARGET_SYSTEM}" MATCHES "MSVC")
    ENABLE_LANGUAGE(ASM_MASM)
endif()

include(CheckTypeSize)
check_type_size(void* ptr_size)
if(${ptr_size} MATCHES "^8$") ## if it's 64-bit OS
    set(HOST_BITNESS 64)
    set(MULTILIB_SUFFIX 32)
else()
    set(HOST_BITNESS 32)
    set(MULTILIB_SUFFIX 64)
endif()

set(SHARED_LIBS_SUPPORTED OFF)
if("${TARGET_SYSTEM}" MATCHES "Linux|FreeBSD|DragonFly|APPLE"
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

set(SHARED_LIB_SUFFIX "")
if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
    if(NOT SHARED_LIBS_SUPPORTED)
        message(FATAL_ERROR "Shared libraries (BUILD_SHARED_LIBS) are only supported on Linux, macOS, FreeBSD and DragonFly for the time being.")
    endif()
    set(SHARED_LIB_SUFFIX "-shared")
endif()

# Auto-detect C system libraries
if("${C_SYSTEM_LIBS}" STREQUAL "AUTO")
    if("${TARGET_SYSTEM}" MATCHES "MSVC")
        set(C_SYSTEM_LIBS libvcruntime libcmt)
    elseif("${TARGET_SYSTEM}" MATCHES "Android")
        set(C_SYSTEM_LIBS m c)
    elseif("${TARGET_SYSTEM}" MATCHES "Linux")
        set(C_SYSTEM_LIBS m pthread rt dl)
    else()
        set(C_SYSTEM_LIBS m pthread)
    endif()
endif()

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
# FIXME: Remove all modules in core/stdcpp/ due to alpha quality.
#        See PR #1917; partly fixed upstream.
file(GLOB_RECURSE DRUNTIME_D_STDCPP ${RUNTIME_DIR}/src/core/stdcpp/*.d)
list(REMOVE_ITEM DRUNTIME_D ${DRUNTIME_D_STDCPP})
# remove some modules in rt/
list(REMOVE_ITEM DRUNTIME_D
    ${RUNTIME_DIR}/src/rt/alloca.d
    ${RUNTIME_DIR}/src/rt/llmath.d
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
if(NOT "${TARGET_SYSTEM}" MATCHES "MSVC")
    list(REMOVE_ITEM DRUNTIME_C ${RUNTIME_DIR}/src/rt/msvc.c ${RUNTIME_DIR}/src/rt/msvc_math.c)
endif()

# druntime ASM parts
set(DRUNTIME_ASM)
if("${TARGET_SYSTEM}" MATCHES "UNIX")
    list(APPEND DRUNTIME_ASM ${RUNTIME_DIR}/src/core/threadasm.S ${RUNTIME_DIR}/src/ldc/eh_asm.S)
endif()

if(PHOBOS2_DIR)
    # Phobos D parts
    file(GLOB_RECURSE PHOBOS2_D ${PHOBOS2_DIR}/*.d)
    # remove top-level modules index.d and unittest.d
    list(REMOVE_ITEM PHOBOS2_D ${PHOBOS2_DIR}/index.d ${PHOBOS2_DIR}/unittest.d)
    # remove D files in test/
    file(GLOB_RECURSE PHOBOS2_D_TEST ${PHOBOS2_DIR}/test/*.d)
    list(REMOVE_ITEM PHOBOS2_D ${PHOBOS2_D_TEST})
    # only include std/windows/ modules on Windows
    if(NOT "${TARGET_SYSTEM}" MATCHES "Windows")
        file(GLOB_RECURSE PHOBOS2_D_WINDOWS ${PHOBOS2_DIR}/std/windows/*.d)
        list(REMOVE_ITEM PHOBOS2_D ${PHOBOS2_D_WINDOWS})
    endif()

    # Phobos C parts
    file(GLOB_RECURSE PHOBOS2_C ${PHOBOS2_DIR}/*.c)
    # remove some obsolete zlib modules
    list(REMOVE_ITEM PHOBOS2_C ${PHOBOS2_DIR}/etc/c/zlib/minigzip.c ${PHOBOS2_DIR}/etc/c/zlib/example.c)
endif()

#
# Create configuration files.
#

# Add extra paths on Linux and disable linker arch mismatch warnings (like
# DMD and GDC do). OS X doesn't need extra configuration due to the use of
# fat binaries. Other Posixen might need to be added here.
if(MULTILIB AND NOT "${TARGET_SYSTEM}" MATCHES "APPLE")
    set(ADDITIONAL_DEFAULT_LDC_SWITCHES  "${ADDITIONAL_DEFAULT_LDC_SWITCHES}\n        \"-L--no-warn-search-mismatch\",")
    set(MULTILIB_ADDITIONAL_PATH         "\n        \"${CMAKE_BINARY_DIR}/lib${MULTILIB_SUFFIX}\",")
    set(MULTILIB_ADDITIONAL_INSTALL_PATH "\n        \"${CMAKE_INSTALL_PREFIX}/lib${MULTILIB_SUFFIX}\",")
endif()

# Default -rpath linker option when linking against shared libraries.
if(SHARED_LIBS_SUPPORTED)
    if(MULTILIB AND NOT "${TARGET_SYSTEM}" MATCHES "APPLE")
        set(SHARED_LIBS_RPATH         "${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}:${CMAKE_BINARY_DIR}/lib${MULTILIB_SUFFIX}")
        set(SHARED_LIBS_INSTALL_RPATH "${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/lib${MULTILIB_SUFFIX}")
    else()
        set(SHARED_LIBS_RPATH         "${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}")
        set(SHARED_LIBS_INSTALL_RPATH "${CMAKE_INSTALL_LIBDIR}")
    endif()
endif()

# Only have either shared or static libs?
# Then explicitly default to linking against them via default LDC switch.
if(${BUILD_SHARED_LIBS} STREQUAL "ON")
    set(ADDITIONAL_DEFAULT_LDC_SWITCHES "${ADDITIONAL_DEFAULT_LDC_SWITCHES}\n        \"-link-defaultlib-shared\",")
elseif(${BUILD_SHARED_LIBS} STREQUAL "OFF")
    set(ADDITIONAL_DEFAULT_LDC_SWITCHES "${ADDITIONAL_DEFAULT_LDC_SWITCHES}\n        \"-link-defaultlib-shared=false\",")
endif()

if(LDC_WITH_LLD)
    set(WASM_DEFAULT_LDC_SWITCHES "${WASM_DEFAULT_LDC_SWITCHES}\n        \"-link-internally\",")
endif()
# LLD 8+ requires (new) `--export-dynamic` for WebAssembly (https://github.com/ldc-developers/ldc/issues/3023).
if(NOT (LDC_LLVM_VER LESS 800))
    set(WASM_DEFAULT_LDC_SWITCHES "${WASM_DEFAULT_LDC_SWITCHES}\n        \"-L--export-dynamic\",")
endif()

# Directory filled with auto-generated import files
set(LDC_BUILD_IMPORT_DIR "${PROJECT_BINARY_DIR}/import")

# Only generate the config files if this CMake project is embedded in the LDC CMake project.
if(LDC_EXE)
    if(PHOBOS2_DIR)
        set(CONFIG_NAME ${LDC_EXE}_phobos)
    else()
        set(CONFIG_NAME ${LDC_EXE})
    endif()
    configure_file(${PROJECT_PARENT_DIR}/${CONFIG_NAME}.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}.conf @ONLY)
    configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}_install.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}_install.conf @ONLY)
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
    )
  endfunction()

  set(target_arch "AArch64;ARM;Mips;PowerPC;SystemZ;X86")
  set(target_name "aarch64;arm;mips;ppc;s390;x86")

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
macro(dc src_files src_basedir d_flags output_basedir emit_bc all_at_once outlist_o outlist_bc)
    set(dc_flags -c --output-o ${d_flags})
    if(${emit_bc})
        list(APPEND dc_flags -flto=thin --output-bc)
    endif()

    set(dc_deps ${LDC_EXE}
                ${LDC_EXE_FULL}
                ${GCCBUILTINS}
    )

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
                set(renameCommand mv ${output_basedir}/${output_root}.bc ${output_bc})
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
        set(renameCommand "")
        if(${emit_bc})
            if(WIN32)
                set(renameCommand find ${output_basedir} -iname '*.bc' -exec mv '{}' '{}${CMAKE_C_OUTPUT_EXTENSION}' "\;")
            else()
                set(renameCommand find ${output_basedir} -iname '*.bc' -exec mv '{}' '{}${CMAKE_C_OUTPUT_EXTENSION}' "\\\\;")
            endif()
        endif()

        add_custom_command(
            OUTPUT  ${new_o} ${new_bc}
            COMMAND ${LDC_EXE_FULL} ${dc_flags} -od=${output_basedir} -op ${relative_src_files}
            COMMAND ${renameCommand}
            WORKING_DIRECTORY ${src_basedir}
            DEPENDS ${src_files} ${dc_deps}
        )
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
macro(compile_druntime d_flags lib_suffix path_suffix emit_bc all_at_once outlist_o outlist_bc)
    get_target_suffix("${lib_suffix}" "${path_suffix}" target_suffix)
    dc("${DRUNTIME_D}"
       "${RUNTIME_DIR}/src"
       "-conf=;${d_flags};-I${RUNTIME_DIR}/src"
       "${PROJECT_BINARY_DIR}/objects${target_suffix}"
       "${emit_bc}"
       "${all_at_once}"
       ${outlist_o}
       ${outlist_bc}
    )
endmacro()

# Sets up the targets for building the Phobos D object files, appending the
# names of the (bitcode) files to link into the library to outlist_o (outlist_bc).
macro(compile_phobos2 d_flags lib_suffix path_suffix emit_bc all_at_once outlist_o outlist_bc)
    get_target_suffix("${lib_suffix}" "${path_suffix}" target_suffix)

    # Special case for Windows and emit_bc=ON:
    # LLVM before v9.0 doesn't support naked DMD-style asm when generating bitcode.
    # For 1 (Win64) / 3 (Win32) remaining modules, compile separately without generating
    # the bitcode file (and adding the binary object file to outlist_bc instead).
    set(phobos_src_files ${PHOBOS2_D})
    if(${emit_bc} AND (LDC_LLVM_VER LESS 900 AND "${TARGET_SYSTEM}" MATCHES "Windows"))
        set(naked_asm_src_files "${PHOBOS2_DIR}/std/internal/digest/sha_SSSE3.d")
        if(CMAKE_SIZEOF_VOID_P EQUAL 4)
            list(APPEND naked_asm_src_files
                "${PHOBOS2_DIR}/std/internal/math/biguintx86.d"
                "${PHOBOS2_DIR}/std/internal/math/biguintcore.d"
            )
        endif()

        list(REMOVE_ITEM phobos_src_files ${naked_asm_src_files})

        set(naked_asm_o "")
        dc("${naked_asm_src_files}"
           "${PHOBOS2_DIR}"
           "-conf=;${d_flags};-I${RUNTIME_DIR}/src;-I${PHOBOS2_DIR}"
           "${PROJECT_BINARY_DIR}/objects${target_suffix}"
           "OFF"
           "${all_at_once}"
           naked_asm_o
           naked_asm_o
        )

        list(APPEND ${outlist_o} ${naked_asm_o})
        list(APPEND ${outlist_bc} ${naked_asm_o})
    endif()

    dc("${phobos_src_files}"
       "${PHOBOS2_DIR}"
       "-conf=;${d_flags};-I${RUNTIME_DIR}/src;-I${PHOBOS2_DIR}"
       "${PROJECT_BINARY_DIR}/objects${target_suffix}"
       "${emit_bc}"
       "${all_at_once}"
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
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME                 ${name}
        ARCHIVE_OUTPUT_DIRECTORY    ${output_dir}
        LIBRARY_OUTPUT_DIRECTORY    ${output_dir}
        RUNTIME_OUTPUT_DIRECTORY    ${output_dir}
        COMPILE_DEFINITIONS         "LDC=1"
        COMPILE_FLAGS               "${c_flags}"
        LINK_FLAGS                  "${ld_flags}"
        VERSION                     ${DMDFE_VERSION}
        SOVERSION                   ${DMDFE_PATCH_VERSION}
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
        target_link_libraries(druntime-ldc${target_suffix} ${C_SYSTEM_LIBS})
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
                druntime-ldc${target_suffix_without_unittest} ${C_SYSTEM_LIBS})
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
macro(build_runtime_variant d_flags c_flags ld_flags lib_suffix path_suffix all_d_files_at_once outlist_targets)
    set(emit_bc OFF)
    if(BUILD_LTO_LIBS AND NOT "${lib_suffix}" MATCHES "-unittest")
        set(emit_bc ON)
    endif()

    # Only compile Phobos modules once for static+shared libs.
    # If a shared Phobos lib is generated (too), compile explicitly with `-relocation-model=pic`.
    set(phobos2_d_flags "${d_flags}")
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
        list(APPEND phobos2_d_flags -relocation-model=pic)
    endif()

    set(phobos2_o "")
    set(phobos2_bc "")
    compile_phobos2("${phobos2_d_flags}" "${lib_suffix}" "${path_suffix}" "${emit_bc}" "${all_d_files_at_once}" phobos2_o phobos2_bc)

    # Use a dummy custom target ('phobos2-ldc…-common') depending solely on the Phobos D objects
    # (custom commands) as dependency for static+shared Phobos libs.
    # Otherwise building both libs in parallel may result in conflicting Phobos module compilations
    # (at least with the make generator), a known CMake issue.
    set(phobos2_common phobos2-ldc${target_suffix}-common)
    add_custom_target(${phobos2_common} DEPENDS ${phobos2_o} ${phobos2_bc})

    # static druntime/Phobos
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "ON")
        set(druntime_o "")
        set(druntime_bc "")
        compile_druntime("${d_flags}" "${lib_suffix}"
                         "${path_suffix}" "${emit_bc}" "${all_d_files_at_once}" druntime_o druntime_bc)

        build_runtime_libs("${druntime_o}" "${druntime_bc}" "${phobos2_o}" "${phobos2_bc}"
                           "${c_flags}" "${ld_flags}" "${lib_suffix}" "${path_suffix}"
                           "OFF" "${emit_bc}" ${outlist_targets})
        add_dependencies(phobos2-ldc${target_suffix} ${phobos2_common})
    endif()
    # shared druntime/Phobos
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
        set(druntime_o "")
        set(druntime_bc "")
        compile_druntime("${d_flags};-relocation-model=pic;-d-version=Shared" "${lib_suffix}${SHARED_LIB_SUFFIX}"
                         "${path_suffix}" "OFF" "${all_d_files_at_once}" druntime_o druntime_bc)

        build_runtime_libs("${druntime_o}" "${druntime_bc}" "${phobos2_o}" "${phobos2_bc}"
                           "${c_flags}" "${ld_flags}" "${lib_suffix}${SHARED_LIB_SUFFIX}" "${path_suffix}"
                           "ON" "OFF" ${outlist_targets})
        add_dependencies(phobos2-ldc${target_suffix} ${phobos2_common})
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
        "${COMPILE_ALL_D_FILES_AT_ONCE}"
        ${outlist_targets}
    )
endmacro()

# Setup the build of jit runtime
include(jit-rt/DefineBuildJitRT.cmake)

#
# Set up build and install targets
#

if(MULTILIB AND "${TARGET_SYSTEM}" MATCHES "APPLE")
    # On OS X, build "fat" libraries containing code for both architectures.

    # Some suffix for the target/file names of the host-native arch so
    # that they don't collide with the final combined ones.
    set(hostsuffix "${LIB_SUFFIX}${HOST_BITNESS}")

    set(libtargets)
    build_runtime_variants("" "${RT_CFLAGS}" "${LD_FLAGS}" "${hostsuffix}" libtargets)
    build_runtime_variants("-m${MULTILIB_SUFFIX}" "-m${MULTILIB_SUFFIX} ${RT_CFLAGS}" "-m${MULTILIB_SUFFIX} ${LD_FLAGS}" "${MULTILIB_SUFFIX}" libtargets)

    # Only build the host version of the jit runtime due to LLVM dependency.
    set(libtargets_jit)
    build_jit_runtime("${D_FLAGS};${D_FLAGS_RELEASE}" "${RT_CFLAGS}" "${LD_FLAGS}" "${LIB_SUFFIX}" libtargets_jit)
    # Install separately; also manually copy libldc-jit-rt.a to prevent ranlib from
    # potentially corrupting the file during installation.
    if(LDC_DYNAMIC_COMPILE)
        install(FILES       $<TARGET_FILE:ldc-jit-rt>
                DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
        list(REMOVE_ITEM libtargets_jit ldc-jit-rt)
    endif()
    install(TARGETS     ${libtargets_jit}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})

    # KLUDGE: Cannot use `$<TARGET_LINKER_FILE:target>` in custom command.
    # Set up the list of generated libs (without 'lib' prefix) to be merged manually.
    set(libs_to_merge)
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "ON")
        list(APPEND libs_to_merge druntime-ldc.a druntime-ldc-debug.a
                                  phobos2-ldc.a  phobos2-ldc-debug.a
        )
        # Only install the release versions of the (static-only) bitcode libraries.
        if(BUILD_LTO_LIBS)
            list(APPEND libs_to_merge druntime-ldc-lto.a phobos2-ldc-lto.a)
        endif()
    endif()
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
        set(suffix ${SHARED_LIB_SUFFIX}.dylib)
        list(APPEND libs_to_merge druntime-ldc${suffix} druntime-ldc-debug${suffix}
                                  phobos2-ldc${suffix}  phobos2-ldc-debug${suffix}
        )
    endif()

    foreach(lib ${libs_to_merge})
        set(target_name "")
        set(lib_extension "")
        get_filename_component(target_name ${lib} NAME_WE)
        get_filename_component(lib_extension ${lib} EXT)

        set(host_path  ${CMAKE_BINARY_DIR}/lib${hostsuffix}/lib${lib})
        set(multi_path ${CMAKE_BINARY_DIR}/lib${MULTILIB_SUFFIX}/lib${lib})

        # Create (& later install) symbolic links for merged dylibs.
        set(final_files_to_install)
        if(${lib_extension} STREQUAL ".dylib")
            set(final_path     ${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/lib${target_name}.${DMDFE_VERSION}.dylib)
            set(dylib_symlink1 ${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/lib${target_name}.${DMDFE_PATCH_VERSION}.dylib)
            set(dylib_symlink2 ${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/lib${target_name}.dylib)
            set(final_files_to_install ${final_path} ${dylib_symlink1} ${dylib_symlink2})
            add_custom_command(
                OUTPUT  ${final_files_to_install}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}
                COMMAND lipo ${multi_path} ${host_path} -create -output ${final_path}
                COMMAND ln -s lib${target_name}.${DMDFE_VERSION}.dylib ${dylib_symlink1}
                COMMAND ln -s lib${target_name}.${DMDFE_PATCH_VERSION}.dylib ${dylib_symlink2}
                DEPENDS ${libtargets}
            )
        else()
            set(final_path ${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/lib${lib})
            set(final_files_to_install ${final_path})
            add_custom_command(
                OUTPUT  ${final_path}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}
                COMMAND lipo ${multi_path} ${host_path} -create -output ${final_path}
                DEPENDS ${libtargets}
            )
        endif()

        add_custom_target(${target_name} ALL DEPENDS ${final_files_to_install})

        install(FILES       ${final_files_to_install}
                DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
    endforeach()
else()
    set(libs_to_install)
    # build host versions
    build_runtime_variants("" "${RT_CFLAGS}" "${LD_FLAGS}" "${LIB_SUFFIX}" libs_to_install)
    if(MULTILIB)
        # build multi versions
        build_runtime_variants("-m${MULTILIB_SUFFIX}" "-m${MULTILIB_SUFFIX} ${RT_CFLAGS}" "-m${MULTILIB_SUFFIX} ${LD_FLAGS}" "${MULTILIB_SUFFIX}" libs_to_install)
    endif()

    # Only build the host version of the jit runtime due to LLVM dependency.
    build_jit_runtime("${D_FLAGS};${D_FLAGS_RELEASE}" "${RT_CFLAGS}" "${LD_FLAGS}" "${LIB_SUFFIX}" libs_to_install)

    # Only install the release versions of the (static-only) bitcode libraries.
    if(BUILD_LTO_LIBS AND (NOT ${BUILD_SHARED_LIBS} STREQUAL "ON"))
        list(APPEND libs_to_install druntime-ldc-lto phobos2-ldc-lto)
    endif()

    foreach(libname ${libs_to_install})
        # On Mac, manually copy static libraries to prevent ranlib from
        # potentially corrupting the file during installation.
        # See https://bugs.llvm.org/show_bug.cgi?id=34808.
        set(copy_manually OFF)
        if("${TARGET_SYSTEM}" MATCHES "APPLE")
            set(target_type)
            get_target_property(target_type ${libname} TYPE)
            if("${target_type}" STREQUAL "STATIC_LIBRARY")
                set(copy_manually ON)
            endif()
        endif()

        set(destination_dir ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
        if(${libname} MATCHES "_${MULTILIB_SUFFIX}$")
            set(destination_dir ${CMAKE_INSTALL_PREFIX}/lib${MULTILIB_SUFFIX})
        endif()

        if(copy_manually)
            install(FILES       $<TARGET_FILE:${libname}>
                    DESTINATION ${destination_dir})
        else()
            install(TARGETS     ${libname}
                    DESTINATION ${destination_dir})
        endif()
    endforeach()
endif()

set(DRUNTIME_PACKAGES core etc ldc)

install(FILES ${RUNTIME_DIR}/src/object.d DESTINATION ${INCLUDE_INSTALL_DIR})
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

set(_GLOBAL_TESTRUNNERS "" CACHE INTERNAL "List of all test runner build targets")

# Generates targets for a pair of druntime/Phobos test runners.
# The build targets are also appended to the _GLOBAL_TESTRUNNERS list.
function(build_test_runners name_suffix path_suffix d_flags linkflags is_shared)
    set(target_suffix)
    get_target_suffix("${name_suffix}" "${path_suffix}" target_suffix)

    set(test_runner_o  "")
    set(test_runner_bc "")
    dc("${RUNTIME_DIR}/src/test_runner.d"
       "${RUNTIME_DIR}/src"
       "${d_flags}"
       "${PROJECT_BINARY_DIR}/objects-unittest${target_suffix}"
       "OFF"
       "OFF"
       test_runner_o
       test_runner_bc
    )
    add_custom_target(test_runner${target_suffix} DEPENDS ${test_runner_o})

    set(c_libs "")
    if("${is_shared}" STREQUAL "OFF")
        set(c_libs ${C_SYSTEM_LIBS})
    endif()

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
        set(phobos_name phobos2-test-runner${target_suffix})
        add_executable(${phobos_name} EXCLUDE_FROM_ALL ${test_runner_o})
        add_dependencies(${phobos_name} test_runner${target_suffix} phobos2-ldc-unittest${target_suffix})
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
    set(full_d_flags ${D_FLAGS} ${d_flags} -unittest -d-version=StdUnittest)
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
        file_to_module_name(${file} module)
        add_test(NAME "${module}${target_suffix}"
            WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
            COMMAND ${runner}-test-runner${target_suffix} ${module}
        )
        set_tests_properties("${module}${target_suffix}" PROPERTIES
            DEPENDS build-${runner}-test-runner${target_suffix}
        )
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

# Add the standalone druntime tests.
include(StandaloneDRuntimeTests)
