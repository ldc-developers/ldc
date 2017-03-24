cmake_minimum_required(VERSION 2.8)
project(ldc)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

include(FindDCompiler)
include(CheckIncludeFile)
include(CheckLibraryExists)
include(CheckCXXCompilerFlag)
include(CheckDSourceCompiles)

# The script currently only supports the DMD-style commandline interface
if (NOT D_COMPILER_DMD_COMPAT)
    message(FATAL_ERROR "We currently only support building using a D compiler with a DMD-compatible commandline interface. (try 'ldmd2' or 'gdmd')")
endif()

#
# Locate LLVM.
#

find_package(LLVM 3.5 REQUIRED
    all-targets analysis asmparser asmprinter bitreader bitwriter codegen core debuginfocodeview debuginfodwarf debuginfomsf debuginfopdb globalisel instcombine ipa ipo instrumentation irreader linker lto mc mcdisassembler mcparser objcarcopts object option profiledata scalaropts selectiondag support tablegen target transformutils vectorize ${EXTRA_LLVM_MODULES})
math(EXPR LDC_LLVM_VER ${LLVM_VERSION_MAJOR}*100+${LLVM_VERSION_MINOR})
# Remove LLVMTableGen library from list of libraries
string(REGEX MATCH "^-.*LLVMTableGen[^;]*;|;-.*LLVMTableGen[^;]*" LLVM_TABLEGEN_LIBRARY "${LLVM_LIBRARIES}")
string(REGEX REPLACE "^-.*LLVMTableGen[^;]*;|;-.*LLVMTableGen[^;]*" "" LLVM_LIBRARIES "${LLVM_LIBRARIES}")

foreach(LLVM_SUPPORTED_TARGET ${LLVM_TARGETS_TO_BUILD})
    add_definitions("-DLDC_LLVM_SUPPORTED_TARGET_${LLVM_SUPPORTED_TARGET}")
endforeach()

#
# Get info about used Linux distribution.
#
include(GetLinuxDistribution)

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

#
# Main configuration.
#

# Version information
set(LDC_VERSION "1.3.0") # May be overridden by git hash tag
set(DMDFE_MAJOR_VERSION   2)
set(DMDFE_MINOR_VERSION   0)
set(DMDFE_PATCH_VERSION   73)
set(DMDFE_FIX_LEVEL       2) # Comment out if not used

set(DMD_VERSION ${DMDFE_MAJOR_VERSION}.${DMDFE_MINOR_VERSION}${DMDFE_PATCH_VERSION})
if(DEFINED DMDFE_FIX_LEVEL)
    set(DMD_VERSION ${DMD_VERSION}.${DMDFE_FIX_LEVEL})
endif()

# Generally, we want to install everything into CMAKE_INSTALL_PREFIX, but when
# it is /usr, put the config files into /etc to meet common practice.
if(NOT DEFINED SYSCONF_INSTALL_DIR)
    if(CMAKE_INSTALL_PREFIX STREQUAL "/usr")
        set(SYSCONF_INSTALL_DIR "/etc")
    else()
        set(SYSCONF_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/etc")
    endif()
endif()

set(D_VERSION ${DMDFE_MAJOR_VERSION} CACHE STRING "D language version")
set(PROGRAM_PREFIX "" CACHE STRING "Prepended to ldc/ldmd binary names")
set(PROGRAM_SUFFIX "" CACHE STRING "Appended to ldc/ldmd binary names")
set(CONF_INST_DIR ${SYSCONF_INSTALL_DIR} CACHE PATH "Directory ldc.conf is installed to")

# Note: LIB_SUFFIX should perhaps be renamed to LDC_LIBDIR_SUFFIX.
set(LIB_SUFFIX "" CACHE STRING "Appended to the library installation directory. Set to '64' to install libraries into ${PREFIX}/lib64.")

# The following flags are currently not well tested, expect the build to fail.
option(GENERATE_OFFTI "generate complete ClassInfo.offTi arrays")
mark_as_advanced(GENERATE_OFFTI)

if(D_VERSION EQUAL 1)
    message(FATAL_ERROR "D version 1 is no longer supported.
Please consider using D version 2 or checkout the 'd1' git branch for the last version supporting D version 1.")
elseif(D_VERSION EQUAL 2)
    set(DDMDFE_PATH ddmd)
    set(LDC_EXE ldc2)
    set(LDMD_EXE ldmd2)
    set(LDCPROFDATA_EXE ldc-profdata)
    set(RUNTIME druntime)
    append("-DDMDV2" CMAKE_CXX_FLAGS)
else()
    message(FATAL_ERROR "unsupported D version")
endif()

set(LDC_EXE_NAME ${PROGRAM_PREFIX}${LDC_EXE}${PROGRAM_SUFFIX})
set(LDMD_EXE_NAME ${PROGRAM_PREFIX}${LDMD_EXE}${PROGRAM_SUFFIX})
set(LDCPROFDATA_EXE_NAME ${PROGRAM_PREFIX}${LDCPROFDATA_EXE}${PROGRAM_SUFFIX})

file(MAKE_DIRECTORY
    ${PROJECT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/${DDMDFE_PATH}
)

if(UNIX)
    ENABLE_LANGUAGE(ASM)
elseif(MSVC)
    ENABLE_LANGUAGE(ASM_MASM)
endif()

# Setup D compiler flags (DMD syntax, which also works with LDMD).
set(DDMD_DFLAGS "-wi")
set(DDMD_LFLAGS "")
if(NOT MSVC_IDE)
    # for multi-config builds, these options have to be added later to the custom command
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        append("-g" DDMD_DFLAGS)
        if(${D_COMPILER_ID} STREQUAL "LDMD")
            append("-link-debuglib" DDMD_DFLAGS)
        endif()
    elseif(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
        append("-g -O -inline -release" DDMD_DFLAGS)
    else()
        # Default to a Release build type
        append("-O -inline -release" DDMD_DFLAGS)
    endif()
endif()

if(MSVC)
    if(${D_COMPILER_ID} STREQUAL "DigitalMars")
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            message(STATUS "Let DMD output 64bit object files")
            append("-m64" DDMD_DFLAGS)
        else()
            message(STATUS "Let DMD output 32bit COFF object files")
            append("-m32mscoff" DDMD_DFLAGS)
        endif()

        if(MSVC_VERSION GREATER 1800) # VS 2015+
            append("-Llegacy_stdio_definitions.lib" DDMD_DFLAGS)
        endif()
    endif()

    # LDC and LDMD are both linked against the dynamic MSVC runtime by
    # default (due to default CMAKE_C[XX]_FLAGS_*).
    # Host DMD/LDMD will default to linking against the static MSVC runtime.
    # So disable some default libs based on CMAKE_C_FLAGS_RELEASE.
    if(CMAKE_C_FLAGS_RELEASE MATCHES "(^| )[/-]MD( |$)")
        append("-L/NODEFAULTLIB:libcmt" DDMD_LFLAGS)
        if(MSVC_VERSION GREATER 1800) # VS 2015+
            append("-L/NODEFAULTLIB:libvcruntime" DDMD_LFLAGS)
        endif()
    else()
        append("-L/NODEFAULTLIB:msvcrt" DDMD_LFLAGS)
        if(MSVC_VERSION GREATER 1800) # VS 2015+
            append("-L/NODEFAULTLIB:vcruntime" DDMD_LFLAGS)
        endif()
    endif()
endif()

append("-J${PROJECT_SOURCE_DIR}/${DDMDFE_PATH} -J${PROJECT_SOURCE_DIR}/res" DDMD_DFLAGS) # Needed for importing text files
string(STRIP "${DDMD_DFLAGS}" DDMD_DFLAGS)

# Use separate compiler flags for the frontend and for the LDC-specific parts,
# as enabling warnings on the DMD frontend only leads to a lot of clutter in
# the output (LLVM_CXXFLAGS sometimes already includes -Wall).
set(LDC_CXXFLAGS)
if(CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang"))
    # Disable some noisy warnings:
    #  * -Wunused-parameter triggers for LLVM headers
    #  * -Wmissing-field-initializer leads to reams of warnings in gen/asm-*.h
    #  * -Wnon-virtual-dtor is something Walter has declined to let us fix upstream
    #    and it triggers for the visitors we need in our glue code
    #  * -Wpedantic warns on trailing commas in initializer lists and casting
    #    function pointers to void*.
    #  * -Wgnu-anonymous-struct and -Wnested-anon-types trigger for tokens.h.
    #  * -Wgnu-redeclared-enum triggers for various frontend headers.
    append("-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wno-non-virtual-dtor" LDC_CXXFLAGS)
    if ((${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang"))
        append("-Wno-gnu-anonymous-struct -Wno-nested-anon-types -Wno-gnu-redeclared-enum" LDC_CXXFLAGS)
    endif()
    if(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_C_COMPILER_VERSION VERSION_LESS "4.7.0")
        append("-Wno-pedantic" LDC_CXXFLAGS)
    endif()
elseif(MSVC)
    # Remove flags here, for exceptions and RTTI.
    # CL.EXE complains to override flags like "/GR /GR-".
    string(REGEX REPLACE "(^| )[/-]EH[-cs]*( |$)" "\\2" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "(^| )[/-]GR-?( |$)" "\\2" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    append("/GR- /EHs-c-" CMAKE_CXX_FLAGS)
    append("/D_HAS_EXCEPTIONS=0" CMAKE_CXX_FLAGS)

    # warning C4018: signed/unsigned mismatch
    # warning C4101: unreferenced local variable
    # warning C4102: unreferenced label
    # warning C4146: unary minus operator applied to unsigned type, result still unsigned
    # warnings C4244 and C4267: conversion from '...' to '...', possible loss of data
    # warnings C4456-4459: declaration of '...' hides ...
    # warning C4624: destructor was implicitly defined as deleted because a base class destructor is inaccessible or deleted
    # warning C4800: forcing value to bool 'true' or 'false' (performance warning)
    # warning C4996: we're not using Microsoft's secure stringOp_s() functions
    append("/wd4018 /wd4101 /wd4102 /wd4146 /wd4244 /wd4267 /wd4456 /wd4457 /wd4458 /wd4459 /wd4624 /wd4800 /wd4996" LDC_CXXFLAGS)
    if(LDC_LLVM_VER GREATER 307)
        # Suppress noisy warning C4141 'modifier' used more than once', because of __forceinline combined with inline in LLVM headers
        append("/wd4141" LDC_CXXFLAGS)
    endif()
endif()
# Append -mminimal-toc for gcc 4.0.x - 4.5.x on ppc64
if( CMAKE_COMPILER_IS_GNUCXX
    AND CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64|powerpc64"
    AND CMAKE_C_COMPILER_VERSION VERSION_LESS "4.6.0" )
    append("-mminimal-toc" LDC_CXXFLAGS)
endif()
# Do not use doubledouble on ppc
if( CMAKE_SYSTEM_PROCESSOR MATCHES "ppc|powerpc")
    append("-mlong-double-64" LDC_CXXFLAGS)
endif()
if(UNIX)
    append("-DLDC_POSIX" LDC_CXXFLAGS)
endif()
set(SANITIZE_CXXFLAGS)
set(SANITIZE_LDFLAGS)
if(SANITIZE)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        append("-fsanitize=address" SANITIZE_CXXFLAGS)
        append("-fsanitize=address" SANITIZE_LDFLAGS)
    else()
        message(WARNING "Option SANITIZE specified but compiler is not clang.")
    endif()
endif()
append("${SANITIZE_CXXFLAGS}" LDC_CXXFLAGS)
# LLVM_CXXFLAGS may contain -Werror which causes compile errors with dmd source
string(REPLACE "-Werror " "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
if (UNIX AND NOT "${LLVM_LDFLAGS}" STREQUAL "")
    # LLVM_LDFLAGS may contain -l-lld which is a wrong library reference (AIX)
    string(REPLACE "-l-lld " "-lld " LLVM_LDFLAGS ${LLVM_LDFLAGS})
endif()
if(MSVC)
    separate_arguments(LLVM_LDFLAGS WINDOWS_COMMAND "${LLVM_LDFLAGS}")
else()
    separate_arguments(LLVM_LDFLAGS UNIX_COMMAND "${LLVM_LDFLAGS}")
endif()
# LLVM_CXXFLAGS may contain -Wcovered-switch-default and -fcolor-diagnostics
# which are clang-only options
if(CMAKE_COMPILER_IS_GNUCXX)
    string(REPLACE "-Wcovered-switch-default " "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
    string(REPLACE "-fcolor-diagnostics " "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
endif()
# LLVM_CXXFLAGS may contain -Wno-maybe-uninitialized
# which is gcc-only options
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    string(REPLACE "-Wno-maybe-uninitialized " "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
endif()


# Build (generate executable) of D source
macro(build_idgen  input_d  output_exec  extra_d_flags  link_flags  extra_deps)
    separate_arguments(FLAG_LIST WINDOWS_COMMAND "${D_COMPILER_FLAGS} ${extra_d_flags} ${link_flags}")
    add_custom_command(
        OUTPUT ${output_exec}
        COMMAND ${D_COMPILER} ${FLAG_LIST} -I${PROJECT_SOURCE_DIR}/${DDMDFE_PATH} -of${output_exec} ${input_d}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        DEPENDS ${input_d} ${extra_deps}
    )
endmacro()


#
# Build idgen.
#
build_idgen(${DDMDFE_PATH}/idgen.d ${PROJECT_BINARY_DIR}/${DDMDFE_PATH}/idgen${CMAKE_EXECUTABLE_SUFFIX}  ${DDMD_DFLAGS} "" "")
# Run idgen.
add_custom_command(
    OUTPUT
        ${PROJECT_BINARY_DIR}/${DDMDFE_PATH}/id.d
        ${PROJECT_BINARY_DIR}/${DDMDFE_PATH}/id.h
    COMMAND ${PROJECT_BINARY_DIR}/${DDMDFE_PATH}/idgen  #provide full path to avoid clash with idgen on path
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${DDMDFE_PATH}
    DEPENDS ${PROJECT_BINARY_DIR}/${DDMDFE_PATH}/idgen${CMAKE_EXECUTABLE_SUFFIX}
)
set(LDC_CXX_GENERATED
    ${PROJECT_BINARY_DIR}/${DDMDFE_PATH}/id.h
)
set(LDC_D_GENERATED
    ${PROJECT_BINARY_DIR}/${DDMDFE_PATH}/id.d
)

#
# Gather source files.
#
include(GetGitRevisionDescription)
git_get_exact_tag(TAG)
if(NOT TAG MATCHES "NOTFOUND")
    if(TAG MATCHES "v[0-9].*")
        # For a version tag, remove the leading 'v'. CMake 2.8.0 (e.g. Ubuntu
        # 10.04 LTS) doesn't support -1 in string(SUBSTRING ...), so spell it
        # out.
        string(LENGTH "${TAG}" taglen)
        MATH(EXPR taglen "${taglen} - 1")
        string(SUBSTRING "${TAG}" 1 ${taglen} LDC_VERSION)
    else()
        set(LDC_VERSION "${TAG}")
    endif()
else()
    get_git_head_revision(REFSPEC HASH)
    if(NOT HASH STREQUAL "GITDIR-NOTFOUND")
        # Append git hash to LDC_VERSION
        string(SUBSTRING "${HASH}" 0 7 LDC_VERSION_HASH)
        set(LDC_VERSION "${LDC_VERSION}git-${LDC_VERSION_HASH}")

        # Append "-dirty" when the working copy is dirty
        git_describe(GIT_DIRTY --dirty)
        if (GIT_DIRTY MATCHES ".*-dirty")
            set(LDC_VERSION "${LDC_VERSION}-dirty")
        endif()
    endif()
endif()
message(STATUS "LDC version identifier: ${LDC_VERSION}")
configure_file(driver/ldc-version.cpp.in driver/ldc-version.cpp)

# Also add the header files to the build so that they are available in IDE
# project files generated via CMake.
file(GLOB_RECURSE FE_SRC_D ${DDMDFE_PATH}/*.d)
file(GLOB_RECURSE FE_HDR   ${DDMDFE_PATH}/*.h)
file(GLOB_RECURSE FE_RES   res/*.*)
file(GLOB_RECURSE GEN_SRC gen/*.cpp)
file(GLOB_RECURSE GEN_HDR gen/*.h)
file(GLOB_RECURSE GEN_SRC_D gen/*.d)
file(GLOB_RECURSE DRV_SRC_D driver/*.d)
file(GLOB_RECURSE IR_SRC_D ir/*.d)
file(GLOB IR_SRC ir/*.cpp)
file(GLOB IR_HDR ir/*.h)
set(DRV_SRC
    driver/cache.cpp
    driver/cl_options.cpp
    driver/codegenerator.cpp
    driver/configfile.cpp
    driver/dcomputecodegenerator.cpp
    driver/exe_path.cpp
    driver/targetmachine.cpp
    driver/toobj.cpp
    driver/tool.cpp
    driver/linker.cpp
    driver/main.cpp
    ${CMAKE_BINARY_DIR}/driver/ldc-version.cpp
)
set(DRV_HDR
    driver/cache.h
    driver/cache_pruning.h
    driver/cl_options.h
    driver/codegenerator.h
    driver/configfile.h
    driver/dcomputecodegenerator.h
    driver/exe_path.h
    driver/ldc-version.h
    driver/linker.h
    driver/targetmachine.h
    driver/toobj.h
    driver/tool.h
)
# exclude idgen and man.d
list(REMOVE_ITEM FE_SRC_D
    ${PROJECT_SOURCE_DIR}/${DDMDFE_PATH}/idgen.d
    ${PROJECT_SOURCE_DIR}/${DDMDFE_PATH}/root/man.d
)
# exclude ldmd.d from ldc
list(REMOVE_ITEM DRV_SRC_D
    ${PROJECT_SOURCE_DIR}/driver/ldmd.d
)
set(LDC_CXX_SOURCE_FILES
    ${LDC_CXX_GENERATED}
    ${FE_HDR}
    ${GEN_SRC}
    ${GEN_HDR}
    ${IR_SRC}
    ${IR_HDR}
)
set(LDC_D_SOURCE_FILES
    ${LDC_D_GENERATED}
    ${FE_SRC_D}
    ${GEN_SRC_D}
    ${DRV_SRC_D}
    ${IR_SRC_D}
)

source_group("Source Files\\${DDMDFE_PATH}" FILES ${FE_SRC_D})
source_group("Header Files\\${DDMDFE_PATH}" FILES ${FE_HDR})
source_group("Source Files\\gen" FILES ${GEN_SRC})
source_group("Header Files\\gen" FILES ${GEN_HDR})
source_group("Source Files\\ir" FILES ${IR_SRC})
source_group("Header Files\\ir" FILES ${IR_HDR})
source_group("Generated Files" REGULAR_EXPRESSION "(id\\.[cdh]|impcnvtab\\.c)$")


#
# Configure the build system to use LTO and/or PGO while building LDC
#
include(HandleLTOPGOBuildOptions)

#
# Enable PGO if supported for this platform and LLVM version.
# LLVM >= 3.7 is required for PGO.
#
set(LDC_WITH_PGO False)  # must be a valid Python boolean constant (case sensitive)
if (NOT (LDC_LLVM_VER LESS 307))
    message(STATUS "Building LDC with PGO support")
    add_definitions(-DLDC_WITH_PGO)
    set(LDC_WITH_PGO True)
endif()

#
# Includes, defines.
#

include_directories(
    .
    ${DDMDFE_PATH}
    ${DDMDFE_PATH}/root
    ${PROJECT_BINARY_DIR}/${DDMDFE_PATH}
    ${PROJECT_SOURCE_DIR}
)
include_directories( SYSTEM
    ${LLVM_INCLUDE_DIRS}
)
append("-I${PROJECT_SOURCE_DIR}" DDMD_DFLAGS)
append("-I${PROJECT_BINARY_DIR}" DDMD_DFLAGS)


if(MSVC)
    include_directories(${PROJECT_SOURCE_DIR}/vcbuild)
endif()

if(MSVC)
    append("-version=IN_LLVM_MSVC" DDMD_DFLAGS)
endif()
append("-version=IN_LLVM" DDMD_DFLAGS)
append("-DIN_LLVM" CMAKE_CXX_FLAGS)
append("-DOPAQUE_VTBLS" CMAKE_CXX_FLAGS)
append("\"-DLDC_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}\"" CMAKE_CXX_FLAGS)
append("-DLDC_LLVM_VER=${LDC_LLVM_VER}" CMAKE_CXX_FLAGS)
append("\"-DLDC_LIBDIR_SUFFIX=\\\"${LIB_SUFFIX}\\\"\"" CMAKE_CXX_FLAGS)

if(GENERATE_OFFTI)
    append("-DGENERATE_OFFTI" CMAKE_CXX_FLAGS)
endif()

option(RISCV_LLVM_DEV, "full RISC-V support with riscv-llvm")
mark_as_advanced(RISCV_LLVM_DEV)

#
# Enable building with riscv-llvm, for full RISC-V support.
#
if(RISCV_LLVM_DEV)
    append("-DRISCV_LLVM_DEV" CMAKE_CXX_FLAGS)
endif()

# if llvm was built with assertions we have to do the same
# as there are some headers with differing behavior based on NDEBUG
if(LLVM_ENABLE_ASSERTIONS)
    append("-UNDEBUG" EXTRA_CXXFLAGS)
    # avoid MSVC warning D9025 about "-DNDEBUG ... -UNDEBUG"
    string(REGEX REPLACE "(^| )[/-]D *NDEBUG( |$)" "\\1-UNDEBUG\\2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REGEX REPLACE "(^| )[/-]D *NDEBUG( |$)" "\\1-UNDEBUG\\2" CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}")
    string(REGEX REPLACE "(^| )[/-]D *NDEBUG( |$)" "\\1-UNDEBUG\\2" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

#
# Check if libpthread is available.
# FIXME: Guard with LLVM_ENABLE_THREADS
#
if( NOT WIN32 OR CYGWIN )
    check_include_file(pthread.h HAVE_PTHREAD_H)
    check_library_exists(pthread pthread_create "" HAVE_LIBPTHREAD)
    if(HAVE_LIBPTHREAD)
        set(PTHREAD_LIBS -lpthread)
    endif()
endif()

#
# Check if terminfo is available.
# FIXME: Guard with LLVM_ENABLE_TERMINFO
#
if( NOT WIN32 OR CYGWIN )
    set(HAVE_TERMINFO 0)
    foreach(library tinfo terminfo curses ncurses ncursesw)
        string(TOUPPER ${library} library_suffix)
        check_library_exists(${library} setupterm "" HAVE_TERMINFO_${library_suffix})
        if(HAVE_TERMINFO_${library_suffix})
            set(HAVE_TERMINFO 1)
            set(TERMINFO_LIBS "${library}")
        break()
        endif()
    endforeach()
endif()

#
# Enable instrumentation for code coverage analysis
#
set(TEST_COVERAGE OFF CACHE BOOL "instrument compiler for code coverage analysis")
if(TEST_COVERAGE)
    if(CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang"))
        append("-O0 -g -fprofile-arcs -ftest-coverage" EXTRA_CXXFLAGS)
        list(APPEND LLVM_LDFLAGS "-lgcov")
    else()
        message(WARNING "Coverage testing is not available.")
    endif()
endif()

#
# Set up the main ldc/ldc2 target.
#
if(BUILD_SHARED)
    set(LDC_LIB_TYPE SHARED)
else()
    set(LDC_LIB_TYPE STATIC)
endif()

set(LDC_LIB LDCShared)
add_library(${LDC_LIB} ${LDC_LIB_TYPE} ${LDC_CXX_SOURCE_FILES} ${DRV_SRC} ${DRV_HDR})
set_target_properties(
    ${LDC_LIB} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}
    ARCHIVE_OUTPUT_NAME ldc
    LIBRARY_OUTPUT_NAME ldc
    RUNTIME_OUTPUT_NAME ldc
    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS} ${EXTRA_CXXFLAGS}"
    LINK_FLAGS "${SANITIZE_LDFLAGS}"
)
# LDFLAGS should actually be in target property LINK_FLAGS, but this works, and gets around linking problems
target_link_libraries(${LDC_LIB} ${LLVM_LIBRARIES} ${PTHREAD_LIBS} ${TERMINFO_LIBS} ${LLVM_LDFLAGS})
if(WIN32)
    target_link_libraries(${LDC_LIB} imagehlp psapi)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(${LDC_LIB} dl)
endif()

set(LDC_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${LDC_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX})
set(LDMD_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${LDMD_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX})
add_custom_target(${LDC_EXE} ALL DEPENDS ${LDC_EXE_FULL})
add_custom_target(${LDMD_EXE} ALL DEPENDS ${LDMD_EXE_FULL})

# Figure out how to link the main LDC executable, for which we need to take the
# LLVM flags into account.
set(LDC_LINKERFLAG_LIST "${SANITIZE_LDFLAGS};${LLVM_LIBRARIES};${LLVM_LDFLAGS}")

set(LDC_LINK_MANUALLY OFF)
if(UNIX AND (CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")))
    # On Unix-like systems, DMD and LDC will use the C compiler for linking, but
    # will pass on -L options prefixed by -Xlinker to directly forward them to
    # the underlying ld. Since there are some flags the GCC driver handles itself
    # rather than passing them to ld, we cannot just directly translate
    # LDC_LINKERFLAG_LIST to -L options. To be able to handle general linker flags,
    # we manually invoke the linker instead of using the D compiler to do so.
    set(LDC_LINK_MANUALLY ON)

    # gdmd's -of option is incompatible with DMD/LDMD when -c is given. We could
    # detect this and modify the command line accordingly, but for now, just
    # disallow gdmd.
    if(D_COMPILER_ID STREQUAL "GDMD")
        message(FATAL_ERROR "GDMD currently not supported due to http://bugzilla.gdcproject.org/show_bug.cgi?id=232.")
    endif()

    include(ExtractDMDSystemLinker)
    message(STATUS "Host D compiler linker program: ${D_LINKER_COMMAND}")
    message(STATUS "Host D compiler linker flags: ${D_LINKER_ARGS}")
    list(APPEND LDC_LINKERFLAG_LIST ${D_LINKER_ARGS})

    if(NOT "${CMAKE_EXE_LINKER_FLAGS}" STREQUAL "")
        separate_arguments(flags UNIX_COMMAND "${CMAKE_EXE_LINKER_FLAGS}")
        list(APPEND LDC_LINKERFLAG_LIST ${flags})
    endif()
else()
    # Use D compiler for linking, trying to translate a few common linker flags.
    set(LDC_TRANSLATED_LINKER_FLAGS "")
    foreach(f ${LDC_LINKERFLAG_LIST})
        string(REPLACE "-LIBPATH:" "/LIBPATH:" f ${f})
        append("\"-L${f}\"" LDC_TRANSLATED_LINKER_FLAGS)
    endforeach()

    if(MSVC)
        # Issue 1297 – set LDC's stack to 8 MiB like on Linux and Mac (default: 1 MiB).
        append("-L/STACK:8388608" LDC_TRANSLATED_LINKER_FLAGS)
    endif()
endif()

function(build_d_executable output_exe compiler_args linker_args compile_deps link_deps)
    if(LDC_LINK_MANUALLY)
        set(object_file ${output_exe}${CMAKE_CXX_OUTPUT_EXTENSION})
        separate_arguments(dmd_flags UNIX_COMMAND "${D_COMPILER_FLAGS} ${DDMD_DFLAGS}")
        add_custom_command(
            OUTPUT ${object_file}
            COMMAND ${D_COMPILER} -c ${dmd_flags} -I${PROJECT_SOURCE_DIR}/${DDMDFE_PATH} -of${object_file} ${compiler_args}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            DEPENDS ${compile_deps}
        )
        add_custom_command(
            OUTPUT ${output_exe}
            COMMAND ${CMAKE_CXX_COMPILER} -o ${output_exe} ${object_file} ${linker_args} ${LDC_LINKERFLAG_LIST}
            DEPENDS ${object_file} ${link_deps}
        )
    else()
        set(dflags "${D_COMPILER_FLAGS} ${DDMD_DFLAGS} ${DDMD_LFLAGS}")
        set(lflags ${LDC_TRANSLATED_LINKER_FLAGS})
        if(UNIX)
          separate_arguments(dflags UNIX_COMMAND "${dflags}")
          separate_arguments(lflags UNIX_COMMAND "${lflags}")
        else()
          separate_arguments(dflags WINDOWS_COMMAND "${dflags}")
          separate_arguments(lflags WINDOWS_COMMAND "${lflags}")
        endif()
        foreach(f ${linker_args})
            append("\"-L${f}\"" lflags)
        endforeach()
        add_custom_command(
            OUTPUT ${output_exe}
            COMMAND ${D_COMPILER} ${dflags} ${lflags} -I${PROJECT_SOURCE_DIR}/${DDMDFE_PATH} -of${output_exe} ${compiler_args} ${linker_args}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            DEPENDS ${compile_deps} ${link_deps}
        )
    endif()
endfunction()

# CONFIG generator expressions need to be repeated due to https://cmake.org/Bug/view.php?id=14353
if(MSVC_IDE)
    separate_arguments(LDC_FLAG_LIST WINDOWS_COMMAND "${LDC_TRANSLATED_LINKER_FLAGS} ${D_COMPILER_FLAGS} ${DDMD_DFLAGS} ${DDMD_LFLAGS}")
    add_custom_command(
        OUTPUT ${LDC_EXE_FULL}
        COMMAND ${D_COMPILER} -L$<TARGET_LINKER_FILE:${LDC_LIB}> ${LDC_FLAG_LIST} -I${PROJECT_SOURCE_DIR}/${DDMDFE_PATH} -of${LDC_EXE_FULL} ${LDC_D_SOURCE_FILES} $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:-g> $<$<NOT:$<CONFIG:Debug>>:-O> $<$<NOT:$<CONFIG:Debug>>:-inline> $<$<NOT:$<CONFIG:Debug>>:-release>
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        DEPENDS ${LDC_D_SOURCE_FILES} ${FE_RES} ${PROJECT_BINARY_DIR}/${DDMDFE_PATH}/id.d ${LDC_LIB}
    )
else()
    build_d_executable(
        "${LDC_EXE_FULL}"
        "${LDC_D_SOURCE_FILES}"
        "$<TARGET_LINKER_FILE:${LDC_LIB}>"
        "${LDC_D_SOURCE_FILES};${FE_RES};${PROJECT_BINARY_DIR}/${DDMDFE_PATH}/id.d"
        "${LDC_LIB}"
    )
endif()

if(MSVC_IDE)
    # the IDE generator is a multi-config one
    # so copy the config file into the correct bin subfolder
    # (different outputs no longer feasible for custom commands, so disabled)
    #    add_custom_command(TARGET ${LDC_EXE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/bin/${LDC_EXE}.conf $<TARGET_FILE_DIR:${LDC_EXE}> COMMENT "Copy config file ${LDC_EXE}.conf")
endif()


#
# LDMD
#
include(CheckSymbolExists)
CHECK_SYMBOL_EXISTS(_SC_ARG_MAX "unistd.h" HAVE_SC_ARG_MAX)
if (HAVE_SC_ARG_MAX)
   append("-DHAVE_SC_ARG_MAX" CMAKE_CXX_FLAGS)
endif()

set_source_files_properties(driver/exe_path.cpp driver/ldmd.cpp driver/response.cpp PROPERTIES
    COMPILE_FLAGS "${LDC_CXXFLAGS} ${LLVM_CXXFLAGS}"
    COMPILE_DEFINITIONS LDC_EXE_NAME="${LDC_EXE_NAME}"
)

add_library(LDMD_CXX_LIB ${LDC_LIB_TYPE} driver/exe_path.cpp driver/ldmd.cpp driver/response.cpp driver/exe_path.h)
set_target_properties(
    LDMD_CXX_LIB PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}
    ARCHIVE_OUTPUT_NAME ldmd
    LIBRARY_OUTPUT_NAME ldmd
)
set(LDMD_D_SOURCE_FILES ${DDMDFE_PATH}/root/man.d driver/ldmd.d)
build_d_executable(
    "${LDMD_EXE_FULL}"
    "${LDMD_D_SOURCE_FILES}"
    "$<TARGET_LINKER_FILE:LDMD_CXX_LIB>"
    "${LDMD_D_SOURCE_FILES}"
    "LDMD_CXX_LIB;${LDC_LIB}"
)

#
# Locate LLVM's LTO binary and use it (LLVM >= 3.9)
#
if(APPLE)
    set(LDC_INSTALL_LTOPLUGIN_DEFAULT ON)
else()
    set(LDC_INSTALL_LTOPLUGIN_DEFAULT OFF)
endif()
set(LDC_INSTALL_LTOPLUGIN ${LDC_INSTALL_LTOPLUGIN_DEFAULT} CACHE BOOL "Copy/install the LTO plugin from the LLVM package when available (LLVM >= 3.9).")
if (NOT (LDC_LLVM_VER LESS 309) AND LDC_INSTALL_LTOPLUGIN)
    if(APPLE)
        set(LLVM_LTO_BINARY ${LLVM_LIBRARY_DIRS}/libLTO.dylib)
        set(LDC_LTO_BINARY_NAME libLTO-ldc.dylib)
    elseif(UNIX)
        set(LLVM_LTO_BINARY ${LLVM_LIBRARY_DIRS}/LLVMgold.so)
        set(LDC_LTO_BINARY_NAME LLVMgold-ldc.so)
    endif()
    if(EXISTS ${LLVM_LTO_BINARY})
        message(STATUS "Also installing LTO binary: ${LLVM_LTO_BINARY}")
        file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX})
        configure_file(${LLVM_LTO_BINARY} ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}/${LDC_LTO_BINARY_NAME} COPYONLY)
    endif()
endif()

#
# Auxiliary build and test utils.
#
add_subdirectory(utils)

#
# Auxiliary tools.
#
add_subdirectory(tools)

#
# Test and runtime targets. Note that enable_testing() is order-sensitive!
#
enable_testing()

# LDC unittest executable (D unittests only).
set(LDC_UNITTEST_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${LDC_EXE_NAME}-unittest${CMAKE_EXECUTABLE_SUFFIX})
build_d_executable(
    "${LDC_UNITTEST_EXE_FULL}"
    "-unittest;${LDC_D_SOURCE_FILES}"
    "$<TARGET_LINKER_FILE:${LDC_LIB}>"
    "${LDC_D_SOURCE_FILES}"
    "${LDC_LIB}"
)
add_custom_target(ldc2-unittest DEPENDS ${LDC_UNITTEST_EXE_FULL})
add_test(NAME build-ldc2-unittest COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ldc2-unittest)
add_test(NAME ldc2-unittest COMMAND ${LDC_UNITTEST_EXE_FULL} --version)
set_tests_properties(ldc2-unittest PROPERTIES DEPENDS build-ldc2-unittest)

add_subdirectory(runtime)
if(D_VERSION EQUAL 2)
    add_subdirectory(tests/d2)
endif()
add_subdirectory(tests)

#
# Install target.
#

install(PROGRAMS ${LDC_EXE_FULL} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
install(PROGRAMS ${LDMD_EXE_FULL} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
if(${BUILD_SHARED})
    # For now, only install libldc if explicitly building the shared library.
    # While it might theoretically be possible to use LDC as a static library
    # as well, for the time being this just bloats the normal packages.
    install(TARGETS ${LDC_LIB} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
endif()
install(FILES ${PROJECT_BINARY_DIR}/bin/${LDC_EXE}_install.conf DESTINATION ${CONF_INST_DIR} RENAME ${LDC_EXE}.conf)

if(MSVC)
    install(DIRECTORY vcbuild/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.bat")
    # Also put the VCBuild scripts in the build/bin folder, so that ${PROJECT_BINARY_DIR}/bin/ldc2 is functional.
    # This is necessary for the IR tests that use ${PROJECT_BINARY_DIR}/bin/ldc2.
    configure_file(vcbuild/amd64.bat   ${PROJECT_BINARY_DIR}/bin/amd64.bat   COPYONLY)
    configure_file(vcbuild/msvcEnv.bat ${PROJECT_BINARY_DIR}/bin/msvcEnv.bat COPYONLY)
    configure_file(vcbuild/x86.bat     ${PROJECT_BINARY_DIR}/bin/x86.bat     COPYONLY)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    find_package(bash-completion QUIET)
    if(NOT BASH_COMPLETION_FOUND)
        set(BASH_COMPLETION_COMPLETIONSDIR "${CONF_INST_DIR}/bash_completion.d")
        if(LINUX_DISTRIBUTION_IS_GENTOO)
            set(BASH_COMPLETION_COMPLETIONSDIR "/usr/share/bash-completion")
        endif()
    endif()
    install(DIRECTORY bash_completion.d/ DESTINATION ${BASH_COMPLETION_COMPLETIONSDIR})
endif()

# Also install LLVM's LTO binary if available
if(EXISTS ${LLVM_LTO_BINARY})
    install(PROGRAMS ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}/${LDC_LTO_BINARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
endif()

#
# Packaging
#

include (CMakeCPack.cmake)
include (CPack)
