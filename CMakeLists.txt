cmake_minimum_required(VERSION 2.8)
project(ldc)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

if(MSVC)
    set(LIBCONFIG_DLL OFF CACHE BOOL "Use libconfig DLL instead of static library")
endif()

include(FindDCompiler)
include(CheckIncludeFile)
include(CheckLibraryExists)
include(CheckCXXCompilerFlag)

#
# Locate LLVM.
#

find_package(LLVM 3.5 REQUIRED
    all-targets analysis asmparser asmprinter bitreader bitwriter codegen core debuginfocodeview debuginfodwarf debuginfopdb globalisel instcombine ipa ipo instrumentation irreader linker lto mc mcdisassembler mcparser objcarcopts object option profiledata scalaropts selectiondag support tablegen target transformutils vectorize ${EXTRA_LLVM_MODULES})
math(EXPR LDC_LLVM_VER ${LLVM_VERSION_MAJOR}*100+${LLVM_VERSION_MINOR})
# Remove LLVMTableGen library from list of libraries
string(REGEX MATCH "^-.*LLVMTableGen[^;]*;|;-.*LLVMTableGen[^;]*" LLVM_TABLEGEN_LIBRARY "${LLVM_LIBRARIES}")
string(REGEX REPLACE "^-.*LLVMTableGen[^;]*;|;-.*LLVMTableGen[^;]*" "" LLVM_LIBRARIES "${LLVM_LIBRARIES}")

#
# Locate libconfig.
#
find_package(LibConfig REQUIRED)

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
set(LDC_VERSION "1.0.0") # May be overridden by git hash tag
set(DMDFE_MAJOR_VERSION   2)
set(DMDFE_MINOR_VERSION   0)
set(DMDFE_PATCH_VERSION   70)
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

# Setup D compiler flags and linker flags for phobos (system linker) etc. We use
# DMD syntax for all the flags so we can work with both DMD and LDMD.
set(DDMD_DFLAGS "-wi")
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    append("-g -link-debuglib" DDMD_FLAGS)
elseif(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
    append("-g -O -inline -release" DDMD_DFLAGS)
else()
    # Default to a Release build type
    append("-O -inline -release" DDMD_DFLAGS)
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
        append("-L/DEFAULTLIB:msvcrt -L/NODEFAULTLIB:libcmt" DDMD_DFLAGS)
        if(MSVC_VERSION GREATER 1800) # VS 2015+
            append("-L/NODEFAULTLIB:libvcruntime" DDMD_DFLAGS)
        endif()
    else()
        append("-L/DEFAULTLIB:libcmt -L/NODEFAULTLIB:msvcrt" DDMD_DFLAGS)
        if(MSVC_VERSION GREATER 1800) # VS 2015+
            append("-L/NODEFAULTLIB:vcruntime" DDMD_DFLAGS)
        endif()
    endif()
endif()

append("-J${PROJECT_SOURCE_DIR}/${DDMDFE_PATH}" DDMD_DFLAGS) # Needed for importing text files
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
    # LLVM_LDFLAGS may have a space at the end
    string(REGEX REPLACE " $" "" LLVM_LDFLAGS ${LLVM_LDFLAGS})
endif()
# LLVM_CXXFLAGS may contain -Wcovered-switch-default and -fcolor-diagnostics
# which are clang-only options
if(CMAKE_COMPILER_IS_GNUCXX)
    string(REPLACE "-Wcovered-switch-default " "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
    string(REPLACE "-fcolor-diagnostics " "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
endif()


# Compiles the given D module into an object file.
macro(Dcompile  input_d  output_dir  extra_d_flags  outlist_o extra_deps)
    if (IS_ABSOLUTE ${input_d})
        file(RELATIVE_PATH output ${output_dir} ${input_d})
    else()
        set(output ${input_d})
    endif()

    get_filename_component(name ${output} NAME)
    get_filename_component(path ${output} PATH)
    if("${path}" STREQUAL "")
        set(output_root ${name})
    else()
        set(output_root ${path}/${name})
    endif()

    set(output_o  ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${output_root}${CMAKE_C_OUTPUT_EXTENSION})
    list(APPEND ${outlist_o} ${output_o})

    separate_arguments(FLAG_LIST WINDOWS_COMMAND "${D_COMPILER_FLAGS} ${extra_d_flags}")
    add_custom_command(
        OUTPUT ${output_o}
        COMMAND ${D_COMPILER} ${FLAG_LIST} -c -I${PROJECT_SOURCE_DIR}/${DDMDFE_PATH} -of${output_o} ${input_d}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        DEPENDS ${input_d} ${extra_deps}
    )
endmacro()
macro(Dcompilelib  input_d  output_dir  extra_d_flags  outlist_o)
    if (IS_ABSOLUTE ${input_d})
        file(RELATIVE_PATH output ${output_dir} ${input_d})
    else()
        set(output ${input_d})
    endif()

    get_filename_component(name ${output} NAME_WE)
    get_filename_component(path ${output} PATH)
    if("${path}" STREQUAL "")
        set(output_root ${name})
    else()
        set(output_root ${path}/${name})
    endif()

    set(output_lib  ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${output_root}${CMAKE_STATIC_LIBRARY_SUFFIX})
    list(APPEND ${outlist_o} ${output_lib})

    separate_arguments(FLAG_LIST WINDOWS_COMMAND "${D_COMPILER_FLAGS} ${extra_d_flags}")
    add_custom_command(
        OUTPUT ${output_lib}
        COMMAND ${D_COMPILER} ${FLAG_LIST} -lib -I${PROJECT_SOURCE_DIR}/${DDMDFE_PATH} -of${output_lib} ${input_d}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        DEPENDS ${input_d}
    )
endmacro()

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
        string(SUBSTRING "${HASH}" 0 6 LDC_VERSION)
    endif()
endif()
message(STATUS "LDC version identifier: ${LDC_VERSION}")
configure_file(driver/ldc-version.cpp.in driver/ldc-version.cpp)

# Also add the header files to the build so that they are available in IDE
# project files generated via CMake.
file(GLOB_RECURSE FE_SRC_D ${DDMDFE_PATH}/*.d)
file(GLOB_RECURSE FE_HDR   ${DDMDFE_PATH}/*.h)
file(GLOB_RECURSE GEN_SRC gen/*.cpp)
file(GLOB_RECURSE GEN_HDR gen/*.h)
file(GLOB_RECURSE GEN_SRC_D gen/*.d)
file(GLOB_RECURSE DRV_SRC_D driver/*.d)
file(GLOB_RECURSE IR_SRC_D ir/*.d)
file(GLOB IR_SRC ir/*.cpp)
file(GLOB IR_HDR ir/*.h)
set(DRV_SRC
    driver/cl_options.cpp
    driver/codegenerator.cpp
    driver/configfile.cpp
    driver/exe_path.cpp
    driver/targetmachine.cpp
    driver/toobj.cpp
    driver/tool.cpp
    driver/linker.cpp
    driver/main.cpp
    ${CMAKE_BINARY_DIR}/driver/ldc-version.cpp
)
set(DRV_HDR
    driver/linker.h
    driver/cl_options.h
    driver/codegenerator.h
    driver/configfile.h
    driver/exe_path.h
    driver/ldc-version.h
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
# Enable PGO if supported for this platform and LLVM version.
# LLVM >= 3.7 is required for PGO.
#
set(LDC_WITH_PGO False)  # must be a valid Python boolean constant (case sensitive)
if (NOT (LDC_LLVM_VER LESS 307))
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
    ${LIBCONFIG_INCLUDE_DIR}
)
append("-I${PROJECT_SOURCE_DIR}" DDMD_DFLAGS)
append("-I${PROJECT_BINARY_DIR}" DDMD_DFLAGS)


if(MSVC)
    include_directories(${PROJECT_SOURCE_DIR}/vcbuild)
    if(NOT LIBCONFIG_DLL)
        append("-DLIBCONFIG_STATIC" CMAKE_CXX_FLAGS)
    endif()
endif()

if(MSVC)
    append("-version=IN_LLVM_MSVC" DDMD_DFLAGS)
endif()
append("-version=IN_LLVM" DDMD_DFLAGS)
append("-DIN_LLVM" CMAKE_CXX_FLAGS)
append("-DOPAQUE_VTBLS" CMAKE_CXX_FLAGS)
append("-DLDC_INSTALL_PREFIX=\\\"${CMAKE_INSTALL_PREFIX}\\\"" CMAKE_CXX_FLAGS)
append("-DLDC_LLVM_VER=${LDC_LLVM_VER}" CMAKE_CXX_FLAGS)

if(GENERATE_OFFTI)
    append("-DGENERATE_OFFTI" CMAKE_CXX_FLAGS)
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
# Check endianess.
# There is no realiable way to delegate the work to the compiler.
# E.g. gcc up to version 4.6 defines __LITTLE_ENDIAN, but not 4.7
#
include(TestBigEndian)
test_big_endian(BIGENDIAN)
if(${BIGENDIAN})
    append("-D__BIG_ENDIAN__" CMAKE_CXX_FLAGS)
else()
    append("-D__LITTLE_ENDIAN__" CMAKE_CXX_FLAGS)
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
        append("-lgcov" LLVM_LDFLAGS)
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
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_NAME ldc
    LIBRARY_OUTPUT_NAME ldc
    RUNTIME_OUTPUT_NAME ldc
    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS} ${EXTRA_CXXFLAGS}"
    LINK_FLAGS "${SANITIZE_LDFLAGS}"
)
# LDFLAGS should actually be in target property LINK_FLAGS, but this works, and gets around linking problems
target_link_libraries(${LDC_LIB} ${LLVM_LIBRARIES} ${PTHREAD_LIBS} ${TERMINFO_LIBS} "${LLVM_LDFLAGS}")
if(WIN32)
    target_link_libraries(${LDC_LIB} imagehlp psapi)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(${LDC_LIB} dl)
endif()

if(WIN32)
    set(EXECUTABLE_EXTENSION ".exe")
endif()
set(LDC_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${LDC_EXE_NAME}${EXECUTABLE_EXTENSION})
set(LDMD_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${LDMD_EXE_NAME}${EXECUTABLE_EXTENSION})
add_custom_target(${LDC_EXE} ALL DEPENDS ${LDC_EXE_FULL})
add_custom_target(${LDMD_EXE} ALL DEPENDS ${LDMD_EXE_FULL})
string (REPLACE ";" " " LDC_LINKERFLAG_LIST "${SANITIZE_LDFLAGS} ${WINDOWS_STACK_SIZE} ${LIBCONFIG_LIBRARY} ${LLVM_LIBRARIES} ${LLVM_LDFLAGS}")
string (REPLACE "-Wl," "" LDC_LINKERFLAG_LIST ${LDC_LINKERFLAG_LIST})
separate_arguments(LDC_LINKERFLAG_LIST WINDOWS_COMMAND ${LDC_LINKERFLAG_LIST})
set(tempVar "")
FOREACH(f ${LDC_LINKERFLAG_LIST})
      append("-L${f}" tempVar)
ENDFOREACH(f)
if(MSVC)
    # Issue 1297
    # The default system-allocated stack size is 8MB on Linux and Mac, but only 1MB on Windows
    # Set LDC's stack to 8MB also on Windows:
    append("-L/STACK:8388608" tempVar)
else()
    if (${CMAKE_EXE_LINKER_FLAGS} MATCHES ".*stdlib=libc\\+\\+.*")
        message(STATUS "Linking with libc++")
        append("-L-lc++" tempVar)
    else()
        append("-L-lstdc++" tempVar)
    endif()
endif()

separate_arguments(LDC_FLAG_LIST WINDOWS_COMMAND "${tempVar} ${D_COMPILER_FLAGS} ${DDMD_DFLAGS}")
add_custom_command(
    OUTPUT ${LDC_EXE_FULL}
    COMMAND ${D_COMPILER} -L$<TARGET_LINKER_FILE:${LDC_LIB}> ${LDC_FLAG_LIST} -I${PROJECT_SOURCE_DIR}/${DDMDFE_PATH} -of${LDC_EXE_FULL} ${LDC_D_SOURCE_FILES}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS ${LDC_D_SOURCE_FILES} ${PROJECT_BINARY_DIR}/${DDMDFE_PATH}/id.d ${LDC_LIB}
)


if(MSVC_IDE)
    # the IDE generator is a multi-config one
    # so copy the config file into the correct bin subfolder
    add_custom_command(TARGET ${LDC_EXE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/bin/${LDC_EXE}.conf $<TARGET_FILE_DIR:${LDC_EXE}> COMMENT "Copy config file ${LDC_EXE}.conf")
endif()


#
# Intrinsics module generation tools.
#

# The LLVM_INCLUDE_DIR definition is not always set, e.g. on Windows.
# strip off anything but the first path
string(REGEX REPLACE ";.*$" "" LLVM_INC_DIR "${LLVM_INCLUDE_DIRS}")
find_path(LLVM_INTRINSIC_TD_PATH "Intrinsics.td" PATHS ${LLVM_INC_DIR}/llvm ${LLVM_INC_DIR}/llvm/IR NO_DEFAULT_PATH)
if (${LLVM_INTRINSIC_TD_PATH} STREQUAL "LLVM_INTRINSIC_TD_PATH-NOTFOUND")
    message(SEND_ERROR "File Intrinsics.td not found")
else()
    string(REGEX REPLACE "/llvm(/IR)?$" "" LLVM_INTRINSIC_TD_PATH ${LLVM_INTRINSIC_TD_PATH})
    message(STATUS "Using path for Intrinsics.td: ${LLVM_INTRINSIC_TD_PATH}")
endif()
append("-DLLVM_INTRINSIC_TD_PATH=\\\"${LLVM_INTRINSIC_TD_PATH}\\\"" CMAKE_CXX_FLAGS)

add_executable(gen_gccbuiltins utils/gen_gccbuiltins.cpp)

set_target_properties(
    gen_gccbuiltins PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS}"
    LINK_FLAGS "${SANITIZE_LDFLAGS}"
)
target_link_libraries(gen_gccbuiltins ${LLVM_TABLEGEN_LIBRARY} ${LLVM_LIBRARIES} ${PTHREAD_LIBS} ${TERMINFO_LIBS} ${CMAKE_DL_LIBS} "${LLVM_LDFLAGS}")
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(gen_gccbuiltins dl)
endif()


# Build FileCheck for testing (build source version depending on LLVM version)
set(FILECHECK_SRC utils/FileCheck-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.cpp)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${FILECHECK_SRC})
    add_executable(FileCheck ${FILECHECK_SRC})
    set_target_properties(
        FileCheck PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
        COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS}"
        LINK_FLAGS "${SANITIZE_LDFLAGS}"
    )
    target_link_libraries(FileCheck  ${LLVM_LIBRARIES} ${TERMINFO_LIBS} ${CMAKE_DL_LIBS} "${LLVM_LDFLAGS}")
else()
    message(STATUS "Skip building FileCheck, assuming it can be found in LLVM bin directory")
endif()

# Build `not` for testing
add_executable(not utils/not.cpp)
set_target_properties(
    not PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS}"
    LINK_FLAGS "${SANITIZE_LDFLAGS}"
)
target_link_libraries(not  ${LLVM_LIBRARIES} ${TERMINFO_LIBS} ${CMAKE_DL_LIBS} "${LLVM_LDFLAGS}")

# Build ldc-profdata for converting profile data formats (source version depends on LLVM version)
set(LDCPROFDATA_SRC utils/llvm-profdata-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.cpp)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LDCPROFDATA_SRC})
    add_executable(ldc-profdata ${LDCPROFDATA_SRC})
    set_target_properties(
        ldc-profdata PROPERTIES
        OUTPUT_NAME "${LDCPROFDATA_EXE_NAME}"
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
        COMPILE_FLAGS "${LLVM_CXXFLAGS} ${CMAKE_CXX_FLAGS_RELEASE}"
    )
    target_link_libraries(ldc-profdata  ${LLVM_LIBRARIES} ${TERMINFO_LIBS} ${CMAKE_DL_LIBS} "${LLVM_LDFLAGS}")
    install(TARGETS ${LDCPROFDATA_EXE} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
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
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_NAME ldmd
    LIBRARY_OUTPUT_NAME ldmd
)
add_custom_command(
    OUTPUT ${LDMD_EXE_FULL}
    COMMAND ${D_COMPILER} -L$<TARGET_LINKER_FILE:LDMD_CXX_LIB> ${LDC_FLAG_LIST} -I${PROJECT_SOURCE_DIR}/${DDMDFE_PATH} -of${LDMD_EXE_FULL} ${DDMDFE_PATH}/root/man.d driver/ldmd.d
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS LDMD_CXX_LIB ${LDC_LIB}
)


#
# Test and runtime targets. Note that enable_testing() is order-sensitive!
#
enable_testing()
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
    install(TARGETS ${LDC_LIB} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
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

#
# Packaging
#

include (CMakeCPack.cmake)
include (CPack)
