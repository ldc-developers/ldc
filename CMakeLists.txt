cmake_minimum_required(VERSION 2.8)
if(POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif()

project(ldc)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

include(FindDCompiler)
include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckLibraryExists)
include(CheckCXXCompilerFlag)
include(CheckDSourceCompiles)
include(CheckLinkFlag)
include(BuildDExecutable)

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
# Locate LLVM.
#

find_package(LLVM 3.9 REQUIRED
    all-targets analysis asmparser asmprinter bitreader bitwriter codegen core
    debuginfodwarf debuginfomsf debuginfopdb demangle
    instcombine ipo instrumentation irreader libdriver linker lto mc
    mcdisassembler mcparser objcarcopts object option profiledata scalaropts
    selectiondag support tablegen target transformutils vectorize
    windowsmanifest ${EXTRA_LLVM_MODULES})
math(EXPR LDC_LLVM_VER ${LLVM_VERSION_MAJOR}*100+${LLVM_VERSION_MINOR})
# Remove LLVMTableGen library from list of libraries
string(REGEX MATCH "[^;]*LLVMTableGen[^;]*" LLVM_TABLEGEN_LIBRARY "${LLVM_LIBRARIES}")
string(REGEX REPLACE "[^;]*LLVMTableGen[^;]*;?" "" LLVM_LIBRARIES "${LLVM_LIBRARIES}")

# Information about which targets LLVM was built to target
foreach(LLVM_SUPPORTED_TARGET ${LLVM_TARGETS_TO_BUILD})
    add_definitions("-DLDC_LLVM_SUPPORTED_TARGET_${LLVM_SUPPORTED_TARGET}=1")
endforeach()

# Check and adapt for LLVMSPIRVLib (Khronos SPIRV-LLVM-Translator)
set(LLVM_SPIRV_FOUND OFF)
if(MSVC)
    if(EXISTS "${LLVM_LIBRARY_DIRS}/LLVMSPIRVLib.lib")
        set(LLVM_SPIRV_FOUND ON)
        set(LLVM_LIBRARIES "${LLVM_LIBRARY_DIRS}/LLVMSPIRVLib.lib" ${LLVM_LIBRARIES})
    endif()
else()
    if((EXISTS "${LLVM_LIBRARY_DIRS}/libLLVMSPIRVLib.a") OR
       (EXISTS "${LLVM_LIBRARY_DIRS}/libLLVMSPIRVLib.so") OR
       (EXISTS "${LLVM_LIBRARY_DIRS}/libLLVMSPIRVLib.dylib"))
        set(LLVM_SPIRV_FOUND ON)
        set(LLVM_LIBRARIES -lLLVMSPIRVLib ${LLVM_LIBRARIES})
    endif()
endif()
if(NOT LLVM_SPIRV_FOUND)
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        if(MSVC)
            # make pkg-config use -LC:\path\to\build\LLVMSPIRVLib.lib not -L-lLLVMSPIRVLib
            set(PKG_CONFIG_EXECUTABLE "${PKG_CONFIG_EXECUTABLE} --msvc-syntax")
        endif()
        pkg_check_modules(LLVM_SPIRV LLVMSPIRVLib)
        if(LLVM_SPIRV_FOUND)
            set(LLVM_SPIRV_FOUND ON) # translate 1 to ON
            foreach(flag ${LLVM_SPIRV_LDFLAGS})
                append("${flag}" LLVM_LDFLAGS)
            endforeach(flag)
        else()
            set(LLVM_SPIRV_FOUND OFF)
        endif()
    endif()
endif()
if(LLVM_SPIRV_FOUND)
    message(STATUS "Building with SPIR-V support")
    add_definitions("-DLDC_LLVM_SUPPORTED_TARGET_SPIRV=1")
endif()

#
# Get info about used Linux distribution.
#
include(GetLinuxDistribution)


#
# Main configuration.
#

# Version information
set(LDC_VERSION "1.20.0") # May be overridden by git hash tag
set(DMDFE_MAJOR_VERSION   2)
set(DMDFE_MINOR_VERSION   0)
set(DMDFE_PATCH_VERSION   90)
set(DMDFE_FIX_LEVEL       1)

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

set(COMPILE_D_MODULES_SEPARATELY OFF CACHE BOOL "Compile each D module separately (instead of all at once). Useful for many CPU cores and/or iterative development; generated executables will be somewhat slower.")

# The following flag is currently not well tested, expect the build to fail.
option(GENERATE_OFFTI "generate complete ClassInfo.offTi arrays")
mark_as_advanced(GENERATE_OFFTI)

if(D_VERSION EQUAL 1)
    message(FATAL_ERROR "D version 1 is no longer supported.
Please consider using D version 2 or checkout the 'd1' git branch for the last version supporting D version 1.")
elseif(D_VERSION EQUAL 2)
    set(LDC_EXE ldc2)
    set(LDMD_EXE ldmd2)
    set(RUNTIME druntime)
    append("-DDMDV2" CMAKE_CXX_FLAGS)
else()
    message(FATAL_ERROR "unsupported D version")
endif()

set(LDC_EXE_NAME ${PROGRAM_PREFIX}${LDC_EXE}${PROGRAM_SUFFIX})
set(LDMD_EXE_NAME ${PROGRAM_PREFIX}${LDMD_EXE}${PROGRAM_SUFFIX})

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

    if(LLVM_ENABLE_ASSERTIONS)
        string(REPLACE " -release" "" DDMD_DFLAGS "${DDMD_DFLAGS}")
    endif()
endif()

if(MSVC)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(STATUS "Let D host compiler output 64bit object files")
        append("-m64" DDMD_DFLAGS)
    else()
        message(STATUS "Let D host compiler output 32bit COFF object files")
        if(${D_COMPILER_ID} STREQUAL "DigitalMars")
            append("-m32mscoff" DDMD_DFLAGS)
        else()
            append("-m32" DDMD_DFLAGS)
        endif()
    endif()

    if(${D_COMPILER_ID} STREQUAL "DigitalMars" AND (MSVC_VERSION GREATER 1800)) # VS 2015+
        append("-Llegacy_stdio_definitions.lib" DDMD_DFLAGS)
    endif()

    # Link against the static MSVC runtime; CMake's C(++) flags apparently default to the dynamic one.
    # Host DMD/LDMD already defaults to linking against the static MSVC runtime.
    if(${LLVM_CXXFLAGS} MATCHES "(^| )/MDd?( |$)")
        message(FATAL_ERROR "LLVM must be built with CMake option LLVM_USE_CRT_<CMAKE_BUILD_TYPE>=MT[d]")
    endif()
    set(llvm_ob_flag)
    string(REGEX MATCH "/Ob[0-2]" llvm_ob_flag "${LLVM_CXXFLAGS}")
    foreach(flag_var
            CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        # CMake defaults to /W3, LLVM uses /W4 => MS compiler warns about overridden option.
        # Simply replace with /W4.
        string(REGEX REPLACE "/W[0-3]" "/W4" ${flag_var} "${${flag_var}}")
        # Some CMake configs default to /Ob1, LLVM uses /Ob2. Replace with LLVM's option.
        if(NOT llvm_ob_flag STREQUAL "")
            string(REGEX REPLACE "/Ob[0-2]" "${llvm_ob_flag}" ${flag_var} "${${flag_var}}")
        endif()
    endforeach()
endif()

append("-J${PROJECT_SOURCE_DIR}/res" DDMD_DFLAGS) # Needed for importing text files
string(STRIP "${DDMD_DFLAGS}" DDMD_DFLAGS)

# Use separate compiler flags for the frontend and for the LDC-specific parts,
# as enabling warnings on the DMD frontend only leads to a lot of clutter in
# the output (LLVM_CXXFLAGS sometimes already includes -Wall).
set(LDC_CXXFLAGS)
if(CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
    if(NOT MSVC) # not for Windows-clang
        append("-Wall -Wextra" LDC_CXXFLAGS)
    endif()
    # Disable some noisy warnings:
    #  * -Wunused-parameter triggers for LLVM headers
    #  * -Wmissing-field-initializer leads to reams of warnings in gen/asm-*.h
    #  * -Wnon-virtual-dtor is something Walter has declined to let us fix upstream
    #    and it triggers for the visitors we need in our glue code
    #  * -Wpedantic warns on trailing commas in initializer lists and casting
    #    function pointers to void*.
    #  * -Wgnu-anonymous-struct and -Wnested-anon-types trigger for tokens.h.
    #  * -Wgnu-redeclared-enum triggers for various frontend headers.
    #  * -Wunused-private-field triggers for expression.h.
    append("-Wno-unused-parameter -Wno-missing-field-initializers -Wno-non-virtual-dtor" LDC_CXXFLAGS)
    if ((${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
        append("-Wno-gnu-anonymous-struct -Wno-nested-anon-types -Wno-gnu-redeclared-enum -Wno-unused-private-field" LDC_CXXFLAGS)
        # clang trying to eagerly anticipate linker errors wrt. static class template
        # members leads to false positives (e.g., instantiated/defined in D):
        # 'instantiation of variable required here, but no definition is available'
        append("-Wno-undefined-var-template" LDC_CXXFLAGS)
    endif()
    if(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_C_COMPILER_VERSION VERSION_LESS "4.7.0")
        append("-Wno-pedantic" LDC_CXXFLAGS)
    endif()
endif()

if(NOT WIN32 AND NOT CYGWIN)
    # Unify symbol visibility with LLVM to silence linker warning "direct access in function X to global
    # weak symbol Y means the weak symbol cannot be overridden at runtime. This was likely caused by
    # different translation units being compiled with different visibility settings."
    # See LLVM's cmake/modules/HandleLLVMOptions.cmake.
    check_cxx_compiler_flag("-fvisibility-inlines-hidden" SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
    if (${SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG})
        append("-fvisibility-inlines-hidden" LDC_CXXFLAGS)
    endif()
endif()

if(MSVC)
    # Remove flags here, for exceptions and RTTI.
    # CL.EXE complains to override flags like "/GR /GR-".
    string(REGEX REPLACE "(^| )[/-]EH[-cs]*( |$)" "\\2" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "(^| )[/-]GR-?( |$)" "\\2" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    append("/GR- /EHs-c-" CMAKE_CXX_FLAGS)
    append("/D_HAS_EXCEPTIONS=0" CMAKE_CXX_FLAGS)

    # disable warning C4201: nonstandard extension used: nameless struct/union
    append("/wd4201" LDC_CXXFLAGS)
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
    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
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
    # LLVM 5.0+ requires diaguids.lib from MS Debug Interface Access SDK
    if(NOT (LDC_LLVM_VER LESS 500))
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            list(APPEND LLVM_LDFLAGS "$ENV{VSINSTALLDIR}DIA SDK\\lib\\amd64\\diaguids.lib")
        else()
            list(APPEND LLVM_LDFLAGS "$ENV{VSINSTALLDIR}DIA SDK\\lib\\diaguids.lib")
        endif()
    endif()
else()
    separate_arguments(LLVM_LDFLAGS UNIX_COMMAND "${LLVM_LDFLAGS}")
endif()

# Suppress superfluous randlib warnings about "*.a" having no symbols on MacOSX.
if (APPLE)
    set(CMAKE_C_ARCHIVE_CREATE   "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_C_ARCHIVE_FINISH   "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
    set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
endif()

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
    get_git_head_revision(REFSPEC HASH FALSE)
    if(NOT HASH STREQUAL "GITDIR-NOTFOUND")
        # Append git hash to LDC_VERSION
        string(SUBSTRING "${HASH}" 0 7 LDC_VERSION_HASH)
        set(LDC_VERSION "${LDC_VERSION}-git-${LDC_VERSION_HASH}")

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
file(GLOB_RECURSE FE_SRC_D dmd/*.d)
file(GLOB_RECURSE FE_HDR   dmd/*.h)
file(GLOB_RECURSE FE_RES   res/*.*)
file(GLOB_RECURSE GEN_SRC gen/*.cpp)
file(GLOB_RECURSE GEN_HDR gen/*.h)
file(GLOB_RECURSE GEN_SRC_D gen/*.d)
file(GLOB_RECURSE DRV_SRC_D driver/*.d)
file(GLOB_RECURSE IR_SRC_D ir/*.d)
file(GLOB IR_SRC ir/*.cpp)
file(GLOB IR_HDR ir/*.h)
set(DRV_SRC
    driver/args.cpp
    driver/cache.cpp
    driver/cl_options.cpp
    driver/cl_options_instrumentation.cpp
    driver/cl_options_sanitizers.cpp
    driver/cl_options-llvm.cpp
    driver/codegenerator.cpp
    driver/configfile.cpp
    driver/dcomputecodegenerator.cpp
    driver/exe_path.cpp
    driver/targetmachine.cpp
    driver/toobj.cpp
    driver/tool.cpp
    driver/archiver.cpp
    driver/linker.cpp
    driver/linker-gcc.cpp
    driver/linker-msvc.cpp
    driver/main.cpp
    driver/plugins.cpp
    ${CMAKE_BINARY_DIR}/driver/ldc-version.cpp
)
set(DRV_HDR
    driver/args.h
    driver/cache.h
    driver/cache_pruning.h
    driver/cl_options.h
    driver/cl_options_instrumentation.h
    driver/cl_options_sanitizers.h
    driver/cl_options-llvm.h
    driver/codegenerator.h
    driver/configfile.h
    driver/dcomputecodegenerator.h
    driver/exe_path.h
    driver/ldc-version.h
    driver/archiver.h
    driver/linker.h
    driver/plugins.h
    driver/targetmachine.h
    driver/toobj.h
    driver/tool.h
)
# exclude man.d from ldc (only required by ldmd)
list(REMOVE_ITEM FE_SRC_D
    ${PROJECT_SOURCE_DIR}/dmd/root/man.d
)
set(LDC_CXX_SOURCE_FILES
    ${FE_HDR}
    ${GEN_SRC}
    ${GEN_HDR}
    ${IR_SRC}
    ${IR_HDR}
)
set(LDC_D_SOURCE_FILES
    ${FE_SRC_D}
    ${GEN_SRC_D}
    ${DRV_SRC_D}
    ${IR_SRC_D}
)

source_group("Source Files\\dmd" FILES ${FE_SRC_D})
source_group("Header Files\\dmd" FILES ${FE_HDR})
source_group("Source Files\\gen" FILES ${GEN_SRC} ${GEN_SRC_D})
source_group("Header Files\\gen" FILES ${GEN_HDR})
source_group("Source Files\\ir" FILES ${IR_SRC} ${IR_SRC_D})
source_group("Header Files\\ir" FILES ${IR_HDR})
source_group("Source Files" FILES ${DRV_SRC_D})


#
# Configure the build system to use LTO and/or PGO while building LDC
#
include(HandleLTOPGOBuildOptions)

#
# Enable Dynamic compilation if supported for this platform and LLVM version.
#
set(LDC_DYNAMIC_COMPILE "AUTO" CACHE STRING "Support dynamic compilation (True|False). Enabled by default.")
option(LDC_DYNAMIC_COMPILE_USE_CUSTOM_PASSES "Use custom LDC passes in jit" ON)
if(LDC_DYNAMIC_COMPILE STREQUAL "AUTO")
    set(LDC_DYNAMIC_COMPILE False)  # must be a valid Python boolean constant (case sensitive)
    if (NOT (LDC_LLVM_VER LESS 500))
        set(LDC_DYNAMIC_COMPILE True)
        add_definitions(-DLDC_DYNAMIC_COMPILE)
        add_definitions(-DLDC_DYNAMIC_COMPILE_API_VERSION=3)
    endif()
endif()
message(STATUS "Building LDC with dynamic compilation support: ${LDC_DYNAMIC_COMPILE} (LDC_DYNAMIC_COMPILE=${LDC_DYNAMIC_COMPILE})")

#
# Includes, defines.
#

include_directories(. dmd)
append("-I${PROJECT_SOURCE_DIR}" DDMD_DFLAGS)
append("-I${PROJECT_BINARY_DIR}" DDMD_DFLAGS)

append("-version=IN_LLVM" DDMD_DFLAGS)
append("-DIN_LLVM" LDC_CXXFLAGS)
append("-DOPAQUE_VTBLS" LDC_CXXFLAGS)
# Predefine LDC_INSTALL_PREFIX as raw string literal, requiring shell + CMake escaping.
# E.g., for CMAKE_INSTALL_PREFIX=`C:\dir with space`:
#   g++ "-DLDC_INSTALL_PREFIX=R\"(C:\dir with space)\"" ...
#   => LDC_INSTALL_PREFIX defined as `R"(C:\dir with space)"`
append("\"-DLDC_INSTALL_PREFIX=R\\\"(${CMAKE_INSTALL_PREFIX})\\\"\"" LDC_CXXFLAGS)
append("-DLDC_LLVM_VER=${LDC_LLVM_VER}" LDC_CXXFLAGS)
append("\"-DLDC_LIBDIR_SUFFIX=R\\\"(${LIB_SUFFIX})\\\"\"" LDC_CXXFLAGS)
append("-DLDC_HOST_${D_COMPILER_ID}=1" LDC_CXXFLAGS)
append("-DLDC_HOST_FE_VER=${D_COMPILER_FE_VERSION}" LDC_CXXFLAGS)

if(GENERATE_OFFTI)
    append("-DGENERATE_OFFTI" LDC_CXXFLAGS)
endif()

#
# LLD integration (requires headers & libs)
#
if(NOT DEFINED LDC_WITH_LLD)
    if(((LDC_LLVM_VER LESS 600) AND (EXISTS "${LLVM_INCLUDE_DIRS}/lld/Driver/Driver.h")) OR
       ((NOT LDC_LLVM_VER LESS 600) AND (EXISTS "${LLVM_INCLUDE_DIRS}/lld/Common/Driver.h")))
        set(LDC_WITH_LLD ON)
    else()
        set(LDC_WITH_LLD OFF)
    endif()
endif()
if(LDC_WITH_LLD)
    append("-DLDC_WITH_LLD" LDC_CXXFLAGS)
endif()
message(STATUS "Building LDC with integrated LLD: ${LDC_WITH_LLD} (LDC_WITH_LLD=${LDC_WITH_LLD})")

#
# Enable building with riscv-llvm, for full RISC-V support.
#
option(RISCV_LLVM_DEV "full RISC-V support with riscv-llvm")
mark_as_advanced(RISCV_LLVM_DEV)
if(RISCV_LLVM_DEV)
    append("-DRISCV_LLVM_DEV" LDC_CXXFLAGS)
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
# Enable instrumentation for code coverage analysis
#
set(TEST_COVERAGE OFF CACHE BOOL "instrument compiler for code coverage analysis")
if(TEST_COVERAGE)
    if(CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
        append("-O0 -g -fprofile-arcs -ftest-coverage" EXTRA_CXXFLAGS)
        list(APPEND LLVM_LDFLAGS "-lgcov")
    else()
        message(WARNING "Coverage testing is not available.")
    endif()
endif()

#
# Set up the main ldc/ldc2 target.
#
set(LDC_LIB_LANGUAGE CXX)
if(BUILD_SHARED)
    set(LDC_LIB_TYPE SHARED)
else()
    set(LDC_LIB_TYPE STATIC)
    if("${D_COMPILER_ID}" STREQUAL "LDMD" AND D_COMPILER_FE_VERSION GREATER 2074)
        # Define a 'HOST_D' CMake linker language for the static LDCShared
        # library, using the host ldmd2 compiler ≥ v1.5 as archiver, which
        # supports LTO objects and cross-archiving.
        set(CMAKE_HOST_D_CREATE_STATIC_LIBRARY "${D_COMPILER} -lib ${D_COMPILER_FLAGS} ${DDMD_DFLAGS} -of=<TARGET> <OBJECTS>")
        set(LDC_LIB_LANGUAGE HOST_D)
    endif()
endif()

set(LDC_LIB LDCShared)
set(LDC_LIB_EXTRA_SOURCES "")
if(MSVC_IDE)
    set(LDC_LIB_EXTRA_SOURCES ${LDC_D_SOURCE_FILES})
endif()
add_library(${LDC_LIB} ${LDC_LIB_TYPE} ${LDC_CXX_SOURCE_FILES} ${DRV_SRC} ${DRV_HDR} ${LDC_LIB_EXTRA_SOURCES})
set_target_properties(
    ${LDC_LIB} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}
    ARCHIVE_OUTPUT_NAME ldc
    LIBRARY_OUTPUT_NAME ldc
    RUNTIME_OUTPUT_NAME ldc
    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS} ${EXTRA_CXXFLAGS}"
    LINKER_LANGUAGE ${LDC_LIB_LANGUAGE}
    LINK_FLAGS "${SANITIZE_LDFLAGS}"
)
# LDFLAGS should actually be in target property LINK_FLAGS, but this works, and gets around linking problems
target_link_libraries(${LDC_LIB} ${LLVM_LIBRARIES} ${LLVM_LDFLAGS})
if(WIN32)
    target_link_libraries(${LDC_LIB} imagehlp psapi)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(${LDC_LIB} dl)
endif()

set(LDC_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${LDC_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX})
set(LDMD_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${LDMD_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX})

# Figure out how to link the main LDC executable, for which we need to take the
# LLVM flags into account.
set(LDC_LINKERFLAG_LIST ${SANITIZE_LDFLAGS} ${LLVM_LIBRARIES} ${LLVM_LDFLAGS})
if(MSVC)
    # Issue 1297 – set LDC's stack to 8 MiB like on Linux and Mac (default: 1 MiB).
    list(APPEND LDC_LINKERFLAG_LIST "/STACK:8388608")
    # VS 2017+: Use undocumented /NOOPTTLS MS linker switch to keep on emitting
    # a .tls section. Required for older host druntime versions, otherwise the
    # GC TLS ranges are garbage starting with VS 2017 Update 15.3.
    if(MSVC_VERSION GREATER 1900 AND D_COMPILER_FE_VERSION LESS 2076)
        list(APPEND LDC_LINKERFLAG_LIST "/NOOPTTLS")
    endif()
endif()
if(LDC_WITH_LLD)
    if(NOT (LDC_LLVM_VER LESS 1000))
        if(MSVC)
            list(APPEND LDC_LINKERFLAG_LIST LLVMSymbolize.lib)
        else()
            set(LDC_LINKERFLAG_LIST -lLLVMSymbolize ${LDC_LINKERFLAG_LIST})
        endif()
    endif()
    # ELF, Mach-O, MinGW and WebAssembly formats supported since LLD 6.0.0, otherwise just Windows COFF
    if(NOT (LDC_LLVM_VER LESS 600))
        if(MSVC)
            list(APPEND LDC_LINKERFLAG_LIST lldDriver.lib lldMinGW.lib lldCOFF.lib lldELF.lib lldMachO.lib lldWasm.lib lldYAML.lib lldReaderWriter.lib lldCommon.lib lldCore.lib)
        else()
            set(LDC_LINKERFLAG_LIST -llldDriver -llldMinGW -llldCOFF -llldELF -llldMachO -llldWasm -llldYAML -llldReaderWriter -llldCommon -llldCore ${LDC_LINKERFLAG_LIST})
        endif()
        if(APPLE) # bug, should be fixed in LLVM 6.0.1
            list(APPEND LDC_LINKERFLAG_LIST -lxml2)
        endif()
    else()
        if(MSVC)
            list(APPEND LDC_LINKERFLAG_LIST lldCOFF.lib lldCore.lib lldDriver.lib)
        else()
            set(LDC_LINKERFLAG_LIST -llldCOFF -llldCore -llldDriver ${LDC_LINKERFLAG_LIST})
        endif()
    endif()
endif()

# Plugin support
if(UNIX)
    set(LDC_ENABLE_PLUGINS_DEFAULT ON)
else()
    set(LDC_ENABLE_PLUGINS_DEFAULT OFF)
endif()
set(LDC_ENABLE_PLUGINS ${LDC_ENABLE_PLUGINS_DEFAULT} CACHE BOOL "Build LDC with plugin support (increases binary size)")
if(LDC_ENABLE_PLUGINS)
    add_definitions(-DLDC_ENABLE_PLUGINS)

    if(APPLE)
        # No extra link flags needed.
    elseif(UNIX)
        # For plugin support, we need to link with --export-dynamic on Unix.
        # Make sure the linker supports --export-dynamic (on Solaris it is not supported and also not needed).
        CHECK_LINK_FLAG("--export-dynamic" LINKER_ACCEPTS_EXPORT_DYNAMIC_FLAG)
        if(LINKER_ACCEPTS_EXPORT_DYNAMIC_FLAG)
            set(LDC_LINKERFLAG_LIST "${LDC_LINKERFLAG_LIST};-Wl,--export-dynamic")
        endif()
    endif()
endif()
message(STATUS "Building LDC with plugin support: ${LDC_ENABLE_PLUGINS} (LDC_ENABLE_PLUGINS=${LDC_ENABLE_PLUGINS})")

set(LDC_LINK_MANUALLY OFF)
if(UNIX AND (CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")))
    # On Unix-like systems, DMD and LDC will use the C compiler for linking, but
    # will pass on -L options prefixed by -Xlinker to directly forward them to
    # the underlying ld. Since there are some flags the GCC driver handles itself
    # rather than passing them to ld, we cannot just directly translate
    # LDC_LINKERFLAG_LIST to -L options. To be able to handle general linker flags,
    # we manually invoke the linker instead of using the D compiler to do so.
    set(LDC_LINK_MANUALLY ON)

    if(NOT DEFINED D_LINKER_ARGS)
        include(ExtractDMDSystemLinker)
        message(STATUS "Host D compiler linker program: ${D_LINKER_COMMAND}")
        message(STATUS "Host D compiler linker flags: ${D_LINKER_ARGS}")
    endif()
endif()

# CONFIG generator expressions need to be repeated due to https://cmake.org/Bug/view.php?id=14353
if(MSVC_IDE)
    translate_linker_args(linker_args translated_linker_args)
    separate_arguments(LDC_FLAG_LIST WINDOWS_COMMAND "${D_COMPILER_FLAGS} ${DDMD_DFLAGS} ${DDMD_LFLAGS} ${translated_linker_args}")
    add_custom_command(
        OUTPUT ${LDC_EXE_FULL}
        COMMAND ${D_COMPILER} -L$<TARGET_LINKER_FILE:${LDC_LIB}> ${LDC_FLAG_LIST} -of${LDC_EXE_FULL} ${LDC_D_SOURCE_FILES} $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:-g> $<$<NOT:$<CONFIG:Debug>>:-O> $<$<NOT:$<CONFIG:Debug>>:-inline> $<$<NOT:$<CONFIG:Debug>>:-release>
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        DEPENDS ${LDC_D_SOURCE_FILES} ${FE_RES} ${LDC_LIB}
    )
else()
    build_d_executable(
        "${LDC_EXE}"
        "${LDC_EXE_FULL}"
        "${LDC_D_SOURCE_FILES}"
        ""
        "${LDC_LINKERFLAG_LIST}"
        "${FE_RES}"
        "${LDC_LIB}"
        ${COMPILE_D_MODULES_SEPARATELY}
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

set_source_files_properties(driver/args.cpp driver/exe_path.cpp driver/ldmd.cpp driver/response.cpp PROPERTIES
    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS}"
    COMPILE_DEFINITIONS LDC_EXE_NAME="${LDC_EXE_NAME}"
)

add_library(LDMD_CXX_LIB ${LDC_LIB_TYPE} driver/args.cpp driver/exe_path.cpp driver/ldmd.cpp driver/response.cpp driver/args.h driver/exe_path.h)
set_target_properties(
    LDMD_CXX_LIB PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}
    ARCHIVE_OUTPUT_NAME ldmd
    LIBRARY_OUTPUT_NAME ldmd
)
set(LDMD_D_SOURCE_FILES ${PROJECT_SOURCE_DIR}/dmd/root/man.d ${PROJECT_SOURCE_DIR}/driver/main.d)
build_d_executable(
    "${LDMD_EXE}"
    "${LDMD_EXE_FULL}"
    "${LDMD_D_SOURCE_FILES}"
    ""
    "${LDC_LINKERFLAG_LIST}"
    ""
    "LDMD_CXX_LIB"
    ${COMPILE_D_MODULES_SEPARATELY}
)

# Little helper.
function(copy_and_rename_file source_path target_path)
    get_filename_component(source_name ${source_path} NAME)
    get_filename_component(target_dir ${target_path} DIRECTORY)
    file(MAKE_DIRECTORY ${target_dir})
    # don't preserve source file permissions, see https://github.com/ldc-developers/ldc/issues/2337
    file(COPY ${source_path} DESTINATION ${target_dir} NO_SOURCE_PERMISSIONS)
    file(RENAME ${target_dir}/${source_name} ${target_path})
endfunction()

#
# Locate LLVM's LTO binary and use it (LLVM >= 3.9)
#
if(APPLE)
    set(LDC_INSTALL_LTOPLUGIN_DEFAULT ON)
else()
    set(LDC_INSTALL_LTOPLUGIN_DEFAULT OFF)
endif()
set(LDC_INSTALL_LTOPLUGIN ${LDC_INSTALL_LTOPLUGIN_DEFAULT} CACHE BOOL "Copy/install the LTO plugin from the LLVM package when available (LLVM >= 3.9).")
if (LDC_INSTALL_LTOPLUGIN)
    if(APPLE)
        set(LLVM_LTO_BINARY ${LLVM_LIBRARY_DIRS}/libLTO.dylib)
        set(LDC_LTO_BINARY_NAME libLTO-ldc.dylib)
    elseif(UNIX)
        set(LLVM_LTO_BINARY ${LLVM_LIBRARY_DIRS}/LLVMgold.so)
        set(LDC_LTO_BINARY_NAME LLVMgold-ldc.so)
    endif()
    if(EXISTS ${LLVM_LTO_BINARY})
        message(STATUS "Also installing LTO binary: ${LLVM_LTO_BINARY} (LDC_INSTALL_LTOPLUGIN=${LDC_INSTALL_LTOPLUGIN})")
        copy_and_rename_file(${LLVM_LTO_BINARY} ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}/${LDC_LTO_BINARY_NAME})
        install(PROGRAMS ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}/${LDC_LTO_BINARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
    else()
        message(STATUS "Not found: ${LLVM_LTO_BINARY}")
    endif()
endif()

#
# Locate ASan and other LLVM compiler-rt libraries, and copy them to our lib folder
# Location is LLVM_LIBRARY_DIRS/clang/<version>/lib/<OS>/ , for example LLVM_LIBRARY_DIRS/clang/4.0.0/lib/darwin/
#
if(APPLE OR WIN32)
    set(LDC_INSTALL_LLVM_RUNTIME_LIBS_DEFAULT ON)
else()
    set(LDC_INSTALL_LLVM_RUNTIME_LIBS_DEFAULT OFF)
endif()
set(LDC_INSTALL_LLVM_RUNTIME_LIBS ${LDC_INSTALL_LLVM_RUNTIME_LIBS_DEFAULT} CACHE BOOL "Copy/install LLVM compiler-rt libraries (ASan, libFuzzer, ...) from LLVM/Clang into LDC lib dir when available.")
function(copy_compilerrt_lib llvm_lib_name ldc_lib_name fixup_dylib)
    set(llvm_lib_path ${LLVM_LIBRARY_DIRS}/clang/${LLVM_VERSION_BASE_STRING}/lib/${llvm_lib_name})
    if(EXISTS ${llvm_lib_path})
        message(STATUS "Copying runtime library: ${llvm_lib_path} --> ${ldc_lib_name}")
        set(ldc_lib_path ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}/${ldc_lib_name})
        copy_and_rename_file(${llvm_lib_path} ${ldc_lib_path})
        if (fixup_dylib)
            execute_process(COMMAND install_name_tool -id @rpath/${ldc_lib_name} ${ldc_lib_path})
        endif()
        install(FILES ${ldc_lib_path} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
    else()
        message(STATUS "Not found: ${llvm_lib_path}")
    endif()
endfunction()
include(CheckTypeSize)
check_type_size(void* ptr_size)
if (LDC_INSTALL_LLVM_RUNTIME_LIBS)
    # Locate LLVM sanitizer runtime libraries, and copy them to our lib folder
    # Note: libFuzzer is part of compiler-rt version >= 6.0, but was part of LLVM =< 5.0

    if(APPLE)
        copy_compilerrt_lib("darwin/libclang_rt.asan_osx_dynamic.dylib" "libldc_rt.asan.dylib" TRUE)
        copy_compilerrt_lib("darwin/libclang_rt.osx.a"                  "libldc_rt.builtins.a" FALSE)
        copy_compilerrt_lib("darwin/libclang_rt.profile_osx.a"          "libldc_rt.profile.a"  FALSE)
        if(NOT (LDC_LLVM_VER LESS 600))
            copy_compilerrt_lib("darwin/libclang_rt.fuzzer_osx.a"       "libldc_rt.fuzzer.a"   FALSE)
            copy_compilerrt_lib("darwin/libclang_rt.xray_osx.a"         "libldc_rt.xray.a"     FALSE)
        endif()
        if(NOT (LDC_LLVM_VER LESS 700))
            copy_compilerrt_lib("darwin/libclang_rt.xray-basic_osx.a"      "libldc_rt.xray-basic.a"     FALSE)
            copy_compilerrt_lib("darwin/libclang_rt.xray-fdr_osx.a"        "libldc_rt.xray-fdr.a"       FALSE)
            copy_compilerrt_lib("darwin/libclang_rt.xray-profiling_osx.a"  "libldc_rt.xray-profiling.a" FALSE)
        endif()
    elseif(UNIX)
        set(LDC_INSTALL_LLVM_RUNTIME_LIBS_OS   "linux"  CACHE STRING "Non-Mac Posix: OS used as directory name for the compiler-rt source libraries, e.g., 'freebsd'.")
        set(LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH "x86_64" CACHE STRING "Non-Mac Posix: architecture used as libname suffix for the compiler-rt source libraries, e.g., 'aarch64'.")

        copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.asan-${LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH}.a"       "libldc_rt.asan.a"     FALSE)
        copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.builtins-${LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH}.a"   "libldc_rt.builtins.a" FALSE)
        copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.profile-${LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH}.a"    "libldc_rt.profile.a"  FALSE)
        if(NOT (LDC_LLVM_VER LESS 500))
            copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.xray-${LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH}.a"   "libldc_rt.xray.a"     FALSE)
        endif()
        if(NOT (LDC_LLVM_VER LESS 600))
            copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.fuzzer-${LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH}.a" "libldc_rt.fuzzer.a"   FALSE)
        endif()
        if(NOT (LDC_LLVM_VER LESS 700))
            copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.xray-basic-${LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH}.a"      "libldc_rt.xray-basic.a"     FALSE)
            copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.xray-fdr-${LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH}.a"        "libldc_rt.xray-fdr.a"       FALSE)
            copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.xray-profiling-${LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH}.a"  "libldc_rt.xray-profiling.a" FALSE)
        endif()
    elseif(WIN32)
        set(compilerrt_arch_suffix "x86_64")
        if(${ptr_size} EQUAL 4)
            set(compilerrt_arch_suffix "i386")
        endif()
        copy_compilerrt_lib("windows/clang_rt.asan-${compilerrt_arch_suffix}.lib"       "ldc_rt.asan.lib"     FALSE)
        copy_compilerrt_lib("windows/clang_rt.builtins-${compilerrt_arch_suffix}.lib"   "ldc_rt.builtins.lib" FALSE)
        copy_compilerrt_lib("windows/clang_rt.profile-${compilerrt_arch_suffix}.lib"    "ldc_rt.profile.lib"  FALSE)
        if(NOT (LDC_LLVM_VER LESS 600))
            copy_compilerrt_lib("windows/clang_rt.fuzzer-${compilerrt_arch_suffix}.lib" "ldc_rt.fuzzer.lib"   FALSE)
            copy_compilerrt_lib("windows/clang_rt.xray-${compilerrt_arch_suffix}.lib"   "ldc_rt.xray.lib"     FALSE)
        endif()
        if(NOT (LDC_LLVM_VER LESS 700))
            copy_compilerrt_lib("windows/clang_rt.xray-basic-${compilerrt_arch_suffix}.lib"      "ldc_rt.xray-basic.lib"     FALSE)
            copy_compilerrt_lib("windows/clang_rt.xray-fdr-${compilerrt_arch_suffix}.lib"        "ldc_rt.xray-fdr.lib"       FALSE)
            copy_compilerrt_lib("windows/clang_rt.xray-profiling-${compilerrt_arch_suffix}.lib"  "ldc_rt.xray-profiling.lib" FALSE)
        endif()
    endif()

    if(LDC_LLVM_VER LESS 600)
        set(LLVM_LIBFUZZER_PATH ${LLVM_LIBRARY_DIRS}/libFuzzer.a)
        if(EXISTS ${LLVM_LIBFUZZER_PATH})
            message(STATUS "Copying libFuzzer library: ${LLVM_LIBFUZZER_PATH} --> libFuzzer.a")
            file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX})
            file(COPY ${LLVM_LIBFUZZER_PATH} DESTINATION ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX})
            install(FILES ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}/libFuzzer.a DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
        else()
            message(STATUS "Not found: ${LLVM_LIBFUZZER_PATH}")
        endif()
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
set(LDC_UNITTEST_EXE ${LDC_EXE}-unittest)
set(LDC_UNITTEST_EXE_NAME ${PROGRAM_PREFIX}${LDC_UNITTEST_EXE}${PROGRAM_SUFFIX})
set(LDC_UNITTEST_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${LDC_UNITTEST_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX})
build_d_executable(
    "${LDC_UNITTEST_EXE}"
    "${LDC_UNITTEST_EXE_FULL}"
    "${LDC_D_SOURCE_FILES}"
    "-unittest"
    "${LDC_LINKERFLAG_LIST}"
    ""
    "${LDC_LIB}"
    ON # always build separately (faster with parallelization)
)
set_target_properties("${LDC_UNITTEST_EXE}" PROPERTIES EXCLUDE_FROM_ALL ON)
add_test(NAME build-ldc2-unittest COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ldc2-unittest)
add_test(NAME ldc2-unittest COMMAND ${LDC_UNITTEST_EXE_FULL} --version)
set_tests_properties(ldc2-unittest PROPERTIES DEPENDS build-ldc2-unittest)

if(EXISTS "${PROJECT_SOURCE_DIR}/runtime/druntime/src/object.d")
    add_subdirectory(runtime)
else()
    message(STATUS "Runtime file runtime/druntime/src/object.d not found, will build ldc binaries but not the standard library.")
endif()
if(D_VERSION EQUAL 2)
    add_subdirectory(tests/d2)
endif()
add_subdirectory(tests)

# ldc-build-runtime tool
configure_file(${PROJECT_SOURCE_DIR}/runtime/ldc-build-runtime.d.in ${PROJECT_BINARY_DIR}/ldc-build-runtime.d @ONLY)
set(LDC_BUILD_RUNTIME_EXE ldc-build-runtime)
set(LDC_BUILD_RUNTIME_EXE_NAME ${PROGRAM_PREFIX}${LDC_BUILD_RUNTIME_EXE}${PROGRAM_SUFFIX})
set(LDC_BUILD_RUNTIME_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${LDC_BUILD_RUNTIME_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX})
build_d_executable(
    "${LDC_BUILD_RUNTIME_EXE}"
    "${LDC_BUILD_RUNTIME_EXE_FULL}"
    "${PROJECT_BINARY_DIR}/ldc-build-runtime.d"
    ""
    ""
    "${PROJECT_SOURCE_DIR}/runtime/ldc-build-runtime.d.in"
    ""
    ${COMPILE_D_MODULES_SEPARATELY}
)

#
# Install target.
#

install(PROGRAMS ${LDC_EXE_FULL} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
install(PROGRAMS ${LDMD_EXE_FULL} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
install(PROGRAMS ${LDC_BUILD_RUNTIME_EXE_FULL} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
if(${BUILD_SHARED})
    # For now, only install libldc if explicitly building the shared library.
    # While it might theoretically be possible to use LDC as a static library
    # as well, for the time being this just bloats the normal packages.
    install(TARGETS ${LDC_LIB} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
endif()
install(FILES ${PROJECT_BINARY_DIR}/bin/${LDC_EXE}_install.conf DESTINATION ${CONF_INST_DIR} RENAME ${LDC_EXE}.conf)

if(MSVC)
    file(COPY vcbuild/ DESTINATION ${PROJECT_BINARY_DIR}/bin FILES_MATCHING PATTERN "*.bat")
    install(DIRECTORY vcbuild/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.bat")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    if(NOT DEFINED BASH_COMPLETION_COMPLETIONSDIR)
        find_package(bash-completion QUIET)
        if((NOT BASH_COMPLETION_FOUND) OR (NOT BASH_COMPLETION_PREFIX STREQUAL CMAKE_INSTALL_PREFIX))
            set(BASH_COMPLETION_COMPLETIONSDIR "${CONF_INST_DIR}/bash_completion.d")
            if(LINUX_DISTRIBUTION_IS_GENTOO AND CMAKE_INSTALL_PREFIX STREQUAL "/usr")
                set(BASH_COMPLETION_COMPLETIONSDIR "/usr/share/bash-completion")
            endif()
        endif()
    endif()
    install(DIRECTORY packaging/bash_completion.d/ DESTINATION ${BASH_COMPLETION_COMPLETIONSDIR})
endif()

#
# Packaging
#

include (CMakeCPack.cmake)
include (CPack)
