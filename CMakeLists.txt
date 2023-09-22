cmake_minimum_required(VERSION 3.4.3)
if(POLICY CMP0025)
    cmake_policy(SET CMP0025 NEW)
endif()
if(${CMAKE_VERSION} VERSION_GREATER "3.26.9")
    # Prevent implicit dependencies for custom commands, e.g.,
    # `obj/ldc2.o` depending on `lib/libldc.a` with LDC_LINK_MANUALLY=ON.
    # Only supported since CMake v3.27 unfortunately.
    set(CMAKE_ADD_CUSTOM_COMMAND_DEPENDS_EXPLICIT_ONLY ON)
endif()

project(ldc)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

include(FindDCompiler)
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

find_package(LLVM 11.0 REQUIRED
    all-targets analysis asmparser asmprinter bitreader bitwriter codegen core
    debuginfodwarf debuginfomsf debuginfopdb demangle
    instcombine ipo instrumentation irreader libdriver linker lto mc
    mcdisassembler mcparser objcarcopts object option profiledata scalaropts
    selectiondag support tablegen target transformutils vectorize
    windowsdriver windowsmanifest ${EXTRA_LLVM_MODULES})
math(EXPR LDC_LLVM_VER ${LLVM_VERSION_MAJOR}*100+${LLVM_VERSION_MINOR})
message(STATUS "Using LLVM Version ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}")
# Remove LLVMTableGen library from list of libraries
string(REGEX MATCH "[^;]*LLVMTableGen[^;]*" LLVM_TABLEGEN_LIBRARY "${LLVM_LIBRARIES}")
string(REGEX REPLACE "[^;]*LLVMTableGen[^;]*;?" "" LLVM_LIBRARIES "${LLVM_LIBRARIES}")

# Information about which targets LLVM was built to target
foreach(LLVM_SUPPORTED_TARGET ${LLVM_TARGETS_TO_BUILD})
    add_definitions("-DLDC_LLVM_SUPPORTED_TARGET_${LLVM_SUPPORTED_TARGET}=1")
endforeach()

# Set MLIR support variables if it is found.
# FIXME: LLVM 14+ (`mlir::OwningModuleRef` replacement)
if(NOT LDC_WITH_MLIR STREQUAL "OFF" AND LDC_LLVM_VER LESS 1400)
    include(FindMLIR)
    if(MLIR_FOUND)
        message(STATUS "-- Building LDC with MLIR support (${MLIR_ROOT_DIR})")
        include_directories(${MLIR_INCLUDE_DIR})
        add_definitions("-DLDC_MLIR_ENABLED")
        set(LLVM_LIBRARIES "${MLIR_LIBRARIES}" ${LLVM_LIBRARIES})
    else()
        message(STATUS "-- Building LDC without MLIR support: not found")
    endif()
endif()

# Check and adapt for LLVMSPIRVLib (Khronos SPIRV-LLVM-Translator)
set(LLVM_SPIRV_FOUND OFF)
if (LDC_LLVM_VER LESS 1600)
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
            include_directories(${LLVM_SPIRV_INCLUDE_DIRS})
        else()
            set(LLVM_SPIRV_FOUND OFF)
        endif()
    endif()
endif()
if(LLVM_SPIRV_FOUND)
    message(STATUS "-- Building LDC with SPIR-V support")
    add_definitions("-DLDC_LLVM_SUPPORTED_TARGET_SPIRV=1")
else()
    message(STATUS "-- Building LDC without SPIR-V support: not found")
endif()
endif()

#
# Get info about used Linux distribution.
#
include(GetLinuxDistribution)


#
# Main configuration.
#

# Version information
set(LDC_VERSION "1.35.0") # May be overridden by git hash tag
set(DMDFE_MAJOR_VERSION   2)
set(DMDFE_MINOR_VERSION   105)
set(DMDFE_PATCH_VERSION   2)

set(DMD_VERSION ${DMDFE_MAJOR_VERSION}.${DMDFE_MINOR_VERSION}.${DMDFE_PATCH_VERSION})

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

set(LDC_ENABLE_ASSERTIONS "${LLVM_ENABLE_ASSERTIONS}" CACHE BOOL "Enable LDC assertions. Defaults to the LLVM assertions mode; overriding may cause LDC segfaults!")

# Allow user to specify mimalloc.o location, to be linked with `ldc2` only
set(ALTERNATIVE_MALLOC_O   ""     CACHE STRING "If specified, adds ALTERNATIVE_MALLOC_O object file to LDC link, to override the CRT malloc.")

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
set(DFLAGS_BASE "-wi")
set(DFLAGS_BUILD_TYPE "") # DFLAGS derived from CMAKE_BUILD_TYPE
if(NOT MSVC_IDE)
    # for multi-config builds, these options have to be added later to the custom command
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        append("-g" DFLAGS_BUILD_TYPE)
        if(${D_COMPILER_ID} STREQUAL "LDMD")
            append("-link-debuglib" DFLAGS_BASE)
        endif()
    elseif(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
        append("-g -O -inline -release" DFLAGS_BUILD_TYPE)
    else()
        # Default to a Release build type
        append("-O -inline -release" DFLAGS_BUILD_TYPE)
    endif()

    if(LDC_ENABLE_ASSERTIONS)
        string(REPLACE " -release" "" DFLAGS_BUILD_TYPE "${DFLAGS_BUILD_TYPE}")
    endif()
endif()

if(MSVC)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(STATUS "Let D host compiler output 64-bit object files")
        append("-m64" DFLAGS_BASE)
    else()
        message(STATUS "Let D host compiler output 32-bit COFF object files")
        if(${D_COMPILER_ID} STREQUAL "DigitalMars")
            append("-m32mscoff" DFLAGS_BASE)
        else()
            append("-m32" DFLAGS_BASE)
        endif()
    endif()

    if(${D_COMPILER_ID} STREQUAL "DigitalMars" AND (MSVC_VERSION GREATER 1800)) # VS 2015+
        append("-Llegacy_stdio_definitions.lib" DFLAGS_BASE)
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

# Use separate compiler flags for the frontend and for the LDC-specific parts,
# as enabling warnings on the DMD frontend only leads to a lot of clutter in
# the output (LLVM_CXXFLAGS sometimes already includes -Wall).
set(LDC_CXXFLAGS)
if(CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
    if(NOT MSVC) # not for Windows-clang
        append("-Wall -Wextra" LDC_CXXFLAGS)
    endif()
    # Disable some noisy warnings:
    #  * -Wunused-parameter and -Wcomment trigger for LLVM headers
    #  * -Wmissing-field-initializer leads to reams of warnings in gen/asm-*.h
    #  * -Wnon-virtual-dtor is something Walter has declined to let us fix upstream
    #    and it triggers for the visitors we need in our glue code
    #  * -Wpedantic warns on trailing commas in initializer lists and casting
    #    function pointers to void*.
    #  * -Wgnu-anonymous-struct and -Wnested-anon-types trigger for tokens.h.
    #  * -Wgnu-redeclared-enum triggers for various frontend headers.
    #  * -Wunused-private-field triggers for expression.h.
    append("-Wno-unused-parameter -Wno-comment -Wno-missing-field-initializers -Wno-non-virtual-dtor" LDC_CXXFLAGS)
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
    if(NOT MSVC_IDE) # apparently not needed for VS (and spaces in path are problematic)
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
configure_file(driver/ldc_version.d.in driver/ldc_version.d)

# Also add the header files to the build so that they are available in IDE
# project files generated via CMake.
file(GLOB_RECURSE FE_SRC_D   dmd/*.d)
file(GLOB_RECURSE FE_HDR     dmd/*.h)
file(GLOB_RECURSE FE_RES     dmd/res/*.*)
file(GLOB_RECURSE GEN_SRC    gen/*.cpp gen/abi/*.cpp)
file(GLOB_RECURSE GEN_HDR    gen/*.h gen/abi/*.h)
file(GLOB_RECURSE GEN_SRC_D  gen/*.d)
file(GLOB_RECURSE IR_SRC     ir/*.cpp)
file(GLOB_RECURSE IR_HDR     ir/*.h)
file(GLOB_RECURSE IR_SRC_D   ir/*.d)
file(GLOB_RECURSE DRV_SRC_D  driver/*.d)
set(DRV_SRC
    driver/args.cpp
    driver/cache.cpp
    driver/cl_helpers.cpp
    driver/cl_options.cpp
    driver/cl_options_instrumentation.cpp
    driver/cl_options_sanitizers.cpp
    driver/cl_options-llvm.cpp
    driver/codegenerator.cpp
    driver/configfile.cpp
    driver/cpreprocessor.cpp
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
)
set(DRV_SRC_EXTRA ${CMAKE_BINARY_DIR}/driver/ldc-version.cpp)
set(DRV_HDR
    driver/args.h
    driver/cache.h
    driver/cache_pruning.h
    driver/cl_helpers.h
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
    driver/timetrace.h
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
    ${DRV_SRC}
    ${DRV_SRC_EXTRA}
    ${DRV_HDR}
)
set(LDC_D_SOURCE_FILES
    ${FE_SRC_D}
    ${GEN_SRC_D}
    ${IR_SRC_D}
    ${DRV_SRC_D}
)

# source_group(TREE ...) requires CMake v3.8+
IF("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" VERSION_GREATER 3.7)
    source_group(TREE "${PROJECT_SOURCE_DIR}" PREFIX "Source Files" FILES ${FE_SRC_D} ${GEN_SRC} ${GEN_SRC_D} ${IR_SRC} ${IR_SRC_D} ${DRV_SRC} ${DRV_SRC_D})
    source_group("Source Files\\driver"                             FILES ${DRV_SRC_EXTRA})
    source_group(TREE "${PROJECT_SOURCE_DIR}" PREFIX "Header Files" FILES ${FE_HDR} ${GEN_HDR} ${IR_HDR} ${DRV_HDR})
endif()


#
# Configure the build system to use LTO and/or PGO while building LDC
#
include(HandleLTOPGOBuildOptions)

#
# Enable Dynamic compilation if supported for this platform and LLVM version.
#
set(LDC_DYNAMIC_COMPILE "AUTO" CACHE STRING "Support dynamic compilation (ON|OFF). Enabled by default; not supported for LLVM >= 12.")
option(LDC_DYNAMIC_COMPILE_USE_CUSTOM_PASSES "Use custom LDC passes in jit" ON)
if(LDC_DYNAMIC_COMPILE STREQUAL "AUTO")
    if(LDC_LLVM_VER LESS 1200)
        set(LDC_DYNAMIC_COMPILE ON)
    else()
        # TODO: port from ORCv1 API (dropped with LLVM 12) to ORCv2 (added with LLVM 7)
        set(LDC_DYNAMIC_COMPILE OFF)
    endif()
endif()
message(STATUS "-- Building LDC with dynamic compilation support (LDC_DYNAMIC_COMPILE): ${LDC_DYNAMIC_COMPILE}")
if(LDC_DYNAMIC_COMPILE)
    add_definitions(-DLDC_DYNAMIC_COMPILE)
    add_definitions(-DLDC_DYNAMIC_COMPILE_API_VERSION=3)
endif()

#
# Includes, defines.
#

include_directories(. dmd)
append("-I${PROJECT_SOURCE_DIR}" DFLAGS_LDC)
append("-I${PROJECT_BINARY_DIR}" DFLAGS_LDC)
append("-J${PROJECT_SOURCE_DIR}/dmd/res" DFLAGS_LDC)

append("-version=IN_LLVM" DFLAGS_LDC)
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
# If the LLVM is shared, add its lib dir to the hardcoded list used for library lookups.
if(LLVM_IS_SHARED)
    append("\"-DLDC_LLVM_LIBDIR=R\\\"(${LLVM_LIBRARY_DIRS})\\\"\"" LDC_CXXFLAGS)
endif()

#
# LLD integration (requires headers & libs)
#
if(NOT DEFINED LDC_WITH_LLD)
    if(EXISTS "${LLVM_INCLUDE_DIRS}/lld/Common/Driver.h")
        set(LDC_WITH_LLD ON)
    else()
        set(LDC_WITH_LLD OFF)
    endif()
endif()
if(LDC_WITH_LLD)
    append("-DLDC_WITH_LLD" LDC_CXXFLAGS)
endif()
message(STATUS "-- Building LDC with integrated LLD linker (LDC_WITH_LLD): ${LDC_WITH_LLD}")

message(STATUS "-- Building LDC with enabled assertions (LDC_ENABLE_ASSERTIONS): ${LDC_ENABLE_ASSERTIONS}")
if(LDC_ENABLE_ASSERTIONS)
    append("-UNDEBUG" EXTRA_CXXFLAGS)
    # avoid MSVC warning D9025 about "-DNDEBUG ... -UNDEBUG"
    string(REGEX REPLACE "(^| )[/-]D *NDEBUG( |$)" "\\1-UNDEBUG\\2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REGEX REPLACE "(^| )[/-]D *NDEBUG( |$)" "\\1-UNDEBUG\\2" CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}")
    string(REGEX REPLACE "(^| )[/-]D *NDEBUG( |$)" "\\1-UNDEBUG\\2" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
else()
    append("-DNDEBUG" EXTRA_CXXFLAGS)
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
        set(CMAKE_HOST_D_CREATE_STATIC_LIBRARY "${D_COMPILER} -lib ${D_COMPILER_FLAGS} ${DFLAGS_BASE} -of=<TARGET> <OBJECTS>")
        set(LDC_LIB_LANGUAGE HOST_D)
    endif()
endif()

set(LDC_LIB LDCShared)
set(LDC_LIB_EXTRA_SOURCES "")
if(MSVC_IDE) # Visual Studio generator
    # Add the .d files as (Visual D) source files to this lib, so that they show up somewhere.
    set(LDC_LIB_EXTRA_SOURCES ${LDC_D_SOURCE_FILES})
    set_property(SOURCE ${LDC_LIB_EXTRA_SOURCES} PROPERTY VS_TOOL_OVERRIDE "DCompile")
    # 'Clear' the original list for the custom commands below, producing ldc2.exe and ldc2-unittest.exe -
    # we still need a dummy .d file.
    set(LDC_D_SOURCE_FILES "${PROJECT_SOURCE_DIR}/dmd/root/man.d")
    # Mark this main library target as (bold) startup project for the generated Visual Studio solution.
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${LDC_LIB})
endif()
add_library(${LDC_LIB} ${LDC_LIB_TYPE} ${LDC_CXX_SOURCE_FILES} ${LDC_LIB_EXTRA_SOURCES})
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
    # Use a custom .props file to set up Visual D (import paths, predefined versions...).
    VS_USER_PROPS "${PROJECT_SOURCE_DIR}/cmake/VisualD.props"
)
# LDFLAGS should actually be in target property LINK_FLAGS, but this works, and gets around linking problems
target_link_libraries(${LDC_LIB} ${LLVM_LIBRARIES} ${LLVM_LDFLAGS})
if(WIN32)
    target_link_libraries(${LDC_LIB} imagehlp psapi)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(${LDC_LIB} dl)
endif()
if(LLVM_SPIRV_FOUND)
    target_link_libraries(${LDC_LIB} ${LLVM_SPIRV_LIBRARIES} ${LLVM_SPIRV_LDFLAGS})
endif()

set(LDC_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${LDC_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX})
set(LDMD_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${LDMD_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX})

# Figure out how to link the main LDC executable, for which we need to take the
# LLVM flags into account.
set(LDC_LINKERFLAG_LIST ${SANITIZE_LDFLAGS} ${LLVM_LIBRARIES} ${LLVM_LDFLAGS})
if(MSVC)
    # Issue 1297 – set LDC's stack to 16 MiB for Windows builds (default: 1 MiB).
    list(APPEND LDC_LINKERFLAG_LIST "/STACK:16777216")
    # VS 2017+: Use undocumented /NOOPTTLS MS linker switch to keep on emitting
    # a .tls section. Required for older host druntime versions, otherwise the
    # GC TLS ranges are garbage starting with VS 2017 Update 15.3.
    if(MSVC_VERSION GREATER 1900 AND D_COMPILER_FE_VERSION LESS 2076)
        list(APPEND LDC_LINKERFLAG_LIST "/NOOPTTLS")
    endif()
endif()
if(LDC_WITH_LLD)
    if(MSVC)
        list(APPEND LDC_LINKERFLAG_LIST LLVMSymbolize.lib)
    else()
        set(LDC_LINKERFLAG_LIST -lLLVMSymbolize ${LDC_LINKERFLAG_LIST})
    endif()
    if (LDC_LLVM_VER LESS 1200 OR NOT LDC_LLVM_VER LESS 1400)
        set(LLD_MACHO lldMachO)
    else()
        set(LLD_MACHO lldMachO2)
    endif()
    if(MSVC)
        if(NOT (LDC_LLVM_VER LESS 1400))
            list(APPEND LDC_LINKERFLAG_LIST lldMinGW.lib lldCOFF.lib lldELF.lib ${LLD_MACHO}.lib lldWasm.lib lldCommon.lib)
        else()
            list(APPEND LDC_LINKERFLAG_LIST lldDriver.lib lldMinGW.lib lldCOFF.lib lldELF.lib ${LLD_MACHO}.lib lldWasm.lib lldYAML.lib lldReaderWriter.lib lldCommon.lib lldCore.lib)
        endif()
    else()
        if(NOT (LDC_LLVM_VER LESS 1400))
            set(LDC_LINKERFLAG_LIST -llldMinGW -llldCOFF -llldELF -l${LLD_MACHO} -llldWasm -llldCommon ${LDC_LINKERFLAG_LIST})
        else()
            set(LDC_LINKERFLAG_LIST -llldDriver -llldMinGW -llldCOFF -llldELF -l${LLD_MACHO} -llldWasm -llldYAML -llldReaderWriter -llldCommon -llldCore ${LDC_LINKERFLAG_LIST})
        endif()
    endif()
    if(APPLE)
        if(NOT (LDC_LLVM_VER LESS 1300)) # LLD 13.0.0 on Mac needs libxar
            list(APPEND LDC_LINKERFLAG_LIST -lxar)
        endif()
    endif()
endif()

if(NOT DEFINED LDC_LINK_MANUALLY)
    if(MSVC)
        # Use the D host compiler for linking D executables.
        set(LDC_LINK_MANUALLY OFF)
    else()
        # On Unix-like systems, default to having CMake link the D executables via the C++ compiler.
        # (Using the D compiler needs -Xcc and -gcc support, see file BuildDExecutable.cmake.)
        set(LDC_LINK_MANUALLY ON)
    endif()
endif()
if(LDC_LINK_MANUALLY AND NOT DEFINED D_LINKER_ARGS)
    include(ExtractDMDSystemLinker)
    message(STATUS "Host D compiler linker program: ${D_LINKER_COMMAND}")
    message(STATUS "Host D compiler linker flags: ${D_LINKER_ARGS}")
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
        # Need to disable dead_strip with LDC host compilers.
        if("${D_COMPILER_ID}" STREQUAL "LDMD")
            if(LDC_LINK_MANUALLY)
                # suboptimal - applies to all D executables (incl. ldmd2, ldc-build-runtime, ldc-prune-cache)
                list(REMOVE_ITEM D_LINKER_ARGS "-Wl,-dead_strip")
            else()
                # just for ldc2 (and ldc2-unittest)
                append("-disable-linker-strip-dead" DFLAGS_LDC)
            endif()
        endif()
    elseif(UNIX)
        # For plugin support, we need to link with --export-dynamic on Unix.
        # Make sure the linker supports --export-dynamic (on Solaris it is not supported and also not needed).
        set(CMAKE_REQUIRED_QUIET_BAK ${CMAKE_REQUIRED_QUIET})
        set(CMAKE_REQUIRED_QUIET ON) # suppress status messages
        CHECK_LINK_FLAG("--export-dynamic" LINKER_ACCEPTS_EXPORT_DYNAMIC_FLAG)
        set(CMAKE_REQUIRED_QUIET ${CMAKE_REQUIRED_QUIET_BAK})

        if(LINKER_ACCEPTS_EXPORT_DYNAMIC_FLAG)
            set(LDC_LINKERFLAG_LIST "${LDC_LINKERFLAG_LIST};-Wl,--export-dynamic")
        else()
            message(WARNING "Linker does not accept --export-dynamic, user plugins may give missing symbol errors upon load")
        endif()
    endif()
endif()
message(STATUS "-- Building LDC with plugin support (LDC_ENABLE_PLUGINS): ${LDC_ENABLE_PLUGINS}")
message(STATUS "-- Linking LDC with flags: ${ALTERNATIVE_MALLOC_O};${LDC_LINKERFLAG_LIST}")

if(NOT WIN32 AND NOT CYGWIN)
    # Unify symbol visibility with LLVM to silence linker warning "direct access in function X to global
    # weak symbol Y means the weak symbol cannot be overridden at runtime. This was likely caused by
    # different translation units being compiled with different visibility settings."
    # See LLVM's cmake/modules/HandleLLVMOptions.cmake.
    check_cxx_compiler_flag("-fvisibility-inlines-hidden" SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
    if (LDC_ENABLE_PLUGINS AND NOT APPLE)
        # For plugins, we shouldn't apply this flag because it hides the inline methods of e.g. Visitor. On macOS it's OK to add.
    elseif (${SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG})
        append("-fvisibility-inlines-hidden" LDC_CXXFLAGS)
    endif()
endif()

build_d_executable(
    "${LDC_EXE}"
    "${LDC_EXE_FULL}"
    "${LDC_D_SOURCE_FILES}"
    "${DFLAGS_BUILD_TYPE} ${DFLAGS_LDC}"
    "${ALTERNATIVE_MALLOC_O};${LDC_LINKERFLAG_LIST}"
    "${FE_RES}"
    "${LDC_LIB}"
    ${COMPILE_D_MODULES_SEPARATELY}
)

if(MSVC_IDE)
    # the IDE generator is a multi-config one
    # so copy the config file into the correct bin subfolder
    # (different outputs no longer feasible for custom commands, so disabled)
    #    add_custom_command(TARGET ${LDC_EXE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/bin/${LDC_EXE}.conf $<TARGET_FILE_DIR:${LDC_EXE}> COMMENT "Copy config file ${LDC_EXE}.conf")
endif()


#
# LDMD
#
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
    "${DFLAGS_BUILD_TYPE}"
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
# Locate LLVM's LTO binary and use it
#
if(WIN32 OR LLVM_IS_SHARED)
    set(LDC_INSTALL_LTOPLUGIN_DEFAULT OFF)
else()
    set(LDC_INSTALL_LTOPLUGIN_DEFAULT ON)
endif()
set(LDC_INSTALL_LTOPLUGIN ${LDC_INSTALL_LTOPLUGIN_DEFAULT} CACHE BOOL "Copy/install the LTO plugin from the LLVM package when available.")
if (LDC_INSTALL_LTOPLUGIN)
    if(APPLE)
        set(LLVM_LTO_BINARY ${LLVM_LIBRARY_DIRS}/libLTO.dylib)
        set(LDC_LTO_BINARY_NAME libLTO-ldc.dylib)
    elseif(UNIX)
        set(LLVM_LTO_BINARY ${LLVM_LIBRARY_DIRS}/LLVMgold.so)
        set(LDC_LTO_BINARY_NAME LLVMgold-ldc.so)
    endif()
    if(EXISTS ${LLVM_LTO_BINARY})
        message(STATUS "-- Including LTO linker plugin (LDC_INSTALL_LTOPLUGIN): ON (${LLVM_LTO_BINARY})")
        copy_and_rename_file(${LLVM_LTO_BINARY} ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}/${LDC_LTO_BINARY_NAME})
        install(PROGRAMS ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}/${LDC_LTO_BINARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
    else()
        message(STATUS "-- Including LTO linker plugin (LDC_INSTALL_LTOPLUGIN): OFF (cannot find ${LLVM_LTO_BINARY})")
    endif()
else()
    message(STATUS "-- Including LTO linker plugin (LDC_INSTALL_LTOPLUGIN): ${LDC_INSTALL_LTOPLUGIN}")
endif()

#
# Locate ASan and other LLVM compiler-rt libraries, and copy them to our lib folder
# Location is LLVM_LIBRARY_DIRS/clang/<version>/lib/<OS>/ , for example LLVM_LIBRARY_DIRS/clang/4.0.0/lib/darwin/
#
if(LLVM_IS_SHARED)
    set(LDC_INSTALL_LLVM_RUNTIME_LIBS_DEFAULT OFF)
else()
    set(LDC_INSTALL_LLVM_RUNTIME_LIBS_DEFAULT ON)
endif()
set(LDC_INSTALL_LLVM_RUNTIME_LIBS ${LDC_INSTALL_LLVM_RUNTIME_LIBS_DEFAULT} CACHE BOOL "Copy/install LLVM compiler-rt libraries (ASan, libFuzzer, ...) from LLVM/Clang into LDC lib dir when available.")
function(copy_compilerrt_lib llvm_lib_name ldc_lib_name fixup_dylib)
    if(LDC_LLVM_VER LESS 1600)
        set(llvm_lib_path ${LLVM_LIBRARY_DIRS}/clang/${LLVM_VERSION_BASE_STRING}/lib/${llvm_lib_name})
    else()
        set(llvm_lib_path ${LLVM_LIBRARY_DIRS}/clang/${LLVM_VERSION_MAJOR}/lib/${llvm_lib_name})
    endif()
    if(EXISTS ${llvm_lib_path})
        message(STATUS "--  - ${llvm_lib_path} --> ${ldc_lib_name}")
        set(ldc_lib_path ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX}/${ldc_lib_name})
        copy_and_rename_file(${llvm_lib_path} ${ldc_lib_path})
        if (fixup_dylib)
            execute_process(COMMAND install_name_tool -id @rpath/${ldc_lib_name} ${ldc_lib_path})
        endif()
        install(FILES ${ldc_lib_path} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
    else()
        message(STATUS "--  - not found: ${llvm_lib_path}")
    endif()
endfunction()
message(STATUS "-- Including LLVM compiler-rt libraries (LDC_INSTALL_LLVM_RUNTIME_LIBS): ${LDC_INSTALL_LLVM_RUNTIME_LIBS}")
if (LDC_INSTALL_LLVM_RUNTIME_LIBS)
    # Locate LLVM sanitizer runtime libraries, and copy them to our lib folder

    if(APPLE)
        copy_compilerrt_lib("darwin/libclang_rt.asan_osx_dynamic.dylib" "libldc_rt.asan.dylib" TRUE)
        copy_compilerrt_lib("darwin/libclang_rt.lsan_osx_dynamic.dylib" "libldc_rt.lsan.dylib" TRUE)
        copy_compilerrt_lib("darwin/libclang_rt.tsan_osx_dynamic.dylib" "libldc_rt.tsan.dylib" TRUE)
        copy_compilerrt_lib("darwin/libclang_rt.osx.a"                  "libldc_rt.builtins.a" FALSE)
        copy_compilerrt_lib("darwin/libclang_rt.profile_osx.a"          "libldc_rt.profile.a"  FALSE)
        copy_compilerrt_lib("darwin/libclang_rt.fuzzer_osx.a"           "libldc_rt.fuzzer.a"   FALSE)
        copy_compilerrt_lib("darwin/libclang_rt.xray_osx.a"             "libldc_rt.xray.a"     FALSE)
        copy_compilerrt_lib("darwin/libclang_rt.xray-basic_osx.a"      "libldc_rt.xray-basic.a"     FALSE)
        copy_compilerrt_lib("darwin/libclang_rt.xray-fdr_osx.a"        "libldc_rt.xray-fdr.a"       FALSE)
        copy_compilerrt_lib("darwin/libclang_rt.xray-profiling_osx.a"  "libldc_rt.xray-profiling.a" FALSE)
    elseif(UNIX)
        if(LDC_LLVM_VER LESS 1500)
            set(LDC_INSTALL_LLVM_RUNTIME_LIBS_OS_DEFAULT "linux")
            set(LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH_DEFAULT "x86_64")
        else()
            set(LDC_INSTALL_LLVM_RUNTIME_LIBS_OS_DEFAULT "x86_64-unknown-linux-gnu")
            set(LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH_DEFAULT "")
        endif()
        set(LDC_INSTALL_LLVM_RUNTIME_LIBS_OS   "${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS_DEFAULT}"   CACHE STRING "Non-Mac Posix: OS used as directory name for the compiler-rt source libraries, e.g., 'freebsd'.")
        set(LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH "${LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH_DEFAULT}" CACHE STRING "Non-Mac Posix: architecture used as libname suffix for the compiler-rt source libraries, e.g., 'aarch64'.")
        if(LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH STREQUAL "")
            set(compilerrt_suffix "")
        else()
            set(compilerrt_suffix "-${LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH}")
        endif()

        copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.asan${compilerrt_suffix}.a"       "libldc_rt.asan.a"     FALSE)
        copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.lsan${compilerrt_suffix}.a"       "libldc_rt.lsan.a"     FALSE)
        copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.msan${compilerrt_suffix}.a"       "libldc_rt.msan.a"     FALSE)
        copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.tsan${compilerrt_suffix}.a"       "libldc_rt.tsan.a"     FALSE)
        copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.builtins${compilerrt_suffix}.a"   "libldc_rt.builtins.a" FALSE)
        copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.profile${compilerrt_suffix}.a"    "libldc_rt.profile.a"  FALSE)
        copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.xray${compilerrt_suffix}.a"       "libldc_rt.xray.a"     FALSE)
        copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.fuzzer${compilerrt_suffix}.a"     "libldc_rt.fuzzer.a"   FALSE)
        copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.xray-basic${compilerrt_suffix}.a"      "libldc_rt.xray-basic.a"     FALSE)
        copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.xray-fdr${compilerrt_suffix}.a"        "libldc_rt.xray-fdr.a"       FALSE)
        copy_compilerrt_lib("${LDC_INSTALL_LLVM_RUNTIME_LIBS_OS}/libclang_rt.xray-profiling${compilerrt_suffix}.a"  "libldc_rt.xray-profiling.a" FALSE)
    elseif(WIN32)
        set(compilerrt_arch_suffix "x86_64")
        if(CMAKE_SIZEOF_VOID_P EQUAL 4)
            set(compilerrt_arch_suffix "i386")
        endif()
        copy_compilerrt_lib("windows/clang_rt.asan-${compilerrt_arch_suffix}.lib"       "ldc_rt.asan.lib"     FALSE)
        copy_compilerrt_lib("windows/clang_rt.lsan-${compilerrt_arch_suffix}.lib"       "ldc_rt.lsan.lib"     FALSE)
        copy_compilerrt_lib("windows/clang_rt.builtins-${compilerrt_arch_suffix}.lib"   "ldc_rt.builtins.lib" FALSE)
        copy_compilerrt_lib("windows/clang_rt.profile-${compilerrt_arch_suffix}.lib"    "ldc_rt.profile.lib"  FALSE)
        copy_compilerrt_lib("windows/clang_rt.fuzzer-${compilerrt_arch_suffix}.lib"     "ldc_rt.fuzzer.lib"   FALSE)
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
    "-g -unittest ${DFLAGS_LDC}"
    "${LDC_LINKERFLAG_LIST}"
    ""
    "${LDC_LIB}"
    ${COMPILE_D_MODULES_SEPARATELY}
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
    add_subdirectory(tests/dmd)
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
    "${DFLAGS_BUILD_TYPE}"
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
