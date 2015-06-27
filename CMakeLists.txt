project(ldc)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

if(MSVC)
    set(LIBCONFIG_DLL OFF CACHE BOOL "Use libconfig DLL instead of static library")
endif()

include(CheckIncludeFile)
include(CheckLibraryExists)
include(CheckCXXCompilerFlag)

#
# Locate LLVM.
#

find_package(LLVM 3.1 REQUIRED
    all-targets analysis asmparser asmprinter bitreader bitwriter codegen core debuginfodwarf instcombine ipa ipo instrumentation linker lto mc mcdisassembler mcparser objcarcopts object option profiledata scalaropts selectiondag support tablegen target transformutils vectorize ${EXTRA_LLVM_MODULES})
math(EXPR LDC_LLVM_VER ${LLVM_VERSION_MAJOR}*100+${LLVM_VERSION_MINOR})

#
# Locate libconfig.
#
find_package(LibConfig REQUIRED)

#
# Get info about used Linux distribution.
#
include(GetLinuxDistribution)

#
# Main configuration.
#

# Version information
set(LDC_VERSION "0.15.2") # May be overridden by git hash tag
set(DMDFE_MAJOR_VERSION   2)
set(DMDFE_MINOR_VERSION   0)
set(DMDFE_PATCH_VERSION   66)
set(DMDFE_FIX_LEVEL       1) # Comment out if not used

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
    set(DMDFE_PATH dmd2)
    set(LDC_EXE ldc2)
    set(LDMD_EXE ldmd2)
    set(RUNTIME druntime)
    add_definitions(-DDMDV2)
else()
    message(FATAL_ERROR "unsupported D version")
endif()

set(LDC_EXE_NAME ${PROGRAM_PREFIX}${LDC_EXE}${PROGRAM_SUFFIX})
set(LDMD_EXE_NAME ${PROGRAM_PREFIX}${LDMD_EXE}${PROGRAM_SUFFIX})

file(MAKE_DIRECTORY
    ${PROJECT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/${DMDFE_PATH}
)

if(UNIX)
    ENABLE_LANGUAGE(ASM)
elseif(MSVC)
    ENABLE_LANGUAGE(ASM_MASM)
endif()

function(append value)
    foreach(variable ${ARGN})
        if(${variable} STREQUAL "")
            set(${variable} "${value}" PARENT_SCOPE)
        else()
            set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
        endif()
    endforeach(variable)
endfunction()

# Use separate compiler flags for the frontend and for the LDC-specific parts,
# as enabling warnings on the DMD frontend only leads to a lot of clutter in
# the output (LLVM_CXXFLAGS sometimes already includes -Wall).
set(DMD_CXXFLAGS)
set(LDC_CXXFLAGS)
if(CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang"))
    append("-w" DMD_CXXFLAGS)

    # Disable some noisy warnings:
    #  * -Wunused-parameter triggers for LLVM headers
    #  * -Wmissing-field-initializer leads to reams of warnings in gen/asm-*.h
    #  * -Wnon-virtual-dtor is something Walter has declined to let us fix upstream
    #    and it triggers for the visitors we need in our glue code
    #  * -Wpedantic warns on trailing commas in initializer lists and casting
    #    function pointers to void*.
    append("-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wno-non-virtual-dtor" LDC_CXXFLAGS)
    if(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_C_COMPILER_VERSION VERSION_LESS "4.7.0")
        append("-Wno-pedantic" LDC_CXXFLAGS)
    endif()
endif()
# Append -mminimal-toc for gcc 4.0.x - 4.5.x on ppc64
if( CMAKE_COMPILER_IS_GNUCXX
    AND CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64|powerpc64"
    AND CMAKE_C_COMPILER_VERSION VERSION_LESS "4.6.0" )
    append("-mminimal-toc" DMD_CXXFLAGS LDC_CXXFLAGS)
endif()
# Do not use doubledouble on ppc
if( CMAKE_SYSTEM_PROCESSOR MATCHES "ppc|powerpc")
    append("-mlong-double-64" DMD_CXXFLAGS LDC_CXXFLAGS)
endif()
if(UNIX)
    append("-DPOSIX -DLDC_POSIX" DMD_CXXFLAGS)
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
append("${SANITIZE_CXXFLAGS}" DMD_CXXFLAGS)
append("${SANITIZE_CXXFLAGS}" LDC_CXXFLAGS)
# LLVM_CXXFLAGS may contain -Werror which causes compile errors with dmd source
string(REPLACE "-Werror " "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
if (UNIX AND NOT "${LLVM_LDFLAGS}" STREQUAL "")
    # LLVM_LDFLAGS may contain -l-lld which is a wrong library reference (AIX)
    string(REPLACE "-l-lld " "-lld " LLVM_LDFLAGS ${LLVM_LDFLAGS})
endif()

#
# Run idgen and impcnvgen.
#
set_source_files_properties(
    ${DMDFE_PATH}/idgen.c
    ${DMDFE_PATH}/impcnvgen.c
    PROPERTIES LANGUAGE CXX
)
add_executable(idgen ${DMDFE_PATH}/idgen.c)
add_executable(impcnvgen ${DMDFE_PATH}/impcnvgen.c)
# cmake 2.4
set_target_properties(
    idgen impcnvgen PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${DMDFE_PATH}
    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${DMD_CXXFLAGS}"
    LINK_FLAGS "${SANITIZE_LDFLAGS}"
)

add_custom_command(
    OUTPUT
        ${PROJECT_BINARY_DIR}/${DMDFE_PATH}/id.c
        ${PROJECT_BINARY_DIR}/${DMDFE_PATH}/id.h
    COMMAND idgen
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${DMDFE_PATH}
    DEPENDS idgen
)
add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/${DMDFE_PATH}/impcnvtab.c
    COMMAND impcnvgen
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${DMDFE_PATH}
    DEPENDS impcnvgen
)
set(LDC_GENERATED
    ${PROJECT_BINARY_DIR}/${DMDFE_PATH}/id.c
    ${PROJECT_BINARY_DIR}/${DMDFE_PATH}/id.h
    ${PROJECT_BINARY_DIR}/${DMDFE_PATH}/impcnvtab.c
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
file(GLOB_RECURSE FE_SRC ${DMDFE_PATH}/*.c)
file(GLOB_RECURSE FE_HDR ${DMDFE_PATH}/*.h)
file(GLOB_RECURSE GEN_SRC gen/*.cpp)
file(GLOB_RECURSE GEN_HDR gen/*.h)
file(GLOB IR_SRC ir/*.cpp)
file(GLOB IR_HDR ir/*.h)
set(DRV_SRC
    driver/cl_options.cpp
    driver/configfile.cpp
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
    driver/configfile.h
    driver/ldc-version.h
    driver/targetmachine.h
    driver/toobj.h
    driver/tool.h
)
# exclude idgen and impcnvgen and generated sources, just in case
list(REMOVE_ITEM FE_SRC
    ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/idgen.c
    ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/impcnvgen.c
    ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/id.c
    ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/impcnvtab.c
)
# Add special LLVM fixes
if(LLVM_VERSION_MAJOR EQUAL 3 AND LLVM_VERSION_MINOR EQUAL 6)
    message(STATUS "Adding fix for LLVM 3.6")
    set(LLVM_FIX llvm_fix/LinkModules_36.cpp)
endif()

set(LDC_SOURCE_FILES
    ${LDC_GENERATED}
    ${FE_SRC}
    ${FE_HDR}
    ${GEN_SRC}
    ${GEN_HDR}
    ${IR_SRC}
    ${IR_HDR}
    ${LLVM_FIX}
)

# DMD source files have a .c extension, but are actually C++ code.
foreach(file ${LDC_SOURCE_FILES} ${DRV_SRC} ${DRV_HDR})
    if(file MATCHES ".*\\.c$")
        set_source_files_properties(${file} PROPERTIES
            LANGUAGE CXX
            COMPILE_FLAGS "${DMD_CXXFLAGS}"
        )
    else()
        set_source_files_properties(${file} PROPERTIES
            COMPILE_FLAGS "${LDC_CXXFLAGS}"
        )
    endif()
endforeach()

source_group("Source Files\\${DMDFE_PATH}" FILES ${FE_SRC})
source_group("Header Files\\${DMDFE_PATH}" FILES ${FE_HDR})
source_group("Source Files\\gen" FILES ${GEN_SRC})
source_group("Header Files\\gen" FILES ${GEN_HDR})
source_group("Source Files\\ir" FILES ${IR_SRC})
source_group("Header Files\\ir" FILES ${IR_HDR})
source_group("Generated Files" REGULAR_EXPRESSION "(id\\.[ch]|impcnvtab\\.c)$")

#
# Includes, defines.
#

include_directories(
    .
    ${DMDFE_PATH}
    ${DMDFE_PATH}/root
    ${PROJECT_BINARY_DIR}/${DMDFE_PATH}
    ${PROJECT_SOURCE_DIR}
)
include_directories( SYSTEM
    ${LLVM_INCLUDE_DIRS}
    ${LIBCONFIG_INCLUDE_DIR}
)

if(MSVC)
    include_directories(${PROJECT_SOURCE_DIR}/vcbuild)
    if(NOT LIBCONFIG_DLL)
        add_definitions(-DLIBCONFIG_STATIC)
    endif()
endif()

add_definitions(
    -DIN_LLVM
    -DOPAQUE_VTBLS
    -DLDC_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
    -DLDC_LLVM_VER=${LDC_LLVM_VER}
)

if(GENERATE_OFFTI)
    add_definitions(-DGENERATE_OFFTI)
endif()

if(MSVC)
    # Remove flags here, for exceptions and RTTI.
    # CL.EXE complains to override flags like "/GR /GR-".
    string(REGEX REPLACE "(^| ) */EH[-cs]+ *( |$)" "\\1 \\2" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "(^| ) */GR-? *( |$)" "\\1 \\2" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    add_definitions(-D_HAS_EXCEPTIONS=0)
    set(EXTRA_CXXFLAGS "/W0 /wd4996 /GF /GR- /EHs-c- /MP")
else()
    set(EXTRA_CXXFLAGS "")
endif()

# if llvm was built with assertions we have to do the same
# as there are some headers with differing behavior based on NDEBUG
if(LLVM_ENABLE_ASSERTIONS)
    set(EXTRA_CXXFLAGS "${EXTRA_CXXFLAGS} -UNDEBUG")
    # avoid msvc warning D9025
    string(REGEX REPLACE "(^| )[/-]D *NDEBUG( |$)" " " CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REGEX REPLACE "(^| )[/-]D *NDEBUG( |$)" " " CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

#
# Check endianess.
# There is no realiable way to delegate the work to the compiler.
# E.g. gcc up to version 4.6 defines __LITTLE_ENDIAN, but not 4.7
#
include(TestBigEndian)
test_big_endian(BIGENDIAN)
if(${BIGENDIAN})
    add_definitions(-D__BIG_ENDIAN__)
else()
    add_definitions(-D__LITTLE_ENDIAN__)
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
        append("--coverage" LLVM_LDFLAGS)
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
add_library(${LDC_LIB} ${LDC_LIB_TYPE} ${LDC_SOURCE_FILES})
set_target_properties(
    ${LDC_LIB} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_NAME ldc
    LIBRARY_OUTPUT_NAME ldc
    RUNTIME_OUTPUT_NAME ldc
    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${EXTRA_CXXFLAGS}"
    LINK_FLAGS "${SANITIZE_LDFLAGS}"
)

# LDFLAGS should actually be in target property LINK_FLAGS, but this works, and gets around linking problems
target_link_libraries(${LDC_LIB} ${LLVM_LIBRARIES} ${PTHREAD_LIBS} ${TERMINFO_LIBS} "${LLVM_LDFLAGS}")
if(WIN32)
    target_link_libraries(${LDC_LIB} imagehlp psapi)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(${LDC_LIB} dl)
endif()


add_executable(${LDC_EXE} ${DRV_SRC} ${DRV_HDR})
set_target_properties(
    ${LDC_EXE} PROPERTIES
    OUTPUT_NAME ${LDC_EXE_NAME}
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${EXTRA_CXXFLAGS}"
    LINK_FLAGS "${SANITIZE_LDFLAGS}"
)
target_link_libraries(${LDC_EXE} ${LDC_LIB} ${LIBCONFIG_LIBRARY} ${PTHREAD_LIBS} ${CMAKE_DL_LIBS} ${TERMINFO_LIBS})

if(MSVC_IDE)
    # the IDE generator is a multi-config one
    # so copy the config file into the correct bin subfolder
    add_custom_command(TARGET ${LDC_EXE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/bin/${LDC_EXE}.conf $<TARGET_FILE_DIR:${LDC_EXE}> COMMENT "Copy config file ${LDC_EXE}.conf")
endif()


#
# Intrinsics module generation tools.
#

# The LLVM_INCLUDE_DIR definition is not always set, e.g. on Windows.
find_path(LLVM_INTRINSIC_TD_PATH "Intrinsics.td" PATHS ${LLVM_INCLUDE_DIRS}/llvm ${LLVM_INCLUDE_DIRS}/llvm/IR NO_DEFAULT_PATH)
if (${LLVM_INTRINSIC_TD_PATH} STREQUAL "LLVM_INTRINSIC_TD_PATH-NOTFOUND")
    message(SEND_ERROR "File Intrinsics.td not found")
else()
    string(REGEX REPLACE "/llvm(/IR)?$" "" LLVM_INTRINSIC_TD_PATH ${LLVM_INTRINSIC_TD_PATH})
    message(STATUS "Using path for Intrinsics.td: ${LLVM_INTRINSIC_TD_PATH}")
endif()
add_definitions(-DLLVM_INTRINSIC_TD_PATH="${LLVM_INTRINSIC_TD_PATH}")

add_executable(gen_gccbuiltins utils/gen_gccbuiltins.cpp)

# Prior to LLVM 3.2, TableGen still uses RTTI, contrary to the rest of LLVM.
# Thus, we have to remove the -fno-rtti flag from the llvm-config output.
set(TABLEGEN_CXXFLAGS "${LLVM_CXXFLAGS}")
if(${LDC_LLVM_VER} LESS 302)
    string(REPLACE "-fno-rtti" "" TABLEGEN_CXXFLAGS ${TABLEGEN_CXXFLAGS})
endif()

set_target_properties(
    gen_gccbuiltins PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    COMPILE_FLAGS "${TABLEGEN_CXXFLAGS} ${LDC_CXXFLAGS}"
    LINK_FLAGS "${SANITIZE_LDFLAGS}"
)
target_link_libraries(gen_gccbuiltins ${LLVM_LIBRARIES} ${PTHREAD_LIBS} ${TERMINFO_LIBS} ${CMAKE_DL_LIBS} "${LLVM_LDFLAGS}")
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(gen_gccbuiltins dl)
endif()

#
# LDMD
#
include(CheckSymbolExists)
CHECK_SYMBOL_EXISTS(_SC_ARG_MAX "unistd.h" HAVE_SC_ARG_MAX)
if (HAVE_SC_ARG_MAX)
   add_definitions(-DHAVE_SC_ARG_MAX)
endif()

set_source_files_properties(driver/ldmd.cpp PROPERTIES
    COMPILE_FLAGS "${LDC_CXXFLAGS}"
)
add_executable(${LDMD_EXE} dmd2/root/man.c driver/ldmd.cpp driver/response.cpp)
set_target_properties(${LDMD_EXE} PROPERTIES
    COMPILE_DEFINITIONS LDC_EXE_NAME="${LDC_EXE_NAME}"
    COMPILE_FLAGS "${LLVM_CXXFLAGS}"
    LINK_FLAGS "${SANITIZE_LDFLAGS}"
    OUTPUT_NAME "${LDMD_EXE_NAME}"
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
# Same as above, LLVM_LDFLAGS should really be in LINK_FLAGS, but the LLVM libs
# use symbols from libdl, ..., so LLVM_LDFLAGS must come _after_ them in the
# command line. Maybe this could be improved using library groups, at least with
# GNU ld.
target_link_libraries(${LDMD_EXE} ${LLVM_LIBRARIES} ${PTHREAD_LIBS} ${TERMINFO_LIBS} ${CMAKE_DL_LIBS} "${LLVM_LDFLAGS}")

#
# Test and runtime targets. Note that enable_testing() is order-sensitive!
#
enable_testing()
add_subdirectory(runtime)
if(D_VERSION EQUAL 2)
    add_subdirectory(tests/d2)
endif()

#
# Install target.
#

install(TARGETS ${LDC_EXE} ${LDMD_EXE} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
if(${BUILD_SHARED})
    # For now, only install libldc if explicitely building the shared library.
    # While it might theoretically be possible to use LDC as a static library
    # as well, for the time being this just bloats the normal packages.
    install(TARGETS ${LDC_LIB} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
install(FILES ${PROJECT_BINARY_DIR}/bin/${LDC_EXE}_install.conf DESTINATION ${CONF_INST_DIR} RENAME ${LDC_EXE}.conf)

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
