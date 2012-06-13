# - Find LLVM headers and libraries.
# This module locates LLVM and adapts the llvm-config output for use with
# CMake.
#
# A given list of COMPONENTS is passed to llvm-config.
#
# The following variables are defined:
#  LLVM_FOUND          - true if LLVM was found
#  LLVM_CXXFLAGS       - C++ compiler flags for files that include LLVM headers.
#  LLVM_HOST_TARGET    - Target triple used to configure LLVM.
#  LLVM_INCLUDE_DIRS   - Directory containing LLVM include files.
#  LLVM_LDFLAGS        - Linker flags to add when linking against LLVM
#                        (includes -LLLVM_LIBRARY_DIRS).
#  LLVM_LIBRARIES      - Full paths to the library files to link against.
#  LLVM_LIBRARY_DIRS   - Directory containing LLVM libraries.
#  LLVM_ROOT_DIR       - The root directory of the LLVM installation.
#                        llvm-config is searched for in ${LLVM_ROOT_DIR}/bin.
#  LLVM_VERSION_MAJOR  - Major version of LLVM.
#  LLVM_VERSION_MINOR  - Minor version of LLVM.
#  LLVM_VERSION_STRING - Full LLVM version string (e.g. 2.9).
#
# Note: The variable names were chosen in conformance with the offical CMake
# guidelines, see ${CMAKE_ROOT}/Modules/readme.txt.

find_program(LLVM_CONFIG llvm-config ${LLVM_ROOT_DIR}/bin
    DOC "Path to llvm-config tool.")

if (NOT LLVM_CONFIG)
    if (WIN32)
        # A bit of a sanity check:
        if( NOT EXISTS ${LLVM_ROOT_DIR}/include/llvm )
            message(FATAL_ERROR "LLVM_ROOT_DIR (${LLVM_ROOT_DIR}) is not a valid LLVM install")
        endif()
        # We incorporate the CMake features provided by LLVM:
        set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${LLVM_ROOT_DIR}/share/llvm/cmake")
        include(LLVMConfig)
        # Set properties
        set(LLVM_HOST_TARGET ${TARGET_TRIPLE})
        set(LLVM_VERSION_STRING ${LLVM_PACKAGE_VERSION})
        set(LLVM_CXXFLAGS ${LLVM_DEFINITIONS})
        set(LLVM_LDFLAGS "")
        llvm_map_components_to_libraries(LLVM_LIBRARIES ${LLVM_FIND_COMPONENTS})
    else()
        if (NOT FIND_LLVM_QUIETLY)
            message(WARNING "Could not find llvm-config. Consider manually setting LLVM_ROOT_DIR.")
        endif()
    endif()
else()
    macro(llvm_set var flag)
   	if(LLVM_FIND_QUIETLY)
            set(_quiet_arg ERROR_QUIET)
        endif()
        execute_process(
            COMMAND ${LLVM_CONFIG} --${flag}
            OUTPUT_VARIABLE LLVM_${var}
            OUTPUT_STRIP_TRAILING_WHITESPACE
	    ${_quiet_arg}
        )
    endmacro()
    macro(llvm_set_libs var flag)
   	if(LLVM_FIND_QUIETLY)
            set(_quiet_arg ERROR_QUIET)
        endif()
        execute_process(
            COMMAND ${LLVM_CONFIG} --${flag} ${LLVM_FIND_COMPONENTS}
            OUTPUT_VARIABLE LLVM_${var}
            OUTPUT_STRIP_TRAILING_WHITESPACE
	    ${_quiet_arg}
        )
    endmacro()

    llvm_set(CXXFLAGS cxxflags)
    llvm_set(HOST_TARGET host-target)
    llvm_set(INCLUDE_DIRS includedir)
    llvm_set(LDFLAGS ldflags)
    llvm_set_libs(LIBRARIES libfiles)
    llvm_set(LIBRARY_DIRS libdir)
    llvm_set(ROOT_DIR prefix)
    llvm_set(VERSION_STRING version)
endif()

string(REGEX REPLACE "([0-9]+).*" "\\1" LLVM_VERSION_MAJOR "${LLVM_VERSION_STRING}" )
string(REGEX REPLACE "[0-9]+\\.([0-9]+).*[A-Za-z]" "\\1" LLVM_VERSION_MINOR "${LLVM_VERSION_STRING}" )

# Use the default CMake facilities for handling QUIET/REQUIRED.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LLVM
    REQUIRED_VARS LLVM_ROOT_DIR LLVM_HOST_TARGET
    VERSION_VAR LLVM_VERSION_STRING)
