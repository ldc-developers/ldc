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
        list(REMOVE_ITEM LLVM_FIND_COMPONENTS "all-targets" index)
        list(APPEND LLVM_FIND_COMPONENTS ${LLVM_TARGETS_TO_BUILD})
        list(REMOVE_ITEM LLVM_FIND_COMPONENTS "backend" index)
        llvm_map_components_to_libraries(tmplibs ${LLVM_FIND_COMPONENTS})
        foreach(lib ${tmplibs})
            list(APPEND LLVM_LIBRARIES "${LLVM_LIBRARY_DIRS}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_STATIC_LIBRARY_SUFFIX}")
        endforeach()
    else()
        if (NOT FIND_LLVM_QUIETLY)
            message(WARNING "Could not find llvm-config. Try manually setting LLVM_CONFIG to the llvm-config executable of the installation to use.")
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
    macro(llvm_set_libs var flag prefix)
       if(LLVM_FIND_QUIETLY)
            set(_quiet_arg ERROR_QUIET)
        endif()
        execute_process(
            COMMAND ${LLVM_CONFIG} --${flag} ${LLVM_FIND_COMPONENTS}
            OUTPUT_VARIABLE tmplibs
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ${_quiet_arg}
        )
        string(REGEX REPLACE "([$^.[|*+?()]|])" "\\\\\\1" pattern ${prefix})
        string(REGEX MATCHALL "${pattern}[^ ]+" LLVM_${var} ${tmplibs})
    endmacro()

    llvm_set(VERSION_STRING version)
    if(${LLVM_VERSION_STRING} MATCHES "3.0[A-Za-z]*")
        # Version 3.0 does not support component all-targets
        llvm_set(TARGETS_TO_BUILD targets-built)
        string(REGEX MATCHALL "[^ ]+" LLVM_TARGETS_TO_BUILD ${LLVM_TARGETS_TO_BUILD})
        list(REMOVE_ITEM LLVM_FIND_COMPONENTS "all-targets" index)
        list(APPEND LLVM_FIND_COMPONENTS ${LLVM_TARGETS_TO_BUILD})
    else()
        # Version 3.1+ does not supoort component backend
        list(REMOVE_ITEM LLVM_FIND_COMPONENTS "backend" index)
    endif()
    llvm_set(CXXFLAGS cxxflags)
    llvm_set(HOST_TARGET host-target)
    llvm_set(INCLUDE_DIRS includedir)
    llvm_set(LDFLAGS ldflags)
    llvm_set(LIBRARY_DIRS libdir)
    llvm_set_libs(LIBRARIES libfiles "${LLVM_LIBRARY_DIRS}/")
    llvm_set(ROOT_DIR prefix)

    # On CMake builds of LLVM, the output of llvm-config --cxxflags does not
    # include -fno-rtti, leading to linker errors. Be sure to add it.
    if(CMAKE_COMPILER_IS_GNUCXX OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
        if(NOT ${LLVM_CXXFLAGS} MATCHES "-fno-rtti")
            set(LLVM_CXXFLAGS "${LLVM_CXXFLAGS} -fno-rtti")
        endif()
    endif()
endif()

string(REGEX REPLACE "([0-9]+).*" "\\1" LLVM_VERSION_MAJOR "${LLVM_VERSION_STRING}" )
string(REGEX REPLACE "[0-9]+\\.([0-9]+).*[A-Za-z]*" "\\1" LLVM_VERSION_MINOR "${LLVM_VERSION_STRING}" )

# Use the default CMake facilities for handling QUIET/REQUIRED.
include(FindPackageHandleStandardArgs)

if(${CMAKE_VERSION} VERSION_LESS "2.8.4")
  # The VERSION_VAR argument is not supported on pre-2.8.4, work around this.
  set(VERSION_VAR dummy)
endif()

find_package_handle_standard_args(LLVM
    REQUIRED_VARS LLVM_ROOT_DIR LLVM_HOST_TARGET
    VERSION_VAR LLVM_VERSION_STRING)
