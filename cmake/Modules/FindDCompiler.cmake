# Find a D compiler!
#
# use environment variable DMD first if defined by user, next use a list of common compiler executables.
#
# The following variables are defined:
#  D_COMPILER_FOUND          - true if a D compiler was found
#  D_COMPILER          - D compiler
#  D_COMPILER_FLAGS    - D compiler flags (could be passed in the DMD environment variable)
#  D_COMPILER_ID       = {"DigitalMars", "LDMD", "LDC", "GDC"}
#  D_COMPILER_VERSION_STRING - String containing the compiler version, e.g. "DMD64 D Compiler v2.070.2"


set(D_COMPILER_FOUND "FALSE")

set(COMMON_D_COMPILERS "ldmd2" "dmd")
set(COMMON_D_COMPILER_PATHS "/usr/bin" "/usr/local/bin" "C:\\d\\dmd2\\windows\\bin")

if($ENV{DMD} MATCHES ".+")
    get_filename_component(D_COMPILER $ENV{DMD} PROGRAM PROGRAM_ARGS D_COMPILER_FLAGS_ENV_INIT CACHE)
    if(D_COMPILER_FLAGS_ENV_INIT)
        set(D_COMPILER_FLAGS "${D_COMPILER_FLAGS_ENV_INIT}" CACHE STRING "Default flags for D compiler")
    endif()
    if(NOT EXISTS ${D_COMPILER})
        message(FATAL_ERROR "Could not find compiler set in environment variable $ENV{DMD}.")
    endif()
else()
    # "NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH" is necessary, otherwise CMake will find the compiler in the install prefix path!
    find_program(D_COMPILER NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH NAMES ${COMMON_D_COMPILERS} PATHS ${COMMON_D_COMPILER_PATHS} DOC "D compiler")
endif()

# TODO: Test compiler and set compiler ID
if (D_COMPILER)
    set(D_COMPILER_FOUND "TRUE")

    get_filename_component(__D_COMPILER_NAME ${D_COMPILER} NAME_WE)
    if (__D_COMPILER_NAME STREQUAL "dmd")
        set(D_COMPILER_ID "DigitalMars")
    elseif (__D_COMPILER_NAME STREQUAL "ldmd2")
        set(D_COMPILER_ID "LDMD")
    elseif (__D_COMPILER_NAME STREQUAL "ldc2")
        set(D_COMPILER_ID "LDC")
    elseif (__D_COMPILER_NAME STREQUAL "gdc")
        set(D_COMPILER_ID "GDC")
    endif()

    # Older versions of ldmd do not have --version cmdline option, but the error message still contains the version info in the first line.
    execute_process(COMMAND ${D_COMPILER} --version
                    OUTPUT_VARIABLE D_COMPILER_VERSION_STRING
                    ERROR_VARIABLE D_COMPILER_VERSION_STRING
                    ERROR_QUIET)
    string(REGEX MATCH "^[^\r\n:]*" D_COMPILER_VERSION_STRING "${D_COMPILER_VERSION_STRING}")
endif()


if (D_COMPILER_FOUND)
    message(STATUS "Found D compiler ${D_COMPILER}, with default flags '${D_COMPILER_FLAGS}'")
    message(STATUS "D compiler version: ${D_COMPILER_VERSION_STRING}")
else()
    message(FATAL_ERROR "Did not find D compiler! Try setting the 'DMD' environment variable.")
endif()
