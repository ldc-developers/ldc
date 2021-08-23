# Find a D compiler!
#
# Use variable D_COMPILER first if defined by user, then try the environment variable DMD, next use a list of common compiler executables.
#
# The following variables are defined:
#  D_COMPILER_FOUND    - true if a D compiler with DMD-compatible cmdline interface was found
#  D_COMPILER          - D compiler
#  D_COMPILER_FLAGS    - D compiler flags (could be passed in the DMD environment variable)
#  D_COMPILER_ID       = {"DigitalMars", "LDMD", "GDMD"}
#  D_COMPILER_VERSION_STRING - String containing the compiler version, e.g. "DMD64 D Compiler v2.070.2"
#  D_COMPILER_FE_VERSION - compiler front-end version, e.g. "2070"


set(D_COMPILER_FOUND "FALSE")

set(COMMON_D_COMPILERS "ldmd2" "dmd" "gdmd")
set(COMMON_D_COMPILER_PATHS "/usr/bin" "/usr/local/bin" "C:\\d\\dmd2\\windows\\bin")

if (D_COMPILER)
    get_filename_component(D_COMPILER ${D_COMPILER} PROGRAM PROGRAM_ARGS D_COMPILER_FLAGS_ENV_INIT)
    if(D_COMPILER_FLAGS_ENV_INIT)
        set(D_COMPILER_FLAGS "${D_COMPILER_FLAGS_ENV_INIT}" CACHE STRING "Default flags for D compiler")
    endif()
    if(NOT EXISTS ${D_COMPILER})
        message(FATAL_ERROR "Could not find compiler set in variable D_COMPILER: ${D_COMPILER}.")
    endif()
elseif($ENV{DMD} MATCHES ".+")
    get_filename_component(D_COMPILER $ENV{DMD} PROGRAM PROGRAM_ARGS D_COMPILER_FLAGS_ENV_INIT CACHE)
    if(D_COMPILER_FLAGS_ENV_INIT)
        set(D_COMPILER_FLAGS "${D_COMPILER_FLAGS_ENV_INIT}" CACHE STRING "Default flags for D compiler")
    endif()
    if(NOT EXISTS ${D_COMPILER})
        message(FATAL_ERROR "Could not find compiler set in environment variable DMD: $ENV{DMD}.")
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
    elseif (__D_COMPILER_NAME MATCHES "gdmd")
        set(D_COMPILER_ID "GDMD")
    elseif (NOT D_COMPILER_ID)
        message(FATAL_ERROR "Unsupported D compiler filename (try 'ldmd2' or 'gdmd', or set D_COMPILER_ID manually).")
    endif()

    # Older versions of ldmd do not have --version cmdline option, but the error message still contains the version info in the first line.
    execute_process(COMMAND ${D_COMPILER} --version
                    OUTPUT_VARIABLE D_COMPILER_VERSION_STRING
                    ERROR_VARIABLE D_COMPILER_VERSION_STRING
                    ERROR_QUIET)
    if ("${D_COMPILER_ID}" STREQUAL "GDMD")
        execute_process(COMMAND "${CMAKE_COMMAND}" -E echo "pragma(msg, int(__VERSION__));"
                        COMMAND "${D_COMPILER}" - -o-
                        ERROR_VARIABLE D_COMPILER_FE_VERSION
                        OUTPUT_QUIET)
        string(REGEX MATCH "^[^\r\n:]*" D_COMPILER_FE_VERSION "${D_COMPILER_FE_VERSION}")
    else()
        string(REGEX MATCH " (D Compiler|based on DMD) v([0-9]+)\\.([0-9]+)" D_COMPILER_FE_VERSION "${D_COMPILER_VERSION_STRING}")
        math(EXPR D_COMPILER_FE_VERSION ${CMAKE_MATCH_2}*1000+${CMAKE_MATCH_3}) # e.g., 2079
    endif()
    string(REGEX MATCH "^[^\r\n:]*" D_COMPILER_VERSION_STRING "${D_COMPILER_VERSION_STRING}")
endif()


if (D_COMPILER_FOUND)
    message(STATUS "Found host D compiler ${D_COMPILER}, with default flags '${D_COMPILER_FLAGS}'")
    message(STATUS "Host D compiler ID: ${D_COMPILER_ID}")
    message(STATUS "Host D compiler version: ${D_COMPILER_VERSION_STRING}")
    message(STATUS "Host D compiler front-end version: ${D_COMPILER_FE_VERSION}")
else()
    message(FATAL_ERROR "No supported D compiler (ldmd2, dmd, gdmd) found! Try setting the 'D_COMPILER' variable or 'DMD' environment variable.")
endif()
