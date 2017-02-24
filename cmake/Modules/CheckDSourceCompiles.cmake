# This file is modified from CMake's CheckCxxSourceCompiles.cmake.
# Distributed under the OSI-approved BSD 3-Clause License.  See https://cmake.org/licensing for details.

# Check if given D source compiles and links into an executable
#
# CHECK_D_SOURCE_COMPILES(<code> <var> [FLAGS <flags>])
#
# ::
#
#   <code>       - source code to try to compile
#   <var>        - variable to store whether the source code compiled
#                  Will be created as an internal cache variable.
#   <flags>      - Extra commandline flags passed to the compiler.
#
# The D_COMPILER variable is read and must point to the D compiler executable.

macro(CHECK_D_SOURCE_COMPILES SOURCE VAR)
  if(NOT DEFINED "${VAR}")
    set(_FLAGS)
    set(_key)
    foreach(arg ${ARGN})
      if("${arg}" MATCHES "^(FLAGS)$")
        set(_key "${arg}")
      elseif(_key)
        list(APPEND _${_key} "${arg}")
        set(_key 0)
      else()
        message(FATAL_ERROR "Unknown argument:\n  ${arg}\n")
      endif()
    endforeach()

    file(WRITE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.d"
      "${SOURCE}\n")

    if(NOT CMAKE_REQUIRED_QUIET)
      message(STATUS "Performing Test ${VAR}")
    endif()

    execute_process(COMMAND ${D_COMPILER} ${_FLAGS} ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.d
                OUTPUT_VARIABLE OUTPUT
                ERROR_VARIABLE OUTPUT
                RESULT_VARIABLE RETVAL)

    if(${RETVAL})
      set(${VAR} 0)
    else()
      set(${VAR} 1)
    endif()

    if(${VAR})
      set(${VAR} 1 CACHE INTERNAL "Test ${VAR}")
      if(NOT CMAKE_REQUIRED_QUIET)
        message(STATUS "Performing Test ${VAR} - Success")
      endif()
      file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
        "Performing D SOURCE FILE Test ${VAR} succeeded with the following output:\n"
        "${OUTPUT}\n"
        "Source was:\n${SOURCE}\n")
    else()
      if(NOT CMAKE_REQUIRED_QUIET)
        message(STATUS "Performing Test ${VAR} - Failed")
      endif()
      set(${VAR} "" CACHE INTERNAL "Test ${VAR}")
      file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
        "Performing D SOURCE FILE Test ${VAR} failed with the following output:\n"
        "${OUTPUT}\n"
        "Source was:\n${SOURCE}\n")
    endif()
  endif()
endmacro()
