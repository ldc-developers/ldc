# Determines if host system uses musl libc
#
# CHECK_MUSL(<var>)
#
# ::
#
#   <var>        - variable to store whether the source code compiled
#                  Will be created as an internal cache variable.
#
# The command `ldd /bin/ls` will be used to determine musl shared library usage.

macro(CHECK_MUSL VAR)
  if(NOT DEFINED "${VAR}")
    if(UNIX)

      execute_process(COMMAND ldd /bin/ls
                  OUTPUT_VARIABLE OUTPUT
                  ERROR_VARIABLE OUTPUT
                  RESULT_VARIABLE RETVAL)

      if(${RETVAL})
        set(${VAR} 0)
      else()
        set(${VAR} 1)
      endif()

      if(${VAR} AND "${OUTPUT}" MATCHES "-musl-" )
        set(${VAR} 1 CACHE INTERNAL "Test ${VAR}")
        if(NOT CMAKE_REQUIRED_QUIET)
          message(STATUS "Detected musl libc")
        endif()
      else()
        set(${VAR} 0 CACHE INTERNAL "Test ${VAR}")
      endif()
    else()
      set(${VAR} 0 CACHE INTERNAL "Test ${VAR}")
    endif()
  endif()
endmacro()
