# Check whether the linker supports a given flag.
#
# CHECK_LINK_FLAG(<flag> <var>)
#
#   <flag>       - Commandline flag passed to the linker
#   <var>        - variable to store whether the source code compiled
#                  Will be created as an internal cache variable.

include(CheckCSourceCompiles)

macro (CHECK_LINK_FLAG FLAG VAR)
   set(SAFE_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
   set(CMAKE_REQUIRED_FLAGS "-Wl,${FLAG}")

   CHECK_C_SOURCE_COMPILES("int main(void) { return 0; }" ${VAR})

   set (CMAKE_REQUIRED_FLAGS "${SAFE_CMAKE_REQUIRED_FLAGS}")
endmacro ()
