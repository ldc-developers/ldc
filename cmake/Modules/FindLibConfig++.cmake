# - Find the libconfig++ includes and library
#
# This module defines
# LIBCONFIG++_INCLUDE_DIR, where to find libconfig++ include files, etc.
# LIBCONFIG++_LIBRARY, the library to link against to use libconfig++.
# LIBCONFIG++_FOUND, If false, do not try to use libconfig++.

find_path(LIBCONFIG++_INCLUDE_DIR libconfig.h++)

find_library(LIBCONFIG++_LIBRARY config++)

# Use the default CMake facilities for handling QUIET/REQUIRED.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibConfig++
    REQUIRED_VARS LIBCONFIG++_INCLUDE_DIR LIBCONFIG++_LIBRARY)
