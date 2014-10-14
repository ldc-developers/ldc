# - Find the libconfig includes and library
#
# This module defines
# LIBCONFIG_INCLUDE_DIR, where to find libconfig include files, etc.
# LIBCONFIG_LIBRARY, the library to link against to use libconfig.
# LIBCONFIG_FOUND, If false, do not try to use libconfig.

find_path(LIBCONFIG_INCLUDE_DIR libconfig.h)

find_library(LIBCONFIG_LIBRARY config)

# Use the default CMake facilities for handling QUIET/REQUIRED.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibConfig
    REQUIRED_VARS LIBCONFIG_INCLUDE_DIR LIBCONFIG_LIBRARY)
