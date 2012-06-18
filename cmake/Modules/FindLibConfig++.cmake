# Find the libconfig++ includes and library
#
# This module defines
# LIBCONFIG++_INCLUDE_DIR, where to find libconfig++ include files, etc.
# LIBCONFIG++_LIBRARy, the library to link against to use libconfig++.
# LIBCONFIG++_FOUND, If false, do not try to use libconfig++.

# also defined, but not for general use are
# LIBCONFIG++_LIBRARY, where to find the libconfig++ library.

set(LIBCONFIG++_FOUND TRUE)

find_path(LIBCONFIG++_INCLUDE_DIR libconfig.h++)

find_library(LIBCONFIG++_LIBRARY config++)

if (LIBCONFIG++_INCLUDE_DIR AND LIBCONFIG++_LIBRARY)
   set(LIBCONFIG++_FOUND TRUE)
endif (LIBCONFIG++_INCLUDE_DIR AND LIBCONFIG++_LIBRARY)

# Use the default CMake facilities for handling QUIET/REQUIRED.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LLVM
    REQUIRED_VARS LIBCONFIG++_INCLUDE_DIR LIBCONFIG++_LIBRARY)
