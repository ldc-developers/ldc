# - Try to find MLIR project at LLVM
#
# The following are set after configuration is done:
#   MLIR_FOUND
#   MLIR_ROOT_DIRS
#   MLIR_INCLUDE_DIRS
#   MLIR_BUILD_INCLUDE_DIR
project(ldc)

find_path(MLIR_ROOT_DIR NAMES "CONTRIBUTING.md" HINTS ${LLVM_ROOT_DIR}/../projects/mlir)

#Used to get the main header files
find_path(MLIR_INCLUDE_DIR NAMES "Parser.h" HINTS ${LLVM_ROOT_DIR}/../projects/mlir/include/mlir)

#Used to get StandardOps.h.inc
find_path(MLIR_BUILD_INCLUDE_DIR NAMES "cmake_install.cmake"
        HINTS ${LLVM_ROOT_DIR}/projects/mlir/include/mlir)

message("-- MLIR Dir: ${MLIR_ROOT_DIR}")
message("-- MLIR Include Dir: ${MLIR_INCLUDE_DIR}/..")
message("-- MLIR Build include dir: ${MLIR_BUILD_INCLUDE_DIR}/..")
message("flag")

set(MLIR_ROOT_DIRS ${MLIR_ROOT_DIR})
set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIR}/..)
set(MLIR_BUILD_INCLUDE_DIRS ${MLIR_BUILD_INCLUDE_DIR}/..)

# Handle the QUIETLY and REQUIRED arguments and set the MLIR_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(MLIR DEFAULT_MSG MLIR_ROOT_DIRS MLIR_INCLUDE_DIRS
                                            MLIR_BUILD_INCLUDE_DIRS)

mark_as_advanced(MLIR_ROOT_DIRS MLIR_INCLUDE_DIRS MLIR_BUILD_INCLUDE_DIRS)