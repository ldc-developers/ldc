# - Try to find MLIR project at LLVM
#
# The following are set after configuration is done:
#   MLIR_FOUND
#   MLIR_ROOT_DIRS
#   MLIR_INCLUDE_DIRS
#   MLIR_BUILD_INCLUDE_DIR
project(ldc)

find_path(MLIR_ROOT_DIR NAMES "WritingAPass.md" HINTS ${LLVM_ROOT_DIR}/../mlir/docs)

#Used to get the main header files
find_path(MLIR_INCLUDE_DIR NAMES "DialectConversion.h" HINTS 
          ${MLIR_ROOT_DIR}/include/mlir/Transforms ${LLVM_INCLUDE_DIRS}/mlir/Transforms/)
set(MLIR_INCLUDE_DIR ${MLIR_INCLUDE_DIR}/..)

#Lib directories
find_path(MLIR_LIB_DIR NAMES "AffineExpr.cpp" HINTS
          ${MLIR_ROOT_DIR}/lib/IR ${LLVM_LIBRARY_DIRS})

#Used to get StandardOps.h.inc
find_path(MLIR_BUILD_INCLUDE_DIR NAMES "cmake_install.cmake"
        HINTS ${LLVM_ROOT_DIR}/tools/mlir/include/mlir)

if(EXISTS ${MLIR_ROOT_DIR})
set(MLIR_ROOT_DIR ${MLIR_ROOT_DIR}/..)

message(STATUS "MLIR Dir: ${MLIR_ROOT_DIR}")
message(STATUS "MLIR Include Dir: ${MLIR_INCLUDE_DIR}")
message(STATUS "MLIR Lib Dir: ${MLIR_LIB_DIR}/..")
message(STATUS "MLIR Build Include Dir: ${MLIR_BUILD_INCLUDE_DIR}/..")

set(MLIR_ROOT_DIRS ${MLIR_ROOT_DIR})
set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIR})
set(MLIR_BUILD_INCLUDE_DIRS ${MLIR_BUILD_INCLUDE_DIR}/..)
set(MLIR_LIB_DIRS ${MLIR_LIB_DIR}/..)

else()

message(STATUS "MLIR Include Dir: ${MLIR_INCLUDE_DIR}")
set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIR})

endif()

#Resources to automatically generate Ops.h.inc and Ops.cpp.inc, assuming that
#mlir-tblgen is already built and is on the usual directory at llvm dir
set(MLIR_MAIN_SRC_DIR ${MLIR_INCLUDE_DIR}/.. )
set(MLIR_TABLEGEN_EXE ${LLVM_ROOT_DIR}/bin/mlir-tblgen)

function(mlir_tablegen)
    cmake_parse_arguments(
     ARG
     "NAME"
     "TARGET;OUTS;FLAG;SRCS"
     ${ARGN}
     )

    MESSAGE(STATUS "Setting target for Ops_" ${ARG_TARGET})

    set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRCS}
            PARENT_SCOPE)
    #mlir-tblgen ops.td --gen-op-* -I*-o=ops.*.inc
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_OUTS}
            COMMAND ${MLIR_TABLEGEN_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRCS} -I${MLIR_MAIN_SRC_DIR} -I${MLIR_INCLUDE_DIR}  -o=${CMAKE_CURRENT_SOURCE_DIR}/${ARG_OUTS}
            ARGS ${ARG_FLAG}
    )
    add_custom_target(Ops_${ARG_TARGET} ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_OUTS})
endfunction()

# Handle the QUIETLY and REQUIRED arguments and set the MLIR_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)

if(EXISTS ${MLIR_ROOT_DIR})
find_package_handle_standard_args(MLIR DEFAULT_MSG MLIR_ROOT_DIRS MLIR_INCLUDE_DIRS
                                            MLIR_BUILD_INCLUDE_DIRS MLIR_LIB_DIRS)

mark_as_advanced(MLIR_ROOT_DIRS MLIR_INCLUDE_DIRS MLIR_BUILD_INCLUDE_DIRS MLIR_LIB_DIRS)
else()
find_package_handle_standard_args(MLIR DEFAULT_MSG  MLIR_INCLUDE_DIRS)
mark_as_advanced(MLIR_INCLUDE_DIRS)
endif()

