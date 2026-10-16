# - Find MLIR headers and libraries.
#
# MLIR is a sub-project of LLVM and installs into the same prefix, so it is
# located relative to the LLVM installation that FindLLVM.cmake discovered via
# llvm-config. MLIRConfig.cmake is deliberately not used: it loads
# LLVMConfig.cmake, which overrides the LLVM_* variables derived from
# llvm-config and registers LLVM's own dependencies (zstd, zlib, ...) as CMake
# targets, clashing with LDC's own lookups of those packages.
#
# Inputs (set by FindLLVM.cmake):
#   LLVM_INCLUDE_DIRS  - LLVM include directory (MLIR headers are in <dir>/mlir)
#   LLVM_LIBRARY_DIRS  - LLVM library directory
#
# Outputs:
#   MLIR_FOUND         - ON if MLIR headers and libraries were found
#   MLIR_INCLUDE_DIRS  - directory containing the mlir/ headers
#   MLIR_LIBS          - MLIR libraries in the same format as LLVM_LIBRARIES:
#                        -lMLIRFoo on Posix, MLIRFoo.lib on MSVC

set(MLIR_FOUND OFF)
set(MLIR_INCLUDE_DIRS "")
set(MLIR_LIBS "")

if(IS_DIRECTORY "${LLVM_INCLUDE_DIRS}/mlir")
    set(MLIR_INCLUDE_DIRS "${LLVM_INCLUDE_DIRS}")

    if(MSVC)
        file(GLOB mlir_lib_files "${LLVM_LIBRARY_DIRS}/MLIR*.lib")
    else()
        file(GLOB mlir_lib_files "${LLVM_LIBRARY_DIRS}/libMLIR*.a")
    endif()

    set(mlir_names "")

    foreach(file IN LISTS mlir_lib_files)
        # libMLIRIR.a / MLIRIR.lib -> MLIRIR transformation
        get_filename_component(name_of_file "${file}" NAME_WE)
        string(REGEX REPLACE "^lib" "" name_of_file "${name_of_file}")
        if(name_of_file MATCHES "^MLIR")
            list(APPEND mlir_names "${name_of_file}")
        endif()
    endforeach()

    if(mlir_names)
        set(MLIR_FOUND ON)

        if(MSVC)
            list(TRANSFORM mlir_names APPEND ".lib" OUTPUT_VARIABLE MLIR_LIBS)
        else()
            list(TRANSFORM mlir_names PREPEND "-l" OUTPUT_VARIABLE MLIR_LIBS)
        endif()

        list(LENGTH mlir_names mlir_count)
        message(STATUS "Found MLIR: ${mlir_count} libraries in ${LLVM_LIBRARY_DIRS}")
    else()
        message(STATUS "MLIR headers found, but no MLIR libraries in ${LLVM_LIBRARY_DIRS}")
    endif()
else()
    message(STATUS "Could not find MLIR: no ${LLVM_INCLUDE_DIRS}/mlir directory")
endif()

unset(mlir_lib_files)
unset(mlir_names)
unset(mlir_count)
unset(file)
unset(name_of_file)
