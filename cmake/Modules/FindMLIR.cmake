# - Find MLIR headers and libraries.
#
# The following are set after configuration is done:
#   MLIR_FOUND         - ON if MLIR installation was found
#   MLIR_DIR           - Directory containing MLIRConfig.cmake
#   MLIR_INCLUDE_DIRS  - Directory containing MLIR include files
#   MLIR_INCLUDE_DIR   - Alias to MLIR_INCLUDE_DIRS
#   MLIR_LIBRARIES     - List of MLIR libraries to link against
#   MLIR_TABLEGEN_EXE  - The mlir-tblgen executable (if present)

set(MLIR_FOUND OFF)

# Prefer finding MLIRConfig.cmake via CMake's CONFIG mode
set(_mlir_hints
    ${MLIR_DIR}
    "${LLVM_CMAKEDIR}/../mlir"
    "${LLVM_LIBRARY_DIRS}/cmake/mlir"
    "${LLVM_ROOT_DIR}/lib/cmake/mlir"
)

find_package(MLIR QUIET CONFIG HINTS ${_mlir_hints})

if(MLIR_FOUND)
    set(MLIR_INCLUDE_DIR ${MLIR_INCLUDE_DIRS})
    message(STATUS "Found MLIR: ${MLIR_DIR}")

    set(_mlir_targets
        # Target Translation to LLVM IR
        MLIRTargetLLVMIRExport
        MLIRToLLVMIRTranslationRegistration
        MLIRBuiltinToLLVMIRTranslation
        MLIRLLVMToLLVMIRTranslation
        MLIRTargetLLVMIRImport

        # Dialects
        MLIRLLVMDialect
        MLIRFuncDialect
        MLIRArithDialect
        MLIRControlFlowDialect
        MLIRSCFDialect

        # Conversions to LLVM
        MLIRFuncToLLVM
        MLIRArithToLLVM
        MLIRControlFlowToLLVM
        MLIRReconcileUnrealizedCasts
        MLIRSCFToControlFlow

        # Transforms & Passes
        MLIRPass
        MLIRTransforms

        # Core IR & Parsing
        MLIRParser
        MLIRIR
        MLIRSupport
    )

    set(MLIR_LIBRARIES ${_mlir_targets})
else()
    if(NOT MLIR_FIND_QUIETLY)
        message(STATUS "Could not find MLIR (searched hints: ${_mlir_hints}). Set MLIR_DIR to the directory containing MLIRConfig.cmake.")
    endif()
endif()
