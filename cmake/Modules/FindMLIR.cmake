# - Find MLIR headers and libraries.
#
# The following are set after configuration is done:
#   MLIR_FOUND         - ON if MLIR installation was found
#   MLIR_DIR           - Directory containing MLIRConfig.cmake
#   MLIR_INCLUDE_DIRS  - Directory containing MLIR include files
#   MLIR_INCLUDE_DIR   - Alias to MLIR_INCLUDE_DIRS
#   MLIR_LIBS          - List of MLIR libraries to link against

set(MLIR_FOUND OFF)

# Prefer finding MLIRConfig.cmake via CMake's CONFIG mode
set(_mlir_hints
    ${MLIR_DIR}
    "${LLVM_CMAKEDIR}/../mlir"
    "${LLVM_LIBRARY_DIRS}/cmake/mlir"
    "${LLVM_ROOT_DIR}/lib/cmake/mlir"
)

set(_saved_llvm_config "${LLVM_CONFIG}")

find_package(MLIR QUIET CONFIG HINTS ${_mlir_hints})

# upon calling find_package(MLIR ...) with CONFIG mode
# it loads and executes MLIRConfig.cmake and sets
# ${MLIR_CONFIG} to be path to MLIRConfig.cmake and in return it calls
# find_package(LLVM ...) with config mode which also sets
# ${LLVM_CONFIG} to be the path to LLVMConfig.cmake hence this line is needed
set(LLVM_CONFIG "${_saved_llvm_config}")

if(MLIR_FOUND)
    set(MLIR_INCLUDE_DIR ${MLIR_INCLUDE_DIRS})
    message(STATUS "Found MLIR: ${MLIR_DIR}")

    if(LDC_LINK_MANUALLY)
      set(MLIR_LIBS
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
    else()
      get_property(MLIR_ALL_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)

      set(MLIR_LIBS "")

      foreach(lib IN  LISTS MLIR_ALL_LIBS)
        get_target_property(_type ${lib} TYPE)
        if (_type STREQUAL "STATIC_LIBRARY")
          list(APPEND MLIR_LIBS ${lib})
        endif()
      endforeach()

      list(LENGTH MLIR_LIBS _n)
      message(STATUS "MLIR static libs: ${_n}")
      list(TRANSFORM MLIR_LIBS PREPEND "-l")
    endif()

else()
    if(NOT MLIR_FIND_QUIETLY)
        message(STATUS "Could not find MLIR (searched hints: ${_mlir_hints}). Set MLIR_DIR to the directory containing MLIRConfig.cmake.")
    endif()
endif()
