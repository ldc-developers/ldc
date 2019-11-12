//===-- MLIR/Dialect.h - Generate Statements MLIR code ---*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the IR Dialect for D Programming Language.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/Function.h"

namespace mlir {
namespace D {

/// This is the definition of the Toy dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types (in its
/// constructor). It can also override some general behavior exposed via virtual
/// methods.
class DDialect : public mlir::Dialect{
public:
  explicit DDialect(mlir::MLIRContext *context);

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static llvm::StringRef getDialectNamespace() { return "D"; }
};

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "Ops.h.inc"

} // end namespace toy
} // end namespace mlir

#endif //LDC_MLIR_ENABLED
