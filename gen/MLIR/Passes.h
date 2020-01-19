//
// Created by Roberto Rosmaninho on 18/12/19.
//

// =============================================================================
//
// This file exposes the entry points to create compiler passes for D.
//
//===----------------------------------------------------------------------===/

#ifndef LDC_PASSES_H
#define LDC_PASSES_H

#include <memory>

namespace mlir{
  class Pass;

  namespace D{
    /// Create a pass for lowering to operations in the `Affine` and `Std`
    /// dialects, for a subset of the D IR (e.g. matmul).
    std::unique_ptr<mlir::Pass> createLowerToStandardPass();
  } // end namespace D
} // end namespace mlir


#endif //LDC_PASSES_H
