//===-- gen/trycatch.h - Try-catch scopes -----------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_TRYCATCH_H
#define LDC_GEN_TRYCATCH_H

#include <stddef.h>
#include <vector>

struct IRState;
class TryCatchStatement;

namespace llvm {
class BasicBlock;
class GlobalVariable;
class MDNode;
}

////////////////////////////////////////////////////////////////////////////////

class TryCatchScope {
public:
  /// Stores information to be able to branch to a catch clause if it matches.
  ///
  /// Each catch body is emitted only once, but may be target from many landing
  /// pads (in case of nested catch or cleanup scopes).
  struct CatchBlock {
    /// The ClassInfo reference corresponding to the type to match the
    /// exception object against.
    llvm::GlobalVariable *classInfoPtr;
    /// The block to branch to if the exception type matches.
    llvm::BasicBlock *bodyBB;
    // PGO branch weights for the exception type match branch.
    // (first weight is for match, second is for mismatch)
    llvm::MDNode *branchWeights;
  };

  TryCatchScope(TryCatchStatement *stmt, llvm::BasicBlock *endbb,
                size_t cleanupScope);

  size_t getCleanupScope() const { return cleanupScope; }
  bool isCatchingNonExceptions() const { return catchesNonExceptions; }

  void emitCatchBodies(IRState &irs);
  const std::vector<CatchBlock> &getCatchBlocks() const;

private:
  TryCatchStatement *stmt;
  llvm::BasicBlock *endbb;
  size_t cleanupScope;
  bool catchesNonExceptions;

  std::vector<CatchBlock> catchBlocks;

  void emitCatchBodiesMSVC(IRState &irs);
};

////////////////////////////////////////////////////////////////////////////////

class TryCatchScopes {
public:
  TryCatchScopes(IRState &irs) : irs(irs) {}

  void push(TryCatchStatement *stmt, llvm::BasicBlock *endbb);
  void pop();
  bool empty() const { return tryCatchScopes.empty(); }

  /// Indicates whether there are any active catch blocks that handle
  /// non-Exception Throwables.
  bool isCatchingNonExceptions() const;

  /// Emits a landing pad to honor all the active cleanups and catches.
  llvm::BasicBlock *emitLandingPad();

private:
  IRState &irs;
  std::vector<TryCatchScope> tryCatchScopes;

  llvm::BasicBlock *emitLandingPadMSVC(size_t cleanupScope);
};

#endif
