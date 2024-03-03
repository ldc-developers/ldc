//===-- function-inlining.cpp ---------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/function-inlining.h"

#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/globals.h"
#include "dmd/id.h"
#include "dmd/module.h"
#include "dmd/statement.h"
#include "dmd/template.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "gen/recursivevisitor.h"
#include "gen/uda.h"

using namespace dmd;

namespace {

/// An ASTVisitor that checks whether the number of statements is larger than a
/// certain number.
struct MoreThanXStatements : public StoppableVisitor {
  /// Are there more or fewer statements than `threshold`?.
  unsigned threshold;
  /// The statement count.
  unsigned count;

  explicit MoreThanXStatements(unsigned X) : threshold(X), count(0) {}

  using StoppableVisitor::visit;

  void visit(Statement *stmt) override {
    count++;
    if (count > threshold)
      stop = true;
  }
  void visit(Expression *exp) override {}
  void visit(Declaration *decl) override {}
  void visit(Initializer *init) override {}
  void visit(Dsymbol *) override {}
};

// Use a heuristic to determine if it could make sense to inline this fdecl.
// Note: isInlineCandidate is called _before_ semantic3 analysis of fdecl.
bool isInlineCandidate(FuncDeclaration &fdecl) {
  // Giving maximum inlining potential to LLVM should be possible, but we
  // restrict it to save some compile time.
  // return true;

  // TODO: make the heuristic more sophisticated?
  // In the end, LLVM will make the decision whether to _actually_ inline.
  // The statement count threshold is completely arbitrary. Also, all
  // statements are weighed the same.

  unsigned statementThreshold = 10;
  MoreThanXStatements statementCounter(statementThreshold);
  RecursiveWalker walker(&statementCounter, false);
  fdecl.fbody->accept(&walker);

  IF_LOG Logger::println("Contains %u statements or more (threshold = %u).",
                         statementCounter.count, statementThreshold);
  return statementCounter.count <= statementThreshold;
}

} // end anonymous namespace

bool skipCodegen(FuncDeclaration &fdecl) {
  if (fdecl.isFuncLiteralDeclaration()) // emitted into each referencing CU
    return false;

  for (FuncDeclaration *f = &fdecl; f;) {
    if (f->inNonRoot()) { // false if instantiated
      return true;
    }
    if (f->isNested()) {
      f = f->toParent2()->isFuncDeclaration();
    } else {
      break;
    }
  }
  return false;
}

bool defineAsExternallyAvailable(FuncDeclaration &fdecl) {
  IF_LOG Logger::println("Enter defineAsExternallyAvailable");
  LOG_SCOPE

  // Implementation note: try to do cheap checks first.

  if (fdecl.neverInline || fdecl.inlining == PINLINE::never) {
    IF_LOG Logger::println("pragma(inline, false) specified");
    return false;
  }

  // pragma(inline, true) functions will be inlined even at -O0
  if (fdecl.inlining == PINLINE::always) {
    IF_LOG Logger::println(
        "pragma(inline, true) specified, overrides cmdline flags");
  } else if (!willCrossModuleInline()) {
    IF_LOG Logger::println("Commandline flags indicate no inlining");
    return false;
  }

  if (fdecl.isFuncLiteralDeclaration()) {
    // defined as discardable linkonce_odr in each referencing CU
    IF_LOG Logger::println("isFuncLiteralDeclaration() == true");
    return false;
  }
  if (fdecl.isUnitTestDeclaration()) {
    IF_LOG Logger::println("isUnitTestDeclaration() == true");
    return false;
  }
  if (fdecl.isFuncAliasDeclaration()) {
    IF_LOG Logger::println("isFuncAliasDeclaration() == true");
    return false;
  }
  if (!fdecl.fbody) {
    IF_LOG Logger::println("No function body available for inlining");
    return false;
  }

  // Because the frontend names `__invariant*` functions differently depending
  // on the compilation order, we cannot emit the `__invariant` wrapper that
  // calls the `__invariant*` functions.
  // This is a workaround, the frontend needs to be changed such that the
  // __invariant* names no longer depend on semantic analysis order.
  // See https://github.com/ldc-developers/ldc/issues/1678
  if (fdecl.isInvariantDeclaration()) {
    IF_LOG Logger::println("__invariant cannot be emitted.");
    return false;
  }

  Module *module = fdecl.getModule();
  if (module == gIR->dmodule || (module->isRoot() && global.params.oneobj)) {
    IF_LOG Logger::println(
        "Function will be regularly defined later in this compilation unit.");
    return false;
  }

  // Weak-linkage functions can not be inlined.
  if (hasWeakUDA(&fdecl)) {
    IF_LOG Logger::println("@weak functions cannot be inlined.");
    return false;
  }

  if (fdecl.inlining != PINLINE::always && !isInlineCandidate(fdecl))
    return false;

  IF_LOG Logger::println("Potential inlining candidate");

  if (fdecl.semanticRun < PASS::semantic3) {
    IF_LOG Logger::println("Do semantic analysis");
    LOG_SCOPE

    // The inlining is aggressive and may give semantic errors that are
    // forward referencing errors. Simply avoid those cases for inlining.
    unsigned errors = global.startGagging();
    global.gaggedForInlining = true;

    bool semantic_error = false;
    if (functionSemantic3(&fdecl)) {
      Module::runDeferredSemantic3();
    } else {
      IF_LOG Logger::println("Failed functionSemantic3.");
      semantic_error = true;
    }

    global.gaggedForInlining = false;
    if (global.endGagging(errors) || semantic_error) {
      IF_LOG Logger::println("Errors occured during semantic analysis.");
      return false;
    }
    assert(fdecl.semanticRun >= PASS::semantic3done);
  }

  // FuncDeclaration::naked is set by the AsmParser during semantic3 analysis,
  // and so this check can only be done at this late point.
  if (fdecl.isNaked()) {
    IF_LOG Logger::println("Naked asm functions cannot be inlined.");
    return false;
  }

  IF_LOG Logger::println("defineAsExternallyAvailable? Yes.");
  return true;
}
