//===-- gen/coverage.h - Code Coverage Analysis -----------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/coverage.h"

#include "dmd/module.h"
#include "driver/cl_options.h"
#include "gen/irstate.h"
#include "gen/logger.h"

void emitCoverageLinecountInc(const Loc &loc) {
  Module *m = gIR->dmodule;

  // Only emit coverage increment for locations in the source of the current
  // module
  // (for example, 'inlined' methods from other source files should be skipped).
  if (!global.params.cov || !loc.linnum() || !loc.filename() ||
      !m->d_cover_data || strcmp(m->srcfile.toChars(), loc.filename()) != 0) {
    return;
  }

  const unsigned line = loc.linnum() - 1; // convert to 0-based line# index
  assert(line < m->numlines);

  IF_LOG Logger::println("Coverage: increment _d_cover_data[%d]", line);
  LOG_SCOPE;

  // Increment the line counter:
  // Get GEP into _d_cover_data array...
  LLType *i32Type = LLType::getInt32Ty(gIR->context());
  LLConstant *idxs[] = {DtoConstUint(0), DtoConstUint(line)};
  LLValue *ptr = llvm::ConstantExpr::getGetElementPtr(
      LLArrayType::get(i32Type, m->numlines), m->d_cover_data, idxs, true);
  // ...and generate the "increment" instruction(s)
  switch (opts::coverageIncrement) {
  case opts::CoverageIncrement::_default: // fallthrough
  case opts::CoverageIncrement::atomic:
    // Do an atomic increment, so this works when multiple threads are executed.
    gIR->ir->CreateAtomicRMW(llvm::AtomicRMWInst::Add, ptr, DtoConstUint(1),
#if LDC_LLVM_VER >= 1300
                             llvm::Align(4),
#endif
                             llvm::AtomicOrdering::Monotonic);
    break;
  case opts::CoverageIncrement::nonatomic: {
    // Do a non-atomic increment, user is responsible for correct results with
    // multithreaded execution
    llvm::LoadInst *load =
        gIR->ir->CreateAlignedLoad(i32Type, ptr, llvm::Align(4));
    llvm::StoreInst *store = gIR->ir->CreateAlignedStore(
        gIR->ir->CreateAdd(load, DtoConstUint(1)), ptr, llvm::Align(4));
    // add !nontemporal attribute, to inform the optimizer that caching is not
    // needed
    llvm::MDNode *node = llvm::MDNode::get(
        gIR->context(), llvm::ConstantAsMetadata::get(DtoConstInt(1)));
    load->setMetadata("nontemporal", node);
    store->setMetadata("nontemporal", node);
    break;
  }
  case opts::CoverageIncrement::boolean: {
    // Do a boolean set, avoiding a memory read (blocking) and threading issues
    // at the cost of not "counting"
    llvm::StoreInst *store =
        gIR->ir->CreateAlignedStore(DtoConstUint(1), ptr, llvm::Align(4));
    // add !nontemporal attribute, to inform the optimizer that caching is not
    // needed
    llvm::MDNode *node = llvm::MDNode::get(
        gIR->context(), llvm::ConstantAsMetadata::get(DtoConstInt(1)));
    store->setMetadata("nontemporal", node);
    break;
  }
  }

  // Set the 'counter valid' bit to 1 for this line of code
  unsigned num_sizet_bits = gDataLayout->getTypeSizeInBits(DtoSize_t());
  unsigned idx = line / num_sizet_bits;
  unsigned bitidx = line % num_sizet_bits;
  IF_LOG Logger::println("_d_cover_valid[%d] |= (1 << %d)", idx, bitidx);
  m->d_cover_valid_init[idx] |= (size_t(1) << bitidx);
}
