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
#include "dmd/declaration.h"
#include "ir/irfunction.h"
#include "driver/cl_options.h"
#include "gen/irstate.h"
#include "gen/logger.h"

void createCoverageData(Module *source) {
  auto &data = gIR->coverage[source];
  auto type = llvm::ArrayType::get(LLType::getInt32Ty(gIR->context()), source->numlines);
  llvm::Constant *init = llvm::ConstantAggregateZero::get(type);
  // Imported fragments contribute runtime hits only.
  if (source == gIR->dmodule && source->ctfe_cov) {
    std::vector<unsigned> hits(source->numlines);
    source->initCoverageDataWithCtfeCoverage(hits.data());
    init = llvm::ConstantDataArray::get(gIR->context(), hits);
  }
  data.counts = new llvm::GlobalVariable(gIR->module, type, false,
      LLGlobalValue::InternalLinkage, init, "_d_cover_data");
  const unsigned wordBits = gDataLayout->getTypeSizeInBits(DtoSize_t());
  data.valid.resize((source->numlines + wordBits - 1) / wordBits);
}

void emitCoverageLinecountInc(Loc loc) {
  if (gIR->coverage.empty() || !loc.linnum() || !loc.filename())
    return;

  Module *m = gIR->dmodule;
  if (strcmp(m->srcfile.toChars(), loc.filename()) != 0) {
    if (gIR->funcGenStates.empty())
      return;
    m = gIR->func()->decl->getModule();
    if (!m || strcmp(m->srcfile.toChars(), loc.filename()) != 0)
      return;
  }
  const unsigned line = loc.linnum() - 1;
  if (line >= m->numlines)
    return;
  if (!gIR->coverage.count(m))
    createCoverageData(m);
  auto &data = gIR->coverage.find(m)->second;

  IF_LOG Logger::println("Coverage: increment _d_cover_data[%d]", line);
  LOG_SCOPE;

  // Increment the line counter:
  // Get GEP into _d_cover_data array...
  LLType *i32Type = LLType::getInt32Ty(gIR->context());
  LLConstant *idxs[] = {DtoConstUint(0), DtoConstUint(line)};
  LLValue *ptr = llvm::ConstantExpr::getGetElementPtr(
      LLArrayType::get(i32Type, m->numlines), data.counts, idxs, true);
  // ...and generate the "increment" instruction(s)
  switch (opts::coverageIncrement) {
  case opts::CoverageIncrement::_default: // fallthrough
  case opts::CoverageIncrement::atomic:
    // Do an atomic increment, so this works when multiple threads are executed.
    gIR->ir->CreateAtomicRMW(llvm::AtomicRMWInst::Add, ptr, DtoConstUint(1),
                             llvm::Align(4), llvm::AtomicOrdering::Monotonic);
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
  data.valid[idx] |= uint64_t(1) << bitidx;
}
