//===-- gen/coverage.h - Code Coverage Analysis -----------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/coverage.h"

#include "mars.h"
#include "module.h"
#include "gen/irstate.h"
#include "gen/logger.h"

void emitCoverageLinecountInc(Loc &loc) {
  Module *m = gIR->dmodule;

  // Only emit coverage increment for locations in the source of the current
  // module
  // (for example, 'inlined' methods from other source files should be skipped).
  if (!global.params.cov || !loc.linnum || !loc.filename || !m->d_cover_data ||
      strcmp(m->srcfile->name->toChars(), loc.filename) != 0) {
    return;
  }

  const unsigned line = loc.linnum - 1; // convert to 0-based line# index
  assert(line < m->numlines);

  IF_LOG Logger::println("Coverage: increment _d_cover_data[%d]", line);
  LOG_SCOPE;

  // Get GEP into _d_cover_data array
  LLConstant *idxs[] = {DtoConstUint(0), DtoConstUint(line)};
  LLValue *ptr = llvm::ConstantExpr::getGetElementPtr(
      LLArrayType::get(LLType::getInt32Ty(gIR->context()), m->numlines),
      m->d_cover_data, idxs, true);

  // Do an atomic increment, so this works when multiple threads are executed.
  gIR->ir->CreateAtomicRMW(llvm::AtomicRMWInst::Add, ptr, DtoConstUint(1),
#if LDC_LLVM_VER >= 309
                           llvm::AtomicOrdering::Monotonic
#else
                           llvm::Monotonic
#endif
  );

  unsigned num_sizet_bits = gDataLayout->getTypeSizeInBits(DtoSize_t());
  unsigned idx = line / num_sizet_bits;
  unsigned bitidx = line % num_sizet_bits;

  IF_LOG Logger::println("_d_cover_valid[%d] |= (1 << %d)", idx, bitidx);

  m->d_cover_valid_init[idx] |= (size_t(1) << bitidx);
}
