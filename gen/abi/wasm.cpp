//===-- wasm.cpp ----------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// see https://github.com/WebAssembly/tool-conventions/blob/main/BasicCABI.md
//
//===----------------------------------------------------------------------===//

#include "gen/abi/generic.h"

using namespace dmd;

namespace {
Type *getSingleWrappedScalarType(Type *t) {
  t = t->toBasetype();

  if (auto ts = t->isTypeStruct()) {
    if (ts->sym->fields.length != 1)
      return nullptr;
    return getSingleWrappedScalarType(ts->sym->fields[0]->type);
  }

  if (auto tsa = t->isTypeSArray()) {
    if (tsa->dim->toInteger() != 1)
      return nullptr;
    return getSingleWrappedScalarType(tsa->nextOf());
  }

  return t->isscalar()
                 // some more pointers:
                 || t->ty == TY::Tclass || t->ty == TY::Taarray
             ? t
             : nullptr;
}
}

struct WasmTargetABI : TargetABI {
  static bool isDirectlyPassedAggregate(Type *t) {
    // Structs and static arrays are generally passed byval, except for POD
    // aggregates wrapping a single scalar type.

    if (!DtoIsInMemoryOnly(t)) // not a struct or static array
      return false;

    // max scalar type size is 16 (`real`); return early if larger
    if (size(t) > 16 || !isPOD(t))
      return false;

    Type *singleWrappedScalarType = getSingleWrappedScalarType(t);
    return singleWrappedScalarType &&
           // not passed directly if over-aligned
           DtoAlignment(t) <= DtoAlignment(singleWrappedScalarType);
  }

  bool passByVal(TypeFunction *, Type *t) override {
    return DtoIsInMemoryOnly(t) && isPOD(t) && !isDirectlyPassedAggregate(t);
  }
};

// The public getter for abi.cpp.
TargetABI *getWasmTargetABI() { return new WasmTargetABI; }
