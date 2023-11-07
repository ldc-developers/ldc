//===-- rttibuilder.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/rttibuilder.h"

#include "dmd/aggregate.h"
#include "dmd/mangle.h"
#include "dmd/mtype.h"
#include "gen/arrays.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/linkage.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"
#include "ir/iraggr.h"
#include "ir/irfunction.h"

// in dmd/opover.d:
AggregateDeclaration *isAggregate(Type *t);

RTTIBuilder::RTTIBuilder(Type *baseType) {
  const auto ad = isAggregate(baseType);
  assert(ad && "not an aggregate type");

  DtoResolveDsymbol(ad);

  if (auto cd = ad->isClassDeclaration()) {
    const auto baseir = getIrAggr(cd);
    assert(baseir && "no IrAggr for TypeInfo base class");

    // just start with adding the vtbl
    push(baseir->getVtblSymbol());
    // and monitor
    push_null_vp();
  }
}

void RTTIBuilder::push(llvm::Constant *C) {
  // We need to explicitly zero any padding bytes as per TDPL §7.1.1 (and
  // also match the struct type lowering code here).
  const uint64_t fieldStart = llvm::alignTo(
      prevFieldEnd, gDataLayout->getABITypeAlign(C->getType()).value());

  const uint64_t paddingBytes = fieldStart - prevFieldEnd;
  if (paddingBytes) {
    llvm::Type *const padding = llvm::ArrayType::get(
        llvm::Type::getInt8Ty(gIR->context()), paddingBytes);
    inits.push_back(llvm::Constant::getNullValue(padding));
  }
  inits.push_back(C);
  prevFieldEnd = fieldStart + gDataLayout->getTypeAllocSize(C->getType());
}

void RTTIBuilder::push_null(Type *T) { push(getNullValue(DtoType(T))); }

void RTTIBuilder::push_null_vp() { push(getNullValue(getVoidPtrType())); }

void RTTIBuilder::push_typeinfo(Type *t) { push(DtoTypeInfoOf(Loc(), t)); }

void RTTIBuilder::push_string(const char *str) { push(DtoConstString(str)); }

void RTTIBuilder::push_null_void_array() {
  LLType *T = DtoType(Type::tvoid->arrayOf());
  push(getNullValue(T));
}

void RTTIBuilder::push_void_array(uint64_t dim, llvm::Constant *ptr) {
  push(DtoConstSlice(DtoConstSize_t(dim), DtoBitCast(ptr, getVoidPtrType())));
}

void RTTIBuilder::push_void_array(llvm::Constant *CI, Type *valtype,
                                  Dsymbol *mangle_sym) {
  OutBuffer initname;
  mangleToBuffer(mangle_sym, initname);
  initname.writestring(".rtti.voidarr.data");

  const LinkageWithCOMDAT lwc(TYPEINFO_LINKAGE_TYPE, needsCOMDAT());

  auto G = new LLGlobalVariable(gIR->module, CI->getType(), true, lwc.first, CI,
                                initname.peekChars());
  setLinkage(lwc, G);
  G->setAlignment(llvm::MaybeAlign(DtoAlignment(valtype)));

  push_void_array(getTypeAllocSize(CI->getType()), G);
}

void RTTIBuilder::push_array(llvm::Constant *CI, uint64_t dim, Type *valtype,
                             Dsymbol *mangle_sym) {
  std::string tmpStr(valtype->arrayOf()->toChars());
  tmpStr.erase(remove(tmpStr.begin(), tmpStr.end(), '['), tmpStr.end());
  tmpStr.erase(remove(tmpStr.begin(), tmpStr.end(), ']'), tmpStr.end());
  tmpStr.append("arr");

  OutBuffer initname;
  if (mangle_sym)
    mangleToBuffer(mangle_sym, initname);
  else
    initname.writestring(".ldc");
  initname.writestring(".rtti.");
  initname.writestring(tmpStr.c_str());
  initname.writestring(".data");

  const LinkageWithCOMDAT lwc(TYPEINFO_LINKAGE_TYPE, needsCOMDAT());

  auto G = new LLGlobalVariable(gIR->module, CI->getType(), true, lwc.first, CI,
                                initname.peekChars());
  setLinkage(lwc, G);
  G->setAlignment(llvm::MaybeAlign(DtoAlignment(valtype)));

  push_array(dim, DtoBitCast(G, DtoType(valtype->pointerTo())));
}

void RTTIBuilder::push_array(uint64_t dim, llvm::Constant *ptr) {
  push(DtoConstSlice(DtoConstSize_t(dim), ptr));
}

void RTTIBuilder::push_uint(unsigned u) { push(DtoConstUint(u)); }

void RTTIBuilder::push_size(uint64_t s) { push(DtoConstSize_t(s)); }

void RTTIBuilder::push_size_as_vp(uint64_t s) {
  push(llvm::ConstantExpr::getIntToPtr(DtoConstSize_t(s), getVoidPtrType()));
}

void RTTIBuilder::push_funcptr(FuncDeclaration *fd, Type *castto) {
  if (fd) {
    LLConstant *F = DtoCallee(fd);
    if (castto) {
      F = DtoBitCast(F, DtoType(castto));
    }
    push(F);
  } else if (castto) {
    push_null(castto);
  } else {
    push_null_vp();
  }
}

void RTTIBuilder::finalize(LLGlobalVariable *gvar) {
  LLStructType *st = isaStruct(gvar->getValueType());
  assert(st);

  // finalize the type if opaque (e.g., for ModuleInfos)
  if (st->isOpaque()) {
    std::vector<LLType *> fieldTypes;
    fieldTypes.reserve(inits.size());
    for (auto c : inits) {
      fieldTypes.push_back(c->getType());
    }
    st->setBody(fieldTypes);
  }

  // create the initializer
  LLConstant *tiInit = get_constant(st);

  // set the initializer
  gvar->setInitializer(tiInit);
}

LLConstant *RTTIBuilder::get_constant(LLStructType *initType) {
  assert(initType->getNumElements() == inits.size());

  std::vector<LLConstant *> castInits;
  castInits.reserve(inits.size());
  for (unsigned i = 0; i < inits.size(); ++i) {
    castInits.push_back(DtoBitCast(inits[i], initType->getElementType(i)));
  }

  return LLConstantStruct::get(initType, castInits);
}
