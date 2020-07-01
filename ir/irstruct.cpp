//===-- irstruct.cpp ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/errors.h"
#include "dmd/mangle.h"
#include "dmd/mtype.h"
#include "dmd/template.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/mangling.h"
#include "gen/rttibuilder.h"
#include "gen/runtime.h"
#include "gen/structs.h"
#include "gen/tollvm.h"
#include "gen/typinf.h"
#include "ir/iraggr.h"
#include "ir/irtypeclass.h"

namespace {
LLStructType* getTypeInfoStructMemType() {
  Type *t = getStructTypeInfoType();
  DtoType(t);
  IrTypeClass *tc = t->ctype->isClass();
  assert(tc && "invalid TypeInfo_Struct type");

  return llvm::cast<LLStructType>(tc->getMemoryLLType());
}
}

LLGlobalVariable* IrStruct::getTypeInfoSymbol() {
  if (typeInfo) {
    return typeInfo;
  }

  OutBuffer mangledName;
  mangledName.writestring("TypeInfo_S");
  mangleToBuffer(aggrdecl, &mangledName);
  const auto length = mangledName.length();
  mangledName.prependstring(("_D" + std::to_string(length)).c_str());
  mangledName.writestring("6__initZ");

  const auto irMangle = getIRMangledVarName(mangledName.peekChars(), LINKd);

  // We need to keep the symbol mutable as the type is not declared as
  // immutable on the D side, and e.g. synchronized() can be used on the
  // implicit monitor.
  typeInfo = declareGlobal(aggrdecl->loc, gIR->module,
                           getTypeInfoStructMemType(), irMangle, false);

  emitTypeInfoMetadata(typeInfo, aggrdecl->type);

  return typeInfo;
}

LLConstant *IrStruct::getTypeInfoInit() {
  if (constTypeInfo) {
    return constTypeInfo;
  }

  auto sd = aggrdecl->isStructDeclaration();
  IF_LOG Logger::println("Defining TypeInfo for struct: %s", sd->toChars());
  LOG_SCOPE;

  TypeStruct *ts = sd->type->isTypeStruct();

  // check declaration in object.d
  const auto structTypeInfoType = getStructTypeInfoType();
  const auto structTypeInfoDecl = Type::typeinfostruct;

  // For x86_64 (except Win64) and AAPCS64 targets, class TypeInfo_Struct
  // contains 2 additional fields (m_arg1/m_arg2) which are used for the
  // TypeInfo-based core.stdc.stdarg.va_arg implementations in druntime.
  const auto &triple = *global.params.targetTriple;
  const auto arch = triple.getArch();
  const bool withArgTypes =
      (arch == llvm::Triple::x86_64 && !triple.isOSWindows()) ||
      (!triple.isOSDarwin() && // Apple uses a simpler scheme
       (arch == llvm::Triple::aarch64 || arch == llvm::Triple::aarch64_be));
  const unsigned expectedFields = 11 + (withArgTypes ? 2 : 0);
  const unsigned actualFields =
      structTypeInfoDecl->fields.length -
      1; // union of xdtor/xdtorti counts as 2 overlapping fields
  if (actualFields != expectedFields) {
    error(Loc(), "Unexpected number of `object.TypeInfo_Struct` fields; "
                 "druntime version does not match compiler");
    fatal();
  }

  RTTIBuilder b(structTypeInfoType);

  LLStructType *memoryLLType = getTypeInfoStructMemType();

  // handle opaque structs
  if (!sd->members) {
    Logger::println("is opaque struct, emitting dummy TypeInfo_Struct");

    b.push_null_void_array(); // name
    b.push_null_void_array(); // m_init
    b.push_null_vp();         // xtoHash
    b.push_null_vp();         // xopEquals
    b.push_null_vp();         // xopCmp
    b.push_null_vp();         // xtoString
    b.push_uint(0);           // m_flags
    b.push_null_vp();         // xdtor/xdtorti
    b.push_null_vp();         // xpostblit
    b.push_uint(0);           // m_align
    if (withArgTypes) {
      b.push_null_vp(); // m_arg1
      b.push_null_vp(); // m_arg2
    }
    b.push_null_vp(); // m_RTInfo

    constTypeInfo = b.get_constant(memoryLLType);
    return constTypeInfo;
  }

  // can't emit typeinfo for forward declarations
  if (sd->sizeok != SIZEOKdone) {
    sd->error("cannot emit `TypeInfo` for forward declaration");
    fatal();
  }

  DtoResolveStruct(sd);

  if (TemplateInstance *ti = sd->isInstantiated()) {
    if (!ti->needsCodegen()) {
      assert(ti->minst || sd->requestTypeInfo);

      // We won't emit ti, so emit the special member functions in here.
      if (sd->xeq && sd->xeq != StructDeclaration::xerreq &&
          sd->xeq->semanticRun >= PASSsemantic3) {
        Declaration_codegen(sd->xeq);
      }
      if (sd->xcmp && sd->xcmp != StructDeclaration::xerrcmp &&
          sd->xcmp->semanticRun >= PASSsemantic3) {
        Declaration_codegen(sd->xcmp);
      }
      if (FuncDeclaration *ftostr = search_toString(sd)) {
        if (ftostr->semanticRun >= PASSsemantic3)
          Declaration_codegen(ftostr);
      }
      if (sd->xhash && sd->xhash->semanticRun >= PASSsemantic3) {
        Declaration_codegen(sd->xhash);
      }
      if (sd->postblit && sd->postblit->semanticRun >= PASSsemantic3) {
        Declaration_codegen(sd->postblit);
      }
      if (sd->dtor && sd->dtor->semanticRun >= PASSsemantic3) {
        Declaration_codegen(sd->dtor);
      }
      if (sd->tidtor && sd->tidtor->semanticRun >= PASSsemantic3) {
        Declaration_codegen(sd->tidtor);
      }
    }
  }

  IrStruct *iraggr = getIrAggr(sd);

  // string name
  b.push_string(sd->toPrettyChars());

  // void[] m_init
  // The protocol is to write a null pointer for zero-initialized structs.
  // The length field is always needed for tsize().
  llvm::Constant *initPtr;
  if (ts->isZeroInit(Loc())) {
    initPtr = getNullValue(getVoidPtrType());
  } else {
    initPtr = iraggr->getInitSymbol();
  }
  b.push_void_array(getTypeStoreSize(DtoType(ts)), initPtr);

  // function xtoHash
  FuncDeclaration *fd = sd->xhash;
  b.push_funcptr(fd);

  // function xopEquals
  fd = sd->xeq;
  b.push_funcptr(fd);

  // function xopCmp
  fd = sd->xcmp;
  b.push_funcptr(fd);

  // function xtoString
  fd = search_toString(sd);
  b.push_funcptr(fd);

  // uint m_flags
  unsigned hasptrs = ts->hasPointers() ? 1 : 0;
  b.push_uint(hasptrs);

  // function xdtor/xdtorti
  b.push_funcptr(sd->tidtor);

  // function xpostblit
  FuncDeclaration *xpostblit = sd->postblit;
  if (xpostblit && sd->postblit->storage_class & STCdisable) {
    xpostblit = nullptr;
  }
  b.push_funcptr(xpostblit);

  // uint m_align
  b.push_uint(DtoAlignment(ts));

  if (withArgTypes) {
    // TypeInfo m_arg1
    // TypeInfo m_arg2
    for (unsigned i = 0; i < 2; i++) {
      if (auto t = sd->argType(i)) {
        t = merge(t);
        b.push_typeinfo(t);
      } else {
        b.push_null(getTypeInfoType());
      }
    }
  }

  // immutable(void)* m_RTInfo
  // The cases where getRTInfo is null are not quite here, but the code is
  // modelled after what DMD does.
  if (sd->getRTInfo) {
    b.push(toConstElem(sd->getRTInfo, gIR));
  } else {
    b.push_size_as_vp(ts->hasPointers() ? 1 : 0);
  }

  constTypeInfo = b.get_constant(memoryLLType);

  return constTypeInfo;
}
