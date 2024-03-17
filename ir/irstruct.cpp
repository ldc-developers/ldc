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

using namespace dmd;

namespace {
LLStructType* getTypeInfoStructMemType() {
  Type *t = getStructTypeInfoType();
  IrTypeClass *tc = getIrType(t, true)->isClass();
  assert(tc && "invalid TypeInfo_Struct type");

  return llvm::cast<LLStructType>(tc->getMemoryLLType());
}
}

LLGlobalVariable* IrStruct::getTypeInfoSymbol(bool define) {
  if (!typeInfo) {
    OutBuffer mangledName;
    mangledName.writestring("TypeInfo_S");
    mangleToBuffer(aggrdecl, mangledName);
    const auto length = mangledName.length();
    mangledName.prependstring(("_D" + std::to_string(length)).c_str());
    mangledName.writestring("6__initZ");

    const auto irMangle = getIRMangledVarName(mangledName.peekChars(), LINK::d);

    // We need to keep the symbol mutable as the type is not declared as
    // immutable on the D side, and e.g. synchronized() can be used on the
    // implicit monitor.
    const bool isConstant = false;
    // Struct TypeInfos are emitted into each referencing CU.
    const bool useDLLImport = false;
    typeInfo =
        declareGlobal(aggrdecl->loc, gIR->module, getTypeInfoStructMemType(),
                      irMangle, isConstant, false, useDLLImport);

    emitTypeInfoMetadata(typeInfo, aggrdecl->type);

    if (!define)
      define = defineOnDeclare(aggrdecl, /*isFunction=*/false);
  }

  if (define) {
    auto init = getTypeInfoInit();
    if (!typeInfo->hasInitializer())
      defineGlobal(typeInfo, init, aggrdecl);
  }

  return typeInfo;
}

LLConstant *IrStruct::getTypeInfoInit() {
  // The upstream implementation is in dmd/todt.d,
  // TypeInfoDtVisitor.visit(TypeInfoStructDeclaration).

  if (constTypeInfo) {
    return constTypeInfo;
  }

  auto sd = aggrdecl->isStructDeclaration();
  IF_LOG Logger::println("Defining TypeInfo for struct: %s", sd->toChars());
  LOG_SCOPE;

  // we need (dummy) TypeInfos for opaque structs too
  const bool isOpaque = !sd->members;

  // make sure xtoHash/xopEquals/xopCmp etc. are semantically analyzed
  if (!isOpaque && sd->semanticRun < PASS::semantic3done) {
    Logger::println(
        "Struct hasn't had semantic3 yet, calling semanticTypeInfoMembers()");
    semanticTypeInfoMembers(sd);
  }

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

  // string mangledName
  if (isOpaque) {
    b.push_null_void_array();
  } else {
    b.push_string(ts->deco);
  }

  // void[] m_init
  // The protocol is to write a null pointer for zero-initialized structs.
  // The length field is always needed for tsize().
  if (isOpaque) {
    b.push_null_void_array();
  } else {
    llvm::Constant *initPtr;
    if (ts->isZeroInit(Loc())) {
      initPtr = getNullValue(getVoidPtrType());
    } else {
      initPtr = getInitSymbol();
    }
    b.push_void_array(sd->size(Loc()), initPtr);
  }

  // function xtoHash
  b.push_funcptr(isOpaque ? nullptr : sd->xhash);

  // function xopEquals
  b.push_funcptr(isOpaque ? nullptr : sd->xeq);

  // function xopCmp
  b.push_funcptr(isOpaque ? nullptr : sd->xcmp);

  // function xtoString
  b.push_funcptr(isOpaque ? nullptr : search_toString(sd));

  // StructFlags m_flags
  b.push_uint(!isOpaque && hasPointers(ts) ? 1 : 0);

  // function xdtor/xdtorti
  b.push_funcptr(isOpaque ? nullptr : sd->tidtor);

  // function xpostblit
  FuncDeclaration *xpostblit = isOpaque ? nullptr : sd->postblit;
  if (xpostblit && (xpostblit->storage_class & STCdisable)) {
    xpostblit = nullptr;
  }
  b.push_funcptr(xpostblit);

  // uint m_align
  b.push_uint(isOpaque ? 0 : DtoAlignment(ts));

  if (withArgTypes) {
    // TypeInfo m_arg1
    // TypeInfo m_arg2
    for (unsigned i = 0; i < 2; i++) {
      if (auto t = isOpaque ? nullptr : sd->argType(i)) {
        t = merge(t);
        b.push_typeinfo(t);
      } else {
        b.push_null(getTypeInfoType());
      }
    }
  }

  // immutable(void)* m_RTInfo
  if (!isOpaque && sd->getRTInfo) {
    b.push(toConstElem(sd->getRTInfo, gIR));
  } else {
    b.push_size_as_vp(!isOpaque && hasPointers(ts) ? 1 : 0);
  }

  constTypeInfo = b.get_constant(getTypeInfoStructMemType());

  return constTypeInfo;
}
