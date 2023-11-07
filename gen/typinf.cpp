//===-- typinf.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file mostly consists of code under the BSD-style LDC license, but some
// parts have been derived from DMD as noted below. See the LICENSE file for
// details.
//
//===----------------------------------------------------------------------===//

// Copyright (c) 1999-2004 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

// Modifications for LDC:
// Copyright (c) 2007 by Tomas Lindquist Olsen
// tomas at famolsen dk

#include "gen/typinf.h"

#include "dmd/aggregate.h"
#include "dmd/attrib.h"
#include "dmd/declaration.h"
#include "dmd/enum.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/id.h"
#include "dmd/import.h"
#include "dmd/init.h"
#include "dmd/mangle.h"
#include "dmd/module.h"
#include "dmd/mtype.h"
#include "dmd/scope.h"
#include "dmd/template.h"
#include "dmd/typinf.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/irstate.h"
#include "gen/linkage.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/mangling.h"
#include "gen/passes/metadata.h"
#include "gen/pragma.h"
#include "gen/rttibuilder.h"
#include "gen/runtime.h"
#include "gen/structs.h"
#include "gen/tollvm.h"
#include "ir/irdsymbol.h"
#include "ir/irtype.h"
#include <ir/irtypeclass.h>
#include "ir/irvar.h"
#include <cassert>
#include <cstdio>

TypeInfoDeclaration *getOrCreateTypeInfoDeclaration(const Loc &loc, Type *forType) {
  IF_LOG Logger::println("getOrCreateTypeInfoDeclaration(): %s",
                         forType->toChars());
  LOG_SCOPE

  genTypeInfo(nullptr, loc, forType, nullptr);

  return forType->vtinfo;
}

/* ========================================================================= */

//////////////////////////////////////////////////////////////////////////////
//                             MAGIC   PLACE
//                                (wut?)
//////////////////////////////////////////////////////////////////////////////

void emitTypeInfoMetadata(LLGlobalVariable *typeinfoGlobal, Type *forType) {
  // We don't want to generate metadata for non-concrete types (such as tuple
  // types, slice types, typeof(expr), etc.), void and function types (without
  // an indirection), as there must be a valid LLVM undef value of that type.
  // As those types cannot appear as LLVM values, they are not interesting for
  // the optimizer passes anyway.
  Type *t = forType->toBasetype();
  if (t->ty < TY::Terror && t->ty != TY::Tvoid && t->ty != TY::Tfunction &&
      t->ty != TY::Tident) {
    const auto metaname = getMetadataName(TD_PREFIX, typeinfoGlobal);

    if (!gIR->module.getNamedMetadata(metaname)) {
      // Construct the metadata and insert it into the module.
      auto meta = gIR->module.getOrInsertNamedMetadata(metaname);
      auto val = llvm::UndefValue::get(DtoType(forType));
      meta->addOperand(llvm::MDNode::get(gIR->context(),
                                         llvm::ConstantAsMetadata::get(val)));
      if (TypeArray *ta = t->isTypeDArray()) {
        auto val2 = llvm::UndefValue::get(DtoMemType(ta->nextOf()));
        meta->addOperand(llvm::MDNode::get(gIR->context(),
                                           llvm::ConstantAsMetadata::get(val2)));
      }
    }
  }
}

/* ========================================================================= */

namespace {
// The upstream implementation is in dmd/todt.d, class TypeInfoDtVisitor.
class DefineVisitor : public Visitor {
  LLGlobalVariable *const gvar;

public:
  DefineVisitor(LLGlobalVariable *gvar) : gvar(gvar) {}

  // Import all functions from class Visitor
  using Visitor::visit;

  /* ======================================================================= */

  void visit(TypeInfoDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    RTTIBuilder b(getTypeInfoType());
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoEnumDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoEnumDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    RTTIBuilder b(getEnumTypeInfoType());

    assert(decl->tinfo->ty == TY::Tenum);
    TypeEnum *tc = static_cast<TypeEnum *>(decl->tinfo);
    EnumDeclaration *sd = tc->sym;

    // TypeInfo base
    b.push_typeinfo(sd->memtype);

    // char[] name
    b.push_string(sd->toPrettyChars());

    // void[] init
    // the array is null if the default initializer is zero
    if (!sd->members || decl->tinfo->isZeroInit(decl->loc)) {
      b.push_null_void_array();
    }
    // otherwise emit a void[] with the default initializer
    else {
      Expression *defaultval = sd->getDefaultValue(decl->loc);
      LLConstant *c = toConstElem(defaultval, gIR);
      b.push_void_array(c, sd->memtype, sd);
    }

    // finish
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoPointerDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoPointerDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    RTTIBuilder b(getPointerTypeInfoType());
    // TypeInfo base
    b.push_typeinfo(decl->tinfo->nextOf());
    // finish
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoArrayDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoArrayDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    RTTIBuilder b(getArrayTypeInfoType());
    // TypeInfo base
    b.push_typeinfo(decl->tinfo->nextOf());
    // finish
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoStaticArrayDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoStaticArrayDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    assert(decl->tinfo->ty == TY::Tsarray);
    TypeSArray *tc = static_cast<TypeSArray *>(decl->tinfo);

    RTTIBuilder b(getStaticArrayTypeInfoType());

    // value typeinfo
    b.push_typeinfo(tc->nextOf());

    // length
    b.push(DtoConstSize_t(static_cast<size_t>(tc->dim->toUInteger())));

    // finish
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoAssociativeArrayDeclaration *decl) override {
    IF_LOG Logger::println(
        "TypeInfoAssociativeArrayDeclaration::llvmDefine() %s",
        decl->toChars());
    LOG_SCOPE;

    assert(decl->tinfo->ty == TY::Taarray);
    TypeAArray *tc = static_cast<TypeAArray *>(decl->tinfo);

    RTTIBuilder b(getAssociativeArrayTypeInfoType());

    // value typeinfo
    b.push_typeinfo(tc->nextOf());

    // key typeinfo
    b.push_typeinfo(tc->index);

    // finish
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoFunctionDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoFunctionDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    RTTIBuilder b(getFunctionTypeInfoType());
    // TypeInfo base
    b.push_typeinfo(decl->tinfo->nextOf());
    // string deco
    b.push_string(decl->tinfo->deco);
    // finish
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoDelegateDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoDelegateDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    assert(decl->tinfo->ty == TY::Tdelegate);
    Type *ret_type = decl->tinfo->nextOf()->nextOf();

    RTTIBuilder b(getDelegateTypeInfoType());
    // TypeInfo base
    b.push_typeinfo(ret_type);
    // string deco
    b.push_string(decl->tinfo->deco);
    // finish
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoStructDeclaration *decl) override {
    llvm_unreachable("Should be handled by IrStruct::getTypeInfoInit()");
  }

  /* ======================================================================= */

  void visit(TypeInfoClassDeclaration *decl) override {
    llvm_unreachable("Should be handled by IrClass::getClassInfoInit()");
  }

  /* ======================================================================= */

  void visit(TypeInfoInterfaceDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoInterfaceDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    // make sure interface is resolved
    auto cd = decl->tinfo->isTypeClass()->sym;
    DtoResolveClass(cd);

    RTTIBuilder b(getInterfaceTypeInfoType());

    // TypeInfo_Class info
    b.push(getIrAggr(cd)->getClassInfoSymbol());

    // finish
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoTupleDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoTupleDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    // create elements array
    assert(decl->tinfo->ty == TY::Ttuple);
    TypeTuple *tu = static_cast<TypeTuple *>(decl->tinfo);

    size_t dim = tu->arguments->length;
    std::vector<LLConstant *> arrInits;
    arrInits.reserve(dim);

    LLType *tiTy = DtoType(getTypeInfoType());

    for (auto arg : *tu->arguments) {
      arrInits.push_back(DtoTypeInfoOf(decl->loc, arg->type));
    }

    // build array
    LLArrayType *arrTy = LLArrayType::get(tiTy, dim);
    LLConstant *arrC = LLConstantArray::get(arrTy, arrInits);

    RTTIBuilder b(getTupleTypeInfoType());

    // push TypeInfo[]
    b.push_array(arrC, dim, getTypeInfoType(), nullptr);

    // finish
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoConstDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoConstDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    RTTIBuilder b(getConstTypeInfoType());
    // TypeInfo base
    b.push_typeinfo(merge(decl->tinfo->mutableOf()));
    // finish
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoInvariantDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoInvariantDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    RTTIBuilder b(getInvariantTypeInfoType());
    // TypeInfo base
    b.push_typeinfo(merge(decl->tinfo->mutableOf()));
    // finish
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoSharedDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoSharedDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    RTTIBuilder b(getSharedTypeInfoType());
    // TypeInfo base
    b.push_typeinfo(merge(decl->tinfo->unSharedOf()));
    // finish
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoWildDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoWildDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    RTTIBuilder b(getInoutTypeInfoType());
    // TypeInfo base
    b.push_typeinfo(merge(decl->tinfo->mutableOf()));
    // finish
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoVectorDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoVectorDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    assert(decl->tinfo->ty == TY::Tvector);
    TypeVector *tv = static_cast<TypeVector *>(decl->tinfo);

    RTTIBuilder b(getVectorTypeInfoType());
    // TypeInfo base
    b.push_typeinfo(tv->basetype);
    // finish
    b.finalize(gvar);
  }
};

// Builds all non-struct/class TypeInfos.
void buildTypeInfo(TypeInfoDeclaration *decl) {
  if (decl->ir->isDefined()) {
    return;
  }
  decl->ir->setDefined();

  IF_LOG Logger::println("Building TypeInfo: %s", decl->toPrettyChars());
  LOG_SCOPE;

  Type *forType = decl->tinfo;

  OutBuffer mangleBuf;
  mangleToBuffer(decl, mangleBuf);
  const char *mangled = mangleBuf.peekChars();

  IF_LOG {
    Logger::println("type = '%s'", forType->toChars());
    Logger::println("typeinfo mangle: %s", mangled);
  }

  // Only declare the symbol if it isn't yet, otherwise it may clash with an
  // existing init symbol of a built-in TypeInfo class (in rt.util.typeinfo)
  // when compiling the rt.util.typeinfo unittests.
  const auto irMangle = getIRMangledVarName(mangled, LINK::d);
  LLGlobalVariable *gvar = gIR->module.getGlobalVariable(irMangle);
  const bool isBuiltin = builtinTypeInfo(forType);
  if (gvar) {
    assert(isBuiltin && "existing global expected to be the init symbol of a "
                        "built-in TypeInfo");
  } else {
    TypeClass *tc = decl->type->isTypeClass();
    LLType *type = getIrType(tc->sym->type, true)->isClass()->getMemoryLLType();
    // We need to keep the symbol mutable as the type is not declared as
    // immutable on the D side, and e.g. synchronized() can be used on the
    // implicit monitor.
    const bool isConstant = false;
    bool useDLLImport = isBuiltin && global.params.dllimport != DLLImport::none;
    gvar = declareGlobal(decl->loc, gIR->module, type, irMangle, isConstant,
                         false, useDLLImport);
  }

  emitTypeInfoMetadata(gvar, forType);

  IrGlobal *irg = getIrGlobal(decl, true);
  irg->value = gvar;

  // check if the definition can be elided
  if (isBuiltin || gvar->hasInitializer() || !global.params.useTypeInfo ||
      !Type::dtypeinfo) {
    return;
  }
  if (auto forClassType = forType->isTypeClass()) {
    if (forClassType->sym->llvmInternal == LLVMno_typeinfo)
      return;
  }

  // define the TypeInfo global
  DefineVisitor v(gvar);
  decl->accept(&v);
  setLinkage({TYPEINFO_LINKAGE_TYPE, needsCOMDAT()}, gvar);
}

/* ========================================================================= */

class DeclareOrDefineVisitor : public Visitor {
  using Visitor::visit;

  // Define struct TypeInfos as linkonce_odr in each referencing CU.
  void visit(TypeInfoStructDeclaration *decl) override {
    auto sd = decl->tinfo->isTypeStruct()->sym;
    auto irstruct = getIrAggr(sd, true);
    IrGlobal *irg = getIrGlobal(decl, true);

    auto ti = irstruct->getTypeInfoSymbol();
    irg->value = ti;

    // check if the definition can be elided
    if (ti->hasInitializer() || irstruct->suppressTypeInfo()) {
      return;
    }

    irstruct->getTypeInfoSymbol(/*define=*/true);
    ti->setLinkage(TYPEINFO_LINKAGE_TYPE); // override
  }

  // Only declare class TypeInfos. They are defined once in their owning module
  // as part of ClassDeclaration codegen.
  void visit(TypeInfoClassDeclaration *decl) override {
    auto cd = decl->tinfo->isTypeClass()->sym;
    DtoResolveClass(cd);

    IrGlobal *irg = getIrGlobal(decl, true);
    irg->value = getIrAggr(cd)->getClassInfoSymbol();
  }

  // Build all other TypeInfos.
  void visit(TypeInfoDeclaration *decl) override { buildTypeInfo(decl); }
};
} // anonymous namespace

LLGlobalVariable *DtoResolveTypeInfo(TypeInfoDeclaration *tid) {
  IF_LOG Logger::println("DtoResolveTypeInfo(%s)", tid->toPrettyChars());
  LOG_SCOPE;

  assert(!gIR->dcomputetarget);

  if (!tid->ir->isResolved()) {
    DeclareOrDefineVisitor v;
    tid->accept(&v);

    tid->ir->setResolved();
  }

  return llvm::cast<LLGlobalVariable>(getIrValue(tid));
}
