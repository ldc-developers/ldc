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
#include "ir/irtype.h"
#include <ir/irtypeclass.h>
#include "ir/irvar.h"
#include <cassert>
#include <cstdio>

FuncDeclaration *search_toString(StructDeclaration *sd);

// defined in dmd/typinf.d:
void genTypeInfo(Loc loc, Type *torig, Scope *sc);
bool builtinTypeInfo(Type *t);

TypeInfoDeclaration *getOrCreateTypeInfoDeclaration(const Loc &loc, Type *torig,
                                                    Scope *sc) {
  IF_LOG Logger::println("Type::getTypeInfo(): %s", torig->toChars());
  LOG_SCOPE

  genTypeInfo(loc, torig, sc);

  return torig->vtinfo;
}

/* ========================================================================= */

//////////////////////////////////////////////////////////////////////////////
//                             MAGIC   PLACE
//                                (wut?)
//////////////////////////////////////////////////////////////////////////////

static void emitTypeMetadata(TypeInfoDeclaration *tid) {
  // We don't want to generate metadata for non-concrete types (such as tuple
  // types, slice types, typeof(expr), etc.), void and function types (without
  // an indirection), as there must be a valid LLVM undef value of that type.
  // As those types cannot appear as LLVM values, they are not interesting for
  // the optimizer passes anyway.
  Type *t = tid->tinfo->toBasetype();
  if (t->ty < Terror && t->ty != Tvoid && t->ty != Tfunction &&
      t->ty != Tident) {
    // Add some metadata for use by optimization passes.
    OutBuffer buf;
    buf.writestring(TD_PREFIX);
    mangleToBuffer(tid, &buf);
    const char *metaname = buf.peekChars();

    llvm::NamedMDNode *meta = gIR->module.getNamedMetadata(metaname);

    if (!meta) {
      // Construct the fields
      llvm::Metadata *mdVals[TD_NumFields];
      mdVals[TD_TypeInfo] = llvm::ValueAsMetadata::get(getIrGlobal(tid)->value);
      mdVals[TD_Type] = llvm::ConstantAsMetadata::get(
          llvm::UndefValue::get(DtoType(tid->tinfo)));

      // Construct the metadata and insert it into the module.
      llvm::NamedMDNode *node = gIR->module.getOrInsertNamedMetadata(metaname);
      node->addOperand(llvm::MDNode::get(
          gIR->context(), llvm::makeArrayRef(mdVals, TD_NumFields)));
    }
  }
}

void DtoResolveTypeInfo(TypeInfoDeclaration *tid) {
  if (tid->ir->isResolved()) {
    return;
  }
  tid->ir->setResolved();

  // TypeInfo instances (except ClassInfo ones) are always emitted as weak
  // symbols when they are used. We call semanticTypeInfo() to make sure
  // that the type (e.g. for structs) is semantic3'd and codegen() does not
  // skip it on grounds of being speculative, as DtoResolveTypeInfo() means
  // that we actually need the value somewhere else in codegen.
  // TODO: DMD does not seem to call semanticTypeInfo() from the glue layer,
  // so there might be a structural issue somewhere.
  semanticTypeInfo(nullptr, tid->tinfo);
  Declaration_codegen(tid);
}

/* ========================================================================= */

class LLVMDefineVisitor : public Visitor {
  LLGlobalVariable *const gvar;

public:
  LLVMDefineVisitor(LLGlobalVariable *gvar) : gvar(gvar) {}

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

    assert(decl->tinfo->ty == Tenum);
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

    assert(decl->tinfo->ty == Tsarray);
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

    assert(decl->tinfo->ty == Taarray);
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

    assert(decl->tinfo->ty == Tdelegate);
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
    IF_LOG Logger::println("TypeInfoStructDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    // make sure struct is resolved
    assert(decl->tinfo->ty == Tstruct);
    TypeStruct *tc = static_cast<TypeStruct *>(decl->tinfo);
    StructDeclaration *sd = tc->sym;

    // check declaration in object.d
    const auto structTypeInfoType = getStructTypeInfoType();
    const auto structTypeInfoDecl = Type::typeinfostruct;

    // On x86_64, class TypeInfo_Struct contains 2 additional fields
    // (m_arg1/m_arg2) which are used for the X86_64 System V ABI varargs
    // implementation. They are not present on any other cpu/os.
    const bool isX86_64 =
        global.params.targetTriple->getArch() == llvm::Triple::x86_64;
    const unsigned expectedFields = 11 + (isX86_64 ? 2 : 0);
    const unsigned actualFields =
        structTypeInfoDecl->fields.length -
        1; // union of xdtor/xdtorti counts as 2 overlapping fields
    if (actualFields != expectedFields) {
      error(Loc(), "Unexpected number of `object.TypeInfo_Struct` fields; "
                   "druntime version does not match compiler");
      fatal();
    }

    RTTIBuilder b(structTypeInfoType);

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
      if (isX86_64) {
        b.push_null_vp();       // m_arg1
        b.push_null_vp();       // m_arg2
      }
      b.push_null_vp();         // m_RTInfo

      b.finalize(gvar);
      return;
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

    IrAggr *iraggr = getIrAggr(sd);

    // string name
    b.push_string(sd->toPrettyChars());

    // void[] m_init
    // The protocol is to write a null pointer for zero-initialized arrays. The
    // length field is always needed for tsize().
    llvm::Constant *initPtr;
    if (tc->isZeroInit(Loc())) {
      initPtr = getNullValue(getVoidPtrType());
    } else {
      initPtr = iraggr->getInitSymbol();
    }
    b.push_void_array(getTypeStoreSize(DtoType(tc)), initPtr);

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
    unsigned hasptrs = tc->hasPointers() ? 1 : 0;
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
    b.push_uint(DtoAlignment(tc));

    if (isX86_64) {
      // TypeInfo m_arg1
      // TypeInfo m_arg2
      Type *t = sd->arg1type;
      for (unsigned i = 0; i < 2; i++) {
        if (t) {
          t = merge(t);
          b.push_typeinfo(t);
        } else {
          b.push_null(getTypeInfoType());
        }

        t = sd->arg2type;
      }
    }

    // immutable(void)* m_RTInfo
    // The cases where getRTInfo is null are not quite here, but the code is
    // modelled after what DMD does.
    if (sd->getRTInfo) {
      b.push(toConstElem(sd->getRTInfo, gIR));
    } else if (!tc->hasPointers()) {
      b.push_size_as_vp(0); // no pointers
    } else {
      b.push_size_as_vp(1); // has pointers
    }

    // finish
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoClassDeclaration *decl) override {
    llvm_unreachable(
        "TypeInfoClassDeclaration::llvmDefine() should not be called, "
        "as a custom Dsymbol::codegen() override is used");
  }

  /* ======================================================================= */

  void visit(TypeInfoInterfaceDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoInterfaceDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    // make sure interface is resolved
    assert(decl->tinfo->ty == Tclass);
    TypeClass *tc = static_cast<TypeClass *>(decl->tinfo);
    DtoResolveClass(tc->sym);

    RTTIBuilder b(getInterfaceTypeInfoType());

    // TypeInfo base
    b.push_classinfo(tc->sym);

    // finish
    b.finalize(gvar);
  }

  /* ======================================================================= */

  void visit(TypeInfoTupleDeclaration *decl) override {
    IF_LOG Logger::println("TypeInfoTupleDeclaration::llvmDefine() %s",
                           decl->toChars());
    LOG_SCOPE;

    // create elements array
    assert(decl->tinfo->ty == Ttuple);
    TypeTuple *tu = static_cast<TypeTuple *>(decl->tinfo);

    size_t dim = tu->arguments->length;
    std::vector<LLConstant *> arrInits;
    arrInits.reserve(dim);

    LLType *tiTy = DtoType(getTypeInfoType());

    for (auto arg : *tu->arguments) {
      arrInits.push_back(DtoTypeInfoOf(arg->type));
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

    assert(decl->tinfo->ty == Tvector);
    TypeVector *tv = static_cast<TypeVector *>(decl->tinfo);

    RTTIBuilder b(getVectorTypeInfoType());
    // TypeInfo base
    b.push_typeinfo(tv->basetype);
    // finish
    b.finalize(gvar);
  }
};

/* ========================================================================= */

void TypeInfoDeclaration_codegen(TypeInfoDeclaration *decl, IRState *p) {
  IF_LOG Logger::println("TypeInfoDeclaration_codegen(%s)",
                         decl->toPrettyChars());
  LOG_SCOPE;

  if (decl->ir->isDefined()) {
    return;
  }
  decl->ir->setDefined();

  OutBuffer mangleBuf;
  mangleToBuffer(decl, &mangleBuf);
  const char *mangled = mangleBuf.peekChars();

  IF_LOG {
    Logger::println("type = '%s'", decl->tinfo->toChars());
    Logger::println("typeinfo mangle: %s", mangled);
  }

  // Only declare the symbol if it isn't yet, otherwise the subtype of built-in
  // TypeInfos (rt.typeinfo.*) may clash with the base type when compiling the
  // rt.typeinfo.* modules.
  const auto irMangle = getIRMangledVarName(mangled, LINKd);
  llvm::GlobalVariable *gvar = gIR->module.getGlobalVariable(irMangle);
  if (!gvar) {
    LLType *type = DtoType(decl->type)->getPointerElementType();
    // We need to keep the symbol mutable as the type is not declared as
    // immutable on the D side, and e.g. synchronized() can be used on the
    // implicit monitor.
    gvar = declareGlobal(decl->loc, gIR->module, type, irMangle, false);
  }

  IrGlobal *irg = getIrGlobal(decl, true);
  irg->value = gvar;

  emitTypeMetadata(decl);

  // check if the definition can be elided
  Type *forType = decl->tinfo;
  if (!global.params.useTypeInfo || !Type::dtypeinfo ||
      isSpeculativeType(forType) || builtinTypeInfo(forType)) {
    return;
  }
  if (auto forStructType = forType->isTypeStruct()) {
    if (forStructType->sym->llvmInternal == LLVMno_typeinfo)
      return;
  }
  if (auto forClassType = forType->isTypeClass()) {
    if (forClassType->sym->llvmInternal == LLVMno_typeinfo)
      return;
  }

  // define the TypeInfo global
  LLVMDefineVisitor v(gvar);
  decl->accept(&v);

  setLinkage({TYPEINFO_LINKAGE_TYPE, supportsCOMDAT()}, gvar);
  if (auto forStructType = forType->isTypeStruct())
    setVisibility(forStructType->sym, gvar);
}

/* ========================================================================= */

void TypeInfoClassDeclaration_codegen(TypeInfoDeclaration *decl, IRState *p) {
  IF_LOG Logger::println("TypeInfoClassDeclaration_codegen(%s)",
                         decl->toPrettyChars());
  LOG_SCOPE;

  // For classes, the TypeInfo is in fact a ClassInfo instance and emitted
  // as a __ClassZ symbol. For interfaces, the __InterfaceZ symbol is
  // referenced as "info" member in a (normal) TypeInfo_Interface instance.
  IrGlobal *irg = getIrGlobal(decl, true);

  assert(decl->tinfo->ty == Tclass);
  TypeClass *tc = static_cast<TypeClass *>(decl->tinfo);
  DtoResolveClass(tc->sym);

  irg->value = getIrAggr(tc->sym)->getClassInfoSymbol();

  if (!tc->sym->isInterfaceDeclaration()) {
    emitTypeMetadata(decl);
  }
}
