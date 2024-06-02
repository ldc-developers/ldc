//===-- tollvm.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/tollvm.h"

#include "dmd/aggregate.h"
#include "dmd/declaration.h"
#include "dmd/dsymbol.h"
#include "dmd/expression.h"
#include "dmd/id.h"
#include "dmd/init.h"
#include "dmd/module.h"
#include "dmd/target.h"
#include "driver/cl_options.h"
#include "gen/abi/abi.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/complex.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/linkage.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/pragma.h"
#include "gen/runtime.h"
#include "gen/structs.h"
#include "gen/typinf.h"
#include "gen/uda.h"
#include "ir/irtype.h"
#include "ir/irtypeclass.h"
#include "ir/irtypefunction.h"
#include "ir/irtypestruct.h"

using namespace dmd;

bool DtoIsInMemoryOnly(Type *type) {
  Type *typ = type->toBasetype();
  TY t = typ->ty;
  return (t == TY::Tstruct || t == TY::Tsarray);
}

bool DtoIsReturnInArg(CallExp *ce) {
  Type *t = ce->e1->type->toBasetype();
  if (t->ty == TY::Tfunction && (!ce->f || !DtoIsIntrinsic(ce->f))) {
    return gABI->returnInArg(static_cast<TypeFunction *>(t),
                             ce->f && ce->f->needThis());
  }
  return false;
}

void DtoAddExtendAttr(Type *type, llvm::AttrBuilder &attrs) {
  type = type->toBasetype();
  if (type->isintegral() && type->ty != TY::Tvector && type->size() <= 2) {
    attrs.addAttribute(type->isunsigned() ? LLAttribute::ZExt
                                          : LLAttribute::SExt);
  }
}

LLType *DtoType(Type *t) {
  t = stripModifiers(t);

  if (t->ctype) {
    return t->ctype->getLLType();
  }

  IF_LOG Logger::println("Building type: %s", t->toChars());
  LOG_SCOPE;

  switch (t->ty) {
  // basic types
  case TY::Tvoid:
  case TY::Tint8:
  case TY::Tuns8:
  case TY::Tint16:
  case TY::Tuns16:
  case TY::Tint32:
  case TY::Tuns32:
  case TY::Tint64:
  case TY::Tuns64:
  case TY::Tint128:
  case TY::Tuns128:
  case TY::Tfloat32:
  case TY::Tfloat64:
  case TY::Tfloat80:
  case TY::Timaginary32:
  case TY::Timaginary64:
  case TY::Timaginary80:
  case TY::Tcomplex32:
  case TY::Tcomplex64:
  case TY::Tcomplex80:
  // case TY::Tbit:
  case TY::Tbool:
  case TY::Tchar:
  case TY::Twchar:
  case TY::Tdchar:
  case TY::Tnoreturn: {
    return IrTypeBasic::get(t)->getLLType();
  }

  // pointers
  case TY::Tnull:
  case TY::Tpointer: {
    return IrTypePointer::get(t)->getLLType();
  }

  // arrays
  case TY::Tarray: {
    return IrTypeArray::get(t)->getLLType();
  }

  case TY::Tsarray: {
    return IrTypeSArray::get(t)->getLLType();
  }

  // aggregates
  case TY::Tstruct:
  case TY::Tclass: {
    const auto isStruct = t->ty == TY::Tstruct;
    AggregateDeclaration *ad;
    if (isStruct) {
      ad = static_cast<TypeStruct *>(t)->sym;
    } else {
      ad = static_cast<TypeClass *>(t)->sym;
    }
    if (ad->type->ty == TY::Terror) {
      static LLStructType *opaqueErrorType =
          LLStructType::create(gIR->context(), Type::terror->toChars());
      return opaqueErrorType;
    }
    Type *adType = stripModifiers(ad->type);
    if (adType->ctype) {
      /* This should not happen, but e.g. can for aggregates whose mangled name
       * contains a lambda which got promoted from a delegate to a function.
       * We certainly don't want to override adType->ctype, and not associate
       * an IrType to multiple Types either (see e.g.
       * IrTypeStruct::resetDComputeTypes()).
       * This means there are some aggregate Types which don't have an
       * associated ctype, so getIrType() should always be fed with its
       * AggregateDeclaration::type.
       */
      IF_LOG {
        Logger::println("Aggregate with multiple Types detected: %s (%s)",
                        ad->toPrettyChars(), ad->locToChars());
        LOG_SCOPE;
        Logger::println("Existing deco:    %s", adType->deco);
        Logger::println("Mismatching deco: %s", t->deco);
      }
      return adType->ctype->getLLType();
    }
    return isStruct ? IrTypeStruct::get(ad->isStructDeclaration())->getLLType()
                    : IrTypeClass::get(ad->isClassDeclaration())->getLLType();
  }

  // functions
  case TY::Tfunction: {
    return IrTypeFunction::get(t)->getLLType();
  }

  // delegates
  case TY::Tdelegate: {
    return IrTypeDelegate::get(t)->getLLType();
  }

  // typedefs
  // enum

  // FIXME: maybe just call toBasetype first ?
  case TY::Tenum: {
    Type *bt = t->toBasetype();
    assert(bt);
    if (t == bt) {
      // This is an enum forward reference that is only legal when referenced
      // through an indirection (e.g. "enum E; void foo(E* p);"). For lack of a
      // better choice, make the outer indirection a void pointer.
      return getI8Type();
    }
    return DtoType(bt);
  }

  // associative arrays
  case TY::Taarray:
    return getVoidPtrType();

  case TY::Tvector:
    return IrTypeVector::get(t)->getLLType();

  default:
    llvm_unreachable("Unknown class of D Type!");
  }
  return nullptr;
}

LLType *DtoMemType(Type *t) { return i1ToI8(voidToI8(DtoType(t))); }

LLPointerType *DtoPtrToType(Type *t) { return DtoMemType(t)->getPointerTo(); }

LLType *voidToI8(LLType *t) {
  return t->isVoidTy() ? LLType::getInt8Ty(t->getContext()) : t;
}

LLType *i1ToI8(LLType *t) {
  return t->isIntegerTy(1) ? LLType::getInt8Ty(t->getContext()) : t;
}

////////////////////////////////////////////////////////////////////////////////

LLValue *DtoDelegateEquals(EXP op, LLValue *lhs, LLValue *rhs) {
  Logger::println("Doing delegate equality");
  if (rhs == nullptr) {
    rhs = LLConstant::getNullValue(lhs->getType());
  }

  LLValue *l1 = gIR->ir->CreateExtractValue(lhs, 0);
  LLValue *l2 = gIR->ir->CreateExtractValue(lhs, 1);

  LLValue *r1 = gIR->ir->CreateExtractValue(rhs, 0);
  LLValue *r2 = gIR->ir->CreateExtractValue(rhs, 1);

  return createIPairCmp(op, l1, l2, r1, r2);
}

////////////////////////////////////////////////////////////////////////////////

namespace {
LLGlobalValue::LinkageTypes DtoLinkageOnly(Dsymbol *sym) {
  if (hasWeakUDA(sym))
    return LLGlobalValue::WeakAnyLinkage;

  // static in ImportC translates to internal linkage
  if (auto decl = sym->isDeclaration())
    if ((decl->storage_class & STCstatic) && decl->isCsymbol())
      return LLGlobalValue::InternalLinkage;

  /* Function (incl. delegate) literals are emitted into each referencing
   * compilation unit, so use internal linkage for all lambdas and all global
   * variables they define.
   * This makes sure these symbols don't accidentally collide when linking
   * object files compiled by different compiler invocations (lambda mangles
   * aren't stable - see https://issues.dlang.org/show_bug.cgi?id=23722).
   */
  auto potentialLambda = sym;
  if (auto vd = sym->isVarDeclaration())
    if (vd->isDataseg())
      potentialLambda = vd->toParent2();
  if (potentialLambda->isFuncLiteralDeclaration())
    return LLGlobalValue::InternalLinkage;

  if (sym->isInstantiated())
    return templateLinkage;

  return LLGlobalValue::ExternalLinkage;
}
}

LinkageWithCOMDAT DtoLinkage(Dsymbol *sym) {
  return {DtoLinkageOnly(sym), needsCOMDAT()};
}

bool needsCOMDAT() {
  /* For MSVC targets (and probably MinGW too), linkonce[_odr] and weak[_odr]
   * linkages don't work and need to be emulated via COMDATs to prevent multiple
   * definition errors when linking.
   * Simply emit all functions in COMDATs, not just templates, for aggressive
   * linker stripping (/OPT:REF and /OPT:ICF with MS linker/LLD), analogous to
   * using /Gy with the MS compiler.
   * https://docs.microsoft.com/en-us/cpp/build/reference/opt-optimizations?view=vs-2019
   */
  return global.params.targetTriple->isOSBinFormatCOFF();
}

void setLinkage(LinkageWithCOMDAT lwc, llvm::GlobalObject *obj) {
  obj->setLinkage(lwc.first);
  obj->setComdat(lwc.second ? gIR->module.getOrInsertComdat(obj->getName())
                            : nullptr);
}

void setLinkageAndVisibility(Dsymbol *sym, llvm::GlobalObject *obj) {
  setLinkage(DtoLinkage(sym), obj);
  setVisibility(sym, obj);
}

namespace {
bool hasExportedLinkage(llvm::GlobalObject *obj) {
  const auto l = obj->getLinkage();
  return l == LLGlobalValue::ExternalLinkage ||
         l == LLGlobalValue::WeakODRLinkage ||
         l == LLGlobalValue::WeakAnyLinkage;
}
}

void setVisibility(Dsymbol *sym, llvm::GlobalObject *obj) {
  const auto &triple = *global.params.targetTriple;

  const bool hasHiddenUDA = obj->hasHiddenVisibility();

  if (triple.isOSWindows()) {
    bool isExported = sym->isExport();
    // Also export (non-linkonce_odr) symbols
    // * with -fvisibility=public without @hidden, or
    // * if declared with dllimport (so potentially imported from other object
    //   files / DLLs).
    if (!isExported && ((global.params.dllexport && !hasHiddenUDA) ||
                        obj->hasDLLImportStorageClass())) {
      isExported = hasExportedLinkage(obj);
    }
    // reset default visibility & DSO locality - on Windows, the DLL storage
    // classes matter
    if (hasHiddenUDA) {
      obj->setVisibility(LLGlobalValue::DefaultVisibility);
      obj->setDSOLocal(false);
    }
    obj->setDLLStorageClass(isExported ? LLGlobalValue::DLLExportStorageClass
                                       : LLGlobalValue::DefaultStorageClass);
  } else {
    if (sym->isExport()) {
      obj->setVisibility(LLGlobalValue::DefaultVisibility); // overrides @hidden
    } else if (!obj->hasLocalLinkage() && !hasHiddenUDA) {
      // Hide with -fvisibility=hidden, or linkonce_odr etc.
      // Note that symbols with local linkage cannot be hidden (LLVM assertion).
      // The Apple linker warns about hidden linkonce_odr symbols from object
      // files compiled with -linkonce-templates being folded with *public*
      // weak_odr symbols from non-linkonce-templates code (e.g., Phobos), so
      // don't hide instantiated symbols for Mac.
      if (opts::symbolVisibility == opts::SymbolVisibility::hidden ||
          (!hasExportedLinkage(obj) &&
           !(triple.isOSDarwin() && sym->isInstantiated()))) {
        obj->setVisibility(LLGlobalValue::HiddenVisibility);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

LLIntegerType *DtoSize_t() {
  // the type of size_t does not change once set
  static LLIntegerType *t = nullptr;
  if (t == nullptr) {
    if (target.ptrsize == 8) {
      t = LLType::getInt64Ty(gIR->context());
    } else if (target.ptrsize == 4) {
      t = LLType::getInt32Ty(gIR->context());
    } else if (target.ptrsize == 2) {
      t = LLType::getInt16Ty(gIR->context());
    } else {
      llvm_unreachable("Unsupported size_t width");
    }
  }
  return t;
}

////////////////////////////////////////////////////////////////////////////////

namespace {
llvm::GetElementPtrInst *DtoGEP(LLType *pointeeTy, LLValue *ptr,
                                llvm::ArrayRef<LLValue *> indices,
                                const char *name, llvm::BasicBlock *bb) {
  auto gep = llvm::GetElementPtrInst::Create(pointeeTy, ptr, indices, name,
                                             bb ? bb : gIR->scopebb());
  gep->setIsInBounds(true);
  return gep;
}
}

LLValue *DtoGEP1(LLType *pointeeTy, LLValue *ptr, LLValue *i0, const char *name,
                 llvm::BasicBlock *bb) {
  return DtoGEP(pointeeTy, ptr, i0, name, bb);
}

LLValue *DtoGEP(LLType *pointeeTy, LLValue *ptr, LLValue *i0, LLValue *i1,
                const char *name, llvm::BasicBlock *bb) {
  LLValue *indices[] = {i0, i1};
  return DtoGEP(pointeeTy, ptr, indices, name, bb);
}

LLValue *DtoGEP1(LLType *pointeeTy, LLValue *ptr, unsigned i0, const char *name,
                 llvm::BasicBlock *bb) {
  return DtoGEP(pointeeTy, ptr, DtoConstUint(i0), name, bb);
}

LLValue *DtoGEP(LLType *pointeeTy, LLValue *ptr, unsigned i0, unsigned i1,
                const char *name, llvm::BasicBlock *bb) {
  LLValue *indices[] = {DtoConstUint(i0), DtoConstUint(i1)};
  return DtoGEP(pointeeTy, ptr, indices, name, bb);
}

LLConstant *DtoGEP(LLType *pointeeTy, LLConstant *ptr, unsigned i0,
                   unsigned i1) {
  LLValue *indices[] = {DtoConstUint(i0), DtoConstUint(i1)};
  return llvm::ConstantExpr::getGetElementPtr(pointeeTy, ptr, indices,
                                              /* InBounds = */ true);
}

LLValue *DtoGEP1i64(LLType *pointeeTy, LLValue *ptr, uint64_t i0, const char *name,
                    llvm::BasicBlock *bb) {
  return DtoGEP(pointeeTy, ptr, DtoConstUlong(i0), name, bb);
}

////////////////////////////////////////////////////////////////////////////////

void DtoMemSet(LLValue *dst, LLValue *val, LLValue *nbytes, unsigned align) {
  gIR->ir->CreateMemSet(dst, val, nbytes, llvm::MaybeAlign(align),
                        false /*isVolatile*/);
}

////////////////////////////////////////////////////////////////////////////////

void DtoMemSetZero(LLType *type, LLValue *dst, LLValue *nbytes, unsigned align) {
  DtoMemSet(dst, DtoConstUbyte(0), nbytes, align);
}

void DtoMemSetZero(LLType *type, LLValue *dst, unsigned align) {
  uint64_t n = getTypeStoreSize(type);
  DtoMemSetZero(type, dst, DtoConstSize_t(n), align);
}

////////////////////////////////////////////////////////////////////////////////

void DtoMemCpy(LLValue *dst, LLValue *src, LLValue *nbytes, unsigned align) {
  auto A = llvm::MaybeAlign(align);
  gIR->ir->CreateMemCpy(dst, A, src, A, nbytes, false /*isVolatile*/);
}

void DtoMemCpy(LLType *type, LLValue *dst, LLValue *src, bool withPadding, unsigned align) {
  uint64_t n =
      withPadding ? getTypeAllocSize(type) : getTypeStoreSize(type);
  DtoMemCpy(dst, src, DtoConstSize_t(n), align);
}

////////////////////////////////////////////////////////////////////////////////

LLValue *DtoMemCmp(LLValue *lhs, LLValue *rhs, LLValue *nbytes) {
  // int memcmp ( const void * ptr1, const void * ptr2, size_t num );

  LLType *VoidPtrTy = getVoidPtrType();
  LLFunction *fn = gIR->module.getFunction("memcmp");
  if (!fn) {
    LLType *Tys[] = {VoidPtrTy, VoidPtrTy, DtoSize_t()};
    LLFunctionType *fty =
        LLFunctionType::get(LLType::getInt32Ty(gIR->context()), Tys, false);
    fn = LLFunction::Create(fty, LLGlobalValue::ExternalLinkage, "memcmp",
                            &gIR->module);
  }

  return gIR->ir->CreateCall(fn, {lhs, rhs, nbytes});
}

////////////////////////////////////////////////////////////////////////////////

llvm::ConstantInt *DtoConstSize_t(uint64_t i) {
  return LLConstantInt::get(DtoSize_t(), i, false);
}
llvm::ConstantInt *DtoConstUlong(uint64_t i) {
  return LLConstantInt::get(LLType::getInt64Ty(gIR->context()), i, false);
}
llvm::ConstantInt *DtoConstLong(int64_t i) {
  return LLConstantInt::get(LLType::getInt64Ty(gIR->context()), i, true);
}
llvm::ConstantInt *DtoConstUint(unsigned i) {
  return LLConstantInt::get(LLType::getInt32Ty(gIR->context()), i, false);
}
llvm::ConstantInt *DtoConstInt(int i) {
  return LLConstantInt::get(LLType::getInt32Ty(gIR->context()), i, true);
}
llvm::ConstantInt *DtoConstUshort(uint16_t i) {
  return LLConstantInt::get(LLType::getInt16Ty(gIR->context()), i, false);
}
LLConstant *DtoConstBool(bool b) {
  return LLConstantInt::get(LLType::getInt1Ty(gIR->context()), b, false);
}
llvm::ConstantInt *DtoConstUbyte(unsigned char i) {
  return LLConstantInt::get(LLType::getInt8Ty(gIR->context()), i, false);
}

LLConstant *DtoConstFP(Type *t, const real_t value) {
  LLType *llty = DtoType(t);
  assert(llty->isFloatingPointTy());

  // 1) represent host real_t as llvm::APFloat
  const auto &targetSemantics = llty->getFltSemantics();
  APFloat v(targetSemantics, APFloat::uninitialized);
  CTFloat::toAPFloat(value, v);

  // 2) convert to target format
  if (&v.getSemantics() != &targetSemantics) {
    bool ignored;
    v.convert(targetSemantics, APFloat::rmNearestTiesToEven, &ignored);
  }

  return LLConstantFP::get(gIR->context(), v);
}

////////////////////////////////////////////////////////////////////////////////

LLConstant *DtoConstCString(const char *str) {
  llvm::StringRef s(str ? str : "");
  LLGlobalVariable *gvar = gIR->getCachedStringLiteral(s);
  return gvar;
}

LLConstant *DtoConstString(const char *str) {
  LLConstant *cString = DtoConstCString(str);
  LLConstant *length = DtoConstSize_t(str ? strlen(str) : 0);
  return DtoConstSlice(length, cString);
}

////////////////////////////////////////////////////////////////////////////////

namespace {
llvm::LoadInst *DtoLoadImpl(LLType *type, LLValue *src, const char *name) {
  return gIR->ir->CreateLoad(type, src, name);
}
}

LLValue *DtoLoad(LLType* type, LLValue *src, const char *name) {
  return DtoLoadImpl(type, src, name);
}

LLValue *DtoLoad(DLValue *src, const char *name) {
  return DtoLoadImpl(DtoType(src->type), DtoLVal(src), name);
}

// Like DtoLoad, but the pointer is guaranteed to be aligned appropriately for
// the type.
LLValue *DtoAlignedLoad(LLType *type, LLValue *src, const char *name) {
  llvm::LoadInst *ld = DtoLoadImpl(type, src, name);
  ld->setAlignment(gDataLayout->getABITypeAlign(ld->getType()));
  return ld;
}

LLValue *DtoVolatileLoad(LLType *type, LLValue *src, const char *name) {
  llvm::LoadInst *ld = DtoLoadImpl(type, src, name);
  ld->setVolatile(true);
  return ld;
}

void DtoStore(LLValue *src, LLValue *dst) {
  assert(!src->getType()->isIntegerTy(1) &&
         "Should store bools as i8 instead of i1.");
  gIR->ir->CreateStore(src, dst);
}

void DtoVolatileStore(LLValue *src, LLValue *dst) {
  assert(!src->getType()->isIntegerTy(1) &&
         "Should store bools as i8 instead of i1.");
  gIR->ir->CreateStore(src, dst)->setVolatile(true);
}

void DtoStoreZextI8(LLValue *src, LLValue *dst) {
  if (src->getType()->isIntegerTy(1)) {
    llvm::Type *i8 = llvm::Type::getInt8Ty(gIR->context());
    src = gIR->ir->CreateZExt(src, i8);
  }
  gIR->ir->CreateStore(src, dst);
}

// Like DtoStore, but the pointer is guaranteed to be aligned appropriately for
// the type.
void DtoAlignedStore(LLValue *src, LLValue *dst) {
  assert(!src->getType()->isIntegerTy(1) &&
         "Should store bools as i8 instead of i1.");
  llvm::StoreInst *st = gIR->ir->CreateStore(src, dst);
  st->setAlignment(gDataLayout->getABITypeAlign(src->getType()));
}

////////////////////////////////////////////////////////////////////////////////

LLType *stripAddrSpaces(LLType *t)
{
  // Fastpath for normal compilation.
  if(gIR->dcomputetarget == nullptr)
    return t;

  llvm::PointerType *pt = isaPointer(t);
  if (!pt)
    return t;

  return getVoidPtrType();
}

LLValue *DtoBitCast(LLValue *v, LLType *t, const llvm::Twine &name) {
  // Strip addrspace qualifications from v before comparing types by pointer
  // equality. This avoids the case where the pointer in { T addrspace(n)* }
  // is dereferenced and generates a GEP -> (invalid) bitcast -> load sequence.
  // Bitcasting of pointers between addrspaces is invalid in LLVM IR. Even if
  // it were valid, it wouldn't be the desired outcome as we would always load
  // from addrspace(0), instead of the addrspace of the pointer.
  if (stripAddrSpaces(v->getType()) == t) {
    return v;
  }
  assert(!isaStruct(t));
  return gIR->ir->CreateBitCast(v, t, name);
}

LLConstant *DtoBitCast(LLConstant *v, LLType *t) {
  // Refer to the explanation in the other DtoBitCast overloaded function.
  if (stripAddrSpaces(v->getType()) == t) {
    return v;
  }
  return llvm::ConstantExpr::getBitCast(v, t);
}

////////////////////////////////////////////////////////////////////////////////

LLValue *DtoInsertValue(LLValue *aggr, LLValue *v, unsigned idx,
                        const char *name) {
  return gIR->ir->CreateInsertValue(aggr, v, idx, name);
}

LLValue *DtoExtractValue(LLValue *aggr, unsigned idx, const char *name) {
  return gIR->ir->CreateExtractValue(aggr, idx, name);
}

////////////////////////////////////////////////////////////////////////////////

LLValue *DtoInsertElement(LLValue *vec, LLValue *v, LLValue *idx,
                          const char *name) {
  return gIR->ir->CreateInsertElement(vec, v, idx, name);
}

LLValue *DtoExtractElement(LLValue *vec, LLValue *idx, const char *name) {
  return gIR->ir->CreateExtractElement(vec, idx, name);
}

LLValue *DtoInsertElement(LLValue *vec, LLValue *v, unsigned idx,
                          const char *name) {
  return DtoInsertElement(vec, v, DtoConstUint(idx), name);
}

LLValue *DtoExtractElement(LLValue *vec, unsigned idx, const char *name) {
  return DtoExtractElement(vec, DtoConstUint(idx), name);
}

////////////////////////////////////////////////////////////////////////////////

LLPointerType *isaPointer(LLValue *v) {
  return llvm::dyn_cast<LLPointerType>(v->getType());
}

LLPointerType *isaPointer(LLType *t) {
  return llvm::dyn_cast<LLPointerType>(t);
}

LLArrayType *isaArray(LLValue *v) {
  return llvm::dyn_cast<LLArrayType>(v->getType());
}

LLArrayType *isaArray(LLType *t) { return llvm::dyn_cast<LLArrayType>(t); }

LLStructType *isaStruct(LLValue *v) {
  return llvm::dyn_cast<LLStructType>(v->getType());
}

LLStructType *isaStruct(LLType *t) { return llvm::dyn_cast<LLStructType>(t); }

LLFunctionType *isaFunction(LLValue *v) {
  return llvm::dyn_cast<LLFunctionType>(v->getType());
}

LLFunctionType *isaFunction(LLType *t) {
  return llvm::dyn_cast<LLFunctionType>(t);
}

LLConstant *isaConstant(LLValue *v) {
  return llvm::dyn_cast<llvm::Constant>(v);
}

llvm::ConstantInt *isaConstantInt(LLValue *v) {
  return llvm::dyn_cast<llvm::ConstantInt>(v);
}

llvm::Argument *isaArgument(LLValue *v) {
  return llvm::dyn_cast<llvm::Argument>(v);
}

llvm::GlobalVariable *isaGlobalVar(LLValue *v) {
  return llvm::dyn_cast<llvm::GlobalVariable>(v);
}

////////////////////////////////////////////////////////////////////////////////

LLType *getI8Type() { return LLType::getInt8Ty(gIR->context()); }

LLPointerType *getPtrToType(LLType *t) {
  if (t == LLType::getVoidTy(gIR->context()))
    t = LLType::getInt8Ty(gIR->context());
  return t->getPointerTo();
}

LLPointerType *getVoidPtrType() {
  return getVoidPtrType(gIR->context());
}

LLPointerType *getVoidPtrType(llvm::LLVMContext &C) {
  return LLType::getInt8Ty(C)->getPointerTo();
}

llvm::ConstantPointerNull *getNullPtr(LLType *t) {
  LLPointerType *pt = llvm::cast<LLPointerType>(t);
  return llvm::ConstantPointerNull::get(pt);
}

LLConstant *getNullValue(LLType *t) { return LLConstant::getNullValue(t); }

////////////////////////////////////////////////////////////////////////////////

size_t getTypeBitSize(LLType *t) { return gDataLayout->getTypeSizeInBits(t); }

size_t getTypeStoreSize(LLType *t) { return gDataLayout->getTypeStoreSize(t); }

size_t getTypeAllocSize(LLType *t) { return gDataLayout->getTypeAllocSize(t); }

unsigned int getABITypeAlign(LLType *t) {
  return gDataLayout->getABITypeAlign(t).value();
}

////////////////////////////////////////////////////////////////////////////////

LLStructType *DtoModuleReferenceType() {
  if (gIR->moduleRefType) {
    return gIR->moduleRefType;
  }

  // this is a recursive type so start out with a struct without body
  LLStructType *st = LLStructType::create(gIR->context(), "ModuleReference");

  // add members
  LLType *types[] = {getPtrToType(st), DtoPtrToType(getModuleInfoType())};

  // resolve type
  st->setBody(types);

  // done
  gIR->moduleRefType = st;
  return st;
}

////////////////////////////////////////////////////////////////////////////////

LLValue *DtoAggrPair(LLType *type, LLValue *V1, LLValue *V2, const char *name) {
  LLValue *res = llvm::UndefValue::get(type);
  res = gIR->ir->CreateInsertValue(res, V1, 0);
  return gIR->ir->CreateInsertValue(res, V2, 1, name);
}

LLValue *DtoAggrPair(LLValue *V1, LLValue *V2, const char *name) {
  LLType *types[] = {V1->getType(), V2->getType()};
  LLType *t = LLStructType::get(gIR->context(), types, false);
  return DtoAggrPair(t, V1, V2, name);
}

LLValue *DtoSlicePaint(LLValue *aggr, LLType *as) {
  if (aggr->getType() == as) {
    return aggr;
  }

  LLValue *res = llvm::UndefValue::get(as);

  LLValue *V = gIR->ir->CreateExtractValue(aggr, 0);
  V = DtoBitCast(V, as->getContainedType(0));
  res = gIR->ir->CreateInsertValue(res, V, 0);

  V = gIR->ir->CreateExtractValue(aggr, 1);
  V = DtoBitCast(V, as->getContainedType(1));
  return gIR->ir->CreateInsertValue(res, V, 1);
}
