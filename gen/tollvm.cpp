//===-- tollvm.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/tollvm.h"
#include "aggregate.h"
#include "declaration.h"
#include "dsymbol.h"
#include "id.h"
#include "init.h"
#include "module.h"
#include "gen/abi.h"
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

bool DtoIsInMemoryOnly(Type *type) {
  Type *typ = type->toBasetype();
  TY t = typ->ty;
  return (t == Tstruct || t == Tsarray);
}

RET retStyle(TypeFunction *tf) {
  bool sret = gABI->returnInArg(tf);
  return sret ? RETstack : RETregs;
}

bool DtoIsReturnInArg(CallExp *ce) {
  TypeFunction *tf = static_cast<TypeFunction *>(ce->e1->type->toBasetype());
  if (tf->ty == Tfunction && (!ce->f || !DtoIsIntrinsic(ce->f))) {
    return retStyle(tf) == RETstack;
  }
  return false;
}

LLAttribute DtoShouldExtend(Type *type) {
  type = type->toBasetype();
  if (type->isintegral()) {
    switch (type->ty) {
    case Tint8:
    case Tint16:
      return LLAttribute::SExt;

    case Tuns8:
    case Tuns16:
    case Tchar:
    case Twchar:
      return LLAttribute::ZExt;

    default:
      // Do not extend.
      break;
    }
  }

  return LLAttribute::None;
}

LLType *DtoType(Type *t) {
  t = stripModifiers(t);

  if (t->ctype) {
    return t->ctype->getLLType();
  }

  IF_LOG Logger::println("Building type: %s", t->toChars());
  LOG_SCOPE;

  assert(t);
  switch (t->ty) {
  // basic types
  case Tvoid:
  case Tint8:
  case Tuns8:
  case Tint16:
  case Tuns16:
  case Tint32:
  case Tuns32:
  case Tint64:
  case Tuns64:
  case Tint128:
  case Tuns128:
  case Tfloat32:
  case Tfloat64:
  case Tfloat80:
  case Timaginary32:
  case Timaginary64:
  case Timaginary80:
  case Tcomplex32:
  case Tcomplex64:
  case Tcomplex80:
  // case Tbit:
  case Tbool:
  case Tchar:
  case Twchar:
  case Tdchar: {
    return IrTypeBasic::get(t)->getLLType();
  }

  // pointers
  case Tnull:
  case Tpointer: {
    return IrTypePointer::get(t)->getLLType();
  }

  // arrays
  case Tarray: {
    return IrTypeArray::get(t)->getLLType();
  }

  case Tsarray: {
    return IrTypeSArray::get(t)->getLLType();
  }

  // aggregates
  case Tstruct: {
    TypeStruct *ts = static_cast<TypeStruct *>(t);
    if (ts->sym->type->ctype) {
      // This should not happen, but the frontend seems to be buggy. Not
      // sure if this is the best way to handle the situation, but we
      // certainly don't want to override ts->sym->type->ctype.
      IF_LOG Logger::cout()
          << "Struct with multiple Types detected: " << ts->toChars() << " ("
          << ts->sym->locToChars() << ")" << std::endl;
      return ts->sym->type->ctype->getLLType();
    }
    return IrTypeStruct::get(ts->sym)->getLLType();
  }
  case Tclass: {
    TypeClass *tc = static_cast<TypeClass *>(t);
    if (tc->sym->type->ctype) {
      // See Tstruct case.
      IF_LOG Logger::cout()
          << "Class with multiple Types detected: " << tc->toChars() << " ("
          << tc->sym->locToChars() << ")" << std::endl;
      return tc->sym->type->ctype->getLLType();
    }
    return IrTypeClass::get(tc->sym)->getLLType();
  }

  // functions
  case Tfunction: {
    return IrTypeFunction::get(t)->getLLType();
  }

  // delegates
  case Tdelegate: {
    return IrTypeDelegate::get(t)->getLLType();
  }

  // typedefs
  // enum

  // FIXME: maybe just call toBasetype first ?
  case Tenum: {
    Type *bt = t->toBasetype();
    assert(bt);
    if (t == bt) {
      // This is an enum forward reference that is only legal when referenced
      // through an indirection (e.g. "enum E; void foo(E* p);"). For lack of a
      // better choice, make the outer indirection a void pointer.
      return getVoidPtrType()->getContainedType(0);
    }
    return DtoType(bt);
  }

  // associative arrays
  case Taarray:
    return getVoidPtrType();

  case Tvector: {
    return IrTypeVector::get(t)->getLLType();
  }

  default:
    llvm_unreachable("Unknown class of D Type!");
  }
  return nullptr;
}

LLType *DtoMemType(Type *t) { return i1ToI8(voidToI8(DtoType(t))); }

LLPointerType *DtoPtrToType(Type *t) { return DtoMemType(t)->getPointerTo(); }

LLType *voidToI8(LLType *t) {
  if (t == LLType::getVoidTy(gIR->context())) {
    return LLType::getInt8Ty(gIR->context());
  }
  return t;
}

LLType *i1ToI8(LLType *t) {
  if (t == LLType::getInt1Ty(gIR->context())) {
    return LLType::getInt8Ty(gIR->context());
  }
  return t;
}

////////////////////////////////////////////////////////////////////////////////

LLValue *DtoDelegateEquals(TOK op, LLValue *lhs, LLValue *rhs) {
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

LinkageWithCOMDAT DtoLinkage(Dsymbol *sym) {
  auto linkage = (DtoIsTemplateInstance(sym) ? templateLinkage
                                             : LLGlobalValue::ExternalLinkage);

  // If @(ldc.attributes.weak) is applied, override the linkage to WeakAny
  if (hasWeakUDA(sym)) {
    linkage = LLGlobalValue::WeakAnyLinkage;
  }

  return {linkage, supportsCOMDAT()};
}

bool supportsCOMDAT() {
#if LDC_LLVM_VER >= 307
  return !global.params.targetTriple->isOSBinFormatMachO();
#else
  return false;
#endif
}

void setLinkage(LinkageWithCOMDAT lwc, llvm::GlobalObject *obj) {
  obj->setLinkage(lwc.first);
  if (lwc.second)
    obj->setComdat(gIR->module.getOrInsertComdat(obj->getName()));
}

void setLinkage(Dsymbol *sym, llvm::GlobalObject *obj) {
  setLinkage(DtoLinkage(sym), obj);
}

////////////////////////////////////////////////////////////////////////////////

LLIntegerType *DtoSize_t() {
  // the type of size_t does not change once set
  static LLIntegerType *t = nullptr;
  if (t == nullptr) {
    t = (global.params.isLP64) ? LLType::getInt64Ty(gIR->context())
                               : LLType::getInt32Ty(gIR->context());
  }
  return t;
}

////////////////////////////////////////////////////////////////////////////////

namespace {
llvm::GetElementPtrInst *DtoGEP(LLValue *ptr, llvm::ArrayRef<LLValue *> indices,
                                bool inBounds, const char *name,
                                llvm::BasicBlock *bb) {
  LLPointerType *p = isaPointer(ptr);
  assert(p && "GEP expects a pointer type");
  auto gep = llvm::GetElementPtrInst::Create(
#if LDC_LLVM_VER >= 307
      p->getElementType(),
#endif
      ptr, indices, name, bb ? bb : gIR->scopebb());
  gep->setIsInBounds(inBounds);
  return gep;
}
}

LLValue *DtoGEP1(LLValue *ptr, LLValue *i0, bool inBounds, const char *name,
                 llvm::BasicBlock *bb) {
  return DtoGEP(ptr, i0, inBounds, name, bb);
}

LLValue *DtoGEP(LLValue *ptr, LLValue *i0, LLValue *i1, bool inBounds,
                const char *name, llvm::BasicBlock *bb) {
  LLValue *indices[] = {i0, i1};
  return DtoGEP(ptr, indices, inBounds, name, bb);
}

LLValue *DtoGEPi1(LLValue *ptr, unsigned i0, const char *name,
                  llvm::BasicBlock *bb) {
  return DtoGEP(ptr, DtoConstUint(i0), /* inBounds = */ true, name, bb);
}

LLValue *DtoGEPi(LLValue *ptr, unsigned i0, unsigned i1, const char *name,
                 llvm::BasicBlock *bb) {
  LLValue *indices[] = {DtoConstUint(i0), DtoConstUint(i1)};
  return DtoGEP(ptr, indices, /* inBounds = */ true, name, bb);
}

LLConstant *DtoGEPi(LLConstant *ptr, unsigned i0, unsigned i1) {
  LLPointerType *p = isaPointer(ptr);
  assert(p && "GEP expects a pointer type");
  LLValue *indices[] = {DtoConstUint(i0), DtoConstUint(i1)};
  return llvm::ConstantExpr::getGetElementPtr(
#if LDC_LLVM_VER >= 307
      p->getElementType(),
#endif
      ptr, indices, /* InBounds = */ true);
}

////////////////////////////////////////////////////////////////////////////////

void DtoMemSet(LLValue *dst, LLValue *val, LLValue *nbytes, unsigned align) {
  LLType *VoidPtrTy = getVoidPtrType();

  dst = DtoBitCast(dst, VoidPtrTy);

  gIR->ir->CreateMemSet(dst, val, nbytes, align, false /*isVolatile*/);
}

////////////////////////////////////////////////////////////////////////////////

void DtoMemSetZero(LLValue *dst, LLValue *nbytes, unsigned align) {
  DtoMemSet(dst, DtoConstUbyte(0), nbytes, align);
}

void DtoMemSetZero(LLValue *dst, unsigned align) {
  uint64_t n = getTypeStoreSize(dst->getType()->getContainedType(0));
  DtoMemSetZero(dst, DtoConstSize_t(n), align);
}

////////////////////////////////////////////////////////////////////////////////

void DtoMemCpy(LLValue *dst, LLValue *src, LLValue *nbytes, unsigned align) {
  LLType *VoidPtrTy = getVoidPtrType();

  dst = DtoBitCast(dst, VoidPtrTy);
  src = DtoBitCast(src, VoidPtrTy);

  gIR->ir->CreateMemCpy(dst, src, nbytes, align, false /*isVolatile*/);
}

void DtoMemCpy(LLValue *dst, LLValue *src, bool withPadding, unsigned align) {
  LLType *pointee = dst->getType()->getContainedType(0);
  uint64_t n =
      withPadding ? getTypeAllocSize(pointee) : getTypeStoreSize(pointee);
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

  lhs = DtoBitCast(lhs, VoidPtrTy);
  rhs = DtoBitCast(rhs, VoidPtrTy);

#if LDC_LLVM_VER >= 307
  return gIR->ir->CreateCall(fn, {lhs, rhs, nbytes});
#else
  return gIR->ir->CreateCall3(fn, lhs, rhs, nbytes);
#endif
}

////////////////////////////////////////////////////////////////////////////////

llvm::ConstantInt *DtoConstSize_t(uint64_t i) {
  return LLConstantInt::get(DtoSize_t(), i, false);
}
llvm::ConstantInt *DtoConstUint(unsigned i) {
  return LLConstantInt::get(LLType::getInt32Ty(gIR->context()), i, false);
}
llvm::ConstantInt *DtoConstInt(int i) {
  return LLConstantInt::get(LLType::getInt32Ty(gIR->context()), i, true);
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

  assert(sizeof(real_t) >= 8 && "real_t < 64 bits?");

  if (llty->isFloatTy()) {
    // let host narrow to single-precision target
    return LLConstantFP::get(gIR->context(),
                             APFloat(static_cast<float>(value)));
  }

  if (llty->isDoubleTy()) {
    // let host (potentially) narrow to double-precision target
    return LLConstantFP::get(gIR->context(),
                             APFloat(static_cast<double>(value)));
  }

  // host real_t => target real

  // 1) represent host real_t as llvm::APFloat
  const auto &targetRealSemantics = llty->getFltSemantics();
  APFloat v(targetRealSemantics, APFloat::uninitialized);
  CTFloat::toAPFloat(value, v);

  // 2) convert to target real
  if (&v.getSemantics() != &targetRealSemantics) {
    bool ignored;
    v.convert(targetRealSemantics, APFloat::rmNearestTiesToEven, &ignored);
  }

  return LLConstantFP::get(gIR->context(), v);
}

////////////////////////////////////////////////////////////////////////////////

LLConstant *DtoConstString(const char *str) {
  llvm::StringRef s(str ? str : "");
  llvm::GlobalVariable *gvar = (gIR->stringLiteral1ByteCache.find(s) ==
                                gIR->stringLiteral1ByteCache.end())
                                   ? nullptr
                                   : gIR->stringLiteral1ByteCache[s];
  if (gvar == nullptr) {
    llvm::Constant *init =
        llvm::ConstantDataArray::getString(gIR->context(), s, true);
    gvar = new llvm::GlobalVariable(gIR->module, init->getType(), true,
                                    llvm::GlobalValue::PrivateLinkage, init,
                                    ".str");
#if LDC_LLVM_VER >= 309
    gvar->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
#else
    gvar->setUnnamedAddr(true);
#endif
    gIR->stringLiteral1ByteCache[s] = gvar;
  }
  LLConstant *idxs[] = {DtoConstUint(0), DtoConstUint(0)};
  return DtoConstSlice(DtoConstSize_t(s.size()),
                       llvm::ConstantExpr::getGetElementPtr(
#if LDC_LLVM_VER >= 307
                           gvar->getInitializer()->getType(),
#endif
                           gvar, idxs, true),
                       Type::tchar->arrayOf());
}

////////////////////////////////////////////////////////////////////////////////

LLValue *DtoLoad(LLValue *src, const char *name) {
  return gIR->ir->CreateLoad(src, name);
}

// Like DtoLoad, but the pointer is guaranteed to be aligned appropriately for
// the type.
LLValue *DtoAlignedLoad(LLValue *src, const char *name) {
  llvm::LoadInst *ld = gIR->ir->CreateLoad(src, name);
  ld->setAlignment(getABITypeAlign(ld->getType()));
  return ld;
}

LLValue *DtoVolatileLoad(LLValue *src, const char *name) {
  llvm::LoadInst *ld = gIR->ir->CreateLoad(src, name);
  ld->setVolatile(true);
  return ld;
}

void DtoStore(LLValue *src, LLValue *dst) {
  assert(src->getType() != llvm::Type::getInt1Ty(gIR->context()) &&
         "Should store bools as i8 instead of i1.");
  gIR->ir->CreateStore(src, dst);
}

void DtoVolatileStore(LLValue *src, LLValue *dst) {
  assert(src->getType() != llvm::Type::getInt1Ty(gIR->context()) &&
         "Should store bools as i8 instead of i1.");
  gIR->ir->CreateStore(src, dst)->setVolatile(true);
}

void DtoStoreZextI8(LLValue *src, LLValue *dst) {
  if (src->getType() == llvm::Type::getInt1Ty(gIR->context())) {
    llvm::Type *i8 = llvm::Type::getInt8Ty(gIR->context());
    assert(dst->getType()->getContainedType(0) == i8);
    src = gIR->ir->CreateZExt(src, i8);
  }
  gIR->ir->CreateStore(src, dst);
}

// Like DtoStore, but the pointer is guaranteed to be aligned appropriately for
// the type.
void DtoAlignedStore(LLValue *src, LLValue *dst) {
  assert(src->getType() != llvm::Type::getInt1Ty(gIR->context()) &&
         "Should store bools as i8 instead of i1.");
  llvm::StoreInst *st = gIR->ir->CreateStore(src, dst);
  st->setAlignment(getABITypeAlign(src->getType()));
}

////////////////////////////////////////////////////////////////////////////////

LLValue *DtoBitCast(LLValue *v, LLType *t, const llvm::Twine &name) {
  if (v->getType() == t) {
    return v;
  }
  assert(!isaStruct(t));
  return gIR->ir->CreateBitCast(v, t, name);
}

LLConstant *DtoBitCast(LLConstant *v, LLType *t) {
  if (v->getType() == t) {
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

LLPointerType *getPtrToType(LLType *t) {
  if (t == LLType::getVoidTy(gIR->context()))
    t = LLType::getInt8Ty(gIR->context());
  return t->getPointerTo();
}

LLPointerType *getVoidPtrType() {
  return LLType::getInt8Ty(gIR->context())->getPointerTo();
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
  return gDataLayout->getABITypeAlignment(t);
}

////////////////////////////////////////////////////////////////////////////////

LLStructType *DtoModuleReferenceType() {
  if (gIR->moduleRefType) {
    return gIR->moduleRefType;
  }

  // this is a recursive type so start out with a struct without body
  LLStructType *st = LLStructType::create(gIR->context(), "ModuleReference");

  // add members
  LLType *types[] = {getPtrToType(st),
                     DtoType(Module::moduleinfo->type->pointerTo())};

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

LLValue *DtoAggrPaint(LLValue *aggr, LLType *as) {
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
