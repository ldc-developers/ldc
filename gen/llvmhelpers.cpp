//===-- llvmhelpers.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvmhelpers.h"
#include "gen/cl_helpers.h"
#include "declaration.h"
#include "expression.h"
#include "gen/abi.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/complex.h"
#include "gen/dvalue.h"
#include "gen/funcgenstate.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/logger.h"
#include "gen/nested.h"
#include "gen/mangling.h"
#include "gen/pragma.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "gen/typinf.h"
#include "gen/uda.h"
#include "id.h"
#include "init.h"
#include "ir/irfunction.h"
#include "ir/irmodule.h"
#include "ir/irtypeaggr.h"
#include "mars.h"
#include "module.h"
#include "template.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Support/ManagedStatic.h"
#include <stack>

#include "llvm/Support/CommandLine.h"
#include "gen/dynamiccompile.h"

llvm::cl::opt<llvm::GlobalVariable::ThreadLocalMode> clThreadModel(
    "fthread-model", llvm::cl::ZeroOrMore, llvm::cl::desc("Thread model"),
    llvm::cl::init(llvm::GlobalVariable::GeneralDynamicTLSModel),
    clEnumValues(clEnumValN(llvm::GlobalVariable::GeneralDynamicTLSModel,
                            "global-dynamic",
                            "Global dynamic TLS model (default)"),
                 clEnumValN(llvm::GlobalVariable::LocalDynamicTLSModel,
                            "local-dynamic", "Local dynamic TLS model"),
                 clEnumValN(llvm::GlobalVariable::InitialExecTLSModel,
                            "initial-exec", "Initial exec TLS model"),
                 clEnumValN(llvm::GlobalVariable::LocalExecTLSModel,
                            "local-exec", "Local exec TLS model")));

/******************************************************************************
 * Simple Triple helpers for DFE
 * TODO: find better location for this
 ******************************************************************************/
bool isArchx86_64() {
  return global.params.targetTriple->getArch() == llvm::Triple::x86_64;
}

bool isTargetWindowsMSVC() {
  return global.params.targetTriple->isWindowsMSVCEnvironment();
}

/******************************************************************************
 * Global context
 ******************************************************************************/
static llvm::ManagedStatic<llvm::LLVMContext> GlobalContext;

llvm::LLVMContext &getGlobalContext() { return *GlobalContext; }

/******************************************************************************
 * DYNAMIC MEMORY HELPERS
 ******************************************************************************/

LLValue *DtoNew(Loc &loc, Type *newtype) {
  // get runtime function
  llvm::Function *fn = getRuntimeFunction(loc, gIR->module, "_d_allocmemoryT");
  // get type info
  LLConstant *ti = DtoTypeInfoOf(newtype);
  assert(isaPointer(ti));
  // call runtime allocator
  LLValue *mem = gIR->CreateCallOrInvoke(fn, ti, ".gc_mem").getInstruction();
  // cast
  return DtoBitCast(mem, DtoPtrToType(newtype), ".gc_mem");
}

LLValue *DtoNewStruct(Loc &loc, TypeStruct *newtype) {
  llvm::Function *fn = getRuntimeFunction(
      loc, gIR->module,
      newtype->isZeroInit(newtype->sym->loc) ? "_d_newitemT" : "_d_newitemiT");
  LLConstant *ti = DtoTypeInfoOf(newtype);
  LLValue *mem = gIR->CreateCallOrInvoke(fn, ti, ".gc_struct").getInstruction();
  return DtoBitCast(mem, DtoPtrToType(newtype), ".gc_struct");
}

void DtoDeleteMemory(Loc &loc, DValue *ptr) {
  llvm::Function *fn = getRuntimeFunction(loc, gIR->module, "_d_delmemory");
  LLValue *lval = (ptr->isLVal() ? DtoLVal(ptr) : makeLValue(loc, ptr));
  gIR->CreateCallOrInvoke(
      fn, DtoBitCast(lval, fn->getFunctionType()->getParamType(0)));
}

void DtoDeleteStruct(Loc &loc, DValue *ptr) {
  llvm::Function *fn = getRuntimeFunction(loc, gIR->module, "_d_delstruct");
  LLValue *lval = (ptr->isLVal() ? DtoLVal(ptr) : makeLValue(loc, ptr));
  gIR->CreateCallOrInvoke(
      fn, DtoBitCast(lval, fn->getFunctionType()->getParamType(0)),
      DtoBitCast(DtoTypeInfoOf(ptr->type->nextOf()),
                 fn->getFunctionType()->getParamType(1)));
}

void DtoDeleteClass(Loc &loc, DValue *inst) {
  llvm::Function *fn = getRuntimeFunction(loc, gIR->module, "_d_delclass");
  LLValue *lval = (inst->isLVal() ? DtoLVal(inst) : makeLValue(loc, inst));
  gIR->CreateCallOrInvoke(
      fn, DtoBitCast(lval, fn->getFunctionType()->getParamType(0)));
}

void DtoDeleteInterface(Loc &loc, DValue *inst) {
  llvm::Function *fn = getRuntimeFunction(loc, gIR->module, "_d_delinterface");
  LLValue *lval = (inst->isLVal() ? DtoLVal(inst) : makeLValue(loc, inst));
  gIR->CreateCallOrInvoke(
      fn, DtoBitCast(lval, fn->getFunctionType()->getParamType(0)));
}

void DtoDeleteArray(Loc &loc, DValue *arr) {
  llvm::Function *fn = getRuntimeFunction(loc, gIR->module, "_d_delarray_t");
  llvm::FunctionType *fty = fn->getFunctionType();

  // the TypeInfo argument must be null if the type has no dtor
  Type *elementType = arr->type->nextOf();
  bool hasDtor = (elementType->toBasetype()->ty == Tstruct &&
                  elementType->needsDestruction());
  LLValue *typeInfo = (!hasDtor ? getNullPtr(fty->getParamType(1))
                                : DtoTypeInfoOf(elementType));

  LLValue *lval = (arr->isLVal() ? DtoLVal(arr) : makeLValue(loc, arr));
  gIR->CreateCallOrInvoke(fn, DtoBitCast(lval, fty->getParamType(0)),
                          DtoBitCast(typeInfo, fty->getParamType(1)));
}

/******************************************************************************
 * ALIGNMENT HELPERS
 ******************************************************************************/

unsigned DtoAlignment(Type *type) {
  structalign_t alignment = type->alignment();
  if (alignment == STRUCTALIGN_DEFAULT) {
    alignment = type->alignsize();
  }
  return (alignment == STRUCTALIGN_DEFAULT ? 0 : alignment);
}

unsigned DtoAlignment(VarDeclaration *vd) {
  return vd->alignment == STRUCTALIGN_DEFAULT ? DtoAlignment(vd->type)
                                              : vd->alignment;
}

/******************************************************************************
 * ALLOCA HELPERS
 ******************************************************************************/

llvm::AllocaInst *DtoAlloca(Type *type, const char *name) {
  return DtoRawAlloca(DtoMemType(type), DtoAlignment(type), name);
}

llvm::AllocaInst *DtoAlloca(VarDeclaration *vd, const char *name) {
  return DtoRawAlloca(DtoMemType(vd->type), DtoAlignment(vd), name);
}

llvm::AllocaInst *DtoArrayAlloca(Type *type, unsigned arraysize,
                                 const char *name) {
  LLType *lltype = DtoType(type);
  auto ai = new llvm::AllocaInst(
      lltype,
#if LDC_LLVM_VER >= 500
      gIR->module.getDataLayout().getAllocaAddrSpace(),
#endif
      DtoConstUint(arraysize), name, gIR->topallocapoint());
  ai->setAlignment(DtoAlignment(type));
  return ai;
}

llvm::AllocaInst *DtoRawAlloca(LLType *lltype, size_t alignment,
                               const char *name) {
  auto ai =
      new llvm::AllocaInst(lltype,
#if LDC_LLVM_VER >= 500
                           gIR->module.getDataLayout().getAllocaAddrSpace(),
#endif
                           name, gIR->topallocapoint());
  if (alignment) {
    ai->setAlignment(alignment);
  }
  return ai;
}

LLValue *DtoGcMalloc(Loc &loc, LLType *lltype, const char *name) {
  // get runtime function
  llvm::Function *fn = getRuntimeFunction(loc, gIR->module, "_d_allocmemory");
  // parameters
  LLValue *size = DtoConstSize_t(getTypeAllocSize(lltype));
  // call runtime allocator
  LLValue *mem = gIR->CreateCallOrInvoke(fn, size, name).getInstruction();
  // cast
  return DtoBitCast(mem, getPtrToType(lltype), name);
}

LLValue *DtoAllocaDump(DValue *val, const char *name) {
  return DtoAllocaDump(val, val->type, name);
}

LLValue *DtoAllocaDump(DValue *val, int alignment, const char *name) {
  return DtoAllocaDump(val, DtoType(val->type), alignment, name);
}

LLValue *DtoAllocaDump(DValue *val, Type *asType, const char *name) {
  return DtoAllocaDump(val, DtoType(asType), DtoAlignment(asType), name);
}

LLValue *DtoAllocaDump(DValue *val, LLType *asType, int alignment,
                       const char *name) {
  if (val->isLVal()) {
    LLValue *lval = DtoLVal(val);
    LLType *asMemType = i1ToI8(voidToI8(asType));
    LLValue *copy = DtoRawAlloca(asMemType, alignment, name);
    const auto minSize =
        std::min(getTypeAllocSize(lval->getType()->getPointerElementType()),
                 getTypeAllocSize(asMemType));
    const auto minAlignment =
        std::min(DtoAlignment(val->type), static_cast<unsigned>(alignment));
    DtoMemCpy(copy, lval, DtoConstSize_t(minSize), minAlignment);
    // TODO: zero-out any remaining bytes?
    return copy;
  }

  return DtoAllocaDump(DtoRVal(val), asType, alignment, name);
}

LLValue *DtoAllocaDump(LLValue *val, int alignment, const char *name) {
  return DtoAllocaDump(val, val->getType(), alignment, name);
}

LLValue *DtoAllocaDump(LLValue *val, Type *asType, const char *name) {
  return DtoAllocaDump(val, DtoType(asType), DtoAlignment(asType), name);
}

LLValue *DtoAllocaDump(LLValue *val, LLType *asType, int alignment,
                       const char *name) {
  LLType *memType = i1ToI8(voidToI8(val->getType()));
  LLType *asMemType = i1ToI8(voidToI8(asType));
  LLType *allocaType =
      (getTypeStoreSize(memType) <= getTypeAllocSize(asMemType) ? asMemType
                                                                : memType);
  LLValue *mem = DtoRawAlloca(allocaType, alignment, name);
  DtoStoreZextI8(val, DtoBitCast(mem, memType->getPointerTo()));
  return DtoBitCast(mem, asMemType->getPointerTo());
}

/******************************************************************************
 * ASSERT HELPER
 ******************************************************************************/

void DtoAssert(Module *M, Loc &loc, DValue *msg) {
  // func
  const char *fname = msg ? "_d_assert_msg" : "_d_assert";
  llvm::Function *fn = getRuntimeFunction(loc, gIR->module, fname);

  // Arguments
  llvm::SmallVector<LLValue *, 3> args;

  // msg param
  if (msg) {
    args.push_back(DtoRVal(msg));
  }

  // file param
  args.push_back(DtoModuleFileName(M, loc));

  // line param
  args.push_back(DtoConstUint(loc.linnum));

  // call
  gIR->funcGen().callOrInvoke(fn, args);

  // after assert is always unreachable
  gIR->ir->CreateUnreachable();
}

void DtoCAssert(Module *M, Loc &loc, LLValue *msg) {
  const auto file = DtoConstCString(loc.filename ? loc.filename
                                                 : M->srcfile->name->toChars());
  const auto line = DtoConstUint(loc.linnum);
  const auto fn = getCAssertFunction(loc, gIR->module);

  llvm::SmallVector<LLValue *, 4> args;
  if (global.params.targetTriple->isOSDarwin()) {
    const auto irFunc = gIR->func();
    const auto funcName =
        irFunc && irFunc->decl ? irFunc->decl->toPrettyChars() : "";
    args.push_back(DtoConstCString(funcName));
    args.push_back(file);
    args.push_back(line);
    args.push_back(msg);
  } else if (global.params.targetTriple->isOSSolaris()) {
    const auto irFunc = gIR->func();
    const auto funcName =
        irFunc && irFunc->decl ? irFunc->decl->toPrettyChars() : "";
    args.push_back(msg);
    args.push_back(file);
    args.push_back(line);
    args.push_back(DtoConstCString(funcName));
  } else if (global.params.targetTriple->getEnvironment() ==
             llvm::Triple::Android) {
    args.push_back(file);
    args.push_back(line);
    args.push_back(msg);
  } else {
    args.push_back(msg);
    args.push_back(file);
    args.push_back(line);
  }

  gIR->funcGen().callOrInvoke(fn, args);

  gIR->ir->CreateUnreachable();
}

/******************************************************************************
 * MODULE FILE NAME
 ******************************************************************************/

LLConstant *DtoModuleFileName(Module *M, const Loc &loc) {
  return DtoConstString(loc.filename ? loc.filename
                                     : M->srcfile->name->toChars());
}

/******************************************************************************
 * GOTO HELPER
 ******************************************************************************/

void DtoGoto(Loc &loc, LabelDsymbol *target) {
  assert(!gIR->scopereturned());

  LabelStatement *lblstmt = target->statement;
  if (!lblstmt) {
    error(loc, "the label `%s` does not exist", target->ident->toChars());
    fatal();
  }

  gIR->funcGen().jumpTargets.jumpToLabel(loc, target->ident);
}

/******************************************************************************
 * ASSIGNMENT HELPER (store this in that)
 ******************************************************************************/

// is this a good approach at all ?

void DtoAssign(Loc &loc, DValue *lhs, DValue *rhs, int op,
               bool canSkipPostblit) {
  IF_LOG Logger::println("DtoAssign()");
  LOG_SCOPE;

  Type *t = lhs->type->toBasetype();
  assert(t->ty != Tvoid && "Cannot assign values of type void.");

  if (t->ty == Tbool) {
    DtoStoreZextI8(DtoRVal(rhs), DtoLVal(lhs));
  } else if (t->ty == Tstruct) {
    // don't copy anything to empty structs
    if (static_cast<TypeStruct *>(t)->sym->fields.dim > 0) {
      llvm::Value *src = DtoLVal(rhs);
      llvm::Value *dst = DtoLVal(lhs);

      // Check whether source and destination values are the same at compile
      // time as to not emit an invalid (overlapping) memcpy on trivial
      // struct self-assignments like 'A a; a = a;'.
      if (src != dst)
        DtoMemCpy(dst, src);
    }
  } else if (t->ty == Tarray || t->ty == Tsarray) {
    DtoArrayAssign(loc, lhs, rhs, op, canSkipPostblit);
  } else if (t->ty == Tdelegate) {
    LLValue *l = DtoLVal(lhs);
    LLValue *r = DtoRVal(rhs);
    IF_LOG {
      Logger::cout() << "lhs: " << *l << '\n';
      Logger::cout() << "rhs: " << *r << '\n';
    }
    DtoStore(r, l);
  } else if (t->ty == Tclass) {
    assert(rhs->type->toBasetype()->ty == Tclass);
    LLValue *l = DtoLVal(lhs);
    LLValue *r = DtoRVal(rhs);
    IF_LOG {
      Logger::cout() << "l : " << *l << '\n';
      Logger::cout() << "r : " << *r << '\n';
    }
    r = DtoBitCast(r, l->getType()->getContainedType(0));
    DtoStore(r, l);
  } else if (t->iscomplex()) {
    LLValue *dst = DtoLVal(lhs);
    LLValue *src = DtoRVal(DtoCast(loc, rhs, lhs->type));
    DtoStore(src, dst);
  } else {
    LLValue *l = DtoLVal(lhs);
    LLValue *r = DtoRVal(rhs);
    IF_LOG {
      Logger::cout() << "lhs: " << *l << '\n';
      Logger::cout() << "rhs: " << *r << '\n';
    }
    LLType *lit = l->getType()->getContainedType(0);
    if (r->getType() != lit) {
      r = DtoRVal(DtoCast(loc, rhs, lhs->type));
      IF_LOG {
        Logger::println("Type mismatch, really assigning:");
        LOG_SCOPE
        Logger::cout() << "lhs: " << *l << '\n';
        Logger::cout() << "rhs: " << *r << '\n';
      }
#if 1
      if (r->getType() !=
          lit) { // It's weird but it happens. TODO: try to remove this hack
        r = DtoBitCast(r, lit);
      }
#else
      assert(r->getType() == lit);
#endif
    }
    gIR->ir->CreateStore(r, l);
  }
}

/******************************************************************************
 * NULL VALUE HELPER
 ******************************************************************************/

DValue *DtoNullValue(Type *type, Loc loc) {
  Type *basetype = type->toBasetype();
  TY basety = basetype->ty;
  LLType *lltype = DtoType(basetype);

  // complex, needs to be first since complex are also floating
  if (basetype->iscomplex()) {
    LLType *basefp = DtoComplexBaseType(basetype);
    LLValue *res = DtoAggrPair(DtoType(type), LLConstant::getNullValue(basefp),
                               LLConstant::getNullValue(basefp));
    return new DImValue(type, res);
  }
  // integer, floating, pointer, assoc array, delegate and class have no special
  // representation
  if (basetype->isintegral() || basetype->isfloating() || basety == Tpointer ||
      basety == Tnull || basety == Tclass || basety == Tdelegate ||
      basety == Taarray) {
    return new DNullValue(type, LLConstant::getNullValue(lltype));
  }
  // dynamic array
  if (basety == Tarray) {
    LLValue *len = DtoConstSize_t(0);
    LLValue *ptr = getNullPtr(DtoPtrToType(basetype->nextOf()));
    return new DSliceValue(type, len, ptr);
  }
  error(loc, "`null` not known for type `%s`", type->toChars());
  fatal();
}

/******************************************************************************
 * CASTING HELPERS
 ******************************************************************************/

DValue *DtoCastInt(Loc &loc, DValue *val, Type *_to) {
  LLType *tolltype = DtoType(_to);

  Type *to = _to->toBasetype();
  Type *from = val->type->toBasetype();
  assert(from->isintegral());

  LLValue *rval = DtoRVal(val);
  if (rval->getType() == tolltype) {
    return new DImValue(_to, rval);
  }

  size_t fromsz = from->size();
  size_t tosz = to->size();

  if (to->ty == Tbool) {
    LLValue *zero = LLConstantInt::get(rval->getType(), 0, false);
    rval = gIR->ir->CreateICmpNE(rval, zero);
  } else if (to->isintegral()) {
    if (fromsz < tosz || from->ty == Tbool) {
      IF_LOG Logger::cout() << "cast to: " << *tolltype << '\n';
      if (isLLVMUnsigned(from) || from->ty == Tbool) {
        rval = new llvm::ZExtInst(rval, tolltype, "", gIR->scopebb());
      } else {
        rval = new llvm::SExtInst(rval, tolltype, "", gIR->scopebb());
      }
    } else if (fromsz > tosz) {
      rval = new llvm::TruncInst(rval, tolltype, "", gIR->scopebb());
    } else {
      rval = DtoBitCast(rval, tolltype);
    }
  } else if (to->iscomplex()) {
    return DtoComplex(loc, to, val);
  } else if (to->isfloating()) {
    if (from->isunsigned()) {
      rval = new llvm::UIToFPInst(rval, tolltype, "", gIR->scopebb());
    } else {
      rval = new llvm::SIToFPInst(rval, tolltype, "", gIR->scopebb());
    }
  } else if (to->ty == Tpointer) {
    IF_LOG Logger::cout() << "cast pointer: " << *tolltype << '\n';
    rval = gIR->ir->CreateIntToPtr(rval, tolltype);
  } else {
    error(loc, "invalid cast from `%s` to `%s`", val->type->toChars(),
          _to->toChars());
    fatal();
  }

  return new DImValue(_to, rval);
}

DValue *DtoCastPtr(Loc &loc, DValue *val, Type *to) {
  LLType *tolltype = DtoType(to);

  Type *totype = to->toBasetype();
  Type *fromtype = val->type->toBasetype();
  (void)fromtype;
  assert(fromtype->ty == Tpointer || fromtype->ty == Tfunction);

  LLValue *rval;

  if (totype->ty == Tpointer || totype->ty == Tclass) {
    LLValue *src = DtoRVal(val);
    IF_LOG {
      Logger::cout() << "src: " << *src << '\n';
      Logger::cout() << "to type: " << *tolltype << '\n';
    }
    rval = DtoBitCast(src, tolltype);
  } else if (totype->ty == Tbool) {
    LLValue *src = DtoRVal(val);
    LLValue *zero = LLConstant::getNullValue(src->getType());
    rval = gIR->ir->CreateICmpNE(src, zero);
  } else if (totype->isintegral()) {
    rval = new llvm::PtrToIntInst(DtoRVal(val), tolltype, "", gIR->scopebb());
  } else {
    error(loc, "invalid cast from `%s` to `%s`", val->type->toChars(),
          to->toChars());
    fatal();
  }

  return new DImValue(to, rval);
}

DValue *DtoCastFloat(Loc &loc, DValue *val, Type *to) {
  if (val->type == to) {
    return val;
  }

  LLType *tolltype = DtoType(to);

  Type *totype = to->toBasetype();
  Type *fromtype = val->type->toBasetype();
  assert(fromtype->isfloating());

  size_t fromsz = fromtype->size();
  size_t tosz = totype->size();

  LLValue *rval;

  if (totype->ty == Tbool) {
    rval = DtoRVal(val);
    LLValue *zero = LLConstant::getNullValue(rval->getType());
    rval = gIR->ir->CreateFCmpUNE(rval, zero);
  } else if (totype->iscomplex()) {
    return DtoComplex(loc, to, val);
  } else if (totype->isfloating()) {
    if (fromsz == tosz) {
      rval = DtoRVal(val);
      assert(rval->getType() == tolltype);
    } else if (fromsz < tosz) {
      rval = new llvm::FPExtInst(DtoRVal(val), tolltype, "", gIR->scopebb());
    } else if (fromsz > tosz) {
      rval = new llvm::FPTruncInst(DtoRVal(val), tolltype, "", gIR->scopebb());
    } else {
      error(loc, "invalid cast from `%s` to `%s`", val->type->toChars(),
            to->toChars());
      fatal();
    }
  } else if (totype->isintegral()) {
    if (totype->isunsigned()) {
      rval = new llvm::FPToUIInst(DtoRVal(val), tolltype, "", gIR->scopebb());
    } else {
      rval = new llvm::FPToSIInst(DtoRVal(val), tolltype, "", gIR->scopebb());
    }
  } else {
    error(loc, "invalid cast from `%s` to `%s`", val->type->toChars(),
          to->toChars());
    fatal();
  }

  return new DImValue(to, rval);
}

DValue *DtoCastDelegate(Loc &loc, DValue *val, Type *to) {
  if (to->toBasetype()->ty == Tdelegate) {
    return DtoPaintType(loc, val, to);
  }
  if (to->toBasetype()->ty == Tbool) {
    return new DImValue(to,
                        DtoDelegateEquals(TOKnotequal, DtoRVal(val), nullptr));
  }
  error(loc, "invalid cast from `%s` to `%s`", val->type->toChars(),
        to->toChars());
  fatal();
}

DValue *DtoCastVector(Loc &loc, DValue *val, Type *to) {
  assert(val->type->toBasetype()->ty == Tvector);
  Type *totype = to->toBasetype();
  LLType *tolltype = DtoType(to);

  if (totype->ty == Tsarray) {
    // If possible, we need to cast only the address of the vector without
    // creating a copy, because, besides the fact that this seem to be the
    // language semantics, DMD rewrites e.g. float4.array to
    // cast(float[4])array.
    if (val->isLVal()) {
      LLValue *vector = DtoLVal(val);
      IF_LOG Logger::cout() << "src: " << *vector << "to type: " << *tolltype
                            << " (casting address)\n";
      return new DLValue(to, DtoBitCast(vector, getPtrToType(tolltype)));
    }

    LLValue *vector = DtoRVal(val);
    IF_LOG Logger::cout() << "src: " << *vector << "to type: " << *tolltype
                          << " (creating temporary)\n";
    LLValue *array = DtoAlloca(to);
    DtoStore(vector, DtoBitCast(array, getPtrToType(vector->getType())));
    return new DLValue(to, array);
  }
  if (totype->ty == Tvector && to->size() == val->type->size()) {
    return new DImValue(to, DtoBitCast(DtoRVal(val), tolltype));
  }
  error(loc, "invalid cast from `%s` to `%s`", val->type->toChars(),
        to->toChars());
  fatal();
}

DValue *DtoCastStruct(Loc &loc, DValue *val, Type *to) {
  Type *const totype = to->toBasetype();
  if (totype->ty == Tstruct) {
    // This a cast to repaint a struct to another type, which the language
    // allows for identical layouts (opCast() and so on have been lowered
    // earlier by the frontend).
    llvm::Value *lval = DtoLVal(val);
    llvm::Value *result = DtoBitCast(lval, DtoType(to)->getPointerTo(),
                                     lval->getName() + ".repaint");
    return new DLValue(to, result);
  }

  error(loc, "Internal Compiler Error: Invalid struct cast from `%s` to `%s`",
        val->type->toChars(), to->toChars());
  fatal();
}

DValue *DtoCast(Loc &loc, DValue *val, Type *to) {
  Type *fromtype = val->type->toBasetype();
  Type *totype = to->toBasetype();

  if (fromtype->ty == Taarray) {
    // DMD allows casting AAs to void*, even if they are internally
    // implemented as structs.
    if (totype->ty == Tpointer) {
      IF_LOG Logger::println("Casting AA to pointer.");
      LLValue *rval = DtoBitCast(DtoRVal(val), DtoType(to));
      return new DImValue(to, rval);
    }
    if (totype->ty == Tbool) {
      IF_LOG Logger::println("Casting AA to bool.");
      LLValue *rval = DtoRVal(val);
      LLValue *zero = LLConstant::getNullValue(rval->getType());
      return new DImValue(to, gIR->ir->CreateICmpNE(rval, zero));
    }
  }

  if (fromtype->equals(totype)) {
    return val;
  }

  IF_LOG Logger::println("Casting from '%s' to '%s'", fromtype->toChars(),
                         to->toChars());
  LOG_SCOPE;

  if (fromtype->ty == Tvector) {
    // First, handle vector types (which can also be isintegral()).
    return DtoCastVector(loc, val, to);
  }
  if (fromtype->isintegral()) {
    return DtoCastInt(loc, val, to);
  }
  if (fromtype->iscomplex()) {
    return DtoCastComplex(loc, val, to);
  }
  if (fromtype->isfloating()) {
    return DtoCastFloat(loc, val, to);
  }

  switch (fromtype->ty) {
  case Tclass:
    return DtoCastClass(loc, val, to);
  case Tarray:
  case Tsarray:
    return DtoCastArray(loc, val, to);
  case Tpointer:
  case Tfunction:
    return DtoCastPtr(loc, val, to);
  case Tdelegate:
    return DtoCastDelegate(loc, val, to);
  case Tstruct:
    return DtoCastStruct(loc, val, to);
  case Tnull:
    return DtoNullValue(to, loc);
  case Taarray:
    if (totype->ty == Taarray) {
      // Do nothing, the types will match up anyway.
      return new DImValue(to, DtoRVal(val));
    }
  // fall-through
  default:
    error(loc, "invalid cast from `%s` to `%s`", val->type->toChars(),
          to->toChars());
    fatal();
  }
}

////////////////////////////////////////////////////////////////////////////////

DValue *DtoPaintType(Loc &loc, DValue *val, Type *to) {
  Type *from = val->type->toBasetype();
  IF_LOG Logger::println("repainting from '%s' to '%s'", from->toChars(),
                         to->toChars());

  if (from->ty == Tarray) {
    Type *at = to->toBasetype();
    assert(at->ty == Tarray);
    Type *elem = at->nextOf()->pointerTo();
    if (DSliceValue *slice = val->isSlice()) {
      return new DSliceValue(to, slice->getLength(),
                             DtoBitCast(slice->getPtr(), DtoType(elem)));
    }
    if (val->isLVal()) {
      LLValue *ptr = DtoLVal(val);
      ptr = DtoBitCast(ptr, DtoType(at->pointerTo()));
      return new DLValue(to, ptr);
    }
    LLValue *len, *ptr;
    len = DtoArrayLen(val);
    ptr = DtoArrayPtr(val);
    ptr = DtoBitCast(ptr, DtoType(elem));
    return new DImValue(to, DtoAggrPair(len, ptr));
  }
  if (from->ty == Tdelegate) {
    Type *dgty = to->toBasetype();
    assert(dgty->ty == Tdelegate);
    if (val->isLVal()) {
      LLValue *ptr = DtoLVal(val);
      assert(isaPointer(ptr));
      ptr = DtoBitCast(ptr, DtoPtrToType(dgty));
      IF_LOG Logger::cout() << "dg ptr: " << *ptr << '\n';
      return new DLValue(to, ptr);
    }
    LLValue *dg = DtoRVal(val);
    LLValue *context = gIR->ir->CreateExtractValue(dg, 0, ".context");
    LLValue *funcptr = gIR->ir->CreateExtractValue(dg, 1, ".funcptr");
    funcptr = DtoBitCast(funcptr, DtoType(dgty)->getContainedType(1));
    LLValue *aggr = DtoAggrPair(context, funcptr);
    IF_LOG Logger::cout() << "dg: " << *aggr << '\n';
    return new DImValue(to, aggr);
  }
  if (from->ty == Tpointer || from->ty == Tclass || from->ty == Taarray) {
    Type *b = to->toBasetype();
    assert(b->ty == Tpointer || b->ty == Tclass || b->ty == Taarray);
    LLValue *ptr = DtoBitCast(DtoRVal(val), DtoType(b));
    return new DImValue(to, ptr);
  }
  if (from->ty == Tsarray) {
    assert(to->toBasetype()->ty == Tsarray);
    LLValue *ptr = DtoBitCast(DtoLVal(val), DtoPtrToType(to));
    return new DLValue(to, ptr);
  }
  assert(DtoType(from) == DtoType(to));
  return new DImValue(to, DtoRVal(val));
}

/******************************************************************************
 * TEMPLATE HELPERS
 ******************************************************************************/

TemplateInstance *DtoIsTemplateInstance(Dsymbol *s) {
  if (!s) {
    return nullptr;
  }
  if (s->isTemplateInstance() && !s->isTemplateMixin()) {
    return s->isTemplateInstance();
  }
  if (s->parent) {
    return DtoIsTemplateInstance(s->parent);
  }
  return nullptr;
}

/******************************************************************************
 * PROCESSING QUEUE HELPERS
 ******************************************************************************/

void DtoResolveDsymbol(Dsymbol *dsym) {
  if (StructDeclaration *sd = dsym->isStructDeclaration()) {
    DtoResolveStruct(sd);
  } else if (ClassDeclaration *cd = dsym->isClassDeclaration()) {
    DtoResolveClass(cd);
  } else if (FuncDeclaration *fd = dsym->isFuncDeclaration()) {
    DtoResolveFunction(fd);
  } else if (TypeInfoDeclaration *tid = dsym->isTypeInfoDeclaration()) {
    DtoResolveTypeInfo(tid);
  } else if (VarDeclaration *vd = dsym->isVarDeclaration()) {
    DtoResolveVariable(vd);
  }
}

void DtoResolveVariable(VarDeclaration *vd) {
  if (vd->isTypeInfoDeclaration()) {
    return DtoResolveTypeInfo(static_cast<TypeInfoDeclaration *>(vd));
  }

  IF_LOG Logger::println("DtoResolveVariable(%s)", vd->toPrettyChars());
  LOG_SCOPE;

  // just forward aliases
  // TODO: Is this required here or is the check in VarDeclaration::codegen
  // sufficient?
  if (vd->aliassym) {
    Logger::println("alias sym");
    DtoResolveDsymbol(vd->aliassym);
    return;
  }

  if (AggregateDeclaration *ad = vd->isMember()) {
    DtoResolveDsymbol(ad);
  }

  // global variable
  if (vd->isDataseg()) {
    Logger::println("data segment");

    assert(!(vd->storage_class & STCmanifest) &&
           "manifest constant being codegen'd!");

    // don't duplicate work
    if (vd->ir->isResolved()) {
      return;
    }
    vd->ir->setDeclared();

    getIrGlobal(vd, true);

    IF_LOG {
      if (vd->parent) {
        Logger::println("parent: %s (%s)", vd->parent->toChars(),
                        vd->parent->kind());
      } else {
        Logger::println("parent: null");
      }
    }

    // If a const/immutable value has a proper initializer (not "= void"),
    // it cannot be assigned again in a static constructor. Thus, we can
    // emit it as read-only data.
    const bool isLLConst = (vd->isConst() || vd->isImmutable()) && vd->_init &&
                           !vd->_init->isVoidInitializer();

    assert(!vd->ir->isInitialized());
    if (gIR->dmodule) {
      vd->ir->setInitialized();
    }
    const auto irMangle = getIRMangledName(vd);

    // Since the type of a global must exactly match the type of its
    // initializer, we cannot know the type until after we have emitted the
    // latter (e.g. in case of unions, …). However, it is legal for the
    // initializer to refer to the address of the variable. Thus, we first
    // create a global with the generic type (note the assignment to
    // vd->ir->irGlobal->value!), and in case we also do an initializer
    // with a different type later, swap it out and replace any existing
    // uses with bitcasts to the previous type.

    // We always start out with external linkage; any other type is set
    // when actually defining it in VarDeclaration::codegen.
    llvm::GlobalValue::LinkageTypes linkage =
        llvm::GlobalValue::ExternalLinkage;
    if (vd->llvmInternal == LLVMextern_weak) {
      linkage = llvm::GlobalValue::ExternalWeakLinkage;
    }

    llvm::GlobalVariable *gvar =
        getOrCreateGlobal(vd->loc, gIR->module, DtoMemType(vd->type), isLLConst,
                          linkage, nullptr, irMangle, vd->isThreadlocal());
    auto varIr = getIrGlobal(vd);
    varIr->value = gvar;

    // Set the alignment (it is important not to use type->alignsize because
    // VarDeclarations can have an align() attribute independent of the type
    // as well).
    gvar->setAlignment(DtoAlignment(vd));

    // Windows: initialize DLL storage class with `dllimport` for `export`ed
    // symbols
    if (global.params.isWindows && vd->isExport()) {
      gvar->setDLLStorageClass(LLGlobalValue::DLLImportStorageClass);
    }

    applyVarDeclUDAs(vd, gvar);
    if (varIr->dynamicCompileConst) {
      addDynamicCompiledVar(gIR, varIr);
    }

    IF_LOG Logger::cout() << *gvar << '\n';
  }
}

/******************************************************************************
 * DECLARATION EXP HELPER
 ******************************************************************************/

// TODO: Merge with DtoRawVarDeclaration!
void DtoVarDeclaration(VarDeclaration *vd) {
  assert(!vd->isDataseg() &&
         "Statics/globals are handled in DtoDeclarationExp.");
  assert(!vd->aliassym && "Aliases are handled in DtoDeclarationExp.");

  IF_LOG Logger::println("DtoVarDeclaration(vdtype = %s)", vd->type->toChars());
  LOG_SCOPE

  if (vd->nestedrefs.dim) {
    IF_LOG Logger::println(
        "has nestedref set (referenced by nested function/delegate)");

    // A variable may not be really nested even if nextedrefs is not empty
    // in case it is referenced by a function inside __traits(compile) or
    // typeof.
    // assert(vd->ir->irLocal && "irLocal is expected to be already set by
    // DtoCreateNestedContext");
  }

  if (isIrLocalCreated(vd)) {
    // Nothing to do if it has already been allocated.
  } else if (gIR->func()->sretArg && ((gIR->func()->decl->nrvo_can &&
                                       gIR->func()->decl->nrvo_var == vd) ||
                                      vd->isResult())) {
    // Named Return Value Optimization (NRVO):
    // T f() {
    //   T ret;        // &ret == hidden pointer
    //   ret = ...
    //   return ret;    // NRVO.
    // }
    assert(!isSpecialRefVar(vd) && "Can this happen?");
    getIrLocal(vd, true)->value = gIR->func()->sretArg;
  } else {
    // normal stack variable, allocate storage on the stack if it has not
    // already been done
    IrLocal *irLocal = getIrLocal(vd, true);

    Type *type = isSpecialRefVar(vd) ? vd->type->pointerTo() : vd->type;

    llvm::Value *allocainst;
    LLType *lltype = DtoType(type);
    if (gDataLayout->getTypeSizeInBits(lltype) == 0) {
      allocainst = llvm::ConstantPointerNull::get(getPtrToType(lltype));
    } else if (type != vd->type) {
      allocainst = DtoAlloca(type, vd->toChars());
    } else {
      allocainst = DtoAlloca(vd, vd->toChars());
    }

    irLocal->value = allocainst;

    gIR->DBuilder.EmitLocalVariable(allocainst, vd);
  }

  IF_LOG Logger::cout() << "llvm value for decl: " << *getIrLocal(vd)->value
                        << '\n';

  if (vd->_init) {
    if (ExpInitializer *ex = vd->_init->isExpInitializer()) {
      // TODO: Refactor this so that it doesn't look like toElem has no effect.
      Logger::println("expression initializer");
      toElem(ex->exp);
    }
  }
}

DValue *DtoDeclarationExp(Dsymbol *declaration) {
  IF_LOG Logger::print("DtoDeclarationExp: %s\n", declaration->toChars());
  LOG_SCOPE;

  if (VarDeclaration *vd = declaration->isVarDeclaration()) {
    Logger::println("VarDeclaration");

    // if aliassym is set, this VarDecl is redone as an alias to another symbol
    // this seems to be done to rewrite Tuple!(...) v;
    // as a TupleDecl that contains a bunch of individual VarDecls
    if (vd->aliassym) {
      return DtoDeclarationExp(vd->aliassym);
    }

    if (vd->storage_class & STCmanifest) {
      IF_LOG Logger::println("Manifest constant, nothing to do.");
      return nullptr;
    }

    // static
    if (vd->isDataseg()) {
      Declaration_codegen(vd);
    } else {
      DtoVarDeclaration(vd);
    }
    return makeVarDValue(vd->type, vd);
  }

  if (StructDeclaration *s = declaration->isStructDeclaration()) {
    Logger::println("StructDeclaration");
    Declaration_codegen(s);
  } else if (FuncDeclaration *f = declaration->isFuncDeclaration()) {
    Logger::println("FuncDeclaration");
    Declaration_codegen(f);
  } else if (ClassDeclaration *e = declaration->isClassDeclaration()) {
    Logger::println("ClassDeclaration");
    Declaration_codegen(e);
  } else if (AttribDeclaration *a = declaration->isAttribDeclaration()) {
    Logger::println("AttribDeclaration");
    // choose the right set in case this is a conditional declaration
    if (auto d = a->include(nullptr)) {
      for (unsigned i = 0; i < d->dim; ++i) {
        DtoDeclarationExp((*d)[i]);
      }
    }
  } else if (TemplateMixin *m = declaration->isTemplateMixin()) {
    Logger::println("TemplateMixin");
    for (Dsymbol *mdsym : *m->members) {
      DtoDeclarationExp(mdsym);
    }
  } else if (TupleDeclaration *tupled = declaration->isTupleDeclaration()) {
    Logger::println("TupleDeclaration");
    assert(tupled->isexp && "Non-expression tuple decls not handled yet.");
    assert(tupled->objects);
    for (unsigned i = 0; i < tupled->objects->dim; ++i) {
      auto exp = static_cast<DsymbolExp *>((*tupled->objects)[i]);
      DtoDeclarationExp(exp->s);
    }
  } else {
    // Do nothing for template/alias/enum declarations and static
    // assertions. We cannot detect StaticAssert without RTTI, so don't
    // even bother to check.
    IF_LOG Logger::println("Ignoring Symbol: %s", declaration->kind());
  }

  return nullptr;
}

// does pretty much the same as DtoDeclarationExp, except it doesn't initialize,
// and only handles var declarations
LLValue *DtoRawVarDeclaration(VarDeclaration *var, LLValue *addr) {
  // we don't handle globals with this one
  assert(!var->isDataseg());

  // we don't handle aliases either
  assert(!var->aliassym);

  IrLocal *irLocal = isIrLocalCreated(var) ? getIrLocal(var) : nullptr;

  // alloca if necessary
  if (!addr && (!irLocal || !irLocal->value)) {
    addr = DtoAlloca(var, var->toChars());
    // add debug info
    if (!irLocal) {
      irLocal = getIrLocal(var, true);
    }
    gIR->DBuilder.EmitLocalVariable(addr, var);
  }

  // nested variable?
  // A variable may not be really nested even if nextedrefs is not empty
  // in case it is referenced by a function inside __traits(compile) or typeof.
  if (var->nestedrefs.dim && isIrLocalCreated(var)) {
    if (!irLocal->value) {
      assert(addr);
      irLocal->value = addr;
    } else {
      assert(!addr || addr == irLocal->value);
    }
  }
  // normal local variable
  else {
    // if this already has storage, it must've been handled already
    if (irLocal->value) {
      if (addr && addr != irLocal->value) {
        // This can happen, for example, in scope(exit) blocks which
        // are translated to IR multiple times.
        // That *should* only happen after the first one is completely done
        // though, so just set the address.
        IF_LOG {
          Logger::println("Replacing LLVM address of %s", var->toChars());
          LOG_SCOPE;
          Logger::cout() << "Old val: " << *irLocal->value << '\n';
          Logger::cout() << "New val: " << *addr << '\n';
        }
        irLocal->value = addr;
      }
      return addr;
    }

    assert(addr);
    irLocal->value = addr;
  }

  // return the alloca
  return irLocal->value;
}

/******************************************************************************
 * INITIALIZER HELPERS
 ******************************************************************************/

LLConstant *DtoConstInitializer(Loc &loc, Type *type, Initializer *init) {
  LLConstant *_init = nullptr; // may return zero
  if (!init) {
    IF_LOG Logger::println("const default initializer for %s", type->toChars());
    Expression *initExp = type->defaultInit();
    _init = DtoConstExpInit(loc, type, initExp);
  } else if (ExpInitializer *ex = init->isExpInitializer()) {
    Logger::println("const expression initializer");
    _init = DtoConstExpInit(loc, type, ex->exp);
  } else if (ArrayInitializer *ai = init->isArrayInitializer()) {
    Logger::println("const array initializer");
    _init = DtoConstArrayInitializer(ai, type);
  } else if (init->isVoidInitializer()) {
    Logger::println("const void initializer");
    LLType *ty = DtoMemType(type);
    _init = LLConstant::getNullValue(ty);
  } else {
    // StructInitializer is no longer suposed to make it to the glue layer
    // in DMD 2.064.
    IF_LOG Logger::println("unsupported const initializer: %s",
                           init->toChars());
  }
  return _init;
}

////////////////////////////////////////////////////////////////////////////////

LLConstant *DtoConstExpInit(Loc &loc, Type *targetType, Expression *exp) {
  IF_LOG Logger::println("DtoConstExpInit(targetType = %s, exp = %s)",
                         targetType->toChars(), exp->toChars());
  LOG_SCOPE

  LLConstant *val = toConstElem(exp, gIR);
  Type *baseValType = exp->type->toBasetype();
  Type *baseTargetType = targetType->toBasetype();

  // The situation here is a bit tricky: In an ideal world, we would always
  // have val->getType() == DtoType(targetType). But there are two reasons
  // why this is not true. One is that the LLVM type system cannot represent
  // all the C types, leading to differences in types being necessary e.g. for
  // union initializers. The second is that the frontend actually does not
  // explicitly lower things like initializing an array/vector with a scalar
  // constant, or since 2.061 sometimes does not get implicit conversions for
  // integers right. However, we cannot just rely on the actual Types being
  // equal if there are no rewrites to do because of – as usual – AST
  // inconsistency bugs.

  LLType *llType = val->getType();
  LLType *targetLLType = DtoMemType(baseTargetType);

  // shortcut for zeros
  if (val->isNullValue())
    return llvm::Constant::getNullValue(targetLLType);

  // extend i1 to i8
  if (llType == LLType::getInt1Ty(gIR->context())) {
    llType = LLType::getInt8Ty(gIR->context());
    val = llvm::ConstantExpr::getZExt(val, llType);
  }

  if (llType == targetLLType)
    return val;

  if (baseTargetType->ty == Tsarray) {
    Logger::println("Building constant array initializer from scalar.");

    assert(baseValType->size() > 0);
    const auto numTotalVals = baseTargetType->size() / baseValType->size();
    assert(baseTargetType->size() % baseValType->size() == 0);

    // may be a multi-dimensional array init, e.g., `char[2][3] x = 0xff`
    baseValType = stripModifiers(baseValType);
    LLSmallVector<size_t, 4> dims; // { 3, 2 }
    for (auto t = baseTargetType; t->ty == Tsarray;) {
      dims.push_back(static_cast<TypeSArray *>(t)->dim->toUInteger());
      auto elementType = stripModifiers(t->nextOf()->toBasetype());
      if (elementType->equals(baseValType))
        break;
      t = elementType;
    }

    size_t product = 1;
    for (size_t i = dims.size(); i--;) {
      product *= dims[i];
      auto at = llvm::ArrayType::get(val->getType(), dims[i]);
      LLSmallVector<llvm::Constant *, 16> elements(dims[i], val);
      val = llvm::ConstantArray::get(at, elements);
    }

    (void)numTotalVals;
    assert(product == numTotalVals);
    return val;
  }

  if (baseTargetType->ty == Tvector) {
    Logger::println("Building constant vector initializer from scalar.");

    TypeVector *tv = static_cast<TypeVector *>(baseTargetType);
    assert(tv->basetype->ty == Tsarray);
    dinteger_t elemCount =
        static_cast<TypeSArray *>(tv->basetype)->dim->toInteger();
    return llvm::ConstantVector::getSplat(elemCount, val);
  }

  if (llType->isIntegerTy() && targetLLType->isIntegerTy()) {
    // This should really be fixed in the frontend.
    Logger::println("Fixing up unresolved implicit integer conversion.");

    llvm::IntegerType *source = llvm::cast<llvm::IntegerType>(llType);
    llvm::IntegerType *target = llvm::cast<llvm::IntegerType>(targetLLType);

    (void)source;
    assert(target->getBitWidth() > source->getBitWidth() &&
           "On initializer integer type mismatch, the target should be wider "
           "than the source.");

    return llvm::ConstantExpr::getZExtOrBitCast(val, target);
  }

  Logger::println("Unhandled type mismatch, giving up.");
  return val;
}

////////////////////////////////////////////////////////////////////////////////

LLConstant *DtoTypeInfoOf(Type *type, bool base) {
  IF_LOG Logger::println("DtoTypeInfoOf(type = '%s', base='%d')",
                         type->toChars(), base);
  LOG_SCOPE

  TypeInfoDeclaration *tidecl = getOrCreateTypeInfoDeclaration(type, nullptr);
  assert(tidecl);
  Declaration_codegen(tidecl);
  assert(getIrGlobal(tidecl)->value != NULL);
  LLConstant *c = isaConstant(getIrGlobal(tidecl)->value);
  assert(c != NULL);
  if (base) {
    return llvm::ConstantExpr::getBitCast(c, DtoType(Type::dtypeinfo->type));
  }
  return c;
}

////////////////////////////////////////////////////////////////////////////////
/// Allocates memory and passes on ownership. (never returns null)
static char *DtoOverloadedIntrinsicName(TemplateInstance *ti,
                                        TemplateDeclaration *td) {
  IF_LOG Logger::println("DtoOverloadedIntrinsicName");
  LOG_SCOPE;

  IF_LOG {
    Logger::println("template instance: %s", ti->toChars());
    Logger::println("template declaration: %s", td->toChars());
    Logger::println("intrinsic name: %s", td->intrinsicName);
  }

  // for now use the size in bits of the first template param in the instance
  assert(ti->tdtypes.dim == 1);
  Type *T = static_cast<Type *>(ti->tdtypes.data[0]);

  char prefix = T->isreal() ? 'f' : T->isintegral() ? 'i' : 0;
  if (!prefix) {
    ti->error("has invalid template parameter for intrinsic: `%s`",
              T->toChars());
    fatal(); // or LLVM asserts
  }

  llvm::Type *dtype(DtoType(T));
  char tmp[21]; // probably excessive, but covers a uint64_t
  sprintf(tmp, "%lu",
          static_cast<unsigned long>(gDataLayout->getTypeSizeInBits(dtype)));

  // replace # in name with bitsize
  std::string name(td->intrinsicName);

  std::string needle("#");
  size_t pos;
  while (std::string::npos != (pos = name.find(needle))) {
    if (pos > 0 && name[pos - 1] == prefix) {
      // Check for special PPC128 double
      if (dtype->isPPC_FP128Ty()) {
        name.insert(pos - 1, "ppc");
        pos += 3;
      }
      // Properly prefixed, insert bitwidth.
      name.replace(pos, 1, tmp);
    } else {
      if (pos && (name[pos - 1] == 'i' || name[pos - 1] == 'f')) {
        // Wrong type character.
        ti->error("has invalid parameter type for intrinsic `%s`: `%s` is not "
                  "a%s type",
                  name.c_str(), T->toChars(),
                  (name[pos - 1] == 'i' ? "n integral" : " floating-point"));
      } else {
        // Just plain wrong. (Error in declaration, not instantiation)
        td->error("has an invalid intrinsic name: `%s`", name.c_str());
      }
      fatal(); // or LLVM asserts
    }
  }

  IF_LOG Logger::println("final intrinsic name: %s", name.c_str());

  return strdup(name.c_str());
}

/// For D frontend
/// Fixup an overloaded intrinsic name string.
void DtoSetFuncDeclIntrinsicName(TemplateInstance *ti, TemplateDeclaration *td,
                                 FuncDeclaration *fd) {
  if (fd->llvmInternal == LLVMintrinsic) {
    fd->intrinsicName = DtoOverloadedIntrinsicName(ti, td);
    fd->mangleOverride =
        fd->intrinsicName ? strdup(fd->intrinsicName) : nullptr;
  } else {
    fd->intrinsicName = td->intrinsicName ? strdup(td->intrinsicName) : nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////

size_t getMemberSize(Type *type) {
  const dinteger_t dSize = type->size();
  llvm::Type *const llType = DtoType(type);
  if (!llType->isSized()) {
    // Forward reference in a cycle or similar, we need to trust the D type.
    return dSize;
  }

  const uint64_t llSize = gDataLayout->getTypeAllocSize(llType);
  assert(llSize <= dSize &&
         "LLVM type is bigger than the corresponding D type, "
         "might lead to aggregate layout mismatch.");

  return llSize;
}

////////////////////////////////////////////////////////////////////////////////

Type *stripModifiers(Type *type, bool transitive) {
  if (type->ty == Tfunction) {
    return type;
  }

  if (transitive) {
    return type->unqualify(MODimmutable | MODconst | MODwild);
  }
  return type->castMod(0);
}

////////////////////////////////////////////////////////////////////////////////

LLValue *makeLValue(Loc &loc, DValue *value) {
  if (value->isLVal())
    return DtoLVal(value);

  if (value->isIm() || value->isConst())
    return DtoAllocaDump(value, ".makelvaluetmp");

  LLValue *mem = DtoAlloca(value->type, ".makelvaluetmp");
  DLValue var(value->type, mem);
  DtoAssign(loc, &var, value, TOKblit);
  return mem;
}

////////////////////////////////////////////////////////////////////////////////

void callPostblit(Loc &loc, Expression *exp, LLValue *val) {
  Type *tb = exp->type->toBasetype();
  if ((exp->op == TOKvar || exp->op == TOKdotvar || exp->op == TOKstar ||
       exp->op == TOKthis || exp->op == TOKindex) &&
      tb->ty == Tstruct) {
    StructDeclaration *sd = static_cast<TypeStruct *>(tb)->sym;
    if (sd->postblit) {
      FuncDeclaration *fd = sd->postblit;
      if (fd->storage_class & STCdisable) {
        fd->toParent()->error(
            loc, "is not copyable because it is annotated with `@disable`");
      }
      DtoResolveFunction(fd);
      Expressions args;
      DFuncValue dfn(fd, DtoCallee(fd), val);
      DtoCallFunction(loc, Type::basic[Tvoid], &dfn, &args);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

bool isSpecialRefVar(VarDeclaration *vd) {
  return (vd->storage_class & (STCref | STCparameter)) == STCref;
}

////////////////////////////////////////////////////////////////////////////////

bool isLLVMUnsigned(Type *t) { return t->isunsigned() || t->ty == Tpointer; }

////////////////////////////////////////////////////////////////////////////////

void printLabelName(std::ostream &target, const char *func_mangle,
                    const char *label_name) {
  target << gTargetMachine->getMCAsmInfo()
                ->getPrivateGlobalPrefix()
#if LDC_LLVM_VER >= 400
                .str()
#endif
         << func_mangle << "_" << label_name;
}

////////////////////////////////////////////////////////////////////////////////

void AppendFunctionToLLVMGlobalCtorsDtors(llvm::Function *func,
                                          const uint32_t priority,
                                          const bool isCtor) {
  if (isCtor) {
    llvm::appendToGlobalCtors(gIR->module, func, priority);
  } else {
    llvm::appendToGlobalDtors(gIR->module, func, priority);
  }
}

////////////////////////////////////////////////////////////////////////////////

void tokToICmpPred(TOK op, bool isUnsigned, llvm::ICmpInst::Predicate *outPred,
                   llvm::Value **outConst) {
  switch (op) {
  case TOKlt:
  case TOKul:
    *outPred = isUnsigned ? llvm::ICmpInst::ICMP_ULT : llvm::ICmpInst::ICMP_SLT;
    break;
  case TOKle:
  case TOKule:
    *outPred = isUnsigned ? llvm::ICmpInst::ICMP_ULE : llvm::ICmpInst::ICMP_SLE;
    break;
  case TOKgt:
  case TOKug:
    *outPred = isUnsigned ? llvm::ICmpInst::ICMP_UGT : llvm::ICmpInst::ICMP_SGT;
    break;
  case TOKge:
  case TOKuge:
    *outPred = isUnsigned ? llvm::ICmpInst::ICMP_UGE : llvm::ICmpInst::ICMP_SGE;
    break;
  case TOKue:
    *outPred = llvm::ICmpInst::ICMP_EQ;
    break;
  case TOKlg:
    *outPred = llvm::ICmpInst::ICMP_NE;
    break;
  case TOKleg:
    *outConst = LLConstantInt::getTrue(gIR->context());
    break;
  case TOKunord:
    *outConst = LLConstantInt::getFalse(gIR->context());
    break;
  default:
    llvm_unreachable("Invalid comparison operation");
  }
}

////////////////////////////////////////////////////////////////////////////////

llvm::ICmpInst::Predicate eqTokToICmpPred(TOK op, bool invert) {
  assert(op == TOKequal || op == TOKnotequal || op == TOKidentity ||
         op == TOKnotidentity);

  bool isEquality = (op == TOKequal || op == TOKidentity);
  if (invert)
    isEquality = !isEquality;

  return (isEquality ? llvm::ICmpInst::ICMP_EQ : llvm::ICmpInst::ICMP_NE);
}

////////////////////////////////////////////////////////////////////////////////

LLValue *createIPairCmp(TOK op, LLValue *lhs1, LLValue *lhs2, LLValue *rhs1,
                        LLValue *rhs2) {
  const auto predicate = eqTokToICmpPred(op);

  LLValue *r1 = gIR->ir->CreateICmp(predicate, lhs1, rhs1);
  LLValue *r2 = gIR->ir->CreateICmp(predicate, lhs2, rhs2);

  LLValue *r =
      (predicate == llvm::ICmpInst::ICMP_EQ ? gIR->ir->CreateAnd(r1, r2)
                                            : gIR->ir->CreateOr(r1, r2));

  return r;
}

///////////////////////////////////////////////////////////////////////////////
DValue *DtoSymbolAddress(Loc &loc, Type *type, Declaration *decl) {
  IF_LOG Logger::println("DtoSymbolAddress ('%s' of type '%s')",
                         decl->toChars(), decl->type->toChars());
  LOG_SCOPE

  if (VarDeclaration *vd = decl->isVarDeclaration()) {
    // The magic variable __ctfe is always false at runtime
    if (vd->ident == Id::ctfe) {
      return new DConstValue(type, DtoConstBool(false));
    }

    // this is an error! must be accessed with DotVarExp
    if (vd->needThis()) {
      error(loc, "need `this` to access member `%s`", vd->toChars());
      fatal();
    }

    // _arguments
    if (vd->ident == Id::_arguments && gIR->func()->_arguments) {
      Logger::println("Id::_arguments");
      LLValue *v = gIR->func()->_arguments;
      assert(!isSpecialRefVar(vd) && "Code not expected to handle special ref "
                                     "vars, although it can easily be made "
                                     "to.");
      return new DLValue(type, v);
    }
    // _argptr
    if (vd->ident == Id::_argptr && gIR->func()->_argptr) {
      Logger::println("Id::_argptr");
      LLValue *v = gIR->func()->_argptr;
      assert(!isSpecialRefVar(vd) && "Code not expected to handle special ref "
                                     "vars, although it can easily be made "
                                     "to.");
      return new DLValue(type, v);
    }
    // _dollar
    if (vd->ident == Id::dollar) {
      Logger::println("Id::dollar");
      if (isIrVarCreated(vd)) {
        // This is the length of a range.
        return makeVarDValue(type, vd);
      }
      assert(!gIR->arrays.empty());
      return new DImValue(type, DtoArrayLen(gIR->arrays.back()));
    }
    // typeinfo
    if (TypeInfoDeclaration *tid = vd->isTypeInfoDeclaration()) {
      Logger::println("TypeInfoDeclaration");
      DtoResolveTypeInfo(tid);
      assert(getIrValue(tid));
      LLType *vartype = DtoType(type);
      LLValue *m = getIrValue(tid);
      if (m->getType() != getPtrToType(vartype)) {
        m = gIR->ir->CreateBitCast(m, vartype);
      }
      return new DImValue(type, m);
    }
    // nested variable
    if (vd->nestedrefs.dim) {
      Logger::println("nested variable");
      return DtoNestedVariable(loc, type, vd);
    }
    // function parameter
    if (vd->isParameter()) {
      IF_LOG {
        Logger::println("function param");
        Logger::println("type: %s", vd->type->toChars());
      }
      FuncDeclaration *fd = vd->toParent2()->isFuncDeclaration();
      if (fd && fd != gIR->func()->decl) {
        Logger::println("nested parameter");
        return DtoNestedVariable(loc, type, vd);
      }
      if (vd->storage_class & STClazy) {
        Logger::println("lazy parameter");
        assert(type->ty == Tdelegate);
      }
      assert(!isSpecialRefVar(vd) && "Code not expected to handle special "
                                     "ref vars, although it can easily be "
                                     "made to.");
      return new DLValue(type, DtoBitCast(getIrValue(vd), DtoPtrToType(type)));
    }
    Logger::println("a normal variable");

    // take care of forward references of global variables
    if (vd->isDataseg() || (vd->storage_class & STCextern)) {
      DtoResolveVariable(vd);
    }

    return makeVarDValue(type, vd);
  }

  if (FuncLiteralDeclaration *flitdecl = decl->isFuncLiteralDeclaration()) {
    Logger::println("FuncLiteralDeclaration");

    // We need to codegen the function here, because literals are not added
    // to the module member list.
    DtoDefineFunction(flitdecl);
    assert(DtoCallee(flitdecl));

    return new DFuncValue(flitdecl, DtoCallee(flitdecl));
  }

  if (FuncDeclaration *fdecl = decl->isFuncDeclaration()) {
    Logger::println("FuncDeclaration");
    fdecl = fdecl->toAliasFunc();
    if (fdecl->llvmInternal == LLVMinline_asm) {
      // TODO: Is this needed? If so, what about other intrinsics?
      error(loc, "special ldc inline asm is not a normal function");
      fatal();
    } else if (fdecl->llvmInternal == LLVMinline_ir) {
      // TODO: Is this needed? If so, what about other intrinsics?
      error(loc, "special ldc inline ir is not a normal function");
      fatal();
    }
    DtoResolveFunction(fdecl);
    const auto llValue =
        fdecl->llvmInternal != LLVMva_arg ? DtoCallee(fdecl) : nullptr;
    return new DFuncValue(fdecl, llValue);
  }

  if (SymbolDeclaration *sdecl = decl->isSymbolDeclaration()) {
    // this seems to be the static initialiser for structs
    Type *sdecltype = sdecl->type->toBasetype();
    IF_LOG Logger::print("Sym: type=%s\n", sdecltype->toChars());
    assert(sdecltype->ty == Tstruct);
    TypeStruct *ts = static_cast<TypeStruct *>(sdecltype);
    assert(ts->sym);
    DtoResolveStruct(ts->sym);

    LLValue *initsym = getIrAggr(ts->sym)->getInitSymbol();
    initsym = DtoBitCast(initsym, DtoType(ts->pointerTo()));
    return new DLValue(type, initsym);
  }

  llvm_unreachable("Unimplemented VarExp type");
}

llvm::Constant *DtoConstSymbolAddress(Loc &loc, Declaration *decl) {
  // Make sure 'this' isn't needed.
  // TODO: This check really does not belong here, should be moved to
  // semantic analysis in the frontend.
  if (decl->needThis()) {
    error(loc, "need `this` to access `%s`", decl->toChars());
    fatal();
  }

  // global variable
  if (VarDeclaration *vd = decl->isVarDeclaration()) {
    if (!vd->isDataseg()) {
      // Not sure if this can be triggered from user code, but it is
      // needed for the current hacky implementation of
      // AssocArrayLiteralExp::toElem, which requires on error
      // gagging to check for constantness of the initializer.
      error(loc,
            "cannot use address of non-global variable `%s` as constant "
            "initializer",
            vd->toChars());
      if (!global.gag) {
        fatal();
      }
      return nullptr;
    }

    DtoResolveVariable(vd);
    LLConstant *llc = llvm::dyn_cast<LLConstant>(getIrValue(vd));
    assert(llc);
    return llc;
  }
  // static function
  if (FuncDeclaration *fd = decl->isFuncDeclaration()) {
    DtoResolveFunction(fd);
    return DtoCallee(fd);
  }

  llvm_unreachable("Taking constant address not implemented.");
}

llvm::StringMap<llvm::GlobalVariable *> *
stringLiteralCacheForType(Type *charType) {
  switch (charType->size()) {
  default:
    llvm_unreachable("Unknown char type");
  case 1:
    return &gIR->stringLiteral1ByteCache;
  case 2:
    return &gIR->stringLiteral2ByteCache;
  case 4:
    return &gIR->stringLiteral4ByteCache;
  }
}

llvm::Constant *buildStringLiteralConstant(StringExp *se, bool zeroTerm) {
  Type *dtype = se->type->toBasetype();
  Type *cty = dtype->nextOf()->toBasetype();

  LLType *ct = DtoMemType(cty);
  auto len = se->numberOfCodeUnits();
  if (zeroTerm) {
    len += 1;
  }
  LLArrayType *at = LLArrayType::get(ct, len);

  std::vector<LLConstant *> vals;
  vals.reserve(len);
  for (size_t i = 0; i < se->numberOfCodeUnits(); ++i) {
    vals.push_back(LLConstantInt::get(ct, se->charAt(i), false));
  }
  if (zeroTerm) {
    vals.push_back(LLConstantInt::get(ct, 0, false));
  }
  return LLConstantArray::get(at, vals);
}

llvm::GlobalVariable *getOrCreateGlobal(const Loc &loc, llvm::Module &module,
                                        llvm::Type *type, bool isConstant,
                                        llvm::GlobalValue::LinkageTypes linkage,
                                        llvm::Constant *init,
                                        llvm::StringRef name,
                                        bool isThreadLocal) {
  llvm::GlobalVariable *existing = module.getGlobalVariable(name, true);
  if (existing) {
    if (existing->getType()->getElementType() != type) {
      error(loc,
            "Global variable type does not match previous declaration with "
            "same mangled name: `%s`",
            name.str().c_str());
      fatal();
    }
    return existing;
  }

  // Use a command line option for the thread model.
  // On PPC there is only local-exec available - in this case just ignore the
  // command line.
  const llvm::GlobalVariable::ThreadLocalMode tlsModel =
      isThreadLocal
          ? (global.params.targetTriple->getArch() == llvm::Triple::ppc
                 ? llvm::GlobalVariable::LocalExecTLSModel
                 : clThreadModel.getValue())
          : llvm::GlobalVariable::NotThreadLocal;
  return new llvm::GlobalVariable(module, type, isConstant, linkage, init, name,
                                  nullptr, tlsModel);
}

FuncDeclaration *getParentFunc(Dsymbol *sym) {
  if (!sym) {
    return nullptr;
  }

  // Static functions and function (not delegate) literals don't allow
  // access to a parent context, even if they are nested.
  if (FuncDeclaration *fd = sym->isFuncDeclaration()) {
    bool certainlyNewRoot =
        fd->isStatic() ||
        (fd->isFuncLiteralDeclaration() &&
         static_cast<FuncLiteralDeclaration *>(fd)->tok == TOKfunction);
    if (certainlyNewRoot) {
      return nullptr;
    }
  }
  // Fun fact: AggregateDeclarations are not Declarations.
  else if (AggregateDeclaration *ad = sym->isAggregateDeclaration()) {
    if (!ad->isNested()) {
      return nullptr;
    }
  }

  for (Dsymbol *parent = sym->parent; parent; parent = parent->parent) {
    if (FuncDeclaration *fd = parent->isFuncDeclaration()) {
      return fd;
    }

    if (AggregateDeclaration *ad = parent->isAggregateDeclaration()) {
      if (!ad->isNested()) {
        return nullptr;
      }
    }
  }

  return nullptr;
}

LLValue *DtoIndexAggregate(LLValue *src, AggregateDeclaration *ad,
                           VarDeclaration *vd) {
  IF_LOG Logger::println("Indexing aggregate field %s:", vd->toPrettyChars());
  LOG_SCOPE;

  // Make sure the aggregate is resolved, as subsequent code might expect
  // isIrVarCreated(vd). This is a bit of a hack, we don't actually need this
  // ourselves, DtoType below would be enough.
  DtoResolveDsymbol(ad);

  // Cast the pointer we got to the canonical struct type the indices are
  // based on.
  LLType *st = DtoType(ad->type);
  if (ad->isStructDeclaration()) {
    st = getPtrToType(st);
  }
  src = DtoBitCast(src, st);

  // Look up field to index and any offset to apply.
  unsigned fieldIndex;
  unsigned byteOffset;
  assert(ad->type->ctype->isAggr());
  static_cast<IrTypeAggr *>(ad->type->ctype)
      ->getMemberLocation(vd, fieldIndex, byteOffset);

  LLValue *val = DtoGEPi(src, 0, fieldIndex);

  if (byteOffset) {
    // Cast to void* to apply byte-wise offset.
    val = DtoBitCast(val, getVoidPtrType());
    val = DtoGEPi1(val, byteOffset);
  }

  // Cast the (possibly void*) pointer to the canonical variable type.
  val = DtoBitCast(val, DtoPtrToType(vd->type));

  IF_LOG Logger::cout() << "Value: " << *val << '\n';
  return val;
}

unsigned getFieldGEPIndex(AggregateDeclaration *ad, VarDeclaration *vd) {
  unsigned fieldIndex;
  unsigned byteOffset;
  assert(ad->type->ctype->isAggr());
  static_cast<IrTypeAggr *>(ad->type->ctype)
      ->getMemberLocation(vd, fieldIndex, byteOffset);
  assert(byteOffset == 0 && "Cannot address field by a simple GEP.");
  return fieldIndex;
}

DValue *makeVarDValue(Type *type, VarDeclaration *vd, llvm::Value *storage) {
  auto val = storage;
  if (!val) {
    assert(isIrVarCreated(vd) && "Variable not resolved.");
    val = getIrValue(vd);
  }

  // We might need to cast.
  llvm::Type *expectedType = DtoPtrToType(type);
  const bool isSpecialRef = isSpecialRefVar(vd);
  if (isSpecialRef)
    expectedType = expectedType->getPointerTo();

  if (val->getType() != expectedType) {
    // The type of globals is determined by their initializer, and the front-end
    // may inject implicit casts for class references and static arrays.
    assert(vd->isDataseg() || (vd->storage_class & STCextern) ||
           type->toBasetype()->ty == Tclass ||
           type->toBasetype()->ty == Tsarray);
    llvm::Type *pointeeType = val->getType()->getPointerElementType();
    if (isSpecialRef)
      pointeeType = pointeeType->getPointerElementType();
    // Make sure that the type sizes fit - '==' instead of '<=' should probably
    // work as well.
    assert(getTypeStoreSize(DtoType(type)) <= getTypeStoreSize(pointeeType) &&
           "LValue type mismatch, encountered type too small.");
    val = DtoBitCast(val, expectedType);
  }

  if (isSpecialRef)
    return new DSpecialRefValue(type, val);

  return new DLValue(type, val);
}
