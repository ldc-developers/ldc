//===-- llvmhelpers.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvmhelpers.h"

#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/init.h"
#include "dmd/module.h"
#include "dmd/template.h"
#include "gen/abi/abi.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/complex.h"
#include "gen/dvalue.h"
#include "gen/dynamiccompile.h"
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
#include "ir/irdsymbol.h"
#include "ir/irfunction.h"
#include "ir/irmodule.h"
#include "ir/irtypeaggr.h"
#include "ir/irtypeclass.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <llvm/IR/Constant.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <stack>

using namespace dmd;

llvm::cl::opt<llvm::GlobalVariable::ThreadLocalMode> clThreadModel(
    "fthread-model", llvm::cl::ZeroOrMore, llvm::cl::desc("Thread model"),
    llvm::cl::init(llvm::GlobalVariable::GeneralDynamicTLSModel),
    llvm::cl::values(clEnumValN(llvm::GlobalVariable::GeneralDynamicTLSModel,
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

LLValue *DtoNew(const Loc &loc, Type *newtype) {
  // get runtime function
  llvm::Function *fn = getRuntimeFunction(loc, gIR->module, "_d_allocmemoryT");
  // get type info
  LLConstant *ti = DtoTypeInfoOf(loc, newtype);
  assert(isaPointer(ti));
  // call runtime allocator
  return gIR->CreateCallOrInvoke(fn, ti, ".gc_mem");
}

void DtoDeleteMemory(const Loc &loc, DValue *ptr) {
  llvm::Function *fn = getRuntimeFunction(loc, gIR->module, "_d_delmemory");
  LLValue *lval = (ptr->isLVal() ? DtoLVal(ptr) : makeLValue(loc, ptr));
  gIR->CreateCallOrInvoke(fn, lval);
}

void DtoDeleteStruct(const Loc &loc, DValue *ptr) {
  llvm::Function *fn = getRuntimeFunction(loc, gIR->module, "_d_delstruct");
  LLValue *lval = (ptr->isLVal() ? DtoLVal(ptr) : makeLValue(loc, ptr));
  gIR->CreateCallOrInvoke(fn, lval, DtoTypeInfoOf(loc, ptr->type->nextOf()));
}

void DtoDeleteClass(const Loc &loc, DValue *inst) {
  llvm::Function *fn = getRuntimeFunction(loc, gIR->module, "_d_delclass");
  LLValue *lval = (inst->isLVal() ? DtoLVal(inst) : makeLValue(loc, inst));
  gIR->CreateCallOrInvoke(fn, lval);
}

void DtoDeleteInterface(const Loc &loc, DValue *inst) {
  llvm::Function *fn = getRuntimeFunction(loc, gIR->module, "_d_delinterface");
  LLValue *lval = (inst->isLVal() ? DtoLVal(inst) : makeLValue(loc, inst));
  gIR->CreateCallOrInvoke(fn, lval);
}

void DtoDeleteArray(const Loc &loc, DValue *arr) {
  llvm::Function *fn = getRuntimeFunction(loc, gIR->module, "_d_delarray_t");
  llvm::FunctionType *fty = fn->getFunctionType();

  // the TypeInfo argument must be null if the type has no dtor
  Type *elementType = arr->type->nextOf();
  bool hasDtor = (elementType->toBasetype()->ty == TY::Tstruct &&
                  elementType->needsDestruction());
  LLValue *typeInfo = !hasDtor ? getNullPtr(fty->getParamType(1))
                               : DtoTypeInfoOf(loc, elementType);

  LLValue *lval = (arr->isLVal() ? DtoLVal(arr) : makeLValue(loc, arr));
  gIR->CreateCallOrInvoke(fn, lval, typeInfo);
}

/******************************************************************************
 * ALIGNMENT HELPERS
 ******************************************************************************/

unsigned DtoAlignment(Type *type) {
  const auto alignment = type->alignment();
  if (!alignment.isDefault() && !alignment.isPack())
    return alignment.get();

  auto ts = type->toBasetype()->isTypeStruct();
  return ts && !ts->sym->members ? 0 // opaque struct
                                 : type->alignsize();
}

unsigned DtoAlignment(VarDeclaration *vd) {
  const unsigned typeAlignment = DtoAlignment(vd->type);
  if (vd->alignment.isDefault())
    return typeAlignment;

  const unsigned explicitAlignValue = vd->alignment.get();
  if (vd->alignment.isPack())
    return std::min(typeAlignment, explicitAlignValue);

  return explicitAlignValue;
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
      lltype, gIR->module.getDataLayout().getAllocaAddrSpace(),
      DtoConstUint(arraysize), name, gIR->topallocapoint());
  if (auto alignment = DtoAlignment(type)) {
    ai->setAlignment(llvm::Align(alignment));
  }
  return ai;
}

llvm::AllocaInst *DtoRawAlloca(LLType *lltype, size_t alignment,
                               const char *name) {
  auto ai = new llvm::AllocaInst(
      lltype, gIR->module.getDataLayout().getAllocaAddrSpace(), name,
      gIR->topallocapoint());
  if (alignment) {
    ai->setAlignment(llvm::Align(alignment));
  }
  return ai;
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
        std::min(getTypeAllocSize(DtoType(val->type)),
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
  DtoStoreZextI8(val, mem);
  return mem;
}

/******************************************************************************
 * ASSERT HELPERS
 ******************************************************************************/

void DtoAssert(Module *M, const Loc &loc, DValue *msg) {
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
  args.push_back(DtoConstUint(loc.linnum()));

  // call
  gIR->CreateCallOrInvoke(fn, args);

  // after assert is always unreachable
  gIR->ir->CreateUnreachable();
}

void DtoCAssert(Module *M, const Loc &loc, LLValue *msg) {
  const auto &triple = *global.params.targetTriple;
  const auto file =
      DtoConstCString(loc.filename() ? loc.filename() : M->srcfile.toChars());
  const auto line = DtoConstUint(loc.linnum());
  const auto fn = getCAssertFunction(loc, gIR->module);

  llvm::SmallVector<LLValue *, 4> args;
  if (triple.isOSDarwin()) {
    const auto irFunc = gIR->func();
    const auto funcName =
        irFunc && irFunc->decl ? irFunc->decl->toPrettyChars() : "";
    args.push_back(DtoConstCString(funcName));
    args.push_back(file);
    args.push_back(line);
    args.push_back(msg);
  } else if (triple.isOSSolaris() || triple.isMusl() ||
             global.params.isUClibcEnvironment ||
             triple.isGNUEnvironment()) {
    const auto irFunc = gIR->func();
    const auto funcName =
        (irFunc && irFunc->decl) ? irFunc->decl->toPrettyChars() : "";
    args.push_back(msg);
    args.push_back(file);
    args.push_back(line);
    args.push_back(DtoConstCString(funcName));
  } else if (triple.getEnvironment() == llvm::Triple::Android) {
    args.push_back(file);
    args.push_back(line);
    args.push_back(msg);
  } else if (global.params.isNewlibEnvironment) {
    const auto irFunc = gIR->func();
    const auto funcName =
        irFunc && irFunc->decl ? irFunc->decl->toPrettyChars() : "";
    args.push_back(file);
    args.push_back(line);
    args.push_back(DtoConstCString(funcName));
    args.push_back(msg);
  } else {
    args.push_back(msg);
    args.push_back(file);
    args.push_back(line);
  }

  gIR->CreateCallOrInvoke(fn, args);

  gIR->ir->CreateUnreachable();
}

/******************************************************************************
 * THROW HELPER
 ******************************************************************************/

void DtoThrow(const Loc &loc, DValue *e) {
  LLFunction *fn = getRuntimeFunction(loc, gIR->module, "_d_throw_exception");
  LLValue *arg = DtoRVal(e);

  gIR->CreateCallOrInvoke(fn, arg);
  gIR->ir->CreateUnreachable();

  llvm::BasicBlock *bb = gIR->insertBB("afterthrow");
  gIR->ir->SetInsertPoint(bb);
}

/******************************************************************************
 * MODULE FILE NAME
 ******************************************************************************/

LLConstant *DtoModuleFileName(Module *M, const Loc &loc) {
  return DtoConstString(loc.filename() ? loc.filename() : M->srcfile.toChars());
}

/******************************************************************************
 * GOTO HELPER
 ******************************************************************************/

void DtoGoto(const Loc &loc, LabelDsymbol *target) {
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

void DtoAssign(const Loc &loc, DValue *lhs, DValue *rhs, EXP op,
               bool canSkipPostblit) {
  IF_LOG Logger::println("DtoAssign()");
  LOG_SCOPE;

  Type *t = lhs->type->toBasetype();
  assert(t->ty != TY::Tvoid && "Cannot assign values of type void.");

  if (t->ty == TY::Tnoreturn) {
    // nothing to assign
    return;
  }

  if (auto bfLVal = lhs->isBitFieldLVal()) {
    bfLVal->store(DtoRVal(rhs));
    return;
  }

  if (t->ty == TY::Tbool) {
    DtoStoreZextI8(DtoRVal(rhs), DtoLVal(lhs));
  } else if (t->ty == TY::Tstruct) {
    // don't copy anything to empty structs
    if (static_cast<TypeStruct *>(t)->sym->fields.length > 0) {
      llvm::Value *src = DtoLVal(rhs);
      llvm::Value *dst = DtoLVal(lhs);

      // Check whether source and destination values are the same at compile
      // time as to not emit an invalid (overlapping) memcpy on trivial
      // struct self-assignments like 'A a; a = a;'.
      if (src != dst)
        DtoMemCpy(DtoType(lhs->type), dst, src);
    }
  } else if (t->ty == TY::Tarray || t->ty == TY::Tsarray) {
    DtoArrayAssign(loc, lhs, rhs, op, canSkipPostblit);
  } else if (t->ty == TY::Tdelegate) {
    LLValue *l = DtoLVal(lhs);
    LLValue *r = DtoRVal(rhs);
    IF_LOG {
      Logger::cout() << "lhs: " << *l << '\n';
      Logger::cout() << "rhs: " << *r << '\n';
    }
    DtoStore(r, l);
  } else if (t->ty == TY::Tclass) {
    assert(rhs->type->toBasetype()->ty == TY::Tclass);
    LLValue *l = DtoLVal(lhs);
    LLValue *r = DtoRVal(rhs);
    IF_LOG {
      Logger::cout() << "l : " << *l << '\n';
      Logger::cout() << "r : " << *r << '\n';
    }
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
    LLType *lit = DtoType(lhs->type);
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
  if (basetype->isintegral() || basetype->isfloating() ||
      basety == TY::Tpointer || basety == TY::Tnull || basety == TY::Tclass ||
      basety == TY::Tdelegate || basety == TY::Taarray) {
    return new DNullValue(type, LLConstant::getNullValue(lltype));
  }
  // dynamic array
  if (basety == TY::Tarray) {
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

DValue *DtoCastInt(const Loc &loc, DValue *val, Type *_to) {
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

  if (to->ty == TY::Tbool) {
    LLValue *zero = LLConstantInt::get(rval->getType(), 0, false);
    rval = gIR->ir->CreateICmpNE(rval, zero);
  } else if (to->isintegral()) {
    if (fromsz < tosz || from->ty == TY::Tbool) {
      IF_LOG Logger::cout() << "cast to: " << *tolltype << '\n';
      if (isLLVMUnsigned(from) || from->ty == TY::Tbool) {
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
  } else if (to->ty == TY::Tpointer) {
    IF_LOG Logger::cout() << "cast pointer: " << *tolltype << '\n';
    rval = gIR->ir->CreateIntToPtr(rval, tolltype);
  } else {
    error(loc, "invalid cast from `%s` to `%s`", val->type->toChars(),
          _to->toChars());
    fatal();
  }

  return new DImValue(_to, rval);
}

DValue *DtoCastPtr(const Loc &loc, DValue *val, Type *to) {
  LLType *tolltype = DtoType(to);

  Type *totype = to->toBasetype();
  Type *fromtype = val->type->toBasetype();
  (void)fromtype;
  assert(fromtype->ty == TY::Tpointer || fromtype->ty == TY::Tfunction);

  LLValue *rval;

  if (totype->ty == TY::Tpointer || totype->ty == TY::Tclass ||
      totype->ty == TY::Taarray) {
    LLValue *src = DtoRVal(val);
    IF_LOG {
      Logger::cout() << "src: " << *src << '\n';
      Logger::cout() << "to type: " << *tolltype << '\n';
    }
    rval = DtoBitCast(src, tolltype);
  } else if (totype->ty == TY::Tbool) {
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

DValue *DtoCastFloat(const Loc &loc, DValue *val, Type *to) {
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

  if (totype->ty == TY::Tbool) {
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

DValue *DtoCastDelegate(const Loc &loc, DValue *val, Type *to) {
  if (to->toBasetype()->ty == TY::Tdelegate) {
    return DtoPaintType(loc, val, to);
  }
  if (to->toBasetype()->ty == TY::Tbool) {
    return new DImValue(
        to, DtoDelegateEquals(EXP::notEqual, DtoRVal(val), nullptr));
  }
  error(loc, "invalid cast from `%s` to `%s`", val->type->toChars(),
        to->toChars());
  fatal();
}

DValue *DtoCastVector(const Loc &loc, DValue *val, Type *to) {
  assert(val->type->toBasetype()->ty == TY::Tvector);
  Type *totype = to->toBasetype();
  LLType *tolltype = DtoType(to);

  if (totype->ty == TY::Tsarray) {
    // Reinterpret-cast without copy if the source vector is in memory.
    if (val->isLVal()) {
      LLValue *vector = DtoLVal(val);
      IF_LOG Logger::cout() << "src: " << *vector << " to type: " << *tolltype
                            << " (casting address)\n";
      return new DLValue(to, DtoBitCast(vector, getPtrToType(tolltype)));
    }

    LLValue *vector = DtoRVal(val);
    IF_LOG Logger::cout() << "src: " << *vector << " to type: " << *tolltype
                          << " (creating temporary)\n";
    LLValue *array = DtoAllocaDump(vector, tolltype, DtoAlignment(val->type));
    return new DLValue(to, array);
  }
  if (totype->ty == TY::Tvector && to->size() == val->type->size()) {
    return new DImValue(to, DtoBitCast(DtoRVal(val), tolltype));
  }
  error(loc, "invalid cast from `%s` to `%s`", val->type->toChars(),
        to->toChars());
  fatal();
}

DValue *DtoCastStruct(const Loc &loc, DValue *val, Type *to) {
  Type *const totype = to->toBasetype();
  if (totype->ty == TY::Tstruct) {
    // This a cast to repaint a struct to another type, which the language
    // allows for identical layouts (opCast() and so on have been lowered
    // earlier by the frontend).
    llvm::Value *lval = DtoLVal(val);
    return new DLValue(to, lval);
  }

  error(loc, "Internal Compiler Error: Invalid struct cast from `%s` to `%s`",
        val->type->toChars(), to->toChars());
  fatal();
}

DValue *DtoCast(const Loc &loc, DValue *val, Type *to) {
  Type *fromtype = val->type->toBasetype();
  Type *totype = to->toBasetype();

  if (fromtype->ty == TY::Taarray) {
    if (totype->ty == TY::Taarray) {
      // reinterpret-cast keeping lvalue-ness, IR types will match up
      if (val->isLVal())
        return new DLValue(to, DtoLVal(val));
      return new DImValue(to, DtoRVal(val));
    }
    // DMD allows casting AAs to void*, even if they are internally
    // implemented as structs.
    if (totype->ty == TY::Tpointer) {
      IF_LOG Logger::println("Casting AA to pointer.");
      return new DImValue(to, DtoRVal(val));
    }
    if (totype->ty == TY::Tbool) {
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

  if (fromtype->ty == TY::Tvector) {
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
  case TY::Tclass:
    return DtoCastClass(loc, val, to);
  case TY::Tarray:
  case TY::Tsarray:
    return DtoCastArray(loc, val, to);
  case TY::Tpointer:
  case TY::Tfunction:
    return DtoCastPtr(loc, val, to);
  case TY::Tdelegate:
    return DtoCastDelegate(loc, val, to);
  case TY::Tstruct:
    return DtoCastStruct(loc, val, to);
  case TY::Tnull:
  case TY::Tnoreturn:
    return DtoNullValue(to, loc);
  default:
    error(loc, "invalid cast from `%s` to `%s`", val->type->toChars(),
          to->toChars());
    fatal();
  }
}

////////////////////////////////////////////////////////////////////////////////

DValue *DtoPaintType(const Loc &loc, DValue *val, Type *to) {
  Type *from = val->type->toBasetype();
  IF_LOG Logger::println("repainting from '%s' to '%s'", from->toChars(),
                         to->toChars());

  Type *tb = to->toBasetype();

  if (val->isLVal()) {
    return new DLValue(to, DtoLVal(val));
  }

  if (auto slice = val->isSlice()) {
    if (tb->ty == TY::Tarray) {
      return new DSliceValue(to, slice->getLength(), slice->getPtr());
    }
  } else if (auto func = val->isFunc()) {
    if (tb->ty == TY::Tdelegate) {
      return new DFuncValue(to, func->func, DtoRVal(func), func->vthis);
    }
  } else { // generic rvalue
    LLValue *rval = DtoRVal(val);
    LLType *tll = DtoType(tb);

    if (rval->getType() == tll) {
      return new DImValue(to, rval);
    }
    if (rval->getType()->isPointerTy() && tll->isPointerTy()) {
      return new DImValue(to, rval);
    }
    if (from->ty == TY::Tdelegate && tb->ty == TY::Tdelegate) {
      LLValue *context = gIR->ir->CreateExtractValue(rval, 0, ".context");
      LLValue *funcptr = gIR->ir->CreateExtractValue(rval, 1, ".funcptr");
      return new DImValue(to, DtoAggrPair(context, funcptr));
    }
  }

  error(loc, "ICE: unexpected type repaint from `%s` to `%s`", from->toChars(),
        to->toChars());
  fatal();
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
  if (auto tid = vd->isTypeInfoDeclaration()) {
    DtoResolveTypeInfo(tid);
    return;
  }

  IF_LOG Logger::println("DtoResolveVariable(%s)", vd->toPrettyChars());
  LOG_SCOPE;

  // just forward aliases
  // TODO: Is this required here or is the check in VarDeclaration::codegen
  // sufficient?
  if (vd->aliasTuple) {
    Logger::println("aliasTuple");
    DtoResolveDsymbol(vd->aliasTuple);
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

    auto irGlobal = getIrGlobal(vd, true);
    irGlobal->getValue();
  }
}

/******************************************************************************
 * DECLARATION EXP HELPER
 ******************************************************************************/

// TODO: Merge with DtoRawVarDeclaration!
void DtoVarDeclaration(VarDeclaration *vd) {
  assert(!vd->isDataseg() &&
         "Statics/globals are handled in DtoDeclarationExp.");
  assert(!vd->aliasTuple && "Aliases are handled in DtoDeclarationExp.");

  IF_LOG Logger::println("DtoVarDeclaration(vdtype = %s)", vd->type->toChars());
  LOG_SCOPE

  if (vd->nestedrefs.length) {
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
  } else if (gIR->func()->sretArg &&
             ((gIR->func()->decl->isNRVO() &&
               gIR->func()->decl->nrvo_var == vd) ||
              (vd->isResult() && !isSpecialRefVar(vd)))) {
    // Named Return Value Optimization (NRVO):
    // T f() {
    //   T ret;        // &ret == hidden pointer
    //   ret = ...
    //   return ret;    // NRVO.
    // }
    assert(!isSpecialRefVar(vd) && "Can this happen?");
    getIrLocal(vd, true)->value = gIR->func()->sretArg;
    gIR->DBuilder.EmitLocalVariable(gIR->func()->sretArg, vd);
  } else {
    // normal stack variable, allocate storage on the stack if it has not
    // already been done
    IrLocal *irLocal = getIrLocal(vd, true);

    Type *type = isSpecialRefVar(vd) ? pointerTo(vd->type) : vd->type;

    llvm::Value *allocainst;
    bool isRealAlloca = false;
    LLType *lltype = DtoType(type); // void for noreturn
    if (lltype->isVoidTy() || gDataLayout->getTypeSizeInBits(lltype) == 0) {
      allocainst = llvm::ConstantPointerNull::get(getPtrToType(lltype));
    } else if (type != vd->type) {
      allocainst = DtoAlloca(type, vd->toChars());
      isRealAlloca = true;
    } else {
      allocainst = DtoAlloca(vd, vd->toChars());
      isRealAlloca = true;
    }

    irLocal->value = allocainst;

    if (!lltype->isVoidTy())
      gIR->DBuilder.EmitLocalVariable(allocainst, vd);

    // Lifetime annotation is only valid on alloca.
    if (isRealAlloca) {
      // The lifetime of a stack variable starts from the point it is declared
      gIR->funcGen().localVariableLifetimeAnnotator.addLocalVariable(
          allocainst, DtoConstUlong(type->size()));
    }
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

    // if aliasTuple is set, this VarDecl is redone as an alias to another symbol
    // this seems to be done to rewrite Tuple!(...) v;
    // as a TupleDecl that contains a bunch of individual VarDecls
    if (vd->aliasTuple) {
      return DtoDeclarationExp(vd->aliasTuple);
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
      for (unsigned i = 0; i < d->length; ++i) {
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
    for (unsigned i = 0; i < tupled->objects->length; ++i) {
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
  assert(!var->aliasTuple);

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
  if (var->nestedrefs.length && isIrLocalCreated(var)) {
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

LLConstant *DtoConstInitializer(const Loc &loc, Type *type, Initializer *init,
                                const bool isCfile) {
  LLConstant *_init = nullptr; // may return zero
  if (!init) {
    if (type->toBasetype()->isTypeNoreturn()) {
      Logger::println("const noreturn initializer");
      LLType *ty = DtoMemType(type);
      _init = LLConstant::getNullValue(ty);
    } else {
      IF_LOG Logger::println("const default initializer for %s", type->toChars());
      Expression *initExp = defaultInit(type, loc, isCfile);
      _init = DtoConstExpInit(loc, type, initExp);
    }
  } else if (ExpInitializer *ex = init->isExpInitializer()) {
    Logger::println("const expression initializer");
    _init = DtoConstExpInit(loc, type, ex->exp);
  } else if (ArrayInitializer *ai = init->isArrayInitializer()) {
    Logger::println("const array initializer");
    _init = DtoConstArrayInitializer(ai, type, isCfile);
  } else if (init->isVoidInitializer()) {
    Logger::println("const void initializer");
    LLType *ty = DtoMemType(type);
    _init = LLConstant::getNullValue(ty);
  } else if (init->isCInitializer()) {
    // TODO: ImportC
    error(loc, "LDC doesn't support C initializer lists yet");
    fatal();
  } else {
    // StructInitializer is no longer suposed to make it to the glue layer
    // in DMD 2.064.
    IF_LOG Logger::println("unsupported const initializer: %s",
                           init->toChars());
  }
  return _init;
}

////////////////////////////////////////////////////////////////////////////////

LLConstant *DtoConstExpInit(const Loc &loc, Type *targetType, Expression *exp) {
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
  if (llType->isIntegerTy(1)) {
    llType = LLType::getInt8Ty(gIR->context());
#if LDC_LLVM_VER < 1800
    val = llvm::ConstantExpr::getZExt(val, llType);
#else
    val = llvm::ConstantFoldCastOperand(llvm::Instruction::ZExt, val, llType, *gDataLayout);
#endif
  }

  if (llType == targetLLType)
    return val;

  if (baseTargetType->ty == TY::Tsarray) {
    Logger::println("Building constant array initializer from scalar.");

    assert(baseValType->size() > 0);
    const auto numTotalVals = baseTargetType->size() / baseValType->size();
    assert(baseTargetType->size() % baseValType->size() == 0);

    // may be a multi-dimensional array init, e.g., `char[2][3] x = 0xff`
    baseValType = stripModifiers(baseValType);
    LLSmallVector<size_t, 4> dims; // { 3, 2 }
    for (auto t = baseTargetType; t->ty == TY::Tsarray;) {
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

    (void)numTotalVals; (void) product; // Silence unused variable warning when assert is disabled.
    assert(product == numTotalVals);
    return val;
  }

  if (baseTargetType->ty == TY::Tvector) {
    Logger::println("Building constant vector initializer from scalar.");

    TypeVector *tv = static_cast<TypeVector *>(baseTargetType);
    assert(tv->basetype->ty == TY::Tsarray);
    dinteger_t elemCount =
        static_cast<TypeSArray *>(tv->basetype)->dim->toInteger();
    const auto elementCount = llvm::ElementCount::getFixed(elemCount);
    return llvm::ConstantVector::getSplat(elementCount, val);
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

#if LDC_LLVM_VER < 1800
    return llvm::ConstantExpr::getZExtOrBitCast(val, target);
#else
    return llvm::ConstantFoldCastOperand(llvm::Instruction::ZExt, val, target, *gDataLayout);
#endif
  }

  Logger::println("Unhandled type mismatch, giving up.");
  return val;
}

////////////////////////////////////////////////////////////////////////////////

LLConstant *DtoTypeInfoOf(const Loc &loc, Type *type) {
  IF_LOG Logger::println("DtoTypeInfoOf(type = '%s')", type->toChars());
  LOG_SCOPE

  auto tidecl = getOrCreateTypeInfoDeclaration(loc, type);
  auto tiglobal = DtoResolveTypeInfo(tidecl);
  return tiglobal;
}

////////////////////////////////////////////////////////////////////////////////
/// Allocates memory and passes on ownership. (never returns null)
static char *DtoOverloadedIntrinsicName(TemplateInstance *ti,
                                        TemplateDeclaration *td) {
  IF_LOG Logger::println("DtoOverloadedIntrinsicName");
  LOG_SCOPE;

  assert(td->intrinsicName);

  IF_LOG {
    Logger::println("template instance: %s", ti->toChars());
    Logger::println("template declaration: %s", td->toChars());
    Logger::println("intrinsic name: %s", td->intrinsicName);
  }

  // for now use the size in bits of the first template param in the instance
  assert(ti->tdtypes.length == 1);
  Type *T = static_cast<Type *>(ti->tdtypes[0]);

  char prefix;
  if (T->isfloating() && !T->iscomplex()) {
    prefix = 'f';
  } else if (T->isintegral()) {
    prefix = 'i';
  } else {
    error(ti->loc, "%s `%s` has invalid template parameter for intrinsic: `%s`",
          ti->kind(), ti->toPrettyChars(), T->toChars());
    fatal(); // or LLVM asserts
  }

  std::string name = td->intrinsicName;

  // replace `{f,i}#` by `{f,i}<bitsize>` (int: `i32`) or
  // `v<vector length>{f,i}<vector element bitsize>` (float4: `v4f32`)
  llvm::Type *dtype = DtoType(T);
  std::string replacement;
  if (dtype->isPPC_FP128Ty()) { // special case
    replacement = "ppcf128";
  } else if (dtype->isVectorTy()) {
    auto vectorType = llvm::cast<llvm::FixedVectorType>(dtype);
    llvm::raw_string_ostream stream(replacement);
    stream << 'v' << vectorType->getNumElements() << prefix
           << gDataLayout->getTypeSizeInBits(vectorType->getElementType());
    stream.flush();
  } else {
    replacement = prefix + std::to_string(gDataLayout->getTypeSizeInBits(dtype));
  }

  size_t pos;
  while (std::string::npos != (pos = name.find('#'))) {
    if (pos > 0 && name[pos - 1] == prefix) {
      name.replace(pos - 1, 2, replacement);
    } else {
      if (pos && (name[pos - 1] == 'i' || name[pos - 1] == 'f')) {
        // Wrong type character.
        error(ti->loc,
              "%s `%s` has invalid parameter type for intrinsic `%s`: `%s` is "
              "not a%s type",
              ti->kind(), ti->toPrettyChars(), name.c_str(), T->toChars(),
              (name[pos - 1] == 'i' ? "n integral" : " floating-point"));
      } else {
        // Just plain wrong. (Error in declaration, not instantiation)
        error(td->loc, "%s `%s` has an invalid intrinsic name: `%s`",
              td->kind(), td->toPrettyChars(), name.c_str());
      }
      fatal(); // or LLVM asserts
    }
  }

  IF_LOG Logger::println("final intrinsic name: %s", name.c_str());

  return mem.xstrdup(name.c_str());
}

/// For D frontend
/// Fixup an overloaded intrinsic name string.
void DtoSetFuncDeclIntrinsicName(TemplateInstance *ti, TemplateDeclaration *td,
                                 FuncDeclaration *fd) {
  if (fd->llvmInternal == LLVMintrinsic) {
    const auto cstr = DtoOverloadedIntrinsicName(ti, td);
    assert(cstr);
    fd->mangleOverride = {strlen(cstr), cstr};
  }
}

////////////////////////////////////////////////////////////////////////////////

Type *stripModifiers(Type *type, bool transitive) {
  if (type->ty == TY::Tfunction) {
    return type;
  }

  if (transitive) {
    return unqualify(type, MODimmutable | MODconst | MODwild);
  }
  return castMod(type, 0);
}

////////////////////////////////////////////////////////////////////////////////

LLValue *makeLValue(const Loc &loc, DValue *value) {
  if (value->isLVal())
    return DtoLVal(value);

  if (value->isIm() || value->isConst())
    return DtoAllocaDump(value, ".makelvaluetmp");

  LLValue *mem = DtoAlloca(value->type, ".makelvaluetmp");
  DLValue var(value->type, mem);
  DtoAssign(loc, &var, value, EXP::blit);
  return mem;
}

////////////////////////////////////////////////////////////////////////////////

void callPostblit(const Loc &loc, Expression *exp, LLValue *val) {
  Type *tb = exp->type->toBasetype();
  if ((exp->op == EXP::variable || exp->op == EXP::dotVariable ||
       exp->op == EXP::star || exp->op == EXP::this_ ||
       exp->op == EXP::index) &&
      tb->ty == TY::Tstruct) {
    StructDeclaration *sd = static_cast<TypeStruct *>(tb)->sym;
    if (sd->postblit) {
      FuncDeclaration *fd = sd->postblit;
      if (fd->storage_class & STCdisable) {
        error(loc,
              "%s `%s` is not copyable because it is annotated with `@disable`",
              sd->kind(), sd->toPrettyChars());
      }
      Expressions args;
      DFuncValue dfn(fd, DtoCallee(fd), val);
      DtoCallFunction(loc, Type::tvoid, &dfn, &args);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

bool isSpecialRefVar(VarDeclaration *vd) {
  return (vd->storage_class & (STCref | STCparameter)) == STCref;
}

////////////////////////////////////////////////////////////////////////////////

bool isLLVMUnsigned(Type *t) {
  return t->isunsigned() || t->ty == TY::Tpointer;
}

////////////////////////////////////////////////////////////////////////////////

void printLabelName(std::ostream &target, const char *func_mangle,
                    const char *label_name) {
  target << gTargetMachine->getMCAsmInfo()->getPrivateGlobalPrefix().str()
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

void tokToICmpPred(EXP op, bool isUnsigned, llvm::ICmpInst::Predicate *outPred,
                   llvm::Value **outConst) {
  switch (op) {
  case EXP::lessThan:
    *outPred = isUnsigned ? llvm::ICmpInst::ICMP_ULT : llvm::ICmpInst::ICMP_SLT;
    break;
  case EXP::lessOrEqual:
    *outPred = isUnsigned ? llvm::ICmpInst::ICMP_ULE : llvm::ICmpInst::ICMP_SLE;
    break;
  case EXP::greaterThan:
    *outPred = isUnsigned ? llvm::ICmpInst::ICMP_UGT : llvm::ICmpInst::ICMP_SGT;
    break;
  case EXP::greaterOrEqual:
    *outPred = isUnsigned ? llvm::ICmpInst::ICMP_UGE : llvm::ICmpInst::ICMP_SGE;
    break;
  default:
    llvm_unreachable("Invalid comparison operation");
  }
}

////////////////////////////////////////////////////////////////////////////////

llvm::ICmpInst::Predicate eqTokToICmpPred(EXP op, bool invert) {
  assert(op == EXP::equal || op == EXP::notEqual || op == EXP::identity ||
         op == EXP::notIdentity);

  bool isEquality = (op == EXP::equal || op == EXP::identity);
  if (invert)
    isEquality = !isEquality;

  return (isEquality ? llvm::ICmpInst::ICMP_EQ : llvm::ICmpInst::ICMP_NE);
}

////////////////////////////////////////////////////////////////////////////////

LLValue *createIPairCmp(EXP op, LLValue *lhs1, LLValue *lhs2, LLValue *rhs1,
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
DValue *DtoSymbolAddress(const Loc &loc, Type *type, Declaration *decl) {
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
      LLValue *m = DtoResolveTypeInfo(tid);
      return new DImValue(type, m);
    }
    // special vtbl symbol, used by LDC as alias to the actual vtbl (with
    // different type and mangled name)
    if (vd->isClassMember() && vd == vd->isClassMember()->vtblsym) {
      Logger::println("vtbl symbol");
      auto cd = vd->isClassMember();
      return new DLValue(type, getIrAggr(cd)->getVtblSymbol());
    }
    // nested variable
    if (vd->nestedrefs.length) {
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
        assert(type->ty == TY::Tdelegate);
      }
      assert(!isSpecialRefVar(vd) && "Code not expected to handle special "
                                     "ref vars, although it can easily be "
                                     "made to.");
      return new DLValue(type, getIrValue(vd));
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

    return new DFuncValue(flitdecl, DtoCallee(flitdecl, false));
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
    assert(!DtoIsMagicIntrinsic(fdecl));
    return new DFuncValue(fdecl, DtoCallee(fdecl));
  }

  if (SymbolDeclaration *sdecl = decl->isSymbolDeclaration()) {
    // this is the static initialiser (init symbol) for aggregates
    AggregateDeclaration *ad = sdecl->dsym;
    IF_LOG Logger::print("init symbol of %s\n", ad->toChars());
    DtoResolveDsymbol(ad);
    auto sd = ad->isStructDeclaration();

    // LDC extension: void[]-typed `__traits(initSymbol)`, for classes too
    auto tb = sdecl->type->toBasetype();
    if (tb->ty != TY::Tstruct) {
      assert(tb->ty == TY::Tarray && tb->nextOf()->ty == TY::Tvoid);
      const auto size = DtoConstSize_t(ad->structsize);
      llvm::Constant *ptr = sd && sd->zeroInit()
                                ? getNullValue(getVoidPtrType())
                                : getIrAggr(ad)->getInitSymbol();
      return new DSliceValue(type, size, ptr);
    }

    assert(sd);
    if (sd->zeroInit()) {
      error(loc, "no init symbol for zero-initialized struct");
      fatal();
    }

    LLValue *initsym = getIrAggr(sd)->getInitSymbol();
    return new DLValue(type, initsym);
  }

  llvm_unreachable("Unimplemented VarExp type");
}

llvm::Constant *DtoConstSymbolAddress(const Loc &loc, Declaration *decl) {
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
    return DtoCallee(fd);
  }

  llvm_unreachable("Taking constant address not implemented.");
}

llvm::Constant *buildStringLiteralConstant(StringExp *se,
                                           uint64_t bufferLength) {
  const auto stringLength = se->len;
  assert(bufferLength >= stringLength);

  if (se->sz == 1 && bufferLength <= stringLength + 1) {
    const DString data = se->peekString();
    const bool nullTerminate = (bufferLength == stringLength + 1);
    return llvm::ConstantDataArray::getString(
        gIR->context(), {data.ptr, data.length}, nullTerminate);
  }

  Type *dtype = se->type->toBasetype();
  Type *cty = dtype->nextOf()->toBasetype();
  LLType *ct = DtoMemType(cty);
  LLArrayType *at = LLArrayType::get(ct, bufferLength);

  std::vector<LLConstant *> vals;
  vals.reserve(bufferLength);
  for (uint64_t i = 0; i < stringLength; ++i) {
    vals.push_back(LLConstantInt::get(ct, se->getIndex(i), false));
  }
  const auto nullChar = LLConstantInt::get(ct, 0, false);
  for (uint64_t i = stringLength; i < bufferLength; ++i) {
    vals.push_back(nullChar);
  }
  return LLConstantArray::get(at, vals);
}

std::string llvmTypeToString(llvm::Type *type) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << *type;
  stream.flush();
  return result;
}

// Is the specified symbol defined in the druntime/Phobos libs?
// For instantiated symbols: is the template declared in druntime/Phobos?
static bool isDefaultLibSymbol(Dsymbol *sym) {
  auto mod = sym->getModule();
  if (!mod)
    return false;

  auto md = mod->md;
  if (!md)
    return false;

  if (md->packages.length == 0)
    return md->id == Id::object || md->id == Id::std;

  auto p = md->packages.ptr[0];
  return p == Id::core || p == Id::etc || p == Id::ldc ||
         (p == Id::std &&
          // 3rd-party package: std.io (https://github.com/MartinNowak/io/)
          !((md->packages.length == 1 && md->id == Id::io) ||
            (md->packages.length > 1 && md->packages.ptr[1] == Id::io)));
}

bool defineOnDeclare(Dsymbol *sym, bool isFunction) {
  // -linkonce-templates: all instantiated symbols
  if (global.params.linkonceTemplates != LinkonceTemplates::no)
    return sym->isInstantiated();

  // -dllimport=defaultLibsOnly: all data symbols instantiated from
  // druntime/Phobos templates
  // see https://github.com/ldc-developers/ldc/issues/3931
  return !isFunction && global.params.dllimport == DLLImport::defaultLibsOnly &&
         sym->isInstantiated() && isDefaultLibSymbol(sym);
}

bool dllimportDataSymbol(Dsymbol *sym) {
  if (!global.params.targetTriple->isOSWindows())
    return false;

  if (sym->isExport() || global.params.dllimport == DLLImport::all ||
      (global.params.dllimport == DLLImport::defaultLibsOnly &&
       isDefaultLibSymbol(sym))) {
    // Okay, this symbol is a candidate. Use dllimport unless we have a
    // guaranteed-codegen'd definition in a root module.
    if (auto mod = sym->isModule())
      return !mod->isRoot(); // non-root ModuleInfo symbol
    if (sym->inNonRoot())
      return true; // not instantiated, and defined in non-root
    if (sym->isInstantiated() && !defineOnDeclare(sym, false))
      return true; // instantiated but potentially culled (needsCodegen())
    if (auto vd = sym->isVarDeclaration())
      if (vd->storage_class & STCextern)
        return true; // externally defined global variable
  }

  return false;
}

llvm::GlobalVariable *declareGlobal(const Loc &loc, llvm::Module &module,
                                    llvm::Type *type,
                                    llvm::StringRef mangledName,
                                    bool isConstant, bool isThreadLocal,
                                    bool useDLLImport) {
  // No TLS support for WebAssembly and AVR; spare users from having to add
  // __gshared everywhere.
  const auto arch = global.params.targetTriple->getArch();
  if (arch == llvm::Triple::wasm32 || arch == llvm::Triple::wasm64 ||
      arch == llvm::Triple::avr)
    isThreadLocal = false;

  llvm::GlobalVariable *existing =
      module.getGlobalVariable(mangledName, /*AllowInternal=*/true);
  if (existing) {
    const auto existingType = existing->getValueType();
    if (existingType != type || existing->isConstant() != isConstant ||
        existing->isThreadLocal() != isThreadLocal) {
      error(loc,
            "Global variable type does not match previous declaration with "
            "same mangled name: `%s`",
            mangledName.str().c_str());
      const auto suppl = [&loc](const char *prefix, LLType *type,
                                bool isConstant, bool isThreadLocal) {
        const auto typeName = llvmTypeToString(type);
        errorSupplemental(loc, "%s %s, %s, %s", prefix, typeName.c_str(),
                          isConstant ? "const" : "mutable",
                          isThreadLocal ? "thread-local" : "non-thread-local");
      };
      suppl("Previous IR type:", existingType, existing->isConstant(),
            existing->isThreadLocal());
      suppl("New IR type:     ", type, isConstant, isThreadLocal);
      fatal();
    }
    return existing;
  }

  // Use a command line option for the thread model.
  // On PPC there is only local-exec available - in this case just ignore the
  // command line.
  const auto tlsModel =
      isThreadLocal
          ? (arch == llvm::Triple::ppc ? llvm::GlobalVariable::LocalExecTLSModel
                                       : clThreadModel.getValue())
          : llvm::GlobalVariable::NotThreadLocal;

  auto gvar = new llvm::GlobalVariable(module, type, isConstant,
                                       llvm::GlobalValue::ExternalLinkage,
                                       nullptr, mangledName, nullptr, tlsModel);

  if (useDLLImport && global.params.targetTriple->isOSWindows())
    gvar->setDLLStorageClass(LLGlobalValue::DLLImportStorageClass);

  return gvar;
}

void defineGlobal(llvm::GlobalVariable *global, llvm::Constant *init,
                  Dsymbol *symbolForLinkageAndVisibility) {
  assert(global->isDeclaration() && "Global variable already defined");
  assert(init);
  global->setInitializer(init);
  if (symbolForLinkageAndVisibility)
    setLinkageAndVisibility(symbolForLinkageAndVisibility, global);
}

llvm::GlobalVariable *defineGlobal(const Loc &loc, llvm::Module &module,
                                   llvm::StringRef mangledName,
                                   llvm::Constant *init,
                                   llvm::GlobalValue::LinkageTypes linkage,
                                   bool isConstant, bool isThreadLocal) {
  assert(init);
  auto global =
      declareGlobal(loc, module, init->getType(), mangledName, isConstant,
                    isThreadLocal, /*useDLLImport*/ false);
  defineGlobal(global, init, nullptr);
  global->setLinkage(linkage);
  return global;
}

FuncDeclaration *getParentFunc(Dsymbol *sym) {
  if (!sym) {
    return nullptr;
  }

  // Static functions, non-extern(D) non-member functions and function (not
  // delegate) literals don't allow access to a parent context, even if they are
  // nested.
  if (FuncDeclaration *fd = sym->isFuncDeclaration()) {
    if (auto fld = fd->isFuncLiteralDeclaration()) {
      if (fld->tok == TOK::function_)
        return nullptr;
    } else if (fd->isStatic() || (!fd->isThis() && fd->_linkage != LINK::d)) {
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

DLValue *DtoIndexAggregate(LLValue *src, AggregateDeclaration *ad,
                           VarDeclaration *vd) {
  IF_LOG Logger::println("Indexing aggregate field %s:", vd->toPrettyChars());
  LOG_SCOPE;

  // Make sure the aggregate is resolved, as subsequent code might expect
  // isIrVarCreated(vd). This is a bit of a hack, we don't actually need this
  // ourselves, DtoType below would be enough.
  DtoResolveDsymbol(ad);

  // Look up field to index or offset to apply.
  auto irTypeAggr = getIrType(ad->type)->isAggr();
  assert(irTypeAggr);
  bool isFieldIdx;
  unsigned off = irTypeAggr->getMemberLocation(vd, isFieldIdx);

  LLValue *ptr = src;
  LLType * ty = nullptr;
  if (!isFieldIdx) {
    // apply byte-wise offset from object start
    ptr = DtoGEP1(getI8Type(), ptr, off);
    ty = DtoType(vd->type);
  } else {
    if (ad->structsize == 0) { // can happen for extern(C) structs
      assert(off == 0);
    } else {
      // Cast the pointer we got to the canonical struct type the indices are
      // based on.
      LLType *st = nullptr;
      if (auto irtc = irTypeAggr->isClass()) {
        st = irtc->getMemoryLLType();
      } else {
        st = irTypeAggr->getLLType();
      }
      ptr = DtoGEP(st, ptr, 0, off);
      ty = isaStruct(st)->getElementType(off);
    }
  }

  IF_LOG Logger::cout() << "Pointer: " << *ptr << '\n';
  if (auto p = isaPointer(ty)) {
    if (p->getAddressSpace())
      return new DDcomputeLValue(vd->type, p, ptr);
  }
  return new DLValue(vd->type, ptr);
}

unsigned getFieldGEPIndex(AggregateDeclaration *ad, VarDeclaration *vd) {
  auto irTypeAggr = getIrType(ad->type)->isAggr();
  assert(irTypeAggr);
  bool isFieldIdx;
  unsigned off = irTypeAggr->getMemberLocation(vd, isFieldIdx);
  assert(isFieldIdx && "Cannot address field by a simple GEP.");
  return off;
}

DValue *makeVarDValue(Type *type, VarDeclaration *vd, llvm::Value *storage) {
  auto val = storage;
  if (!val) {
    assert(isIrVarCreated(vd) && "Variable not resolved.");
    val = getIrValue(vd);
  }

  assert(val->getType()->isPointerTy());

  if (isSpecialRefVar(vd))
    return new DSpecialRefValue(type, val);

  return new DLValue(type, val);
}
