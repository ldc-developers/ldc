//===-- toir.cpp ----------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "attrib.h"
#include "enum.h"
#include "hdrgen.h"
#include "id.h"
#include "init.h"
#include "mtype.h"
#include "module.h"
#include "port.h"
#include "rmem.h"
#include "template.h"
#include "gen/aa.h"
#include "gen/abi.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/complex.h"
#include "gen/coverage.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/nested.h"
#include "gen/optimizer.h"
#include "gen/pragma.h"
#include "gen/runtime.h"
#include "gen/structs.h"
#include "gen/tollvm.h"
#include "gen/typeinf.h"
#include "gen/warnings.h"
#include "ir/irfunction.h"
#include "ir/irtypeclass.h"
#include "ir/irtypestruct.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include <fstream>
#include <math.h>
#include <stack>
#include <stdio.h>

// Needs other includes.
#include "ctfe.h"

llvm::cl::opt<bool> checkPrintf(
    "check-printf-calls",
    llvm::cl::desc("Validate printf call format strings against arguments"),
    llvm::cl::ZeroOrMore);

bool walkPostorder(Expression *e, StoppableVisitor *v);
extern LLConstant *get_default_initializer(VarDeclaration *vd);

////////////////////////////////////////////////////////////////////////////////

dinteger_t undoStrideMul(Loc &loc, Type *t, dinteger_t offset) {
  assert(t->ty == Tpointer);
  d_uns64 elemSize = t->nextOf()->size(loc);
  assert((offset % elemSize) == 0 &&
         "Expected offset by an integer amount of elements");

  return offset / elemSize;
}

////////////////////////////////////////////////////////////////////////////////

static LLValue *write_zeroes(LLValue *mem, unsigned start, unsigned end) {
  mem = DtoBitCast(mem, getVoidPtrType());
  LLValue *gep = DtoGEPi1(mem, start, ".padding");
  DtoMemSetZero(gep, DtoConstSize_t(end - start));
  return mem;
}

////////////////////////////////////////////////////////////////////////////////

static void write_struct_literal(Loc loc, LLValue *mem, StructDeclaration *sd,
                                 Expressions *elements) {
  // ready elements data
  assert(elements && "struct literal has null elements");
  const size_t nexprs = elements->dim;
  Expression **exprs = reinterpret_cast<Expression **>(elements->data);

  // might be reset to an actual i8* value so only a single bitcast is emitted.
  LLValue *voidptr = mem;
  unsigned offset = 0;

  // go through fields
  const size_t nfields = sd->fields.dim;
  for (size_t index = 0; index < nfields; ++index) {
    VarDeclaration *vd = sd->fields[index];

    // get initializer expression
    Expression *expr = (index < nexprs) ? exprs[index] : nullptr;
    if (!expr) {
      // In case of an union, we can't simply use the default initializer.
      // Consider the type union U7727A1 { int i; double d; } and
      // the declaration U7727A1 u = { d: 1.225 };
      // The loop will first visit variable i and then d. Since d has an
      // explicit initializer, we must use this one. The solution is to
      // peek at the next variables.
      for (size_t index2 = index + 1; index2 < nfields; ++index2) {
        VarDeclaration *vd2 = sd->fields[index2];
        if (vd->offset != vd2->offset) {
          break;
        }
        ++index; // skip var
        Expression *expr2 = (index2 < nexprs) ? exprs[index2] : nullptr;
        if (expr2) {
          vd = vd2;
          expr = expr2;
          break;
        }
      }
    }

    // don't re-initialize unions
    if (vd->offset < offset) {
      IF_LOG Logger::println("skipping field: %s %s (+%u)", vd->type->toChars(),
                             vd->toChars(), vd->offset);
      continue;
    }

    // initialize any padding so struct comparisons work
    if (vd->offset != offset) {
      voidptr = write_zeroes(voidptr, offset, vd->offset);
    }
    offset = vd->offset + vd->type->size();

    IF_LOG Logger::println("initializing field: %s %s (+%u)",
                           vd->type->toChars(), vd->toChars(), vd->offset);
    LOG_SCOPE

    // get initializer
    DValue *val;
    DConstValue cv(vd->type,
                   nullptr); // Only used in one branch; value is set beforehand
    if (expr) {
      IF_LOG Logger::println("expr %llu = %s",
                             static_cast<unsigned long long>(index),
                             expr->toChars());
      val = toElem(expr);
    } else if (vd == sd->vthis) {
      IF_LOG Logger::println("initializing vthis");
      LOG_SCOPE
      val = new DImValue(
          vd->type, DtoBitCast(DtoNestedContext(loc, sd), DtoType(vd->type)));
    } else {
      if (vd->init && vd->init->isVoidInitializer()) {
        continue;
      }
      IF_LOG Logger::println("using default initializer");
      LOG_SCOPE
      cv.c = get_default_initializer(vd);
      val = &cv;
    }

    // get a pointer to this field
    DVarValue field(vd->type, vd, DtoIndexAggregate(mem, sd, vd));

    // store the initializer there
    DtoAssign(loc, &field, val, TOKconstruct, true);

    if (expr && expr->isLvalue()) {
      callPostblit(loc, expr, field.getLVal());
    }

    // Also zero out padding bytes counted as being part of the type in DMD
    // but not in LLVM; e.g. real/x86_fp80.
    int implicitPadding =
        vd->type->size() - gDataLayout->getTypeStoreSize(DtoType(vd->type));
    assert(implicitPadding >= 0);
    if (implicitPadding > 0) {
      IF_LOG Logger::println("zeroing %d padding bytes", implicitPadding);
      voidptr = write_zeroes(voidptr, offset - implicitPadding, offset);
    }
  }
  // initialize trailing padding
  if (sd->structsize != offset) {
    voidptr = write_zeroes(voidptr, offset, sd->structsize);
  }
}

namespace {
void pushVarDtorCleanup(IRState *p, VarDeclaration *vd) {
  llvm::BasicBlock *beginBB = llvm::BasicBlock::Create(
      p->context(), llvm::Twine("dtor.") + vd->toChars(), p->topfunc());

  // TODO: Clean this up with push/pop insertion point methods.
  IRScope oldScope = p->scope();
  p->scope() = IRScope(beginBB);
  toElemDtor(vd->edtor);
  p->func()->scopes->pushCleanup(beginBB, p->scopebb());
  p->scope() = oldScope;
}
}

////////////////////////////////////////////////////////////////////////////////

// Tries to find the proper lvalue subexpression of an assign/binassign
// expression.
// Returns null if none is found.
static Expression *findLvalueExp(Expression *e) {
  class FindLvalueVisitor : public Visitor {
  public:
    Expression *result;

    FindLvalueVisitor() : result(nullptr) {}

    void visit(Expression *e) LLVM_OVERRIDE {}

#define FORWARD(TYPE)                                                          \
  void visit(TYPE *e) LLVM_OVERRIDE { e->e1->accept(this); }
    FORWARD(AssignExp)
    FORWARD(BinAssignExp)
    FORWARD(CastExp)
#undef FORWARD

#define IMPLEMENT(TYPE)                                                        \
  void visit(TYPE *e) LLVM_OVERRIDE { result = e; }
    IMPLEMENT(VarExp)
    IMPLEMENT(CallExp)
    IMPLEMENT(PtrExp)
    IMPLEMENT(DotVarExp)
    IMPLEMENT(IndexExp)
    IMPLEMENT(CommaExp)
#undef IMPLEMENT
  };

  FindLvalueVisitor v;
  e->accept(&v);
  return v.result;
}

// Evaluates an lvalue expression e and prevents further
// evaluations as long as e->cachedLvalue isn't reset to null.
static DValue *toElemAndCacheLvalue(Expression *e) {
  DValue *value = toElem(e);
  e->cachedLvalue = value->getLVal();
  return value;
}

// Evaluates e and, if tryGetLvalue is true, returns the
// (casted) nested lvalue if one is found.
// Otherwise simply returns the expression's result.
DValue *toElem(Expression *e, bool tryGetLvalue) {
  if (!tryGetLvalue) {
    return toElem(e);
  }

  Expression *lvalExp = findLvalueExp(e); // may be null
  Expression *nestedLvalExp = (lvalExp == e ? nullptr : lvalExp);

  DValue *nestedLval = nullptr;
  if (nestedLvalExp) {
    IF_LOG Logger::println("Caching l-value of %s => %s", e->toChars(),
                           nestedLvalExp->toChars());

    LOG_SCOPE;
    nestedLval = toElemAndCacheLvalue(nestedLvalExp);
  }

  DValue *value = toElem(e);

  if (nestedLvalExp) {
    nestedLvalExp->cachedLvalue = nullptr;
  }

  return !nestedLval ? value : DtoCast(e->loc, nestedLval, e->type);
}

////////////////////////////////////////////////////////////////////////////////

class ToElemVisitor : public Visitor {
  IRState *p;
  bool destructTemporaries;
  CleanupCursor initialCleanupScope;
  DValue *result;

public:
  ToElemVisitor(IRState *p_, bool destructTemporaries_)
      : p(p_), destructTemporaries(destructTemporaries_), result(nullptr) {
    initialCleanupScope = p->func()->scopes->currentCleanupScope();
  }

  DValue *getResult() {
    if (destructTemporaries &&
        p->func()->scopes->currentCleanupScope() != initialCleanupScope) {
      // If the results is an (LLVM) r-value, temporarily store it in an
      // alloca slot to avoid running into instruction dominance issues
      // if we share the cleanups with another exit path (e.g. unwinding).
      if (result && result->getType()->ty != Tvoid &&
          (result->isIm() || result->isSlice())) {
        LLValue *alloca = DtoAllocaDump(result);
        result = new DVarValue(result->getType(), alloca);
      }

      llvm::BasicBlock *endbb = llvm::BasicBlock::Create(
          p->context(), "toElem.success", p->topfunc());
      p->func()->scopes->runCleanups(initialCleanupScope, endbb);
      p->func()->scopes->popCleanups(initialCleanupScope);
      p->scope() = IRScope(endbb);

      destructTemporaries = false;
    }

    return result;
  }

  //////////////////////////////////////////////////////////////////////////////

  // Import all functions from class Visitor
  using Visitor::visit;

  //////////////////////////////////////////////////////////////////////////////

  void visit(DeclarationExp *e) override {
    IF_LOG Logger::print("DeclarationExp::toElem: %s | T=%s\n", e->toChars(),
                         e->type ? e->type->toChars() : "(null)");
    LOG_SCOPE;

    result = DtoDeclarationExp(e->declaration);

    if (result) {
      if (DVarValue *varValue = result->isVar()) {
        VarDeclaration *vd = varValue->var;
        if (!vd->isDataseg() && vd->edtor && !vd->noscope) {
          pushVarDtorCleanup(p, vd);
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(VarExp *e) override {
    IF_LOG Logger::print("VarExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    assert(e->var);

    if (e->cachedLvalue) {
      LLValue *V = e->cachedLvalue;
      result = new DVarValue(e->type, V);
      return;
    }

    result = DtoSymbolAddress(e->loc, e->type, e->var);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(IntegerExp *e) override {
    IF_LOG Logger::print("IntegerExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;
    LLConstant *c = toConstElem(e, p);
    result = new DConstValue(e->type, c);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(RealExp *e) override {
    IF_LOG Logger::print("RealExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;
    LLConstant *c = toConstElem(e, p);
    result = new DConstValue(e->type, c);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(NullExp *e) override {
    IF_LOG Logger::print("NullExp::toElem(type=%s): %s\n", e->type->toChars(),
                         e->toChars());
    LOG_SCOPE;
    LLConstant *c = toConstElem(e, p);
    result = new DNullValue(e->type, c);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(ComplexExp *e) override {
    IF_LOG Logger::print("ComplexExp::toElem(): %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;
    LLConstant *c = toConstElem(e, p);
    LLValue *res;

    if (c->isNullValue()) {
      switch (e->type->toBasetype()->ty) {
      default:
        llvm_unreachable("Unexpected complex floating point type");
      case Tcomplex32:
        c = DtoConstFP(Type::tfloat32, ldouble(0));
        break;
      case Tcomplex64:
        c = DtoConstFP(Type::tfloat64, ldouble(0));
        break;
      case Tcomplex80:
        c = DtoConstFP(Type::tfloat80, ldouble(0));
        break;
      }
      res = DtoAggrPair(DtoType(e->type), c, c);
    } else {
      res = DtoAggrPair(DtoType(e->type), c->getOperand(0), c->getOperand(1));
    }

    result = new DImValue(e->type, res);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(StringExp *e) override {
    IF_LOG Logger::print("StringExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    Type *dtype = e->type->toBasetype();
    Type *cty = dtype->nextOf()->toBasetype();

    LLType *ct = DtoMemType(cty);
    LLArrayType *at = LLArrayType::get(ct, e->len + 1);

    llvm::StringMap<llvm::GlobalVariable *> *stringLiteralCache = nullptr;
    LLConstant *_init;
    switch (cty->size()) {
    default:
      llvm_unreachable("Unknown char type");
    case 1:
      _init =
          toConstantArray(ct, at, static_cast<uint8_t *>(e->string), e->len);
      stringLiteralCache = &(gIR->stringLiteral1ByteCache);
      break;
    case 2:
      _init =
          toConstantArray(ct, at, static_cast<uint16_t *>(e->string), e->len);
      stringLiteralCache = &(gIR->stringLiteral2ByteCache);
      break;
    case 4:
      _init =
          toConstantArray(ct, at, static_cast<uint32_t *>(e->string), e->len);
      stringLiteralCache = &(gIR->stringLiteral4ByteCache);
      break;
    }

    llvm::StringRef key(e->toChars());
    llvm::GlobalVariable *gvar =
        (stringLiteralCache->find(key) == stringLiteralCache->end())
            ? nullptr
            : (*stringLiteralCache)[key];
    if (gvar == nullptr) {
      llvm::GlobalValue::LinkageTypes _linkage =
          llvm::GlobalValue::PrivateLinkage;
      IF_LOG {
        Logger::cout() << "type: " << *at << '\n';
        Logger::cout() << "init: " << *_init << '\n';
      }
      gvar = new llvm::GlobalVariable(gIR->module, at, true, _linkage, _init,
                                      ".str");
      gvar->setUnnamedAddr(true);
      (*stringLiteralCache)[key] = gvar;
    }

    llvm::ConstantInt *zero =
        LLConstantInt::get(LLType::getInt32Ty(gIR->context()), 0, false);
    LLConstant *idxs[2] = {zero, zero};
#if LDC_LLVM_VER >= 307
    LLConstant *arrptr = llvm::ConstantExpr::getGetElementPtr(
        isaPointer(gvar)->getElementType(), gvar, idxs, true);
#else
    LLConstant *arrptr = llvm::ConstantExpr::getGetElementPtr(gvar, idxs, true);
#endif

    if (dtype->ty == Tarray) {
      LLConstant *clen = LLConstantInt::get(DtoSize_t(), e->len, false);
      result = new DImValue(e->type, DtoConstSlice(clen, arrptr, dtype));
    } else if (dtype->ty == Tsarray) {
      LLType *dstType = getPtrToType(LLArrayType::get(ct, e->len));
      LLValue *emem =
          (gvar->getType() == dstType) ? gvar : DtoBitCast(gvar, dstType);
      result = new DVarValue(e->type, emem);
    } else if (dtype->ty == Tpointer) {
      result = new DImValue(e->type, arrptr);
    } else {
      llvm_unreachable("Unknown type for StringExp.");
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(AssignExp *e) override {
    IF_LOG Logger::print("AssignExp::toElem: %s | (%s)(%s = %s)\n",
                         e->toChars(), e->type->toChars(),
                         e->e1->type->toChars(),
                         e->e2->type ? e->e2->type->toChars() : nullptr);
    LOG_SCOPE;

    if (e->e1->op == TOKarraylength) {
      Logger::println("performing array.length assignment");
      ArrayLengthExp *ale = static_cast<ArrayLengthExp *>(e->e1);
      DValue *arr = toElem(ale->e1);
      DVarValue arrval(ale->e1->type, arr->getLVal());
      DValue *newlen = toElem(e->e2);
      DSliceValue *slice = DtoResizeDynArray(e->loc, arrval.getType(), &arrval,
                                             newlen->getRVal());
      DtoAssign(e->loc, &arrval, slice);
      result = newlen;
      return;
    }

    // Initialization of ref variable?
    // Can't just override ConstructExp::toElem because not all TOKconstruct
    // operations are actually instances of ConstructExp... Long live the DMD
    // coding style!
    if (e->op == TOKconstruct && e->e1->op == TOKvar && !(e->ismemset & 2)) {
      Declaration *d = static_cast<VarExp *>(e->e1)->var;
      if (d->storage_class & (STCref | STCout)) {
        Logger::println("performing ref variable initialization");
        // Note that the variable value is accessed directly (instead
        // of via getLVal(), which would perform a load from the
        // uninitialized location), and that rhs is stored as an l-value!
        DVarValue *lhs = toElem(e->e1)->isVar();
        assert(lhs);
        result = toElem(e->e2);

        // We shouldn't really need makeLValue() here, but the 2.063
        // frontend generates ref variables initialized from function
        // calls.
        DtoStore(makeLValue(e->loc, result), lhs->getRefStorage());

        return;
      }
    }

    if (e->e1->op == TOKslice) {
      // Check if this is an initialization of a static array with an array
      // literal that the frontend has foolishly rewritten into an
      // assignment of a dynamic array literal to a slice.
      Logger::println("performing static array literal assignment");
      SliceExp *const se = static_cast<SliceExp *>(e->e1);
      Type *const t2 = e->e2->type->toBasetype();
      Type *const ta = se->e1->type->toBasetype();

      if (se->lwr == nullptr && ta->ty == Tsarray &&
          e->e2->op == TOKarrayliteral &&
          e->op == TOKconstruct && // DMD Bugzilla 11238: avoid aliasing issue
          t2->nextOf()->mutableOf()->implicitConvTo(ta->nextOf())) {
        ArrayLiteralExp *const ale = static_cast<ArrayLiteralExp *>(e->e2);
        initializeArrayLiteral(p, ale, toElem(se->e1)->getLVal());
        result = toElem(e->e1);
        return;
      }
    }

    DValue *l = toElem(e->e1, true);

    // NRVO for object field initialization in constructor
    if (l->isVar() && e->op == TOKconstruct && e->e2->op == TOKcall) {
      CallExp *ce = static_cast<CallExp *>(e->e2);
      if (DtoIsReturnInArg(ce)) {
        DValue *fnval = toElem(ce->e1);
        LLValue *lval = l->getLVal();
        result = DtoCallFunction(ce->loc, ce->type, fnval, ce->arguments, lval);
        return;
      }
    }

    DValue *r = toElem(e->e2);

    if (e->e1->type->toBasetype()->ty == Tstruct && e->e2->op == TOKint64) {
      Logger::println("performing aggregate zero initialization");
      assert(e->e2->toInteger() == 0);
      DtoMemSetZero(l->getLVal());
      TypeStruct *ts = static_cast<TypeStruct *>(e->e1->type);
      if (ts->sym->isNested() && ts->sym->vthis) {
        DtoResolveNestedContext(e->loc, ts->sym, l->getLVal());
      }
      // Return value should be irrelevant.
      result = r;
      return;
    }

    // This matches the logic in AssignExp::semantic.
    // TODO: Should be cached in the frontend to avoid issues with the code
    // getting out of sync?
    bool lvalueElem = false;
    if ((e->e2->op == TOKslice &&
         static_cast<UnaExp *>(e->e2)->e1->isLvalue()) ||
        (e->e2->op == TOKcast &&
         static_cast<UnaExp *>(e->e2)->e1->isLvalue()) ||
        (e->e2->op != TOKslice && e->e2->isLvalue())) {
      lvalueElem = true;
    }

    Logger::println("performing normal assignment (rhs has lvalue elems = %d)",
                    lvalueElem);
    DtoAssign(e->loc, l, r, e->op, !lvalueElem);

    result = l;
  }

  //////////////////////////////////////////////////////////////////////////////

  template <typename BinExp, bool useLvalForBinExpLhs>
  static DValue *binAssign(BinAssignExp *e) {
    Loc loc = e->loc;

    // find the lhs' lvalue expression
    Expression *lvalExp = findLvalueExp(e->e1);
    if (!lvalExp) {
      e->error("expression %s does not mask any l-value", e->e1->toChars());
      fatal();
    }

    // pre-evaluate and cache the lvalue subexpression
    DValue *lval = nullptr;
    {
      IF_LOG Logger::println("Caching l-value of %s => %s", e->toChars(),
                             lvalExp->toChars());

      LOG_SCOPE;
      lval = toElemAndCacheLvalue(lvalExp);
    }

    // evaluate the underlying binary expression
    Expression *lhsForBinExp = (useLvalForBinExpLhs ? lvalExp : e->e1);
    BinExp binExp(loc, lhsForBinExp, e->e2);
    binExp.type = lhsForBinExp->type;
    DValue *result = toElem(&binExp);

    lvalExp->cachedLvalue = nullptr;

    // assign the (casted) result to lval
    DValue *assignedResult = DtoCast(loc, result, lval->type);
    DtoAssign(loc, lval, assignedResult);

    // return the (casted) result
    return e->type == assignedResult->type ? assignedResult
                                           : DtoCast(loc, result, e->type);
  }

#define BIN_ASSIGN(Op, useLvalForBinExpLhs)                                    \
  void visit(Op##AssignExp *e) override {                                      \
    IF_LOG Logger::print(#Op "AssignExp::toElem: %s @ %s\n", e->toChars(),     \
                         e->type->toChars());                                  \
    LOG_SCOPE;                                                                 \
    result = binAssign<Op##Exp, useLvalForBinExpLhs>(e);                       \
  }

  BIN_ASSIGN(Add, false)
  BIN_ASSIGN(Min, false)
  BIN_ASSIGN(Mul, false)
  BIN_ASSIGN(Div, false)
  BIN_ASSIGN(Mod, false)
  BIN_ASSIGN(And, false)
  BIN_ASSIGN(Or, false)
  BIN_ASSIGN(Xor, false)
  BIN_ASSIGN(Shl, true)
  BIN_ASSIGN(Shr, true)
  BIN_ASSIGN(Ushr, true)

#undef BIN_ASSIGN

  //////////////////////////////////////////////////////////////////////////////

  void errorOnIllegalArrayOp(Expression *base, Expression *e1, Expression *e2) {
    Type *t1 = e1->type->toBasetype();
    Type *t2 = e2->type->toBasetype();

    // valid array ops would have been transformed by optimize
    if ((t1->ty == Tarray || t1->ty == Tsarray) &&
        (t2->ty == Tarray || t2->ty == Tsarray)) {
      base->error("Array operation %s not recognized", base->toChars());
      fatal();
    }
  }

  /// Tries to remove a MulExp by a constant value of baseSize from e. Returns
  /// NULL if not possible.
  Expression *extractNoStrideInc(Expression *e, d_uns64 baseSize,
                                 bool &negate) {
    MulExp *mul;
    while (true) {
      if (e->op == TOKneg) {
        negate = !negate;
        e = static_cast<NegExp *>(e)->e1;
        continue;
      }

      if (e->op == TOKmul) {
        mul = static_cast<MulExp *>(e);
        break;
      }

      return nullptr;
    }

    if (!mul->e2->isConst()) {
      return nullptr;
    }
    dinteger_t stride = mul->e2->toInteger();

    if (stride != baseSize) {
      return nullptr;
    }

    return mul->e1;
  }

  DValue *emitPointerOffset(IRState *p, Loc loc, DValue *base,
                            Expression *offset, bool negateOffset,
                            Type *resultType) {
    // The operand emitted by the frontend is in units of bytes, and not
    // pointer elements. We try to undo this before resorting to
    // temporarily bitcasting the pointer to i8.

    llvm::Value *noStrideInc = nullptr;
    if (offset->isConst()) {
      dinteger_t byteOffset = offset->toInteger();
      if (byteOffset == 0) {
        Logger::println("offset is zero");
        return base;
      }
      noStrideInc = DtoConstSize_t(undoStrideMul(loc, base->type, byteOffset));
    } else if (Expression *inc = extractNoStrideInc(
                   offset, base->type->nextOf()->size(loc), negateOffset)) {
      noStrideInc = toElem(inc)->getRVal();
    }

    if (noStrideInc) {
      if (negateOffset) {
        noStrideInc = p->ir->CreateNeg(noStrideInc);
      }
      return new DImValue(base->type,
                          DtoGEP1(base->getRVal(), noStrideInc, false));
    }

    // This might not actually be generated by the frontend, just to be
    // safe.
    llvm::Value *inc = toElem(offset)->getRVal();
    if (negateOffset) {
      inc = p->ir->CreateNeg(inc);
    }
    llvm::Value *bytePtr = DtoBitCast(base->getRVal(), getVoidPtrType());
    DValue *result = new DImValue(Type::tvoidptr, DtoGEP1(bytePtr, inc, false));
    return DtoCast(loc, result, resultType);
  }

  void visit(AddExp *e) override {
    IF_LOG Logger::print("AddExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *l = toElem(e->e1);

    Type *t = e->type->toBasetype();
    Type *e1type = e->e1->type->toBasetype();
    Type *e2type = e->e2->type->toBasetype();

    errorOnIllegalArrayOp(e, e->e1, e->e2);

    if (e1type != e2type && e1type->ty == Tpointer && e2type->isintegral()) {
      Logger::println("Adding integer to pointer");
      result = emitPointerOffset(p, e->loc, l, e->e2, false, e->type);
    } else if (t->iscomplex()) {
      result = DtoComplexAdd(e->loc, e->type, l, toElem(e->e2));
    } else {
      result = DtoBinAdd(l, toElem(e->e2));
    }
  }

  void visit(MinExp *e) override {
    IF_LOG Logger::print("MinExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *l = toElem(e->e1);

    Type *t = e->type->toBasetype();
    Type *t1 = e->e1->type->toBasetype();
    Type *t2 = e->e2->type->toBasetype();

    errorOnIllegalArrayOp(e, e->e1, e->e2);

    if (t1->ty == Tpointer && t2->ty == Tpointer) {
      LLValue *lv = l->getRVal();
      LLValue *rv = toElem(e->e2)->getRVal();
      IF_LOG Logger::cout() << "lv: " << *lv << " rv: " << *rv << '\n';
      lv = p->ir->CreatePtrToInt(lv, DtoSize_t());
      rv = p->ir->CreatePtrToInt(rv, DtoSize_t());
      LLValue *diff = p->ir->CreateSub(lv, rv);
      if (diff->getType() != DtoType(e->type)) {
        diff = p->ir->CreateIntToPtr(diff, DtoType(e->type));
      }
      result = new DImValue(e->type, diff);
    } else if (t1->ty == Tpointer && t2->isintegral()) {
      Logger::println("Subtracting integer from pointer");
      result = emitPointerOffset(p, e->loc, l, e->e2, true, e->type);
    } else if (t->iscomplex()) {
      result = DtoComplexSub(e->loc, e->type, l, toElem(e->e2));
    } else {
      result = DtoBinSub(l, toElem(e->e2));
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(MulExp *e) override {
    IF_LOG Logger::print("MulExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *l = toElem(e->e1);
    DValue *r = toElem(e->e2);

    errorOnIllegalArrayOp(e, e->e1, e->e2);

    if (e->type->iscomplex()) {
      result = DtoComplexMul(e->loc, e->type, l, r);
    } else {
      result = DtoBinMul(e->type, l, r);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(DivExp *e) override {
    IF_LOG Logger::print("DivExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *l = toElem(e->e1);
    DValue *r = toElem(e->e2);

    errorOnIllegalArrayOp(e, e->e1, e->e2);

    if (e->type->iscomplex()) {
      result = DtoComplexDiv(e->loc, e->type, l, r);
    } else {
      result = DtoBinDiv(e->type, l, r);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(ModExp *e) override {
    IF_LOG Logger::print("ModExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *l = toElem(e->e1);
    DValue *r = toElem(e->e2);

    errorOnIllegalArrayOp(e, e->e1, e->e2);

    if (e->type->iscomplex()) {
      result = DtoComplexRem(e->loc, e->type, l, r);
    } else {
      result = DtoBinRem(e->type, l, r);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(CallExp *e) override {
    IF_LOG Logger::print("CallExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    if (e->cachedLvalue) {
      LLValue *V = e->cachedLvalue;
      result = new DVarValue(e->type, V);
      return;
    }

    // handle magic inline asm
    if (e->e1->op == TOKvar) {
      VarExp *ve = static_cast<VarExp *>(e->e1);
      if (FuncDeclaration *fd = ve->var->isFuncDeclaration()) {
        if (fd->llvmInternal == LLVMinline_asm) {
          result = DtoInlineAsmExpr(e->loc, fd, e->arguments);
          return;
        }
      }
    }

    // Check if we are about to construct a just declared temporary. DMD
    // unfortunately rewrites this as
    //   MyStruct(myArgs) => (MyStruct tmp; tmp).this(myArgs),
    // which would lead us to invoke the dtor even if the ctor throws. To
    // work around this, we hold on to the cleanup and push it only after
    // making the function call.
    //
    // The correct fix for this (DMD issue 13095) would have been to adapt
    // the AST, but we are stuck with this as DMD also patched over it with
    // a similar hack.
    VarDeclaration *delayedDtorVar = nullptr;
    Expression *delayedDtorExp = nullptr;
    if (e->f && e->f->isCtorDeclaration() && e->e1->op == TOKdotvar) {
      DotVarExp *dve = static_cast<DotVarExp *>(e->e1);
      if (dve->e1->op == TOKcomma) {
        CommaExp *ce = static_cast<CommaExp *>(dve->e1);
        if (ce->e1->op == TOKdeclaration && ce->e2->op == TOKvar) {
          VarExp *ve = static_cast<VarExp *>(ce->e2);
          if (VarDeclaration *vd = ve->var->isVarDeclaration()) {
            if (vd->edtor && !vd->noscope) {
              Logger::println("Delaying edtor");
              delayedDtorVar = vd;
              delayedDtorExp = vd->edtor;
              vd->edtor = nullptr;
            }
          }
        }
      }
    }

    // get the callee value
    DValue *fnval;
    if (e->directcall) {
      // TODO: Do this as an extra parameter to DotVarExp implementation.
      assert(e->e1->op == TOKdotvar);
      DotVarExp *dve = static_cast<DotVarExp *>(e->e1);
      FuncDeclaration *fdecl = dve->var->isFuncDeclaration();
      assert(fdecl);
      DtoResolveFunction(fdecl);
      fnval = new DFuncValue(fdecl, getIrFunc(fdecl)->func,
                             toElem(dve->e1)->getRVal());
    } else {
      fnval = toElem(e->e1);
    }

    // get func value if any
    DFuncValue *dfnval = fnval->isFunc();

    // handle magic intrinsics (mapping to instructions)
    if (dfnval && dfnval->func) {
      FuncDeclaration *fndecl = dfnval->func;

      // as requested by bearophile, see if it's a C printf call and that it's
      // valid.
      if (global.params.warnings && checkPrintf) {
        if (fndecl->linkage == LINKc &&
            strcmp(fndecl->ident->string, "printf") == 0) {
          warnInvalidPrintfCall(e->loc, (*e->arguments)[0], e->arguments->dim);
        }
      }

      if (DtoLowerMagicIntrinsic(p, fndecl, e, result)) {
        return;
      }
    }

    result = DtoCallFunction(e->loc, e->type, fnval, e->arguments);

    if (delayedDtorVar) {
      delayedDtorVar->edtor = delayedDtorExp;
      pushVarDtorCleanup(p, delayedDtorVar);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(CastExp *e) override {
    IF_LOG Logger::print("CastExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    // get the value to cast
    DValue *u = toElem(e->e1);

    // handle cast to void (usually created by frontend to avoid "has no effect"
    // error)
    if (e->to == Type::tvoid) {
      result = new DImValue(Type::tvoid,
                            llvm::UndefValue::get(DtoMemType(Type::tvoid)));
      return;
    }

    // cast it to the 'to' type, if necessary
    result = u;
    if (!e->to->equals(e->e1->type)) {
      result = DtoCast(e->loc, u, e->to);
    }

    // paint the type, if necessary
    if (!e->type->equals(e->to)) {
      result = DtoPaintType(e->loc, result, e->type);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(SymOffExp *e) override {
    IF_LOG Logger::print("SymOffExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *base = DtoSymbolAddress(e->loc, e->var->type, e->var);

    // This weird setup is required to be able to handle both variables as
    // well as functions and TypeInfo references (which are not a DVarValue
    // as well due to the level-of-indirection hack in Type::getTypeInfo that
    // is unfortunately required by the frontend).
    llvm::Value *baseValue;
    if (base->isLVal()) {
      baseValue = base->getLVal();
    } else {
      baseValue = base->getRVal();
    }
    assert(isaPointer(baseValue));

    llvm::Value *offsetValue;
    Type *offsetType;

    if (e->offset == 0) {
      offsetValue = baseValue;
      offsetType = base->type->pointerTo();
    } else {
      uint64_t elemSize = gDataLayout->getTypeAllocSize(
          baseValue->getType()->getContainedType(0));
      if (e->offset % elemSize == 0) {
        // We can turn this into a "nice" GEP.
        offsetValue = DtoGEPi1(baseValue, e->offset / elemSize);
        offsetType = base->type->pointerTo();
      } else {
        // Offset isn't a multiple of base type size, just cast to i8* and
        // apply the byte offset.
        offsetValue =
            DtoGEPi1(DtoBitCast(baseValue, getVoidPtrType()), e->offset);
        offsetType = Type::tvoidptr;
      }
    }

    // Casts are also "optimized into" SymOffExp by the frontend.
    result = DtoCast(e->loc, new DImValue(offsetType, offsetValue), e->type);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(AddrExp *e) override {
    IF_LOG Logger::println("AddrExp::toElem: %s @ %s", e->toChars(),
                           e->type->toChars());
    LOG_SCOPE;

    // The address of a StructLiteralExp can in fact be a global variable, check
    // for that instead of re-codegening the literal.
    if (e->e1->op == TOKstructliteral) {
      // lvalue literal must be a global, hence we can just use
      // toConstElem on the AddrExp to get the address.
      LLConstant *addr = toConstElem(e, p);
      IF_LOG Logger::cout()
          << "returning address of struct literal global: " << addr << '\n';
      result = new DImValue(e->type, DtoBitCast(addr, DtoType(e->type)));
      return;
    }

    DValue *v = toElem(e->e1, true);
    if (v->isField()) {
      Logger::println("is field");
      result = v;
      return;
    }
    if (DFuncValue *fv = v->isFunc()) {
      Logger::println("is func");
      // Logger::println("FuncDeclaration");
      FuncDeclaration *fd = fv->func;
      assert(fd);
      DtoResolveFunction(fd);
      result = new DFuncValue(fd, getIrFunc(fd)->func);
      return;
    }
    if (v->isIm()) {
      Logger::println("is immediate");
      result = v;
      return;
    }
    Logger::println("is nothing special");

    // we special case here, since apparently taking the address of a slice is
    // ok
    LLValue *lval;
    if (v->isLVal()) {
      lval = v->getLVal();
    } else {
      assert(v->isSlice());
      lval = DtoAllocaDump(v, ".tmp_slice_storage");
    }

    IF_LOG Logger::cout() << "lval: " << *lval << '\n';
    result = new DImValue(e->type, DtoBitCast(lval, DtoType(e->type)));
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(PtrExp *e) override {
    IF_LOG Logger::println("PtrExp::toElem: %s @ %s", e->toChars(),
                           e->type->toChars());
    LOG_SCOPE;

    // function pointers are special
    if (e->type->toBasetype()->ty == Tfunction) {
      assert(!e->cachedLvalue);
      DValue *dv = toElem(e->e1);
      if (DFuncValue *dfv = dv->isFunc()) {
        result = new DFuncValue(e->type, dfv->func, dfv->getRVal());
      } else {
        result = new DImValue(e->type, dv->getRVal());
      }
      return;
    }

    // get the rvalue and return it as an lvalue
    LLValue *V;
    if (e->cachedLvalue) {
      V = e->cachedLvalue;
    } else {
      V = toElem(e->e1)->getRVal();
    }

    // The frontend emits dereferences of class/interfaces types to access the
    // first member, which is the .classinfo property.
    Type *origType = e->e1->type->toBasetype();
    if (origType->ty == Tclass) {
      TypeClass *ct = static_cast<TypeClass *>(origType);

      Type *resultType;
      if (ct->sym->isInterfaceDeclaration()) {
        // For interfaces, the first entry in the vtbl is actually a pointer
        // to an Interface instance, which has the type info as its first
        // member, so we have to add an extra layer of indirection.
        resultType = Type::typeinfointerface->type->pointerTo();
      } else {
        resultType = Type::typeinfointerface->type;
      }

      V = DtoBitCast(V, DtoType(resultType->pointerTo()->pointerTo()));
    }

    result = new DVarValue(e->type, V);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(DotVarExp *e) override {
    IF_LOG Logger::print("DotVarExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    if (e->cachedLvalue) {
      Logger::println("using cached lvalue");
      LLValue *V = e->cachedLvalue;
      VarDeclaration *vd = e->var->isVarDeclaration();
      assert(vd);
      result = new DVarValue(e->type, vd, V);
      return;
    }

    DValue *l = toElem(e->e1);

    Type *e1type = e->e1->type->toBasetype();

    // Logger::println("e1type=%s", e1type->toChars());
    // Logger::cout() << *DtoType(e1type) << '\n';

    if (VarDeclaration *vd = e->var->isVarDeclaration()) {
      LLValue *arrptr;
      // indexing struct pointer
      if (e1type->ty == Tpointer) {
        assert(e1type->nextOf()->ty == Tstruct);
        TypeStruct *ts = static_cast<TypeStruct *>(e1type->nextOf());
        arrptr = DtoIndexAggregate(l->getRVal(), ts->sym, vd);
      }
      // indexing normal struct
      else if (e1type->ty == Tstruct) {
        TypeStruct *ts = static_cast<TypeStruct *>(e1type);
        arrptr = DtoIndexAggregate(l->getRVal(), ts->sym, vd);
      }
      // indexing class
      else if (e1type->ty == Tclass) {
        TypeClass *tc = static_cast<TypeClass *>(e1type);
        arrptr = DtoIndexAggregate(l->getRVal(), tc->sym, vd);
      } else {
        llvm_unreachable("Unknown DotVarExp type for VarDeclaration.");
      }

      // Logger::cout() << "mem: " << *arrptr << '\n';
      result = new DVarValue(e->type, vd, arrptr);
    } else if (FuncDeclaration *fdecl = e->var->isFuncDeclaration()) {
      DtoResolveFunction(fdecl);

      // This is a bit more convoluted than it would need to be, because it
      // has to take templated interface methods into account, for which
      // isFinalFunc is not necessarily true.
      // Also, private methods are always not virtual.
      const bool nonFinal = !fdecl->isFinalFunc() &&
                            (fdecl->isAbstract() || fdecl->isVirtual()) &&
                            fdecl->prot().kind != PROTprivate;

      // Get the actual function value to call.
      LLValue *funcval = nullptr;
      if (nonFinal) {
        DImValue thisVal(e1type, l->getRVal());
        funcval = DtoVirtualFunctionPointer(&thisVal, fdecl, e->toChars());
      } else {
        funcval = getIrFunc(fdecl)->func;
      }
      assert(funcval);

      result = new DFuncValue(fdecl, funcval, l->getRVal());
    } else {
      llvm_unreachable("Unknown target for VarDeclaration.");
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(ThisExp *e) override {
    IF_LOG Logger::print("ThisExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    // special cases: `this(int) { this(); }` and `this(int) { super(); }`
    if (!e->var) {
      Logger::println("this exp without var declaration");
      LLValue *v = p->func()->thisArg;
      result = new DVarValue(e->type, v);
      return;
    }
    // regular this expr
    if (VarDeclaration *vd = e->var->isVarDeclaration()) {
      LLValue *v;
      Dsymbol *vdparent = vd->toParent2();
      Identifier *ident = p->func()->decl->ident;
      // In D1, contracts are treated as normal nested methods, 'this' is
      // just passed in the context struct along with any used parameters.
      if (ident == Id::ensure || ident == Id::require) {
        Logger::println("contract this exp");
        v = p->func()->nestArg;
        v = DtoBitCast(v, DtoType(e->type)->getPointerTo());
      } else if (vdparent != p->func()->decl) {
        Logger::println("nested this exp");
        result = DtoNestedVariable(e->loc, e->type, vd, e->type->ty == Tstruct);
        return;
      } else {
        Logger::println("normal this exp");
        v = p->func()->thisArg;
      }
      result = new DVarValue(e->type, vd, v);
    } else {
      llvm_unreachable("No VarDeclaration in ThisExp.");
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(IndexExp *e) override {
    IF_LOG Logger::print("IndexExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    if (e->cachedLvalue) {
      LLValue *V = e->cachedLvalue;
      result = new DVarValue(e->type, V);
      return;
    }

    DValue *l = toElem(e->e1);

    Type *e1type = e->e1->type->toBasetype();

    p->arrays.push_back(l); // if $ is used it must be an array so this is fine.
    DValue *r = toElem(e->e2);
    p->arrays.pop_back();

    LLValue *arrptr = nullptr;
    if (e1type->ty == Tpointer) {
      arrptr = DtoGEP1(l->getRVal(), r->getRVal(), false);
    } else if (e1type->ty == Tsarray) {
      if (p->emitArrayBoundsChecks() && !e->indexIsInBounds) {
        DtoIndexBoundsCheck(e->loc, l, r);
      }
      arrptr = DtoGEP(l->getRVal(), DtoConstUint(0), r->getRVal(),
                      e->indexIsInBounds);
    } else if (e1type->ty == Tarray) {
      if (p->emitArrayBoundsChecks() && !e->indexIsInBounds) {
        DtoIndexBoundsCheck(e->loc, l, r);
      }
      arrptr = DtoGEP1(DtoArrayPtr(l), r->getRVal(), e->indexIsInBounds);
    } else if (e1type->ty == Taarray) {
      result = DtoAAIndex(e->loc, e->type, l, r, e->modifiable);
      return;
    } else {
      IF_LOG Logger::println("e1type: %s", e1type->toChars());
      llvm_unreachable("Unknown IndexExp target.");
    }
    result = new DVarValue(e->type, arrptr);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(SliceExp *e) override {
    IF_LOG Logger::print("SliceExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    // this is the new slicing code, it's different in that a full slice will no
    // longer retain the original pointer.
    // but this was broken if there *was* no original pointer, ie. a slice of a
    // slice...
    // now all slices have *both* the 'len' and 'ptr' fields set to != null.

    // value being sliced
    LLValue *elen = nullptr;
    LLValue *eptr;
    DValue *v = toElem(e->e1);

    // handle pointer slicing
    Type *etype = e->e1->type->toBasetype();
    if (etype->ty == Tpointer) {
      assert(e->lwr);
      eptr = v->getRVal();
    }
    // array slice
    else {
      eptr = DtoArrayPtr(v);
    }

    // has lower bound, pointer needs adjustment
    if (e->lwr) {
      // must have upper bound too then
      assert(e->upr);

      // get bounds (make sure $ works)
      p->arrays.push_back(v);
      DValue *lo = toElem(e->lwr);
      DValue *up = toElem(e->upr);
      p->arrays.pop_back();
      LLValue *vlo = lo->getRVal();
      LLValue *vup = up->getRVal();

      const bool needCheckUpper =
          (etype->ty != Tpointer) && !e->upperIsInBounds;
      const bool needCheckLower = !e->lowerIsLessThanUpper;
      if (p->emitArrayBoundsChecks() && (needCheckUpper || needCheckLower)) {
        llvm::BasicBlock *failbb =
            llvm::BasicBlock::Create(p->context(), "bounds.fail", p->topfunc());
        llvm::BasicBlock *okbb =
            llvm::BasicBlock::Create(p->context(), "bounds.ok", p->topfunc());

        llvm::Value *okCond = nullptr;
        if (needCheckUpper) {
          okCond = p->ir->CreateICmp(llvm::ICmpInst::ICMP_ULE, vup,
                                     DtoArrayLen(v), "bounds.cmp.lo");
        }

        if (needCheckLower) {
          llvm::Value *cmp = p->ir->CreateICmp(llvm::ICmpInst::ICMP_ULE, vlo,
                                               vup, "bounds.cmp.up");
          if (okCond) {
            okCond = p->ir->CreateAnd(okCond, cmp);
          } else {
            okCond = cmp;
          }
        }

        p->ir->CreateCondBr(okCond, okbb, failbb);

        p->scope() = IRScope(failbb);
        DtoBoundsCheckFailCall(p, e->loc);

        p->scope() = IRScope(okbb);
      }

      // offset by lower
      eptr = DtoGEP1(eptr, vlo, !needCheckLower, "lowerbound");

      // adjust length
      elen = p->ir->CreateSub(vup, vlo);
    }
    // no bounds or full slice -> just convert to slice
    else {
      assert(e->e1->type->toBasetype()->ty != Tpointer);
      // if the sliceee is a static array, we use the length of that as DMD
      // seems
      // to give contrary inconsistent sizesin some multidimensional static
      // array cases.
      // (namely default initialization, int[16][16] arr; -> int[256] arr = 0;)
      if (etype->ty == Tsarray) {
        TypeSArray *tsa = static_cast<TypeSArray *>(etype);
        elen = DtoConstSize_t(tsa->dim->toUInteger());

        // in this case, we also need to make sure the pointer is cast to the
        // innermost element type
        eptr = DtoBitCast(eptr, DtoType(tsa->nextOf()->pointerTo()));
      }
    }

    // The frontend generates a SliceExp of static array type when assigning a
    // fixed-width slice to a static array.
    if (e->type->toBasetype()->ty == Tsarray) {
      LLValue *v = DtoBitCast(eptr, DtoType(e->type->pointerTo()));
      result = new DVarValue(e->type, v);
      return;
    }

    if (!elen) {
      elen = DtoArrayLen(v);
    }
    result = new DSliceValue(e->type, elen, eptr);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(CmpExp *e) override {
    IF_LOG Logger::print("CmpExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *l = toElem(e->e1);
    DValue *r = toElem(e->e2);

    Type *t = e->e1->type->toBasetype();

    LLValue *eval = nullptr;

    if (t->isintegral() || t->ty == Tpointer || t->ty == Tnull) {
      llvm::ICmpInst::Predicate icmpPred;
      tokToIcmpPred(e->op, isLLVMUnsigned(t), &icmpPred, &eval);

      if (!eval) {
        LLValue *a = l->getRVal();
        LLValue *b = r->getRVal();
        IF_LOG {
          Logger::cout() << "type 1: " << *a << '\n';
          Logger::cout() << "type 2: " << *b << '\n';
        }
        if (a->getType() != b->getType()) {
          b = DtoBitCast(b, a->getType());
        }
        eval = p->ir->CreateICmp(icmpPred, a, b);
      }
    } else if (t->isfloating()) {
      llvm::FCmpInst::Predicate cmpop;
      switch (e->op) {
      case TOKlt:
        cmpop = llvm::FCmpInst::FCMP_OLT;
        break;
      case TOKle:
        cmpop = llvm::FCmpInst::FCMP_OLE;
        break;
      case TOKgt:
        cmpop = llvm::FCmpInst::FCMP_OGT;
        break;
      case TOKge:
        cmpop = llvm::FCmpInst::FCMP_OGE;
        break;
      case TOKunord:
        cmpop = llvm::FCmpInst::FCMP_UNO;
        break;
      case TOKule:
        cmpop = llvm::FCmpInst::FCMP_ULE;
        break;
      case TOKul:
        cmpop = llvm::FCmpInst::FCMP_ULT;
        break;
      case TOKuge:
        cmpop = llvm::FCmpInst::FCMP_UGE;
        break;
      case TOKug:
        cmpop = llvm::FCmpInst::FCMP_UGT;
        break;
      case TOKue:
        cmpop = llvm::FCmpInst::FCMP_UEQ;
        break;
      case TOKlg:
        cmpop = llvm::FCmpInst::FCMP_ONE;
        break;
      case TOKleg:
        cmpop = llvm::FCmpInst::FCMP_ORD;
        break;

      default:
        llvm_unreachable("Unsupported floating point comparison operator.");
      }
      eval = p->ir->CreateFCmp(cmpop, l->getRVal(), r->getRVal());
    } else if (t->ty == Tsarray || t->ty == Tarray) {
      Logger::println("static or dynamic array");
      eval = DtoArrayCompare(e->loc, e->op, l, r);
    } else if (t->ty == Taarray) {
      eval = LLConstantInt::getFalse(gIR->context());
    } else if (t->ty == Tdelegate) {
      llvm::ICmpInst::Predicate icmpPred;
      tokToIcmpPred(e->op, isLLVMUnsigned(t), &icmpPred, &eval);

      if (!eval) {
        // First compare the function pointers, then the context ones. This is
        // what DMD does.
        llvm::Value *lhs = l->getRVal();
        llvm::Value *rhs = r->getRVal();

        llvm::BasicBlock *fptreq =
            llvm::BasicBlock::Create(gIR->context(), "fptreq", gIR->topfunc());
        llvm::BasicBlock *fptrneq =
            llvm::BasicBlock::Create(gIR->context(), "fptrneq", gIR->topfunc());
        llvm::BasicBlock *dgcmpend = llvm::BasicBlock::Create(
            gIR->context(), "dgcmpend", gIR->topfunc());

        llvm::Value *lfptr = p->ir->CreateExtractValue(lhs, 1, ".lfptr");
        llvm::Value *rfptr = p->ir->CreateExtractValue(rhs, 1, ".rfptr");

        llvm::Value *fptreqcmp = p->ir->CreateICmp(llvm::ICmpInst::ICMP_EQ,
                                                   lfptr, rfptr, ".fptreqcmp");
        llvm::BranchInst::Create(fptreq, fptrneq, fptreqcmp, p->scopebb());

        p->scope() = IRScope(fptreq);
        llvm::Value *lctx = p->ir->CreateExtractValue(lhs, 0, ".lctx");
        llvm::Value *rctx = p->ir->CreateExtractValue(rhs, 0, ".rctx");
        llvm::Value *ctxcmp =
            p->ir->CreateICmp(icmpPred, lctx, rctx, ".ctxcmp");
        llvm::BranchInst::Create(dgcmpend, p->scopebb());

        p->scope() = IRScope(fptrneq);
        llvm::Value *fptrcmp =
            p->ir->CreateICmp(icmpPred, lfptr, rfptr, ".fptrcmp");
        llvm::BranchInst::Create(dgcmpend, p->scopebb());

        p->scope() = IRScope(dgcmpend);
        llvm::PHINode *phi = p->ir->CreatePHI(ctxcmp->getType(), 2, ".dgcmp");
        phi->addIncoming(ctxcmp, fptreq);
        phi->addIncoming(fptrcmp, fptrneq);
        eval = phi;
      }
    } else {
      llvm_unreachable("Unsupported CmpExp type");
    }

    result = new DImValue(e->type, eval);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(EqualExp *e) override {
    IF_LOG Logger::print("EqualExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *l = toElem(e->e1);
    DValue *r = toElem(e->e2);
    LLValue *lv = l->getRVal();
    LLValue *rv = r->getRVal();

    Type *t = e->e1->type->toBasetype();

    LLValue *eval = nullptr;

    // the Tclass catches interface comparisons, regular
    // class equality should be rewritten as a.opEquals(b) by this time
    if (t->isintegral() || t->ty == Tpointer || t->ty == Tclass ||
        t->ty == Tnull) {
      Logger::println("integral or pointer or interface");
      llvm::ICmpInst::Predicate cmpop;
      switch (e->op) {
      case TOKequal:
        cmpop = llvm::ICmpInst::ICMP_EQ;
        break;
      case TOKnotequal:
        cmpop = llvm::ICmpInst::ICMP_NE;
        break;
      default:
        llvm_unreachable("Unsupported integral type equality comparison.");
      }
      if (rv->getType() != lv->getType()) {
        rv = DtoBitCast(rv, lv->getType());
      }
      IF_LOG {
        Logger::cout() << "lv: " << *lv << '\n';
        Logger::cout() << "rv: " << *rv << '\n';
      }
      eval = p->ir->CreateICmp(cmpop, lv, rv);
    } else if (t->isfloating()) // includes iscomplex
    {
      eval = DtoBinNumericEquals(e->loc, l, r, e->op);
    } else if (t->ty == Tsarray || t->ty == Tarray) {
      Logger::println("static or dynamic array");
      eval = DtoArrayEquals(e->loc, e->op, l, r);
    } else if (t->ty == Taarray) {
      Logger::println("associative array");
      eval = DtoAAEquals(e->loc, e->op, l, r);
    } else if (t->ty == Tdelegate) {
      Logger::println("delegate");
      eval = DtoDelegateEquals(e->op, l->getRVal(), r->getRVal());
    } else if (t->ty == Tstruct) {
      Logger::println("struct");
      // when this is reached it means there is no opEquals overload.
      eval = DtoStructEquals(e->op, l, r);
    } else {
      llvm_unreachable("Unsupported EqualExp type.");
    }

    result = new DImValue(e->type, eval);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(PostExp *e) override {
    IF_LOG Logger::print("PostExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *l = toElem(e->e1);
    toElem(e->e2);

    LLValue *val = l->getRVal();
    LLValue *post = nullptr;

    Type *e1type = e->e1->type->toBasetype();
    Type *e2type = e->e2->type->toBasetype();

    if (e1type->isintegral()) {
      assert(e2type->isintegral());
      LLValue *one =
          LLConstantInt::get(val->getType(), 1, !e2type->isunsigned());
      if (e->op == TOKplusplus) {
        post = llvm::BinaryOperator::CreateAdd(val, one, "", p->scopebb());
      } else if (e->op == TOKminusminus) {
        post = llvm::BinaryOperator::CreateSub(val, one, "", p->scopebb());
      }
    } else if (e1type->ty == Tpointer) {
      assert(e->e2->op == TOKint64);
      LLConstant *offset =
          e->op == TOKplusplus ? DtoConstUint(1) : DtoConstInt(-1);
      post = DtoGEP1(val, offset, false, "", p->scopebb());
    } else if (e1type->isfloating()) {
      assert(e2type->isfloating());
      LLValue *one = DtoConstFP(e1type, ldouble(1.0));
      if (e->op == TOKplusplus) {
        post = llvm::BinaryOperator::CreateFAdd(val, one, "", p->scopebb());
      } else if (e->op == TOKminusminus) {
        post = llvm::BinaryOperator::CreateFSub(val, one, "", p->scopebb());
      }
    } else {
      assert(post);
    }

    DtoStore(post, l->getLVal());
    result = new DImValue(e->type, val);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(NewExp *e) override {
    IF_LOG Logger::print("NewExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    bool isArgprefixHandled = false;

    assert(e->newtype);
    Type *ntype = e->newtype->toBasetype();

    // new class
    if (ntype->ty == Tclass) {
      Logger::println("new class");
      result = DtoNewClass(e->loc, static_cast<TypeClass *>(ntype), e);
      isArgprefixHandled = true; // by DtoNewClass()
    }
    // new dynamic array
    else if (ntype->ty == Tarray) {
      IF_LOG Logger::println("new dynamic array: %s", e->newtype->toChars());
      assert(e->argprefix == NULL);
      // get dim
      assert(e->arguments);
      assert(e->arguments->dim >= 1);
      if (e->arguments->dim == 1) {
        DValue *sz = toElem((*e->arguments)[0]);
        // allocate & init
        result = DtoNewDynArray(e->loc, e->newtype, sz, true);
      } else {
        size_t ndims = e->arguments->dim;
        std::vector<DValue *> dims;
        dims.reserve(ndims);
        for (size_t i = 0; i < ndims; ++i) {
          dims.push_back(toElem((*e->arguments)[i]));
        }
        result = DtoNewMulDimDynArray(e->loc, e->newtype, &dims[0], ndims);
      }
    }
    // new static array
    else if (ntype->ty == Tsarray) {
      llvm_unreachable("Static array new should decay to dynamic array.");
    }
    // new struct
    else if (ntype->ty == Tstruct) {
      IF_LOG Logger::println("new struct on heap: %s\n", e->newtype->toChars());

      TypeStruct *ts = static_cast<TypeStruct *>(ntype);

      // allocate
      LLValue *mem = nullptr;
      if (e->allocator) {
        // custom allocator
        DtoResolveFunction(e->allocator);
        DFuncValue dfn(e->allocator, getIrFunc(e->allocator)->func);
        DValue *res = DtoCallFunction(e->loc, nullptr, &dfn, e->newargs);
        mem = DtoBitCast(res->getRVal(), DtoType(ntype->pointerTo()),
                         ".newstruct_custom");
      } else {
        // default allocator
        mem = DtoNewStruct(e->loc, ts);
      }

      if (!e->member && e->arguments) {
        IF_LOG Logger::println("Constructing using literal");
        write_struct_literal(e->loc, mem, ts->sym, e->arguments);
      } else {
        // set nested context
        if (ts->sym->isNested() && ts->sym->vthis) {
          DtoResolveNestedContext(e->loc, ts->sym, mem);
        }

        // call constructor
        if (e->member) {
          // evaluate argprefix
          if (e->argprefix) {
            toElemDtor(e->argprefix);
            isArgprefixHandled = true;
          }

          IF_LOG Logger::println("Calling constructor");
          assert(e->arguments != NULL);
          DtoResolveFunction(e->member);
          DFuncValue dfn(e->member, getIrFunc(e->member)->func, mem);
          DtoCallFunction(e->loc, ts, &dfn, e->arguments);
        }
      }

      result = new DImValue(e->type, mem);
    }
    // new basic type
    else {
      IF_LOG Logger::println("basic type on heap: %s\n", e->newtype->toChars());
      assert(e->argprefix == NULL);

      // allocate
      LLValue *mem = DtoNew(e->loc, e->newtype);
      DVarValue tmpvar(e->newtype, mem);

      Expression *exp = nullptr;
      if (!e->arguments || e->arguments->dim == 0) {
        IF_LOG Logger::println("default initializer\n");
        // static arrays never appear here, so using the defaultInit is ok!
        exp = e->newtype->defaultInit(e->loc);
      } else {
        IF_LOG Logger::println("uniform constructor\n");
        assert(e->arguments->dim == 1);
        exp = (*e->arguments)[0];
      }

      DValue *iv = toElem(exp);
      DtoAssign(e->loc, &tmpvar, iv);

      // return as pointer-to
      result = new DImValue(e->type, mem);
    }

    assert(e->argprefix == NULL || isArgprefixHandled);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(DeleteExp *e) override {
    IF_LOG Logger::print("DeleteExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *dval = toElem(e->e1);
    Type *et = e->e1->type->toBasetype();

    // pointer
    if (et->ty == Tpointer) {
      Type *elementType = et->nextOf()->toBasetype();
      if (elementType->ty == Tstruct && elementType->needsDestruction()) {
        DtoDeleteStruct(e->loc, dval);
      } else {
        DtoDeleteMemory(e->loc, dval);
      }
    }
    // class
    else if (et->ty == Tclass) {
      bool onstack = false;
      TypeClass *tc = static_cast<TypeClass *>(et);
      if (tc->sym->isInterfaceDeclaration()) {
        DtoDeleteInterface(e->loc, dval);
        onstack = true;
      } else if (DVarValue *vv = dval->isVar()) {
        if (vv->var && vv->var->onstack) {
          DtoFinalizeClass(e->loc, dval->getRVal());
          onstack = true;
        }
      }

      if (!onstack) {
        DtoDeleteClass(e->loc, dval); // sets dval to null
      } else if (dval->isVar()) {
        LLValue *lval = dval->getLVal();
        DtoStore(LLConstant::getNullValue(lval->getType()->getContainedType(0)),
                 lval);
      }
    }
    // dyn array
    else if (et->ty == Tarray) {
      DtoDeleteArray(e->loc, dval);
      if (dval->isLVal()) {
        DtoSetArrayToNull(dval->getLVal());
      }
    }
    // unknown/invalid
    else {
      llvm_unreachable("Unsupported DeleteExp target.");
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(ArrayLengthExp *e) override {
    IF_LOG Logger::print("ArrayLengthExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *u = toElem(e->e1);
    result = new DImValue(e->type, DtoArrayLen(u));
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(AssertExp *e) override {
    IF_LOG Logger::print("AssertExp::toElem: %s\n", e->toChars());
    LOG_SCOPE;

    // DMD allows syntax like this:
    // f() == 0 || assert(false)
    result = new DImValue(e->type, DtoConstBool(false));

    if (!global.params.useAssert) {
      return;
    }

    // condition
    DValue *cond;
    Type *condty;

    // special case for dmd generated assert(this); when not in -release mode
    if (e->e1->op == TOKthis && static_cast<ThisExp *>(e->e1)->var == nullptr) {
      LLValue *thisarg = p->func()->thisArg;
      assert(thisarg && "null thisarg, but we're in assert(this) exp;");
      LLValue *thisptr = DtoLoad(thisarg);
      condty = e->e1->type->toBasetype();
      cond = new DImValue(condty, thisptr);
    } else {
      cond = toElem(e->e1);
      condty = e->e1->type->toBasetype();
    }

    // create basic blocks
    llvm::BasicBlock *passedbb =
        llvm::BasicBlock::Create(gIR->context(), "assertPassed", p->topfunc());
    llvm::BasicBlock *failedbb =
        llvm::BasicBlock::Create(gIR->context(), "assertFailed", p->topfunc());

    // test condition
    LLValue *condval = DtoCast(e->loc, cond, Type::tbool)->getRVal();

    // branch
    llvm::BranchInst::Create(passedbb, failedbb, condval, p->scopebb());

    // failed: call assert runtime function
    p->scope() = IRScope(failedbb);

    /* DMD Bugzilla 8360: If the condition is evaluated to true,
     * msg is not evaluated at all. So should use toElemDtor()
     * instead of toElem().
     */
    DtoAssert(p->func()->decl->getModule(), e->loc,
              e->msg ? toElemDtor(e->msg) : nullptr);

    // passed:
    p->scope() = IRScope(passedbb);

    FuncDeclaration *invdecl;
    // class invariants
    if (global.params.useInvariants && condty->ty == Tclass &&
        !(static_cast<TypeClass *>(condty)->sym->isInterfaceDeclaration()) &&
        !(static_cast<TypeClass *>(condty)->sym->isCPPclass())) {
      Logger::println("calling class invariant");
      llvm::Function *fn = getRuntimeFunction(
          e->loc, gIR->module,
          gABI->mangleForLLVM("_D9invariant12_d_invariantFC6ObjectZv", LINKd)
              .c_str());
      LLValue *arg =
          DtoBitCast(cond->getRVal(), fn->getFunctionType()->getParamType(0));
      gIR->CreateCallOrInvoke(fn, arg);
    }
    // struct invariants
    else if (global.params.useInvariants && condty->ty == Tpointer &&
             condty->nextOf()->ty == Tstruct &&
             (invdecl = static_cast<TypeStruct *>(condty->nextOf())
                            ->sym->inv) != nullptr) {
      Logger::print("calling struct invariant");
      DtoResolveFunction(invdecl);
      DFuncValue invfunc(invdecl, getIrFunc(invdecl)->func, cond->getRVal());
      DtoCallFunction(e->loc, nullptr, &invfunc, nullptr);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(NotExp *e) override {
    IF_LOG Logger::print("NotExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *u = toElem(e->e1);

    LLValue *b = DtoCast(e->loc, u, Type::tbool)->getRVal();

    LLConstant *zero = DtoConstBool(false);
    b = p->ir->CreateICmpEQ(b, zero);

    result = new DImValue(e->type, b);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(AndAndExp *e) override {
    IF_LOG Logger::print("AndAndExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *u = toElem(e->e1);

    llvm::BasicBlock *andand =
        llvm::BasicBlock::Create(gIR->context(), "andand", gIR->topfunc());
    llvm::BasicBlock *andandend =
        llvm::BasicBlock::Create(gIR->context(), "andandend", gIR->topfunc());

    LLValue *ubool = DtoCast(e->loc, u, Type::tbool)->getRVal();

    llvm::BasicBlock *oldblock = p->scopebb();
    llvm::BranchInst::Create(andand, andandend, ubool, p->scopebb());

    p->scope() = IRScope(andand);
    emitCoverageLinecountInc(e->e2->loc);
    DValue *v = toElemDtor(e->e2);

    LLValue *vbool = nullptr;
    if (v && !v->isFunc() && v->getType() != Type::tvoid) {
      vbool = DtoCast(e->loc, v, Type::tbool)->getRVal();
    }

    llvm::BasicBlock *newblock = p->scopebb();
    llvm::BranchInst::Create(andandend, p->scopebb());
    p->scope() = IRScope(andandend);

    LLValue *resval = nullptr;
    if (ubool == vbool || !vbool) {
      // No need to create a PHI node.
      resval = ubool;
    } else {
      llvm::PHINode *phi =
          p->ir->CreatePHI(LLType::getInt1Ty(gIR->context()), 2, "andandval");
      // If we jumped over evaluation of the right-hand side,
      // the result is false. Otherwise it's the value of the right-hand side.
      phi->addIncoming(LLConstantInt::getFalse(gIR->context()), oldblock);
      phi->addIncoming(vbool, newblock);
      resval = phi;
    }

    result = new DImValue(e->type, resval);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(OrOrExp *e) override {
    IF_LOG Logger::print("OrOrExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *u = toElem(e->e1);

    llvm::BasicBlock *oror =
        llvm::BasicBlock::Create(gIR->context(), "oror", gIR->topfunc());
    llvm::BasicBlock *ororend =
        llvm::BasicBlock::Create(gIR->context(), "ororend", gIR->topfunc());

    LLValue *ubool = DtoCast(e->loc, u, Type::tbool)->getRVal();

    llvm::BasicBlock *oldblock = p->scopebb();
    llvm::BranchInst::Create(ororend, oror, ubool, p->scopebb());

    p->scope() = IRScope(oror);
    emitCoverageLinecountInc(e->e2->loc);
    DValue *v = toElemDtor(e->e2);

    LLValue *vbool = nullptr;
    if (v && !v->isFunc() && v->getType() != Type::tvoid) {
      vbool = DtoCast(e->loc, v, Type::tbool)->getRVal();
    }

    llvm::BasicBlock *newblock = p->scopebb();
    llvm::BranchInst::Create(ororend, p->scopebb());
    p->scope() = IRScope(ororend);

    LLValue *resval = nullptr;
    if (ubool == vbool || !vbool) {
      // No need to create a PHI node.
      resval = ubool;
    } else {
      llvm::PHINode *phi =
          p->ir->CreatePHI(LLType::getInt1Ty(gIR->context()), 2, "ororval");
      // If we jumped over evaluation of the right-hand side,
      // the result is true. Otherwise, it's the value of the right-hand side.
      phi->addIncoming(LLConstantInt::getTrue(gIR->context()), oldblock);
      phi->addIncoming(vbool, newblock);
      resval = phi;
    }

    result = new DImValue(e->type, resval);
  }

////////////////////////////////////////////////////////////////////////////////

#define BIN_BLIT_EXP(X, Y)                                                     \
  void visit(X##Exp *e) override {                                             \
    IF_LOG Logger::print("%sExp::toElem: %s @ %s\n", #X, e->toChars(),         \
                         e->type->toChars());                                  \
    LOG_SCOPE;                                                                 \
    DValue *u = toElem(e->e1);                                                 \
    DValue *v = toElem(e->e2);                                                 \
    errorOnIllegalArrayOp(e, e->e1, e->e2);                                    \
    v = DtoCast(e->loc, v, e->e1->type);                                       \
    LLValue *x = llvm::BinaryOperator::Create(                                 \
        llvm::Instruction::Y, u->getRVal(), v->getRVal(), "", p->scopebb());   \
    result = new DImValue(e->type, x);                                         \
  }

  BIN_BLIT_EXP(And, And)
  BIN_BLIT_EXP(Or, Or)
  BIN_BLIT_EXP(Xor, Xor)
  BIN_BLIT_EXP(Shl, Shl)
  BIN_BLIT_EXP(Ushr, LShr)
#undef BIN_BLIT_EXP

  void visit(ShrExp *e) override {
    IF_LOG Logger::print("ShrExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;
    DValue *u = toElem(e->e1);
    DValue *v = toElem(e->e2);
    v = DtoCast(e->loc, v, e->e1->type);
    LLValue *x;
    if (isLLVMUnsigned(e->e1->type)) {
      x = p->ir->CreateLShr(u->getRVal(), v->getRVal());
    } else {
      x = p->ir->CreateAShr(u->getRVal(), v->getRVal());
    }
    result = new DImValue(e->type, x);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(HaltExp *e) override {
    IF_LOG Logger::print("HaltExp::toElem: %s\n", e->toChars());
    LOG_SCOPE;

#if LDC_LLVM_VER >= 307
    p->ir->CreateCall(GET_INTRINSIC_DECL(trap), {});
#else
    p->ir->CreateCall(GET_INTRINSIC_DECL(trap), "");
#endif
    p->ir->CreateUnreachable();

    // this terminated the basicblock, start a new one
    // this is sensible, since someone might goto behind the assert
    // and prevents compiler errors if a terminator follows the assert
    llvm::BasicBlock *bb =
        llvm::BasicBlock::Create(gIR->context(), "afterhalt", p->topfunc());
    p->scope() = IRScope(bb);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(DelegateExp *e) override {
    IF_LOG Logger::print("DelegateExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    if (e->func->isStatic()) {
      e->error("can't take delegate of static function %s, it does not require "
               "a context ptr",
               e->func->toChars());
    }

    LLPointerType *int8ptrty = getPtrToType(LLType::getInt8Ty(gIR->context()));

    assert(e->type->toBasetype()->ty == Tdelegate);
    LLType *dgty = DtoType(e->type);

    DValue *u = toElem(e->e1);
    LLValue *uval;
    if (DFuncValue *f = u->isFunc()) {
      assert(f->func);
      LLValue *contextptr = DtoNestedContext(e->loc, f->func);
      uval = DtoBitCast(contextptr, getVoidPtrType());
    } else {
      uval = u->getRVal();
    }

    IF_LOG Logger::cout() << "context = " << *uval << '\n';

    LLValue *castcontext = DtoBitCast(uval, int8ptrty);

    IF_LOG Logger::println("func: '%s'", e->func->toPrettyChars());

    LLValue *castfptr;

    if (e->e1->op != TOKsuper && e->e1->op != TOKdottype &&
        e->func->isVirtual() && !e->func->isFinalFunc()) {
      castfptr = DtoVirtualFunctionPointer(u, e->func, e->toChars());
    } else if (e->func->isAbstract()) {
      llvm_unreachable("Delegate to abstract method not implemented.");
    } else if (e->func->toParent()->isInterfaceDeclaration()) {
      llvm_unreachable("Delegate to interface method not implemented.");
    } else {
      DtoResolveFunction(e->func);

      // We need to actually codegen the function here, as literals are not
      // added to the module member list.
      if (e->func->semanticRun == PASSsemantic3done) {
        Dsymbol *owner = e->func->toParent();
        while (!owner->isTemplateInstance() && owner->toParent()) {
          owner = owner->toParent();
        }
        if (owner->isTemplateInstance() || owner == p->dmodule) {
          Declaration_codegen(e->func, p);
        }
      }

      castfptr = getIrFunc(e->func)->func;
    }

    castfptr = DtoBitCast(castfptr, dgty->getContainedType(1));

    result = new DImValue(
        e->type, DtoAggrPair(DtoType(e->type), castcontext, castfptr, ".dg"));
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(IdentityExp *e) override {
    IF_LOG Logger::print("IdentityExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *l = toElem(e->e1);
    DValue *r = toElem(e->e2);
    LLValue *lv = l->getRVal();
    LLValue *rv = r->getRVal();

    Type *t1 = e->e1->type->toBasetype();

    // handle dynarray specially
    if (t1->ty == Tarray) {
      result = new DImValue(e->type, DtoDynArrayIs(e->op, l, r));
      return;
    }
    // also structs
    if (t1->ty == Tstruct) {
      result = new DImValue(e->type, DtoStructEquals(e->op, l, r));
      return;
    }

    // FIXME this stuff isn't pretty
    LLValue *eval = nullptr;

    if (t1->ty == Tdelegate) {
      if (r->isNull()) {
        rv = nullptr;
      } else {
        assert(lv->getType() == rv->getType());
      }
      eval = DtoDelegateEquals(e->op, lv, rv);
    } else if (t1->isfloating()) // includes iscomplex
    {
      eval = DtoBinNumericEquals(e->loc, l, r, e->op);
    } else if (t1->ty == Tpointer || t1->ty == Tclass) {
      if (lv->getType() != rv->getType()) {
        if (r->isNull()) {
          rv = llvm::ConstantPointerNull::get(isaPointer(lv->getType()));
        } else {
          rv = DtoBitCast(rv, lv->getType());
        }
      }
      eval = (e->op == TOKidentity) ? p->ir->CreateICmpEQ(lv, rv)
                                    : p->ir->CreateICmpNE(lv, rv);
    } else {
      assert(lv->getType() == rv->getType());
      eval = (e->op == TOKidentity) ? p->ir->CreateICmpEQ(lv, rv)
                                    : p->ir->CreateICmpNE(lv, rv);
    }
    result = new DImValue(e->type, eval);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(CommaExp *e) override {
    IF_LOG Logger::print("CommaExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    if (e->cachedLvalue) {
      LLValue *V = e->cachedLvalue;
      result = new DVarValue(e->type, V);
      return;
    }

    toElem(e->e1);
    result = toElem(e->e2);

    // Actually, we can get qualifier mismatches in the 2.064 frontend:
    // assert(e2->type == type);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(CondExp *e) override {
    IF_LOG Logger::print("CondExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    Type *dtype = e->type->toBasetype();
    LLValue *retPtr = nullptr;
    if (dtype->ty != Tvoid) {
      // allocate a temporary for pointer to the final result.
      retPtr = DtoAlloca(dtype->pointerTo(), "condtmp");
    }

    llvm::BasicBlock *condtrue =
        llvm::BasicBlock::Create(gIR->context(), "condtrue", gIR->topfunc());
    llvm::BasicBlock *condfalse =
        llvm::BasicBlock::Create(gIR->context(), "condfalse", gIR->topfunc());
    llvm::BasicBlock *condend =
        llvm::BasicBlock::Create(gIR->context(), "condend", gIR->topfunc());

    DValue *c = toElem(e->econd);
    LLValue *cond_val = DtoCast(e->loc, c, Type::tbool)->getRVal();
    llvm::BranchInst::Create(condtrue, condfalse, cond_val, p->scopebb());

    p->scope() = IRScope(condtrue);
    DValue *u = toElem(e->e1);
    if (retPtr) {
      LLValue *lval = makeLValue(e->loc, u);
      DtoStore(lval, DtoBitCast(retPtr, lval->getType()->getPointerTo()));
    }
    llvm::BranchInst::Create(condend, p->scopebb());

    p->scope() = IRScope(condfalse);
    DValue *v = toElem(e->e2);
    if (retPtr) {
      LLValue *lval = makeLValue(e->loc, v);
      DtoStore(lval, DtoBitCast(retPtr, lval->getType()->getPointerTo()));
    }
    llvm::BranchInst::Create(condend, p->scopebb());

    p->scope() = IRScope(condend);
    if (retPtr) {
      result = new DVarValue(e->type, DtoLoad(retPtr));
    } else {
      result = new DConstValue(e->type, getNullValue(DtoMemType(dtype)));
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(ComExp *e) override {
    IF_LOG Logger::print("ComExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *u = toElem(e->e1);

    LLValue *value = u->getRVal();
    LLValue *minusone =
        LLConstantInt::get(value->getType(), static_cast<uint64_t>(-1), true);
    value = llvm::BinaryOperator::Create(llvm::Instruction::Xor, value,
                                         minusone, "", p->scopebb());

    result = new DImValue(e->type, value);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(NegExp *e) override {
    IF_LOG Logger::print("NegExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *l = toElem(e->e1);

    if (e->type->iscomplex()) {
      result = DtoComplexNeg(e->loc, e->type, l);
      return;
    }

    LLValue *val = l->getRVal();

    if (e->type->isintegral()) {
      val = p->ir->CreateNeg(val, "negval");
    } else {
      val = p->ir->CreateFNeg(val, "negval");
    }

    result = new DImValue(e->type, val);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(CatExp *e) override {
    IF_LOG Logger::print("CatExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    result = DtoCatArrays(e->loc, e->type, e->e1, e->e2);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(CatAssignExp *e) override {
    IF_LOG Logger::print("CatAssignExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    result = toElem(e->e1);

    Type *e1type = e->e1->type->toBasetype();
    assert(e1type->ty == Tarray);
    Type *elemtype = e1type->nextOf()->toBasetype();
    Type *e2type = e->e2->type->toBasetype();

    if (e1type->ty == Tarray && e2type->ty == Tdchar &&
        (elemtype->ty == Tchar || elemtype->ty == Twchar)) {
      if (elemtype->ty == Tchar) {
        // append dchar to char[]
        DtoAppendDCharToString(e->loc, result, e->e2);
      } else { /*if (elemtype->ty == Twchar)*/
        // append dchar to wchar[]
        DtoAppendDCharToUnicodeString(e->loc, result, e->e2);
      }
    } else if (e1type->equals(e2type)) {
      // append array
      DSliceValue *slice = DtoCatAssignArray(e->loc, result, e->e2);
      DtoAssign(e->loc, result, slice);
    } else {
      // append element
      DtoCatAssignElement(e->loc, e1type, result, e->e2);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(FuncExp *e) override {
    IF_LOG Logger::print("FuncExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    FuncLiteralDeclaration *fd = e->fd;
    assert(fd);

    if ((fd->tok == TOKreserved || fd->tok == TOKdelegate) &&
        e->type->ty == Tpointer) {
      // This is a lambda that was inferred to be a function literal instead
      // of a delegate, so set tok here in order to get correct types/mangling.
      // Horrible hack, but DMD does the same thing.
      fd->tok = TOKfunction;
      fd->vthis = nullptr;
    }

    if (fd->isNested()) {
      Logger::println("nested");
    }
    Logger::println("kind = %s", fd->kind());

    // We need to actually codegen the function here, as literals are not added
    // to the module member list.
    Declaration_codegen(fd, p);
    if (!isIrFuncCreated(fd)) {
      // See DtoDefineFunction for reasons why codegen was suppressed.
      // Instead just declare the function.
      DtoDeclareFunction(fd);
      assert(!fd->isNested());
    }
    assert(getIrFunc(fd)->func);

    if (fd->isNested()) {
      LLType *dgty = DtoType(e->type);

      LLValue *cval;
      IrFunction *irfn = p->func();
      if (irfn->nestedVar && fd->toParent2() == irfn->decl) {
        // We check fd->toParent2() because a frame allocated in one
        // function cannot be used for a delegate created in another
        // function. Happens with anonymous functions.
        cval = irfn->nestedVar;
      } else if (irfn->nestArg) {
        cval = DtoLoad(irfn->nestArg);
      } else if (irfn->thisArg) {
        AggregateDeclaration *ad = irfn->decl->isMember2();
        if (!ad || !ad->vthis) {
          cval = getNullPtr(getVoidPtrType());
        } else {
          cval =
              ad->isClassDeclaration() ? DtoLoad(irfn->thisArg) : irfn->thisArg;
          cval = DtoLoad(
              DtoGEPi(cval, 0, getFieldGEPIndex(ad, ad->vthis), ".vthis"));
        }
      } else {
        cval = getNullPtr(getVoidPtrType());
      }
      cval = DtoBitCast(cval, dgty->getContainedType(0));

      LLValue *castfptr =
          DtoBitCast(getIrFunc(fd)->func, dgty->getContainedType(1));

      result = new DImValue(e->type, DtoAggrPair(cval, castfptr, ".func"));

    } else {
      result = new DFuncValue(e->type, fd, getIrFunc(fd)->func);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(ArrayLiteralExp *e) override {
    IF_LOG Logger::print("ArrayLiteralExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    // D types
    Type *arrayType = e->type->toBasetype();
    Type *elemType = arrayType->nextOf()->toBasetype();

    // is dynamic ?
    bool const dyn = (arrayType->ty == Tarray);
    // length
    size_t const len = e->elements->dim;

    // llvm target type
    LLType *llType = DtoType(arrayType);
    IF_LOG Logger::cout() << (dyn ? "dynamic" : "static")
                          << " array literal with length " << len
                          << " of D type: '" << arrayType->toChars()
                          << "' has llvm type: '" << *llType << "'\n";

    // llvm storage type
    LLType *llElemType = DtoMemType(elemType);
    LLType *llStoType = LLArrayType::get(llElemType, len);
    IF_LOG Logger::cout() << "llvm storage type: '" << *llStoType << "'\n";

    // don't allocate storage for zero length dynamic array literals
    if (dyn && len == 0) {
      // dmd seems to just make them null...
      result = new DSliceValue(e->type, DtoConstSize_t(0),
                               getNullPtr(getPtrToType(llElemType)));
    } else if (dyn) {
      if (arrayType->isImmutable() && isConstLiteral(e)) {
        llvm::Constant *init = arrayLiteralToConst(p, e);
        auto global = new llvm::GlobalVariable(
            gIR->module, init->getType(), true,
            llvm::GlobalValue::InternalLinkage, init, ".immutablearray");
        result = new DSliceValue(arrayType, DtoConstSize_t(e->elements->dim),
                                 DtoBitCast(global, getPtrToType(llElemType)));
      } else {
        DSliceValue *dynSlice = DtoNewDynArray(
            e->loc, arrayType,
            new DConstValue(Type::tsize_t, DtoConstSize_t(len)), false);
        initializeArrayLiteral(
            p, e, DtoBitCast(dynSlice->ptr, getPtrToType(llStoType)));
        result = dynSlice;
      }
    } else {
      llvm::Value *storage =
          DtoRawAlloca(llStoType, DtoAlignment(e->type), "arrayliteral");
      initializeArrayLiteral(p, e, storage);
      result = new DImValue(e->type, storage);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(StructLiteralExp *e) override {
    IF_LOG Logger::print("StructLiteralExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    if (e->sinit) {
      // Copied from VarExp::toElem, need to clean this mess up.
      Type *sdecltype = e->sinit->type->toBasetype();
      IF_LOG Logger::print("Sym: type = %s\n", sdecltype->toChars());
      assert(sdecltype->ty == Tstruct);
      TypeStruct *ts = static_cast<TypeStruct *>(sdecltype);
      assert(ts->sym);
      DtoResolveStruct(ts->sym);

      LLValue *initsym = getIrAggr(ts->sym)->getInitSymbol();
      initsym = DtoBitCast(initsym, DtoType(ts->pointerTo()));
      result = new DVarValue(e->type, initsym);
      return;
    }

    if (e->inProgressMemory) {
      result = new DVarValue(e->type, e->inProgressMemory);
      return;
    }

    // make sure the struct is fully resolved
    DtoResolveStruct(e->sd);

    // alloca a stack slot
    e->inProgressMemory = DtoAlloca(e->type, ".structliteral");

    // fill the allocated struct literal
    write_struct_literal(e->loc, e->inProgressMemory, e->sd, e->elements);

    // return as a var
    result = new DVarValue(e->type, e->inProgressMemory);
    e->inProgressMemory = nullptr;
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(ClassReferenceExp *e) override {
    IF_LOG Logger::print("ClassReferenceExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    result = new DImValue(e->type, toConstElem(e, p));
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(InExp *e) override {
    IF_LOG Logger::print("InExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *key = toElem(e->e1);
    DValue *aa = toElem(e->e2);

    result = DtoAAIn(e->loc, e->type, aa, key);
  }

  void visit(RemoveExp *e) override {
    IF_LOG Logger::print("RemoveExp::toElem: %s\n", e->toChars());
    LOG_SCOPE;

    DValue *aa = toElem(e->e1);
    DValue *key = toElem(e->e2);

    result = DtoAARemove(e->loc, aa, key);
  }

  //////////////////////////////////////////////////////////////////////////////

  /// Constructs an array initializer constant with the given constants as its
  /// elements. If the element types differ (unions, …), an anonymous struct
  /// literal is emitted (as for array constant initializers).
  llvm::Constant *arrayConst(std::vector<llvm::Constant *> &vals,
                             Type *nominalElemType) {
    if (vals.size() == 0) {
      llvm::ArrayType *type = llvm::ArrayType::get(DtoType(nominalElemType), 0);
      return llvm::ConstantArray::get(type, vals);
    }

    llvm::Type *elementType = nullptr;
    bool differentTypes = false;
    for (auto v : vals) {
      if (!elementType) {
        elementType = v->getType();
      } else {
        differentTypes |= (elementType != v->getType());
      }
    }

    if (differentTypes) {
      return llvm::ConstantStruct::getAnon(vals, true);
    }

    llvm::ArrayType *t = llvm::ArrayType::get(elementType, vals.size());
    return llvm::ConstantArray::get(t, vals);
  }

  void visit(AssocArrayLiteralExp *e) override {
    IF_LOG Logger::print("AssocArrayLiteralExp::toElem: %s @ %s\n",
                         e->toChars(), e->type->toChars());
    LOG_SCOPE;

    assert(e->keys);
    assert(e->values);
    assert(e->keys->dim == e->values->dim);

    Type *basetype = e->type->toBasetype();
    Type *aatype = basetype;
    Type *vtype = aatype->nextOf();

    if (!e->keys->dim) {
      goto LruntimeInit;
    }

    if (aatype->ty != Taarray) {
      // It's the AssociativeArray type.
      // Turn it back into a TypeAArray
      vtype = e->values->tdata()[0]->type;
      aatype = new TypeAArray(vtype, e->keys->tdata()[0]->type);
      aatype = aatype->semantic(e->loc, nullptr);
    }

    {
      std::vector<LLConstant *> keysInits, valuesInits;
      keysInits.reserve(e->keys->dim);
      valuesInits.reserve(e->keys->dim);
      for (size_t i = 0, n = e->keys->dim; i < n; ++i) {
        Expression *ekey = e->keys->tdata()[i];
        Expression *eval = e->values->tdata()[i];
        IF_LOG Logger::println("(%llu) aa[%s] = %s",
                               static_cast<unsigned long long>(i),
                               ekey->toChars(), eval->toChars());
        unsigned errors = global.startGagging();
        LLConstant *ekeyConst = toConstElem(ekey, p);
        LLConstant *evalConst = toConstElem(eval, p);
        if (global.endGagging(errors)) {
          goto LruntimeInit;
        }
        assert(ekeyConst && evalConst);
        keysInits.push_back(ekeyConst);
        valuesInits.push_back(evalConst);
      }

      assert(aatype->ty == Taarray);
      Type *indexType = static_cast<TypeAArray *>(aatype)->index;
      assert(indexType && vtype);

      llvm::Function *func =
          getRuntimeFunction(e->loc, gIR->module, "_d_assocarrayliteralTX");
      LLFunctionType *funcTy = func->getFunctionType();
      LLValue *aaTypeInfo =
          DtoBitCast(DtoTypeInfoOf(stripModifiers(aatype)),
                     DtoType(Type::typeinfoassociativearray->type));

      LLConstant *idxs[2] = {DtoConstUint(0), DtoConstUint(0)};

      LLConstant *initval = arrayConst(keysInits, indexType);
      LLConstant *globalstore = new LLGlobalVariable(
          gIR->module, initval->getType(), false,
          LLGlobalValue::InternalLinkage, initval, ".aaKeysStorage");
#if LDC_LLVM_VER >= 307
      LLConstant *slice = llvm::ConstantExpr::getGetElementPtr(
          isaPointer(globalstore)->getElementType(), globalstore, idxs, true);
#else
      LLConstant *slice =
          llvm::ConstantExpr::getGetElementPtr(globalstore, idxs, true);
#endif
      slice = DtoConstSlice(DtoConstSize_t(e->keys->dim), slice);
      LLValue *keysArray = DtoAggrPaint(slice, funcTy->getParamType(1));

      initval = arrayConst(valuesInits, vtype);
      globalstore = new LLGlobalVariable(gIR->module, initval->getType(), false,
                                         LLGlobalValue::InternalLinkage,
                                         initval, ".aaValuesStorage");
#if LDC_LLVM_VER >= 307
      slice = llvm::ConstantExpr::getGetElementPtr(
          isaPointer(globalstore)->getElementType(), globalstore, idxs, true);
#else
      slice = llvm::ConstantExpr::getGetElementPtr(globalstore, idxs, true);
#endif
      slice = DtoConstSlice(DtoConstSize_t(e->keys->dim), slice);
      LLValue *valuesArray = DtoAggrPaint(slice, funcTy->getParamType(2));

      LLValue *aa = gIR->CreateCallOrInvoke(func, aaTypeInfo, keysArray,
                                            valuesArray, "aa")
                        .getInstruction();
      if (basetype->ty != Taarray) {
        LLValue *tmp = DtoAlloca(e->type, "aaliteral");
        DtoStore(aa, DtoGEPi(tmp, 0, 0));
        result = new DVarValue(e->type, tmp);
      } else {
        result = new DImValue(e->type, aa);
      }

      return;
    }

  LruntimeInit:

    // it should be possible to avoid the temporary in some cases
    LLValue *tmp = DtoAllocaDump(LLConstant::getNullValue(DtoType(e->type)),
                                 e->type, "aaliteral");
    result = new DVarValue(e->type, tmp);

    const size_t n = e->keys->dim;
    for (size_t i = 0; i < n; ++i) {
      Expression *ekey = (*e->keys)[i];
      Expression *eval = (*e->values)[i];

      IF_LOG Logger::println("(%llu) aa[%s] = %s",
                             static_cast<unsigned long long>(i),
                             ekey->toChars(), eval->toChars());

      // index
      DValue *key = toElem(ekey);
      DValue *mem = DtoAAIndex(e->loc, vtype, result, key, true);

      // store
      DValue *val = toElem(eval);
      DtoAssign(e->loc, mem, val);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  DValue *toGEP(UnaExp *exp, unsigned index) {
    // (&a.foo).funcptr is a case where toElem(e1) is genuinely not an l-value.
    LLValue *val = makeLValue(exp->loc, toElem(exp->e1));
    LLValue *v = DtoGEPi(val, 0, index);
    return new DVarValue(exp->type, DtoBitCast(v, DtoPtrToType(exp->type)));
  }

  void visit(DelegatePtrExp *e) override {
    IF_LOG Logger::print("DelegatePtrExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    result = toGEP(e, 0);
  }

  void visit(DelegateFuncptrExp *e) override {
    IF_LOG Logger::print("DelegateFuncptrExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    result = toGEP(e, 1);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(BoolExp *e) override {
    IF_LOG Logger::print("BoolExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    result = new DImValue(
        e->type, DtoCast(e->loc, toElem(e->e1), Type::tbool)->getRVal());
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(DotTypeExp *e) override {
    IF_LOG Logger::print("DotTypeExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    assert(e->sym->getType());
    result = toElem(e->e1);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(TypeExp *e) override {
    e->error("type %s is not an expression", e->toChars());
    // TODO: Improve error handling. DMD just returns some value here and hopes
    // some more sensible error messages will be triggered.
    fatal();
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(TupleExp *e) override {
    IF_LOG Logger::print("TupleExp::toElem() %s\n", e->toChars());
    LOG_SCOPE;

    // If there are any side effects, evaluate them first.
    if (e->e0) {
      toElem(e->e0);
    }

    std::vector<LLType *> types;
    types.reserve(e->exps->dim);
    for (size_t i = 0; i < e->exps->dim; i++) {
      types.push_back(DtoMemType((*e->exps)[i]->type));
    }
    LLValue *val =
        DtoRawAlloca(LLStructType::get(gIR->context(), types), 0, ".tuple");
    for (size_t i = 0; i < e->exps->dim; i++) {
      Expression *el = (*e->exps)[i];
      DValue *ep = toElem(el);
      LLValue *gep = DtoGEPi(val, 0, i);
      if (DtoIsInMemoryOnly(el->type)) {
        DtoMemCpy(gep, ep->getRVal());
      } else if (el->type->ty != Tvoid) {
        DtoStoreZextI8(ep->getRVal(), gep);
      } else {
        DtoStore(LLConstantInt::get(LLType::getInt8Ty(p->context()), 0, false),
                 gep);
      }
    }
    result = new DVarValue(e->type, val);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(VectorExp *e) override {
    IF_LOG Logger::print("VectorExp::toElem() %s\n", e->toChars());
    LOG_SCOPE;

    TypeVector *type = static_cast<TypeVector *>(e->to->toBasetype());
    assert(e->type->ty == Tvector);

    LLValue *vector = DtoAlloca(e->to);

    // Array literals are assigned element-wise, other expressions are cast and
    // splat across the vector elements. This is what DMD does.
    if (e->e1->op == TOKarrayliteral) {
      Logger::println("array literal expression");
      ArrayLiteralExp *lit = static_cast<ArrayLiteralExp *>(e->e1);
      assert(lit->elements->dim == e->dim &&
             "Array literal vector initializer "
             "length mismatch, should have been handled in frontend.");
      for (unsigned int i = 0; i < e->dim; ++i) {
        DValue *val = toElem((*lit->elements)[i]);
        LLValue *llval = DtoCast(e->loc, val, type->elementType())->getRVal();
        DtoStore(llval, DtoGEPi(vector, 0, i));
      }
    } else {
      Logger::println("normal (splat) expression");
      DValue *val = toElem(e->e1);
      LLValue *llval = DtoCast(e->loc, val, type->elementType())->getRVal();
      for (unsigned int i = 0; i < e->dim; ++i) {
        DtoStore(llval, DtoGEPi(vector, 0, i));
      }
    }

    result = new DVarValue(e->to, vector);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(PowExp *e) override {
    IF_LOG Logger::print("PowExp::toElem() %s\n", e->toChars());
    LOG_SCOPE;

    e->error("must import std.math to use ^^ operator");
    result = new DNullValue(e->type, llvm::UndefValue::get(DtoType(e->type)));
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(TypeidExp *e) override {
    if (Type *t = isType(e->obj)) {
      result = DtoSymbolAddress(e->loc, e->type,
                                getOrCreateTypeInfoDeclaration(t, nullptr));
      return;
    }
    if (Expression *ex = isExpression(e->obj)) {
      Type *t = ex->type->toBasetype();
      assert(t->ty == Tclass);

      DValue *val = toElem(ex);

      // Get and load vtbl pointer.
      llvm::Value *vtbl = DtoLoad(DtoGEPi(val->getRVal(), 0, 0));

      // TypeInfo ptr is first vtbl entry.
      llvm::Value *typinf = DtoGEPi(vtbl, 0, 0);

      Type *resultType = Type::typeinfoclass->type;
      if (static_cast<TypeClass *>(t)->sym->isInterfaceDeclaration()) {
        // For interfaces, the first entry in the vtbl is actually a pointer
        // to an Interface instance, which has the type info as its first
        // member, so we have to add an extra layer of indirection.
        resultType = Type::typeinfointerface->type;
        typinf = DtoLoad(
            DtoBitCast(typinf, DtoType(resultType->pointerTo()->pointerTo())));
      }

      result = new DVarValue(resultType, typinf);
      return;
    }
    llvm_unreachable("Unknown TypeidExp argument kind");
  }

////////////////////////////////////////////////////////////////////////////////

#define STUB(x)                                                                \
  void visit(x *e) override {                                                  \
    e->error("Internal compiler error: Type " #x " not implemented: %s",       \
             e->toChars());                                                    \
    fatal();                                                                   \
  }
  STUB(Expression)
  STUB(ScopeExp)
  STUB(SymbolExp)
  STUB(PowAssignExp)
#undef STUB
};

////////////////////////////////////////////////////////////////////////////////

DValue *toElem(Expression *e) {
  ToElemVisitor v(gIR, false);
  e->accept(&v);
  return v.getResult();
}

DValue *toElemDtor(Expression *e) {
  ToElemVisitor v(gIR, true);
  e->accept(&v);
  return v.getResult();
}

// FIXME: Implement & place in right module
Symbol *toModuleAssert(Module *m) { return nullptr; }

// FIXME: Implement & place in right module
Symbol *toModuleUnittest(Module *m) { return nullptr; }

// FIXME: Implement & place in right module
Symbol *toModuleArray(Module *m) { return nullptr; }
