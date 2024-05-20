//===-- toir.cpp ----------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/attrib.h"
#include "dmd/ctfe.h"
#include "dmd/enum.h"
#include "dmd/errors.h"
#include "dmd/hdrgen.h"
#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/init.h"
#include "dmd/ldcbindings.h"
#include "dmd/module.h"
#include "dmd/mtype.h"
#include "dmd/root/port.h"
#include "dmd/root/rmem.h"
#include "dmd/target.h"
#include "dmd/template.h"
#include "gen/aa.h"
#include "gen/abi/abi.h"
#include "gen/arrays.h"
#include "gen/binops.h"
#include "gen/classes.h"
#include "gen/complex.h"
#include "gen/coverage.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/funcgenstate.h"
#include "gen/inlineir.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/mangling.h"
#include "gen/nested.h"
#include "gen/optimizer.h"
#include "gen/pragma.h"
#include "gen/runtime.h"
#include "gen/scope_exit.h"
#include "gen/structs.h"
#include "gen/tollvm.h"
#include "gen/typinf.h"
#include "ir/irfunction.h"
#include "ir/irtypeclass.h"
#include "ir/irtypestruct.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include <fstream>
#include <math.h>
#include <stack>
#include <stdio.h>

using namespace dmd;

llvm::cl::opt<bool> checkPrintf(
    "check-printf-calls", llvm::cl::ZeroOrMore, llvm::cl::ReallyHidden,
    llvm::cl::desc("Validate printf call format strings against arguments"));

bool walkPostorder(Expression *e, StoppableVisitor *v);

////////////////////////////////////////////////////////////////////////////////

static LLValue *write_zeroes(LLValue *mem, unsigned start, unsigned end) {
  LLType *i8 = LLType::getInt8Ty(gIR->context());
  LLValue *gep = DtoGEP1(i8, mem, start, ".padding");
  DtoMemSetZero(i8, gep, DtoConstSize_t(end - start));
  return mem;
}

////////////////////////////////////////////////////////////////////////////////

static void write_struct_literal(Loc loc, LLValue *mem, StructDeclaration *sd,
                                 Expressions *elements) {
  assert(elements && "struct literal has null elements");
  const auto numMissingElements = sd->fields.length - elements->length;
  (void)numMissingElements;
  assert(numMissingElements == 0 || (sd->vthis && numMissingElements == 1));

  // might be reset to an actual i8* value so only a single bitcast is emitted
  LLValue *voidptr = mem;

  struct Data {
    VarDeclaration *field;
    Expression *expr;
  };
  LLSmallVector<Data, 16> data;

  // collect init expressions in fields declaration order
  for (size_t index = 0; index < sd->fields.length; ++index) {
    VarDeclaration *field = sd->fields[index];

    // Skip zero-sized fields such as zero-length static arrays: `ubyte[0]
    // data`.
    if (field->type->size() == 0)
      continue;

    // the initializer expression may be null for overridden overlapping fields
    Expression *expr =
        (index < elements->length ? (*elements)[index] : nullptr);
    if (expr || field == sd->vthis) {
      data.push_back({field, expr});
    }
  }

  // sort by offset
  std::sort(data.begin(), data.end(), [](const Data &l, const Data &r) {
    return l.field->offset < r.field->offset;
  });

  unsigned offset = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const auto vd = data[i].field;
    const auto expr = data[i].expr;

    // initialize any padding so struct comparisons work
    if (vd->offset != offset) {
      if (vd->offset < offset) {
        error(loc, "ICE: overlapping initializers for struct literal");
        fatal();
      }
      voidptr = write_zeroes(voidptr, offset, vd->offset);
      offset = vd->offset;
    }

    if (vd->isBitFieldDeclaration()) {
      const auto group = BitFieldGroup::startingFrom(
          i, data.size(), [&data](size_t i) { return data[i].field; });

      IF_LOG Logger::println("initializing bit field group: (+%u, %u bytes)",
                             group.byteOffset, group.sizeInBytes);
      LOG_SCOPE

      // get a pointer to this group's IR field
      const auto ptr = DtoLVal(DtoIndexAggregate(mem, sd, vd));

      // merge all initializers to a single value
      const auto intType =
          LLIntegerType::get(gIR->context(), group.sizeInBytes * 8);
      LLValue *val = LLConstant::getNullValue(intType);
      for (size_t j = 0; j < group.bitFields.size(); ++j) {
        const auto bf = group.bitFields[j];
        const auto bfExpr = data[i + j].expr;
        assert(bfExpr);

        const unsigned bitOffset = group.getBitOffset(bf);
        IF_LOG Logger::println("bit field: %s %s (bit offset %u, width %u): %s",
                               bf->type->toChars(), bf->toChars(), bitOffset,
                               bf->fieldWidth, bfExpr->toChars());
        LOG_SCOPE

        auto bfVal = DtoRVal(bfExpr);
        bfVal = gIR->ir->CreateZExtOrTrunc(bfVal, intType);
        const auto mask =
            llvm::APInt::getLowBitsSet(intType->getBitWidth(), bf->fieldWidth);
        bfVal = gIR->ir->CreateAnd(bfVal, mask);
        if (bitOffset)
          bfVal = gIR->ir->CreateShl(bfVal, bitOffset);
        val = gIR->ir->CreateOr(val, bfVal);
      }

      IF_LOG Logger::cout() << "merged IR value: " << *val << '\n';
      gIR->ir->CreateAlignedStore(val, ptr, llvm::MaybeAlign(1));
      offset += group.sizeInBytes;

      i += group.bitFields.size() - 1; // skip the other bit fields of the group
    } else {
      IF_LOG Logger::println("initializing field: %s %s (+%u)",
                             vd->type->toChars(), vd->toChars(), vd->offset);
      LOG_SCOPE

      const auto field = DtoIndexAggregate(mem, sd, vd);

      // initialize the field
      if (expr) {
        IF_LOG Logger::println("expr = %s", expr->toChars());
        // try to construct it in-place
        if (!toInPlaceConstruction(field, expr)) {
          DtoAssign(loc, field, toElem(expr), EXP::blit);
          if (expr->isLvalue())
            callPostblit(loc, expr, DtoLVal(field));
        }
      } else {
        assert(vd == sd->vthis);
        IF_LOG Logger::println("initializing vthis");
        LOG_SCOPE
        DImValue val(vd->type, DtoNestedContext(loc, sd));
        DtoAssign(loc, field, &val, EXP::blit);
      }

      // Make sure to zero out padding bytes counted as being part of the type
      // in DMD but not in LLVM; e.g. real/x86_fp80.
      offset += gDataLayout->getTypeStoreSize(DtoType(vd->type));
    }
  }

  // initialize trailing padding
  if (sd->structsize != offset) {
    if (sd->structsize < offset) {
      error(loc, "ICE: struct literal size exceeds struct size");
      fatal();
    }
    voidptr = write_zeroes(voidptr, offset, sd->structsize);
  }
}

namespace {
void pushVarDtorCleanup(IRState *p, VarDeclaration *vd) {
  llvm::BasicBlock *beginBB = p->insertBB(llvm::Twine("dtor.") + vd->toChars());

  const auto savedInsertPoint = p->saveInsertPoint();
  p->ir->SetInsertPoint(beginBB);
  toElemDtor(vd->edtor);
  p->funcGen().scopes.pushCleanup(beginBB, p->scopebb());
}

// Zero-extends a scalar i1 to an integer type, or creates a vector mask from an
// i1 vector.
DImValue *zextBool(LLValue *val, Type *to) {
  assert(val->getType()->isIntOrIntVectorTy(1));
  LLType *llTy = DtoType(to);
  if (val->getType() != llTy) {
    if (llTy->isVectorTy()) {
      assert(val->getType()->isVectorTy());
      val = gIR->ir->CreateSExt(val, llTy);
    } else {
      assert(llTy->isIntegerTy());
      val = gIR->ir->CreateZExt(val, llTy);
    }
  }
  return new DImValue(to, val);
}
}

////////////////////////////////////////////////////////////////////////////////

static Expression *skipOverCasts(Expression *e) {
  while (auto ce = e->isCastExp())
    e = ce->e1;
  return e;
}

DValue *toElem(Expression *e, bool doSkipOverCasts) {
  Expression *inner = skipOverCasts(e);
  if (!doSkipOverCasts || inner == e)
    return toElem(e);

  return DtoCast(e->loc, toElem(inner), e->type);
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
    initialCleanupScope = p->funcGen().scopes.currentCleanupScope();
  }

  DValue *getResult() {
    if (destructTemporaries &&
        p->funcGen().scopes.currentCleanupScope() != initialCleanupScope) {
      // We might share the CFG edges through the below cleanup blocks with
      // other paths (e.g. exception unwinding) where the result value has not
      // been constructed. At runtime, the branches will be chosen such that the
      // end bb (which will likely go on to access the value) is never executed
      // in those other cases, but we need to make sure that the SSA is also
      // well-formed statically (i.e. all instructions dominate their uses).
      // Thus, dump the result to a temporary stack slot (created in the entry
      // bb) if it is not guaranteed to dominate the end bb after possibly
      // adding more control flow.
      if (result && result->type->ty != TY::Tvoid &&
          !result->definedInFuncEntryBB()) {
        if (result->isRVal()) {
          LLValue *lval = DtoAllocaDump(result, ".toElemRValResult");
          result = new DLValue(result->type, lval);
        } else {
          LLValue *lval = DtoLVal(result);
          LLValue *lvalPtr = DtoAllocaDump(lval, 0, ".toElemLValResult");
          result = new DSpecialRefValue(result->type, lvalPtr);
        }
      }

      llvm::BasicBlock *endbb = p->insertBB("toElem.success");
      p->funcGen().scopes.runCleanups(initialCleanupScope, endbb);
      p->funcGen().scopes.popCleanups(initialCleanupScope);
      p->ir->SetInsertPoint(endbb);

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

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    result = DtoDeclarationExp(e->declaration);

    if (auto vd = e->declaration->isVarDeclaration()) {
      if (!vd->isDataseg() && vd->needsScopeDtor()) {
        pushVarDtorCleanup(p, vd);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(VarExp *e) override {
    IF_LOG Logger::print("VarExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    assert(e->var);

    if (auto fd = e->var->isFuncLiteralDeclaration()) {
      genFuncLiteral(fd, nullptr);
    }

    if (auto em = e->var->isEnumMember()) {
      IF_LOG Logger::println("Create temporary for enum member");
      // Return the value of the enum member instead of trying to take its
      // address (impossible because we don't emit them as variables)
      // In most cases, the front-end constfolds a VarExp of an EnumMember,
      // leaving the AST free of EnumMembers. However in rare cases,
      // EnumMembers remain and thus we have to deal with them here.
      // See DMD issues 16022 and 16100.
      result = toElem(em->value(), p);
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
      case TY::Tcomplex32:
        c = DtoConstFP(Type::tfloat32, ldouble(0));
        break;
      case TY::Tcomplex64:
        c = DtoConstFP(Type::tfloat64, ldouble(0));
        break;
      case TY::Tcomplex80:
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
    const auto stringLength = e->len;

    if (auto tsa = dtype->isTypeSArray()) {
      const auto arrayLength = tsa->dim->toInteger();
      assert(arrayLength >= stringLength);
      // ImportC: static array length may exceed string length incl. null
      // terminator - bypass string-literal cache and create a separate constant
      // with zero-initialized tail
      if (arrayLength > stringLength + 1) {
        auto constant = buildStringLiteralConstant(e, arrayLength);
        result = new DLValue(e->type, constant);
        return;
      }
    }

    llvm::GlobalVariable *gvar = p->getCachedStringLiteral(e);

    if (dtype->ty == TY::Tarray) {
      result = new DSliceValue(
          e->type, DtoConstSlice(DtoConstSize_t(stringLength), gvar));
    } else if (dtype->ty == TY::Tsarray) {
      // array length matches string length with or without null terminator
      result = new DLValue(e->type, gvar);
    } else if (dtype->ty == TY::Tpointer) {
      result = new DImValue(e->type, gvar);
    } else {
      llvm_unreachable("Unknown type for StringExp.");
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(LoweredAssignExp *e) override {
    IF_LOG Logger::print("LoweredAssignExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    result = toElem(e->lowering);
  }

  void visit(AssignExp *e) override {
    IF_LOG Logger::print("AssignExp::toElem: %s | (%s)(%s = %s)\n",
                         e->toChars(), e->type->toChars(),
                         e->e1->type->toChars(),
                         e->e2->type ? e->e2->type->toChars() : nullptr);
    LOG_SCOPE;

    // Initialization of ref variable?
    // Can't just override ConstructExp::toElem because not all EXP::construct
    // operations are actually instances of ConstructExp... Long live the DMD
    // coding style!
    if (static_cast<int>(e->memset) &
        static_cast<int>(MemorySet::referenceInit)) {
      assert(e->op == EXP::construct || e->op == EXP::blit);
      auto ve = e->e1->isVarExp();
      assert(ve);

      if (ve->var->storage_class & (STCref | STCout)) {
        Logger::println("performing ref variable initialization");
        // Note that the variable value is accessed directly (instead
        // of via getLVal(), which would perform a load from the
        // uninitialized location), and that rhs is stored as an l-value!
        DSpecialRefValue *lhs = toElem(e->e1)->isSpecialRef();
        assert(lhs);
        DValue *rhs = toElem(e->e2);

        // We shouldn't really need makeLValue() here, but the 2.063
        // frontend generates ref variables initialized from function
        // calls.
        DtoStore(makeLValue(e->loc, rhs), lhs->getRefStorage());

        result = lhs;
        return;
      }
    }

    // The front-end sometimes rewrites a static-array-lhs to a slice, e.g.,
    // when initializing a static array with an array literal.
    // Use the static array as lhs in that case.
    DValue *rewrittenLhsStaticArray = nullptr;
    if (auto se = e->e1->isSliceExp()) {
      Type *sliceeBaseType = se->e1->type->toBasetype();
      if (se->lwr == nullptr && sliceeBaseType->ty == TY::Tsarray &&
          se->type->toBasetype()->nextOf() == sliceeBaseType->nextOf())
        rewrittenLhsStaticArray = toElem(se->e1, true);
    }

    DValue *const lhs = (rewrittenLhsStaticArray ? rewrittenLhsStaticArray
                                                 : toElem(e->e1, true));

    // Set the result of the AssignExp to the lhs.
    // Defer this to the end of this function, so that static arrays are
    // rewritten (converted to a slice) after the assignment, primarily for a
    // more intuitive IR order.
    SCOPE_EXIT {
      if (rewrittenLhsStaticArray) {
        result =
            new DSliceValue(e->e1->type, DtoArrayLen(rewrittenLhsStaticArray),
                            DtoArrayPtr(rewrittenLhsStaticArray));
      } else {
        result = lhs;
      }
    };

    // Try to construct the lhs in-place.
    if (lhs->isLVal() && (e->op == EXP::construct || e->op == EXP::blit)) {
      if (toInPlaceConstruction(lhs->isLVal(), e->e2))
        return;
    }

    // Try to assign to the lhs in-place.
    // This extra complication at -O0 is to prevent excessive stack space usage
    // when assigning to large structs.
    // Note: If the assignment is non-trivial, a CallExp to opAssign is
    // generated by the frontend instead of this AssignExp. The in-place
    // construction is not valid if the rhs is not a literal (consider for
    // example `a = foo(a)`), but also not if the rhs contains non-constant
    // elements (consider for example `a = [0, a[0], 2]` or `a = [0, i, 2]`
    // where `i` is a ref variable aliasing with a).
    // Be conservative with this optimization for now: only do the optimization
    // for struct `.init` assignment.
    if (lhs->isLVal() && e->op == EXP::assign) {
      if (auto sle = e->e2->isStructLiteralExp()) {
        if (sle->useStaticInit) {
          if (toInPlaceConstruction(lhs->isLVal(), sle))
            return;
        }
      }
    }

    DValue *r = toElem(e->e2);

    if (e->e1->type->toBasetype()->ty == TY::Tstruct &&
        e->e2->op == EXP::int64) {
      Logger::println("performing aggregate zero initialization");
      assert(e->e2->toInteger() == 0);
      LLValue *lval = DtoLVal(lhs);
      DtoMemSetZero(DtoType(lhs->type), lval);
      TypeStruct *ts = static_cast<TypeStruct *>(e->e1->type);
      if (ts->sym->isNested() && ts->sym->vthis)
        DtoResolveNestedContext(e->loc, ts->sym, lval);
      return;
    }

    // This matches the logic in AssignExp::semantic.
    // TODO: Should be cached in the frontend to avoid issues with the code
    // getting out of sync?
    bool lvalueElem = false;
    if ((e->e2->op == EXP::slice &&
         static_cast<UnaExp *>(e->e2)->e1->isLvalue()) ||
        (e->e2->op == EXP::cast_ &&
         static_cast<UnaExp *>(e->e2)->e1->isLvalue()) ||
        (e->e2->op != EXP::slice && e->e2->isLvalue())) {
      lvalueElem = true;
    }

    Logger::println("performing normal assignment (rhs has lvalue elems = %d)",
                    lvalueElem);
    DtoAssign(e->loc, lhs, r, e->op, !lvalueElem);
  }

  //////////////////////////////////////////////////////////////////////////////

  void errorOnIllegalArrayOp(Expression *base, Expression *e1, Expression *e2) {
    Type *t1 = e1->type->toBasetype();
    Type *t2 = e2->type->toBasetype();

    // valid array ops would have been transformed by optimize
    if ((t1->ty == TY::Tarray || t1->ty == TY::Tsarray) &&
        (t2->ty == TY::Tarray || t2->ty == TY::Tsarray)) {
      error(base->loc, "array operation `%s` not recognized", base->toChars());
      fatal();
    }
  }

  //////////////////////////////////////////////////////////////////////////////

#define BIN_OP(Op, Func)                                                       \
  void visit(Op##Exp *e) override {                                            \
    IF_LOG Logger::print(#Op "Exp::toElem: %s @ %s\n", e->toChars(),           \
                         e->type->toChars());                                  \
    LOG_SCOPE;                                                                 \
                                                                               \
    errorOnIllegalArrayOp(e, e->e1, e->e2);                                    \
                                                                               \
    auto &PGO = gIR->funcGen().pgo;                                            \
    PGO.setCurrentStmt(e);                                                     \
                                                                               \
    result = Func(e->loc, e->type, toElem(e->e1), e->e2);                      \
  }

  BIN_OP(Add, binAdd)
  BIN_OP(Min, binMin)
  BIN_OP(Mul, binMul)
  BIN_OP(Div, binDiv)
  BIN_OP(Mod, binMod)

  BIN_OP(And, binAnd)
  BIN_OP(Or, binOr)
  BIN_OP(Xor, binXor)
  BIN_OP(Shl, binShl)
  BIN_OP(Shr, binShr)
  BIN_OP(Ushr, binUshr)
#undef BIN_OP

  //////////////////////////////////////////////////////////////////////////////

  using BinOpFunc = DValue *(const Loc &, Type *, DValue *, Expression *, bool);

  static Expression *getLValExp(Expression *e) {
    e = skipOverCasts(e);
    if (auto ce = e->isCommaExp()) {
      Expression *newCommaRhs = getLValExp(ce->e2);
      if (newCommaRhs != ce->e2) {
        CommaExp *newComma =
            createCommaExp(ce->loc, ce->e1, newCommaRhs, ce->isGenerated);
        newComma->type = newCommaRhs->type;
        e = newComma;
      }
    }
    return e;
  }

  template <BinOpFunc binOpFunc, bool useLValTypeForBinOp>
  static DValue *binAssign(BinAssignExp *e) {
    Expression *lvalExp = getLValExp(e->e1);
    DValue *lhsLVal = toElem(lvalExp);

    // Use the lhs lvalue for the binop lhs and optionally cast it to the full
    // lhs type (!useLValTypeForBinOp).
    // The front-end apparently likes to specify the binop type via lhs casts,
    // e.g., `byte x; cast(int)x += 5;`.
    // Load the binop lhs AFTER evaluating the rhs.
    Type *opType = (useLValTypeForBinOp ? lhsLVal->type : e->e1->type);
    DValue *opResult = binOpFunc(e->loc, opType, lhsLVal, e->e2, true);

    DValue *assignedResult = DtoCast(e->loc, opResult, lhsLVal->type);
    DtoAssign(e->loc, lhsLVal, assignedResult, EXP::assign);

    if (e->type->equals(lhsLVal->type))
      return lhsLVal;

    return new DLValue(e->type, DtoLVal(lhsLVal));
  }

#define BIN_ASSIGN(Op, Func, useLValTypeForBinOp)                              \
  void visit(Op##AssignExp *e) override {                                      \
    IF_LOG Logger::print(#Op "AssignExp::toElem: %s @ %s\n", e->toChars(),     \
                         e->type->toChars());                                  \
    LOG_SCOPE;                                                                 \
                                                                               \
    errorOnIllegalArrayOp(e, e->e1, e->e2);                                    \
                                                                               \
    auto &PGO = gIR->funcGen().pgo;                                            \
    PGO.setCurrentStmt(e);                                                     \
                                                                               \
    result = binAssign<Func, useLValTypeForBinOp>(e);                          \
  }

  BIN_ASSIGN(Add, binAdd, false)
  BIN_ASSIGN(Min, binMin, false)
  BIN_ASSIGN(Mul, binMul, false)
  BIN_ASSIGN(Div, binDiv, false)
  BIN_ASSIGN(Mod, binMod, false)

  BIN_ASSIGN(And, binAnd, false)
  BIN_ASSIGN(Or, binOr, false)
  BIN_ASSIGN(Xor, binXor, false)
  BIN_ASSIGN(Shl, binShl, true)
  BIN_ASSIGN(Shr, binShr, true)
  BIN_ASSIGN(Ushr, binUshr, true)
#undef BIN_ASSIGN

  //////////////////////////////////////////////////////////////////////////////

  static DValue *call(IRState *p, CallExp *e, LLValue *sretPointer = nullptr) {
    IF_LOG Logger::print("CallExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    // handle magic intrinsics and inline asm/IR
    if (auto ve = e->e1->isVarExp()) {
      if (auto fd = ve->var->isFuncDeclaration()) {
        if (fd->llvmInternal == LLVMinline_asm) {
          return DtoInlineAsmExpr(e->loc, fd, e->arguments, sretPointer);
        }
        if (fd->llvmInternal == LLVMinline_ir) {
          return DtoInlineIRExpr(e->loc, fd, e->arguments, sretPointer);
        }

        DValue *result = nullptr;
        if (DtoLowerMagicIntrinsic(p, fd, e, result))
          return result;
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
    if (e->f && e->f->isCtorDeclaration()) {
      if (auto dve = e->e1->isDotVarExp())
        if (auto ce = dve->e1->isCommaExp())
          if (ce->e1->op == EXP::declaration)
            if (auto ve = ce->e2->isVarExp())
              if (auto vd = ve->var->isVarDeclaration())
                if (vd->needsScopeDtor()) {
                  Logger::println("Delaying edtor");
                  delayedDtorVar = vd;
                  delayedDtorExp = vd->edtor;
                  vd->edtor = nullptr;
                }
    }

    // get the callee value
    DValue *fnval;
    if (e->directcall) {
      // TODO: Do this as an extra parameter to DotVarExp implementation.
      auto dve = e->e1->isDotVarExp();
      assert(dve);
      FuncDeclaration *fdecl = dve->var->isFuncDeclaration();
      assert(fdecl);
      Expression *thisExp = dve->e1;
      LLValue *thisArg = thisExp->type->toBasetype()->ty == TY::Tclass
                             ? DtoRVal(thisExp)
                             : DtoLVal(thisExp); // when calling a struct method
      fnval = new DFuncValue(fdecl, DtoCallee(fdecl), thisArg);
    } else {
      fnval = toElem(e->e1);
    }

    // get func value if any
    DFuncValue *dfnval = fnval->isFunc();

    // If this is a virtual function call, the object is passed by reference
    // through the `this` parameter, and therefore the optimizer has to assume
    // that the vtable field might be overwritten. This prevents optimization of
    // subsequent virtual calls on the same object. We help the optimizer by
    // allowing it to assume that the vtable field contents is the same after
    // the call. Equivalent D code:
    // ```
    //  auto saved_vtable = a.__vptr;     // emitted as part of `a.foo()`,
    //                                    // except when e->directcall==true for
    //                                    // final method calls.
    //  a.foo();
    //  assume(a.__vptr == saved_vtable); // <-- added assumption
    // ```
    // Only emit this extra code from -O2.
    // This optimization is only valid for D class method calls (not C++).
    bool canEmitVTableUnchangedAssumption =
        dfnval && dfnval->func && (dfnval->func->_linkage == LINK::d) &&
        (optLevel() >= 2);

    if (dfnval && dfnval->func) {
      assert(!DtoIsMagicIntrinsic(dfnval->func));

      // If loading the vtable was not needed for function call, we have to load
      // it here to do the "assume" optimization below.
      if (canEmitVTableUnchangedAssumption && !dfnval->vtable &&
          dfnval->vthis && dfnval->func->isVirtual()) {
        dfnval->vtable =
            DtoLoad(getVoidPtrType(), dfnval->vthis, "saved_vtable");
      }
    }

    DValue *result =
        DtoCallFunction(e->loc, e->type, fnval, e->arguments, sretPointer);

    if (canEmitVTableUnchangedAssumption && dfnval->vtable) {
      // Reload vtable ptr. It's the first element so instead of GEP+load we can
      // do a void* load+bitcast (at this point in the code we don't have easy
      // access to the type of the class to do a GEP).
      auto vtable = DtoLoad(dfnval->vtable->getType(), dfnval->vthis);
      auto cmp = p->ir->CreateICmpEQ(vtable, dfnval->vtable);
      p->ir->CreateCall(GET_INTRINSIC_DECL(assume), {cmp});
    }

    if (delayedDtorVar) {
      delayedDtorVar->edtor = delayedDtorExp;
      pushVarDtorCleanup(p, delayedDtorVar);
    }

    return result;
  }

  void visit(CallExp *e) override { result = call(p, e); }

  //////////////////////////////////////////////////////////////////////////////

  void visit(CastExp *e) override {
    IF_LOG Logger::print("CastExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    // get the value to cast
    DValue *u = toElem(e->e1);

    // handle cast to void (usually created by frontend to avoid "has no effect"
    // error)
    if (e->to == Type::tvoid) {
      result = nullptr;
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

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    DValue *base = DtoSymbolAddress(e->loc, e->var->type, e->var);

    // This weird setup is required to be able to handle both variables as
    // well as functions and TypeInfo references (which are not a DLValue
    // as well due to the level-of-indirection hack in Type::getTypeInfo that
    // is unfortunately required by the frontend).
    llvm::Value *baseValue;
    if (base->isLVal()) {
      baseValue = DtoLVal(base);
    } else {
      baseValue = DtoRVal(base);
    }
    assert(isaPointer(baseValue));

    llvm::Value *offsetValue = nullptr;

    if (e->offset == 0) {
      offsetValue = baseValue;
    } else {
      LLType *elemType = DtoType(base->type);
      if (elemType->isSized()) {
        const uint64_t elemSize = gDataLayout->getTypeAllocSize(elemType);
        if (e->offset % elemSize == 0) {
          // We can turn this into a "nice" GEP.
          const uint64_t i = e->offset / elemSize;
          // LLVM getelementptr requires that offsets are 32-bit constants
          // when the base type is a struct.
          if (target.ptrsize == 8 && !elemType->isStructTy()) {
            offsetValue = DtoGEP1i64(elemType, baseValue, i);
          } else {
            offsetValue =
                DtoGEP1(elemType, baseValue, static_cast<unsigned>(i));
          }
        }
      }

      if (!offsetValue) {
        // Offset isn't a multiple of base type size, just cast to i8* and
        // apply the byte offset.
        if (target.ptrsize == 8) {
          offsetValue = DtoGEP1i64(getI8Type(), baseValue, e->offset);
        } else {
          offsetValue =
              DtoGEP1(getI8Type(), baseValue, static_cast<unsigned>(e->offset));
        }
      }
    }

    // Casts are also "optimized into" SymOffExp by the frontend.
    LLValue *llVal = (e->type->toBasetype()->isintegral()
                          ? p->ir->CreatePtrToInt(offsetValue, DtoType(e->type))
                          : DtoBitCast(offsetValue, DtoType(e->type)));
    result = new DImValue(e->type, llVal);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(AddrExp *e) override {
    IF_LOG Logger::println("AddrExp::toElem: %s @ %s", e->toChars(),
                           e->type->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    // The address of a StructLiteralExp can in fact be a global variable, check
    // for that instead of re-codegening the literal.
    if (e->e1->op == EXP::structLiteral) {
      // lvalue literal must be a global, hence we can just use
      // toConstElem on the AddrExp to get the address.
      LLConstant *addr = toConstElem(e, p);
      IF_LOG Logger::cout()
          << "returning address of struct literal global: " << addr << '\n';
      result = new DImValue(e->type, addr);
      return;
    }

    DValue *v = toElem(e->e1, true);
    if (DFuncValue *fv = v->isFunc()) {
      Logger::println("is func");
      // Logger::println("FuncDeclaration");
      FuncDeclaration *fd = fv->func;
      assert(fd);
      result = new DFuncValue(fd, DtoCallee(fd));
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
      lval = DtoLVal(v);
    } else {
      assert(v->isSlice());
      lval = DtoAllocaDump(v, ".tmp_slice_storage");
    }

    IF_LOG Logger::cout() << "lval: " << *lval << '\n';
    result = new DImValue(e->type, lval);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(PtrExp *e) override {
    IF_LOG Logger::println("PtrExp::toElem: %s @ %s", e->toChars(),
                           e->type->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    // function pointers are special
    if (e->type->toBasetype()->ty == TY::Tfunction) {
      DValue *dv = toElem(e->e1);
      LLValue *llVal = DtoRVal(dv);
      if (DFuncValue *dfv = dv->isFunc()) {
        result = new DFuncValue(e->type, dfv->func, llVal);
      } else {
        result = new DImValue(e->type, llVal);
      }
      return;
    }

    // get the rvalue and return it as an lvalue
    LLValue *V = DtoRVal(e->e1);

    result = new DLValue(e->type, V);
  }

  static llvm::PointerType * getWithSamePointeeType(llvm::PointerType *p, unsigned addressSpace) {
#if LDC_LLVM_VER >= 1700
    return llvm::PointerType::get(p->getContext(), addressSpace);
#else
    return llvm::PointerType::getWithSamePointeeType(p, addressSpace);
#endif
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(DotVarExp *e) override {
    IF_LOG Logger::print("DotVarExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    DValue *l = toElem(e->e1);

    Type *e1type = e->e1->type->toBasetype();

    if (VarDeclaration *vd = e->var->isVarDeclaration()) {
      AggregateDeclaration *ad;
      LLValue *aggrPtr;
      // indexing struct pointer
      if (e1type->ty == TY::Tpointer) {
        auto ts = e1type->nextOf()->isTypeStruct();
        assert(ts);
        ad = ts->sym;
        aggrPtr = DtoRVal(l);
      }
      // indexing normal struct
      else if (auto ts = e1type->isTypeStruct()) {
        ad = ts->sym;
        aggrPtr = DtoLVal(l);
      }
      // indexing class
      else if (auto tc = e1type->isTypeClass()) {
        ad = tc->sym;
        aggrPtr = DtoRVal(l);
      } else {
        llvm_unreachable("Unknown DotVarExp type for VarDeclaration.");
      }

      auto ptr = DtoIndexAggregate(aggrPtr, ad, vd);

      // special case for bit fields (no real lvalues), and address spaced pointers
      if (auto bf = vd->isBitFieldDeclaration()) {
        result = new DBitFieldLValue(e->type, DtoLVal(ptr), bf);
      } else if (auto d = ptr->isDDcomputeLVal()) {
        LLType *ptrty = nullptr;
        if (llvm::PointerType *p = isaPointer(d->lltype)) {
          unsigned as = p->getAddressSpace();
          ptrty = getWithSamePointeeType(isaPointer(DtoType(e->type)), as);
        }
        else
           ptrty = DtoType(e->type);
        result = new DDcomputeLValue(e->type, i1ToI8(ptrty), DtoLVal(d));
      } else {
        result = new DLValue(e->type, DtoLVal(ptr));
      }
    } else if (FuncDeclaration *fdecl = e->var->isFuncDeclaration()) {
      // This is a bit more convoluted than it would need to be, because it
      // has to take templated interface methods into account, for which
      // isFinalFunc is not necessarily true.
      // Also, private/package methods are always non-virtual.
      const bool nonFinal = !fdecl->isFinalFunc() &&
                            (fdecl->isAbstract() || fdecl->isVirtual()) &&
                            fdecl->visibility.kind != Visibility::private_ &&
                            fdecl->visibility.kind != Visibility::package_;

      // Get the actual function value to call.
      LLValue *funcval = nullptr;
      LLValue *vtable = nullptr;
      if (nonFinal) {
        DtoResolveFunction(fdecl);
        std::tie(funcval, vtable) = DtoVirtualFunctionPointer(l, fdecl);
      } else {
        funcval = DtoCallee(fdecl);
      }
      assert(funcval);

      LLValue *vthis = (DtoIsInMemoryOnly(l->type) ? DtoLVal(l) : DtoRVal(l));
      result = new DFuncValue(fdecl, funcval, vthis, vtable);
    } else {
      llvm_unreachable("Unknown target for VarDeclaration.");
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(ThisExp *e) override {
    IF_LOG Logger::print("ThisExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    VarDeclaration *vd = nullptr;

    if (e->var) {
      vd = e->var->isVarDeclaration();
    } else {
      // special cases: `this(int) { this(); }` and `this(int) { super(); }`
      Logger::println("this exp without var declaration");
      if (auto thisArg = p->func()->thisArg) {
        result = new DLValue(e->type, thisArg);
        return;
      }
      // use the inner-most parent's `vthis`
      for (auto fd = getParentFunc(p->func()->decl); fd;
           fd = getParentFunc(fd)) {
        if (auto vthis = fd->vthis) {
          vd = vthis;
          break;
        }
      }
    }

    assert(vd);
    assert(!isSpecialRefVar(vd) && "Code not expected to handle special ref "
                                   "vars, although it can easily be made to.");

    const auto ident = p->func()->decl->ident;
    if (ident == Id::ensure || ident == Id::require) {
      Logger::println("contract this exp");
      LLValue *v = p->func()->nestArg; // thisptr lvalue
      result = new DLValue(e->type, v);
    } else if (vd->toParent2() != p->func()->decl) {
      Logger::println("nested this exp");
      result =
          DtoNestedVariable(e->loc, e->type, vd, e->type->ty == TY::Tstruct);
    } else {
      Logger::println("normal this exp");
      LLValue *v = p->func()->thisArg;
      result = new DLValue(e->type, v);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(IndexExp *e) override {
    IF_LOG Logger::print("IndexExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    DValue *l = toElem(e->e1);

    Type *e1type = e->e1->type->toBasetype();

    p->arrays.push_back(l); // if $ is used it must be an array so this is fine.
    DValue *r = toElem(e->e2);
    p->arrays.pop_back();

    LLValue *arrptr = nullptr;
    if (e1type->ty == TY::Tpointer) {
      arrptr = DtoGEP1(DtoMemType(e1type->nextOf()), DtoRVal(l), DtoRVal(r));
    } else if (e1type->ty == TY::Tsarray) {
      if (p->emitArrayBoundsChecks() && !e->indexIsInBounds) {
        DtoIndexBoundsCheck(e->loc, l, r);
      }
      LLType *elt = DtoMemType(e1type->nextOf());
      LLType *arrty = llvm::ArrayType::get(elt, e1type->isTypeSArray()->dim->isIntegerExp()->getInteger());
      arrptr = DtoGEP(arrty, DtoLVal(l), DtoConstUint(0), DtoRVal(r));
    } else if (e1type->ty == TY::Tarray) {
      if (p->emitArrayBoundsChecks() && !e->indexIsInBounds) {
        DtoIndexBoundsCheck(e->loc, l, r);
      }
      arrptr = DtoGEP1(DtoMemType(l->type->nextOf()), DtoArrayPtr(l), DtoRVal(r));
    } else if (e1type->ty == TY::Taarray) {
      result = DtoAAIndex(e->loc, e->type, l, r, e->modifiable);
      return;
    } else {
      IF_LOG Logger::println("e1type: %s", e1type->toChars());
      llvm_unreachable("Unknown IndexExp target.");
    }
    result = new DLValue(e->type, arrptr);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(SliceExp *e) override {
    IF_LOG Logger::print("SliceExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    // value being sliced
    Type *const etype = e->e1->type->toBasetype();
    LLValue *eptr = nullptr;
    LLValue *elen = nullptr;

    // evaluate the base expression but delay getting its pointer until the
    // potential bounds have been evaluated
    DValue *v = toElem(e->e1);
    auto getBasePointer = [v, etype]() {
      if (etype->ty == TY::Tpointer) {
        // pointer slicing
        return DtoRVal(v);
      } else {
        // array slice
        return DtoArrayPtr(v);
      }
    };

    // has lower bound, pointer needs adjustment
    if (e->lwr) {
      // must have upper bound too then
      assert(e->upr);

      // get bounds (make sure $ works)
      // The lower bound expression must be fully evaluated to an RVal before
      // evaluating the upper bound expression, because the lower bound
      // expression might change value after evaluating the upper bound, e.g. in
      // a statement like this: `auto a1 = values[offset .. offset += 2];`
      p->arrays.push_back(v);
      LLValue *vlo = DtoRVal(e->lwr);
      LLValue *vup = DtoRVal(e->upr);
      p->arrays.pop_back();

      const bool hasLength = etype->ty != TY::Tpointer;
      const bool needCheckUpper = hasLength && !e->upperIsInBounds();
      const bool needCheckLower = !e->lowerIsLessThanUpper();
      if (p->emitArrayBoundsChecks() && (needCheckUpper || needCheckLower)) {
        llvm::BasicBlock *okbb = p->insertBB("bounds.ok");
        llvm::BasicBlock *failbb = p->insertBBAfter(okbb, "bounds.fail");

        LLValue *const vlen = hasLength ? DtoArrayLen(v) : nullptr;

        LLValue *okCond = nullptr;
        if (needCheckUpper) {
          okCond = p->ir->CreateICmp(llvm::ICmpInst::ICMP_ULE, vup, vlen,
                                     "bounds.cmp.up");
        }

        if (needCheckLower) {
          LLValue *cmp = p->ir->CreateICmp(llvm::ICmpInst::ICMP_ULE, vlo, vup,
                                           "bounds.cmp.lo");
          if (okCond) {
            okCond = p->ir->CreateAnd(okCond, cmp);
          } else {
            okCond = cmp;
          }
        }

        p->ir->CreateCondBr(okCond, okbb, failbb);

        p->ir->SetInsertPoint(failbb);
        emitArraySliceError(p, e->loc, vlo, vup,
                            vlen ? vlen : DtoConstSize_t(0));

        p->ir->SetInsertPoint(okbb);
      }

      // offset by lower
      eptr = DtoGEP1(DtoMemType(etype->nextOf()), getBasePointer(), vlo, "lowerbound");

      // adjust length
      elen = p->ir->CreateSub(vup, vlo);
    }
    // no bounds or full slice -> just convert to slice
    else {
      assert(etype->ty != TY::Tpointer);
      eptr = getBasePointer();
      // if the slicee is a static array, we use the length of that as DMD seems
      // to give contrary inconsistent sizesin some multidimensional static
      // array cases.
      // (namely default initialization, int[16][16] arr; -> int[256] arr = 0;)
      if (etype->ty == TY::Tsarray) {
        TypeSArray *tsa = static_cast<TypeSArray *>(etype);
        elen = DtoConstSize_t(tsa->dim->toUInteger());
      }
    }

    // The frontend generates a SliceExp of static array type when assigning a
    // fixed-width slice to a static array.
    Type *const ety = e->type->toBasetype();
    if (ety->ty == TY::Tsarray) {
      result = new DLValue(e->type, eptr);
      return;
    }

    assert(ety->ty == TY::Tarray);
    if (!elen)
      elen = DtoArrayLen(v);

    result = new DSliceValue(e->type, elen, eptr);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(CmpExp *e) override {
    IF_LOG Logger::print("CmpExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    DValue *l = toElem(e->e1);
    DValue *r = toElem(e->e2);

    Type *t = e->e1->type->toBasetype();

    LLValue *eval = nullptr;

    if (t->isintegral() || t->ty == TY::Tpointer || t->ty == TY::Tnull) {
      llvm::ICmpInst::Predicate icmpPred;
      tokToICmpPred(e->op, isLLVMUnsigned(t), &icmpPred, &eval);

      if (!eval) {
        LLValue *a = DtoRVal(l);
        LLValue *b = DtoRVal(r);
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
      case EXP::lessThan:
        cmpop = llvm::FCmpInst::FCMP_OLT;
        break;
      case EXP::lessOrEqual:
        cmpop = llvm::FCmpInst::FCMP_OLE;
        break;
      case EXP::greaterThan:
        cmpop = llvm::FCmpInst::FCMP_OGT;
        break;
      case EXP::greaterOrEqual:
        cmpop = llvm::FCmpInst::FCMP_OGE;
        break;

      default:
        llvm_unreachable("Unsupported floating point comparison operator.");
      }
      eval = p->ir->CreateFCmp(cmpop, DtoRVal(l), DtoRVal(r));
    } else if (t->ty == TY::Taarray) {
      eval = LLConstantInt::getFalse(gIR->context());
    } else if (t->ty == TY::Tdelegate) {
      llvm::ICmpInst::Predicate icmpPred;
      tokToICmpPred(e->op, isLLVMUnsigned(t), &icmpPred, &eval);

      if (!eval) {
        // First compare the function pointers, then the context ones. This is
        // what DMD does.
        llvm::Value *lhs = DtoRVal(l);
        llvm::Value *rhs = DtoRVal(r);

        llvm::BasicBlock *fptreq = p->insertBB("fptreq");
        llvm::BasicBlock *fptrneq = p->insertBBAfter(fptreq, "fptrneq");
        llvm::BasicBlock *dgcmpend = p->insertBBAfter(fptrneq, "dgcmpend");

        llvm::Value *lfptr = p->ir->CreateExtractValue(lhs, 1, ".lfptr");
        llvm::Value *rfptr = p->ir->CreateExtractValue(rhs, 1, ".rfptr");

        llvm::Value *fptreqcmp = p->ir->CreateICmp(llvm::ICmpInst::ICMP_EQ,
                                                   lfptr, rfptr, ".fptreqcmp");
        llvm::BranchInst::Create(fptreq, fptrneq, fptreqcmp, p->scopebb());

        p->ir->SetInsertPoint(fptreq);
        llvm::Value *lctx = p->ir->CreateExtractValue(lhs, 0, ".lctx");
        llvm::Value *rctx = p->ir->CreateExtractValue(rhs, 0, ".rctx");
        llvm::Value *ctxcmp =
            p->ir->CreateICmp(icmpPred, lctx, rctx, ".ctxcmp");
        llvm::BranchInst::Create(dgcmpend, p->scopebb());

        p->ir->SetInsertPoint(fptrneq);
        llvm::Value *fptrcmp =
            p->ir->CreateICmp(icmpPred, lfptr, rfptr, ".fptrcmp");
        llvm::BranchInst::Create(dgcmpend, p->scopebb());

        p->ir->SetInsertPoint(dgcmpend);
        llvm::PHINode *phi = p->ir->CreatePHI(ctxcmp->getType(), 2, ".dgcmp");
        phi->addIncoming(ctxcmp, fptreq);
        phi->addIncoming(fptrcmp, fptrneq);
        eval = phi;
      }
    } else {
      llvm_unreachable("Unsupported CmpExp type");
    }

    result = zextBool(eval, e->type);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(EqualExp *e) override {
    IF_LOG Logger::print("EqualExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    DValue *l = toElem(e->e1);
    DValue *r = toElem(e->e2);

    Type *t = e->e1->type->toBasetype();

    LLValue *eval = nullptr;

    // the Tclass catches interface comparisons, regular
    // class equality should be rewritten as a.opEquals(b) by this time
    if (t->isintegral() || t->ty == TY::Tpointer || t->ty == TY::Tclass ||
        t->ty == TY::Tnull) {
      Logger::println("integral or pointer or interface");
      llvm::ICmpInst::Predicate cmpop;
      switch (e->op) {
      case EXP::equal:
        cmpop = llvm::ICmpInst::ICMP_EQ;
        break;
      case EXP::notEqual:
        cmpop = llvm::ICmpInst::ICMP_NE;
        break;
      default:
        llvm_unreachable("Unsupported integral type equality comparison.");
      }
      LLValue *lv = DtoRVal(l);
      LLValue *rv = DtoRVal(r);
      if (rv->getType() != lv->getType()) {
        rv = DtoBitCast(rv, lv->getType());
      }
      IF_LOG {
        Logger::cout() << "lv: " << *lv << '\n';
        Logger::cout() << "rv: " << *rv << '\n';
      }
      eval = p->ir->CreateICmp(cmpop, lv, rv);
    } else if (t->isfloating()) { // includes iscomplex
      eval = DtoBinNumericEquals(e->loc, l, r, e->op);
    } else if (t->ty == TY::Tsarray || t->ty == TY::Tarray) {
      Logger::println("static or dynamic array");
      eval = DtoArrayEquals(e->loc, e->op, l, r);
    } else if (t->ty == TY::Taarray) {
      Logger::println("associative array");
      eval = DtoAAEquals(e->loc, e->op, l, r);
    } else if (t->ty == TY::Tdelegate) {
      Logger::println("delegate");
      eval = DtoDelegateEquals(e->op, DtoRVal(l), DtoRVal(r));
    } else if (t->ty == TY::Tstruct) {
      Logger::println("struct");
      // when this is reached it means there is no opEquals overload.
      eval = DtoStructEquals(e->op, l, r);
    } else {
      llvm_unreachable("Unsupported EqualExp type.");
    }

    result = zextBool(eval, e->type);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(PostExp *e) override {
    IF_LOG Logger::print("PostExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    DValue *const dv = toElem(e->e1);
    LLValue *const lval = DtoLVal(dv);
    toElem(e->e2);

    LLValue *val = DtoLoad(DtoType(dv->type), lval);
    LLValue *post = nullptr;

    Type *e1type = e->e1->type->toBasetype();
    Type *e2type = e->e2->type->toBasetype();

    if (e1type->isintegral()) {
      assert(e2type->isintegral());
      LLValue *one =
          LLConstantInt::get(val->getType(), 1, !e2type->isunsigned());
      if (e->op == EXP::plusPlus) {
        post = llvm::BinaryOperator::CreateAdd(val, one, "", p->scopebb());
      } else if (e->op == EXP::minusMinus) {
        post = llvm::BinaryOperator::CreateSub(val, one, "", p->scopebb());
      }
    } else if (e1type->ty == TY::Tpointer) {
      assert(e->e2->op == EXP::int64);
      LLConstant *offset =
          e->op == EXP::plusPlus ? DtoConstUint(1) : DtoConstInt(-1);
      post = DtoGEP1(DtoMemType(dv->type->nextOf()), val, offset, "", p->scopebb());
    } else if (e1type->iscomplex()) {
      assert(e2type->iscomplex());
      LLValue *one = LLConstantFP::get(DtoComplexBaseType(e1type), 1.0);
      LLValue *re, *im;
      DtoGetComplexParts(e->loc, e1type, dv, re, im);
      if (e->op == EXP::plusPlus) {
        re = llvm::BinaryOperator::CreateFAdd(re, one, "", p->scopebb());
      } else if (e->op == EXP::minusMinus) {
        re = llvm::BinaryOperator::CreateFSub(re, one, "", p->scopebb());
      }
      DtoComplexSet(DtoType(dv->type), lval, re, im);
    } else if (e1type->isfloating()) {
      assert(e2type->isfloating());
      LLValue *one = DtoConstFP(e1type, ldouble(1.0));
      if (e->op == EXP::plusPlus) {
        post = llvm::BinaryOperator::CreateFAdd(val, one, "", p->scopebb());
      } else if (e->op == EXP::minusMinus) {
        post = llvm::BinaryOperator::CreateFSub(val, one, "", p->scopebb());
      }
    } else {
      llvm_unreachable("Unsupported type for PostExp.");
    }

    // The real part of the complex number has already been updated, skip the
    // store
    if (!e1type->iscomplex()) {
      DtoStore(post, lval);
    }
    result = new DImValue(e->type, val);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(NewExp *e) override {
    IF_LOG Logger::print("NewExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    bool isArgprefixHandled = false;

    assert(e->newtype);
    Type *ntype = e->newtype->toBasetype();

    // new class
    if (ntype->ty == TY::Tclass) {
      Logger::println("new class");
      result = DtoNewClass(e->loc, static_cast<TypeClass *>(ntype), e);
      isArgprefixHandled = true; // by DtoNewClass()
    }
    // new dynamic array
    else if (ntype->ty == TY::Tarray) {
      IF_LOG Logger::println("new dynamic array: %s", e->newtype->toChars());
      assert(e->argprefix == NULL);
      // get dim
      assert(e->arguments);
      assert(e->arguments->length >= 1);
      if (e->arguments->length == 1) {
        DValue *sz = toElem((*e->arguments)[0]);
        // allocate & init
        result = DtoNewDynArray(e->loc, e->newtype, sz, true);
      } else {
        assert(e->lowering);
        LLValue *pair = DtoRVal(e->lowering);
        result = new DSliceValue(e->type, pair);
      }
    }
    // new static array
    else if (ntype->ty == TY::Tsarray) {
      llvm_unreachable("Static array new should decay to dynamic array.");
    }
    // new struct
    else if (ntype->ty == TY::Tstruct) {
      IF_LOG Logger::println("new struct on heap: %s\n", e->newtype->toChars());

      TypeStruct *ts = static_cast<TypeStruct *>(ntype);

      // allocate (via _d_newitemT template lowering)
      assert(e->lowering);
      LLValue *mem = DtoRVal(e->lowering);

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
          DFuncValue dfn(e->member, DtoCallee(e->member), mem);
          DtoCallFunction(e->loc, ts, &dfn, e->arguments);
        }
      }

      result = new DImValue(e->type, mem);
    }
    // new AA
    else if (auto taa = ntype->isTypeAArray()) {
      LLFunction *func = getRuntimeFunction(e->loc, gIR->module, "_aaNew");
      LLValue *aaTypeInfo = DtoTypeInfoOf(e->loc, stripModifiers(taa));
      LLValue *aa = gIR->CreateCallOrInvoke(func, aaTypeInfo, "aa");
      result = new DImValue(e->type, aa);
    }
    // new basic type
    else {
      IF_LOG Logger::println("basic type on heap: %s\n", e->newtype->toChars());
      assert(e->argprefix == NULL);

      // allocate
      LLValue *mem = DtoNew(e->loc, e->newtype);
      DLValue tmpvar(e->newtype, mem);

      Expression *exp = nullptr;
      if (!e->arguments || e->arguments->length == 0) {
        IF_LOG Logger::println("default initializer\n");
        // static arrays never appear here, so using the defaultInit is ok!
        exp = defaultInit(e->newtype, e->loc);
      } else {
        IF_LOG Logger::println("uniform constructor\n");
        assert(e->arguments->length == 1);
        exp = (*e->arguments)[0];
      }

      // try to construct it in-place
      if (!toInPlaceConstruction(&tmpvar, exp))
        DtoAssign(e->loc, &tmpvar, toElem(exp), EXP::blit);

      // return as pointer-to
      result = new DImValue(e->type, mem);
    }

    (void)isArgprefixHandled;
    assert(e->argprefix == NULL || isArgprefixHandled);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(DeleteExp *e) override {
    IF_LOG Logger::print("DeleteExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    DValue *dval = toElem(e->e1);
    Type *et = e->e1->type->toBasetype();

    // pointer
    if (et->ty == TY::Tpointer) {
      Type *elementType = et->nextOf()->toBasetype();
      if (elementType->ty == TY::Tstruct && elementType->needsDestruction()) {
        DtoDeleteStruct(e->loc, dval);
      } else {
        DtoDeleteMemory(e->loc, dval);
      }
    }
    // class
    else if (et->ty == TY::Tclass) {
      bool onstack = false;
      TypeClass *tc = static_cast<TypeClass *>(et);
      if (tc->sym->isInterfaceDeclaration()) {
        DtoDeleteInterface(e->loc, dval);
        onstack = true;
      } else if (auto ve = e->e1->isVarExp()) {
        if (auto vd = ve->var->isVarDeclaration()) {
          if (vd->onstack()) {
            DtoFinalizeScopeClass(e->loc, dval,
                                  vd->onstackWithMatchingDynType());
            onstack = true;
          }
        }
      }

      if (!onstack) {
        DtoDeleteClass(e->loc, dval); // sets dval to null
      } else if (dval->isLVal()) {
        LLValue *lval = DtoLVal(dval);
        DtoStore(LLConstant::getNullValue(DtoType(dval->type)),
                 lval);
      }
    }
    // dyn array
    else if (et->ty == TY::Tarray) {
      DtoDeleteArray(e->loc, dval);
      if (DLValue *ldval = dval->isLVal()) {
        DtoSetArrayToNull(ldval);
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

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    DValue *u = toElem(e->e1);
    result = new DImValue(e->type, DtoArrayLen(u));
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(ThrowExp *e) override {
    IF_LOG Logger::print("ThrowExp::toElem: %s\n", e->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    DtoThrow(e->loc, toElem(e->e1));
    result = new DNullValue(e->type, llvm::UndefValue::get(DtoType(e->type)));
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(AssertExp *e) override {
    IF_LOG Logger::print("AssertExp::toElem: %s\n", e->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    if (global.params.useAssert != CHECKENABLEon)
      return;

    // condition
    DValue *cond;
    Type *condty;

    cond = toElem(e->e1);
    condty = e->e1->type->toBasetype();

    // create basic blocks
    llvm::BasicBlock *passedbb = p->insertBB("assertPassed");
    llvm::BasicBlock *failedbb = p->insertBBAfter(passedbb, "assertFailed");

    // test condition
    LLValue *condval = DtoRVal(DtoCast(e->loc, cond, Type::tbool));

    // branch
    llvm::BranchInst::Create(passedbb, failedbb, condval, p->scopebb());
    // The branch does not need instrumentation for PGO because failedbb
    // terminates in unreachable, which means that LLVM will automatically
    // assign branch weights to this branch instruction.

    // failed: call assert runtime function
    p->ir->SetInsertPoint(failedbb);

    if (global.params.checkAction == CHECKACTION_halt) {
      p->ir->CreateCall(GET_INTRINSIC_DECL(trap), {});
      p->ir->CreateUnreachable();
    } else {
      /* DMD Bugzilla 8360: If the condition is evaluated to true,
       * msg is not evaluated at all. So should use toElemDtor()
       * instead of toElem().
       */
      DValue *msg = e->msg ? toElemDtor(e->msg) : nullptr;
      Module *module = p->func()->decl->getModule();
      if (global.params.checkAction == CHECKACTION_C ||
          module->filetype == FileType::c) {
        LLValue *cMsg =
            msg ? DtoArrayPtr(
                      msg) // assuming `msg` is null-terminated, like DMD
                : DtoConstCString(e->e1->toChars());
        DtoCAssert(module, e->e1->loc, cMsg);
      } else {
        DtoAssert(module, e->loc, msg);
      }
    }

    // passed:
    p->ir->SetInsertPoint(passedbb);

    // class/struct invariants
    if (global.params.useInvariants != CHECKENABLEon)
      return;
    if (auto tc = condty->isTypeClass()) {
      const auto sym = tc->sym;
      if (sym->isInterfaceDeclaration() || sym->isCPPclass())
        return;

      Logger::println("calling class invariant");

      const auto fnMangle =
          getIRMangledFuncName("_D9invariant12_d_invariantFC6ObjectZv", LINK::d);
      const auto fn = getRuntimeFunction(e->loc, gIR->module, fnMangle.c_str());

      const auto arg = DtoRVal(cond);

      gIR->CreateCallOrInvoke(fn, arg);
    } else if (condty->ty == TY::Tpointer &&
               condty->nextOf()->ty == TY::Tstruct) {
      const auto invDecl =
          static_cast<TypeStruct *>(condty->nextOf())->sym->inv;
      if (!invDecl)
        return;

      Logger::print("calling struct invariant");

      DFuncValue invFunc(invDecl, DtoCallee(invDecl), DtoRVal(cond));
      DtoCallFunction(e->loc, nullptr, &invFunc, nullptr);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(NotExp *e) override {
    IF_LOG Logger::print("NotExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    DValue *u = toElem(e->e1);

    LLValue *b = DtoRVal(DtoCast(e->loc, u, Type::tbool));

    LLConstant *zero = DtoConstBool(false);
    b = p->ir->CreateICmpEQ(b, zero);

    result = zextBool(b, e->type);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(LogicalExp *e) override {
    IF_LOG Logger::print("LogicalExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    DValue *u = toElem(e->e1);

    const bool isAndAnd = (e->op == EXP::andAnd); // otherwise OrOr
    llvm::BasicBlock *rhsBB = p->insertBB(isAndAnd ? "andand" : "oror");
    llvm::BasicBlock *endBB =
        p->insertBBAfter(rhsBB, isAndAnd ? "andandend" : "ororend");

    LLValue *ubool = DtoRVal(DtoCast(e->loc, u, Type::tbool));

    llvm::BasicBlock *oldblock = p->scopebb();
    uint64_t truecount, falsecount;
    if (isAndAnd) {
      truecount = PGO.getRegionCount(e);
      falsecount = PGO.getCurrentRegionCount() - truecount;
    } else {
      falsecount = PGO.getRegionCount(e);
      truecount = PGO.getCurrentRegionCount() - falsecount;
    }
    auto branchweights = PGO.createProfileWeights(truecount, falsecount);
    p->ir->CreateCondBr(ubool, isAndAnd ? rhsBB : endBB,
                        isAndAnd ? endBB : rhsBB, branchweights);

    p->ir->SetInsertPoint(rhsBB);
    PGO.emitCounterIncrement(e);
    emitCoverageLinecountInc(e->e2->loc);
    DValue *v = toElemDtor(e->e2);

    LLValue *vbool = nullptr;
    if (v && !v->isFunc() && v->type != Type::tvoid) {
      vbool = DtoRVal(DtoCast(e->loc, v, Type::tbool));
    }

    llvm::BasicBlock *newblock = p->scopebb();
    llvm::BranchInst::Create(endBB, p->scopebb());
    p->ir->SetInsertPoint(endBB);

    // DMD allows stuff like `x == 0 && assert(false)`
    if (e->type->toBasetype()->ty == TY::Tvoid) {
      result = nullptr;
      return;
    }

    LLValue *resval = nullptr;
    if (ubool == vbool || !vbool) {
      // No need to create a PHI node.
      resval = ubool;
    } else {
      llvm::PHINode *phi =
          p->ir->CreatePHI(LLType::getInt1Ty(gIR->context()), 2,
                           isAndAnd ? "andandval" : "ororval");
      if (isAndAnd) {
        // If we jumped over evaluation of the right-hand side,
        // the result is false. Otherwise it's the value of the right-hand side.
        phi->addIncoming(LLConstantInt::getFalse(gIR->context()), oldblock);
      } else {
        // If we jumped over evaluation of the right-hand side,
        // the result is true. Otherwise, it's the value of the right-hand side.
        phi->addIncoming(LLConstantInt::getTrue(gIR->context()), oldblock);
      }
      phi->addIncoming(vbool, newblock);
      resval = phi;
    }

    result = zextBool(resval, e->type);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(HaltExp *e) override {
    IF_LOG Logger::print("HaltExp::toElem: %s\n", e->toChars());
    LOG_SCOPE;

    p->ir->CreateCall(GET_INTRINSIC_DECL(trap), {});
    p->ir->CreateUnreachable();

    // this terminated the basicblock, start a new one
    // this is sensible, since someone might goto behind the assert
    // and prevents compiler errors if a terminator follows the assert
    llvm::BasicBlock *bb = p->insertBB("afterhalt");
    p->ir->SetInsertPoint(bb);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(DelegateExp *e) override {
    IF_LOG Logger::print("DelegateExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    if (e->func->isStatic()) {
      error(e->loc,
            "can't take delegate of static function `%s`, it does not "
            "require a context ptr",
            e->func->toChars());
    }

    assert(e->type->toBasetype()->ty == TY::Tdelegate);

    DValue *u = toElem(e->e1);
    LLValue *contextptr;
    if (DFuncValue *f = u->isFunc()) {
      assert(f->func);
      contextptr = DtoNestedContext(e->loc, f->func);
    } else {
      contextptr = (DtoIsInMemoryOnly(u->type) ? DtoLVal(u) : DtoRVal(u));
    }

    IF_LOG Logger::cout() << "context = " << *contextptr << '\n';

    IF_LOG Logger::println("func: '%s'", e->func->toPrettyChars());

    LLValue *fptr;

    if (e->e1->op != EXP::super_ && e->e1->op != EXP::dotType &&
        e->func->isVirtual() && !e->func->isFinalFunc()) {
      fptr = DtoVirtualFunctionPointer(u, e->func).first;
    } else if (e->func->isAbstract()) {
      llvm_unreachable("Delegate to abstract method not implemented.");
    } else if (e->func->toParent()->isInterfaceDeclaration()) {
      llvm_unreachable("Delegate to interface method not implemented.");
    } else {
      DtoResolveFunction(e->func);

      // We need to actually codegen the function here, as literals are not
      // added to the module member list.
      if (e->func->semanticRun == PASS::semantic3done) {
        Dsymbol *owner = e->func->toParent();
        while (!owner->isTemplateInstance() && owner->toParent()) {
          owner = owner->toParent();
        }
        if (owner->isTemplateInstance() || owner == p->dmodule) {
          Declaration_codegen(e->func, p);
        }
      }

      fptr = DtoCallee(e->func);
    }

    result = new DImValue(
        e->type, DtoAggrPair(DtoType(e->type), contextptr, fptr, ".dg"));
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(IdentityExp *e) override {
    IF_LOG Logger::print("IdentityExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    DValue *l = toElem(e->e1);
    DValue *r = toElem(e->e2);

    Type *t1 = e->e1->type->toBasetype();

    // handle dynarray specially
    if (t1->ty == TY::Tarray) {
      result = zextBool(DtoDynArrayIs(e->op, l, r), e->type);
      return;
    }
    // also structs
    if (t1->ty == TY::Tstruct) {
      result = zextBool(DtoStructEquals(e->op, l, r), e->type);
      return;
    }

    // FIXME this stuff isn't pretty
    LLValue *eval = nullptr;

    if (t1->ty == TY::Tdelegate) {
      LLValue *lv = DtoRVal(l);
      LLValue *rv = nullptr;
      if (!r->isNull()) {
        rv = DtoRVal(r);
        assert(lv->getType() == rv->getType());
      }
      eval = DtoDelegateEquals(e->op, lv, rv);
    } else if (t1->isfloating()) { // includes iscomplex
      eval = DtoBinNumericEquals(e->loc, l, r, e->op);
    } else if (t1->ty == TY::Tpointer || t1->ty == TY::Tclass) {
      LLValue *lv = DtoRVal(l);
      LLValue *rv = DtoRVal(r);
      if (lv->getType() != rv->getType()) {
        if (r->isNull()) {
          rv = llvm::ConstantPointerNull::get(isaPointer(lv));
        } else {
          rv = DtoBitCast(rv, lv->getType());
        }
      }
      eval = (e->op == EXP::identity) ? p->ir->CreateICmpEQ(lv, rv)
                                      : p->ir->CreateICmpNE(lv, rv);
    } else if (t1->ty == TY::Tsarray) {
      LLValue *lptr = DtoLVal(l);
      LLValue *rptr = DtoLVal(r);
      assert(lptr->getType() == rptr->getType());
      eval = (e->op == EXP::identity) ? p->ir->CreateICmpEQ(lptr, rptr)
                                      : p->ir->CreateICmpNE(lptr, rptr);
    } else {
      LLValue *lv = DtoRVal(l);
      LLValue *rv = DtoRVal(r);
      assert(lv->getType() == rv->getType());
      eval = (e->op == EXP::identity) ? p->ir->CreateICmpEQ(lv, rv)
                                      : p->ir->CreateICmpNE(lv, rv);
      if (t1->ty == TY::Tvector) {
        eval = mergeVectorEquals(eval, e->op == EXP::identity ? EXP::equal
                                                              : EXP::notEqual);
      }
    }
    result = zextBool(eval, e->type);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(CommaExp *e) override {
    IF_LOG Logger::print("CommaExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

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

    auto &PGO = gIR->funcGen().pgo;
    PGO.setCurrentStmt(e);

    Type *dtype = e->type->toBasetype();
    LLValue *retPtr = nullptr;
    if (!(dtype->ty == TY::Tvoid || dtype->ty == TY::Tnoreturn)) {
      // allocate a temporary for pointer to the final result.
      retPtr = DtoAlloca(pointerTo(dtype), "condtmp");
    }

    llvm::BasicBlock *condtrue = p->insertBB("condtrue");
    llvm::BasicBlock *condfalse = p->insertBBAfter(condtrue, "condfalse");
    llvm::BasicBlock *condend = p->insertBBAfter(condfalse, "condend");

    DValue *c = toElem(e->econd);
    LLValue *cond_val = DtoRVal(DtoCast(e->loc, c, Type::tbool));

    auto truecount = PGO.getRegionCount(e);
    auto falsecount = PGO.getCurrentRegionCount() - truecount;
    auto branchweights = PGO.createProfileWeights(truecount, falsecount);
    p->ir->CreateCondBr(cond_val, condtrue, condfalse, branchweights);

    p->ir->SetInsertPoint(condtrue);
    PGO.emitCounterIncrement(e);
    DValue *u = toElem(e->e1);
    if (retPtr && u->type->toBasetype()->ty != TY::Tnoreturn) {
      LLValue *lval = makeLValue(e->loc, u);
      DtoStore(lval, retPtr);
    }
    llvm::BranchInst::Create(condend, p->scopebb());

    p->ir->SetInsertPoint(condfalse);
    DValue *v = toElem(e->e2);
    if (retPtr && v->type->toBasetype()->ty != TY::Tnoreturn) {
      LLValue *lval = makeLValue(e->loc, v);
      DtoStore(lval, retPtr);
    }
    llvm::BranchInst::Create(condend, p->scopebb());

    p->ir->SetInsertPoint(condend);
    if (retPtr)
      result = new DSpecialRefValue(e->type, retPtr);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(ComExp *e) override {
    IF_LOG Logger::print("ComExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    LLValue *value = DtoRVal(e->e1);
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

    DRValue *dval = toElem(e->e1)->getRVal();

    if (e->type->iscomplex()) {
      result = DtoComplexNeg(e->loc, e->type, dval);
      return;
    }

    LLValue *val = DtoRVal(dval);

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

    if (!global.params.useGC) {
      error(
          e->loc,
          "array concatenation of expression `%s` requires the GC which is not "
          "available with -betterC",
          e->toChars());
      result =
          new DSliceValue(e->type, llvm::UndefValue::get(DtoType(e->type)));
      return;
    }

    if (e->lowering) {
      result = toElem(e->lowering);
      return;
    }

    llvm_unreachable("CatExp should have been lowered");
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(CatAssignExp *e) override {
    IF_LOG Logger::print("CatAssignExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    if (!global.params.useGC) {
      error(e->loc,
            "appending to array in `%s` requires the GC which is not available "
            "with -betterC",
            e->toChars());
      result =
          new DSliceValue(e->type, llvm::UndefValue::get(DtoType(e->type)));
      return;
    }

    if (e->lowering) {
      assert(e->op != EXP::concatenateDcharAssign);
      result = toElem(e->lowering);
      return;
    }

    result = toElem(e->e1);

    Type *e1type = e->e1->type->toBasetype();
    assert(e1type->ty == TY::Tarray);
    Type *elemtype = e1type->nextOf()->toBasetype();
    Type *e2type = e->e2->type->toBasetype();

    if (e1type->ty == TY::Tarray && e2type->ty == TY::Tdchar &&
        (elemtype->ty == TY::Tchar || elemtype->ty == TY::Twchar)) {
      assert(e->op == EXP::concatenateDcharAssign);
      if (elemtype->ty == TY::Tchar) {
        // append dchar to char[]
        DtoAppendDCharToString(e->loc, result, e->e2);
      } else { /*if (elemtype->ty == TY::Twchar)*/
        // append dchar to wchar[]
        DtoAppendDCharToUnicodeString(e->loc, result, e->e2);
      }
    } else {
      error(e->loc, "ICE: array append should have been lowered to `_d_arrayappend{T,cTX}`!");
      fatal();
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void genFuncLiteral(FuncLiteralDeclaration *fd, FuncExp *e) {
    if ((fd->tok == TOK::reserved || fd->tok == TOK::delegate_) &&
        (e && e->type->ty == TY::Tpointer)) {
      // This is a lambda that was inferred to be a function literal instead
      // of a delegate, so set tok here in order to get correct types/mangling.
      // Horrible hack, but DMD does the same thing.
      fd->tok = TOK::function_;
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
    assert(DtoCallee(fd, false));
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(FuncExp *e) override {
    IF_LOG Logger::print("FuncExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    FuncLiteralDeclaration *fd = e->fd;
    assert(fd);

    genFuncLiteral(fd, e);
    LLFunction *callee = DtoCallee(fd, false);

    if (fd->isNested()) {
      LLValue *cval = DtoNestedContext(e->loc, fd);
      result = new DImValue(e->type, DtoAggrPair(cval, callee, ".func"));
    } else {
      result = new DFuncValue(e->type, fd, callee);
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
    bool const dyn = (arrayType->ty == TY::Tarray);
    // length
    size_t const len = e->elements->length;

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
      return;
    }

    // allocated on the stack?
    if (!dyn || e->onstack) {
      llvm::Value *storage =
          DtoRawAlloca(llStoType, DtoAlignment(elemType), "arrayliteral");
      initializeArrayLiteral(p, e, storage, llStoType);

      if (arrayType->ty == TY::Tsarray) {
        result = new DLValue(e->type, storage);
        return;
      }

      if (arrayType->ty == TY::Tarray) {
        result = new DSliceValue(e->type, DtoConstSize_t(len), storage);
      } else if (arrayType->ty == TY::Tpointer) {
        result = new DImValue(e->type, storage);
      } else {
        llvm_unreachable("Unexpected array literal type");
      }
      return;
    }

    // we're dealing with a non-stack dynamic array literal now
    if (arrayType->isImmutable() && isConstLiteral(e, true)) {
      llvm::Constant *init = arrayLiteralToConst(p, e);
      auto global = new llvm::GlobalVariable(gIR->module, init->getType(), true,
                                             llvm::GlobalValue::InternalLinkage,
                                             init, ".immutablearray");
      result = new DSliceValue(arrayType, DtoConstSize_t(len), global);
    } else {
      DSliceValue *dynSlice = DtoNewDynArray(
          e->loc, arrayType,
          new DConstValue(Type::tsize_t, DtoConstSize_t(len)), false);
      initializeArrayLiteral(p, e, dynSlice->getPtr(), llStoType);
      result = dynSlice;
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  static DLValue *emitStructLiteral(StructLiteralExp *e,
                                    LLValue *dstMem = nullptr) {
    IF_LOG Logger::print("StructLiteralExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    if (e->useStaticInit) {
      StructDeclaration *sd = e->sd;
      DtoResolveStruct(sd);

      if (!dstMem)
        dstMem = DtoAlloca(e->type, ".structliteral");

      if (sd->zeroInit()) {
        DtoMemSetZero(DtoType(e->type), dstMem);
      } else {
        LLValue *initsym = getIrAggr(sd)->getInitSymbol();
        assert(dstMem->getType() == initsym->getType());
        DtoMemCpy(DtoType(e->type), dstMem, initsym);
      }

      return new DLValue(e->type, dstMem);
    }

    if (e->inProgressMemory) {
      assert(!dstMem);
      return new DLValue(e->type, e->inProgressMemory);
    }

    // make sure the struct is fully resolved
    DtoResolveStruct(e->sd);

    if (!dstMem)
      dstMem = DtoAlloca(e->type, ".structliteral");

    e->inProgressMemory = dstMem;
    write_struct_literal(e->loc, dstMem, e->sd, e->elements);
    e->inProgressMemory = nullptr;

    return new DLValue(e->type, dstMem);
  }

  void visit(StructLiteralExp *e) override { result = emitStructLiteral(e); }

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
    assert(e->keys->length == e->values->length);

    Type *basetype = e->type->toBasetype();
    Type *aatype = basetype;
    Type *vtype = aatype->nextOf();

    if (!e->keys->length) {
      goto LruntimeInit;
    }

    if (aatype->ty != TY::Taarray) {
      // It's the AssociativeArray type.
      // Turn it back into a TypeAArray
      vtype = e->values->tdata()[0]->type;
      aatype = TypeAArray::create(vtype, e->keys->tdata()[0]->type);
      aatype = typeSemantic(aatype, e->loc, nullptr);
    }

    {
      std::vector<LLConstant *> keysInits, valuesInits;
      keysInits.reserve(e->keys->length);
      valuesInits.reserve(e->keys->length);
      for (size_t i = 0, n = e->keys->length; i < n; ++i) {
        Expression *ekey = (*e->keys)[i];
        Expression *eval = (*e->values)[i];
        IF_LOG Logger::println("(%llu) aa[%s] = %s",
                               static_cast<unsigned long long>(i),
                               ekey->toChars(), eval->toChars());
        LLConstant *ekeyConst = tryToConstElem(ekey, p);
        LLConstant *evalConst = tryToConstElem(eval, p);
        if (!ekeyConst || !evalConst) {
          goto LruntimeInit;
        }
        keysInits.push_back(ekeyConst);
        valuesInits.push_back(evalConst);
      }

      assert(aatype->ty == TY::Taarray);
      Type *indexType = static_cast<TypeAArray *>(aatype)->index;
      assert(indexType && vtype);

      llvm::Function *func =
          getRuntimeFunction(e->loc, gIR->module, "_d_assocarrayliteralTX");
      LLValue *aaTypeInfo = DtoTypeInfoOf(e->loc, stripModifiers(aatype));

      LLConstant *initval = arrayConst(keysInits, indexType);
      LLConstant *globalstore = new LLGlobalVariable(
          gIR->module, initval->getType(), false,
          LLGlobalValue::InternalLinkage, initval, ".aaKeysStorage");
      LLValue *keysArray = DtoConstSlice(DtoConstSize_t(e->keys->length), globalstore);

      initval = arrayConst(valuesInits, vtype);
      globalstore = new LLGlobalVariable(gIR->module, initval->getType(), false,
                                         LLGlobalValue::InternalLinkage,
                                         initval, ".aaValuesStorage");
      LLValue *valuesArray = DtoConstSlice(DtoConstSize_t(e->keys->length), globalstore);

      LLValue *aa = gIR->CreateCallOrInvoke(func, aaTypeInfo, keysArray,
                                            valuesArray, "aa");
      if (basetype->ty != TY::Taarray) {
        LLValue *tmp = DtoAlloca(e->type, "aaliteral");
        DtoStore(aa, DtoGEP(DtoType(e->type), tmp, 0u, 0));
        result = new DLValue(e->type, tmp);
      } else {
        result = new DImValue(e->type, aa);
      }

      return;
    }

  LruntimeInit:

    // it should be possible to avoid the temporary in some cases
    LLValue *tmp = DtoAllocaDump(LLConstant::getNullValue(DtoType(e->type)),
                                 e->type, "aaliteral");
    result = new DLValue(e->type, tmp);

    const size_t n = e->keys->length;
    for (size_t i = 0; i < n; ++i) {
      Expression *ekey = (*e->keys)[i];
      Expression *eval = (*e->values)[i];

      IF_LOG Logger::println("(%llu) aa[%s] = %s",
                             static_cast<unsigned long long>(i),
                             ekey->toChars(), eval->toChars());

      // index
      DValue *key = toElem(ekey);
      DLValue *mem = DtoAAIndex(e->loc, vtype, result, key, true);

      // try to construct it in-place
      if (!toInPlaceConstruction(mem, eval))
        DtoAssign(e->loc, mem, toElem(eval), EXP::blit);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  DValue *toGEP(UnaExp *exp, unsigned index) {
    // (&a.foo).funcptr is a case where toElem(e1) is genuinely not an l-value.
    DValue * dv = toElem(exp->e1);
    LLValue *val = makeLValue(exp->loc, dv);
    LLValue *v = DtoGEP(DtoType(dv->type),  val, 0, index);
    return new DLValue(exp->type, v);
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

  void visit(DotTypeExp *e) override {
    IF_LOG Logger::print("DotTypeExp::toElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    assert(e->sym->getType());
    result = toElem(e->e1);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(TypeExp *e) override {
    error(e->loc, "type `%s` is not an expression", e->toChars());
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
    types.reserve(e->exps->length);
    for (auto exp : *e->exps) {
      types.push_back(DtoMemType(exp->type));
    }
    llvm::StructType *st = llvm::StructType::get(gIR->context(), types);
    LLValue *val = DtoRawAlloca(st, 0, ".tuple");
    for (size_t i = 0; i < e->exps->length; i++) {
      Expression *el = (*e->exps)[i];
      DValue *ep = toElem(el);
      LLValue *gep = DtoGEP(st, val, 0, i);
      if (DtoIsInMemoryOnly(el->type)) {
        DtoMemCpy(st->getContainedType(i), gep, DtoLVal(ep));
      } else if (el->type->ty != TY::Tvoid) {
        DtoStoreZextI8(DtoRVal(ep), gep);
      } else {
        DtoStore(LLConstantInt::get(LLType::getInt8Ty(p->context()), 0, false),
                 gep);
      }
    }
    result = new DLValue(e->type, val);
  }

  //////////////////////////////////////////////////////////////////////////////

  static DLValue *emitVector(VectorExp *e, LLValue *dstMem) {
    TypeVector *type = e->to->toBasetype()->isTypeVector();
    assert(type);

    const unsigned N = e->dim;

    Type *elementType = type->elementType();
    if (elementType->ty == TY::Tvoid)
      elementType = Type::tuns8;

    const auto getCastElement = [e, elementType](DValue *element) {
      return DtoRVal(DtoCast(e->loc, element, elementType));
    };

    LLType *dstType = DtoType(e->type);
    Type *tsrc = e->e1->type->toBasetype();
    if (auto lit = e->e1->isArrayLiteralExp()) {
      // Optimization for array literals: check for a fully static literal and
      // store a vector constant in that case, otherwise emplace element-wise
      // into destination memory.
      Logger::println("array literal expression");
      assert(lit->elements->length == N &&
             "Array literal vector initializer "
             "length mismatch, should have been handled in frontend.");

      std::vector<LLValue *> llElements;
      std::vector<LLConstant *> llElementConstants;
      llElements.reserve(N);
      llElementConstants.reserve(N);
      for (unsigned i = 0; i < N; ++i) {
        DValue *val = toElem(indexArrayLiteral(lit, i));
        LLValue *llVal = getCastElement(val);
        llElements.push_back(llVal);
        if (auto llConstant = isaConstant(llVal))
          llElementConstants.push_back(llConstant);
      }

      if (llElementConstants.size() == N) {
        auto vectorConstant = llvm::ConstantVector::get(llElementConstants);
        DtoStore(vectorConstant, dstMem);
      } else {
        for (unsigned i = 0; i < N; ++i) {
          DtoStore(llElements[i], DtoGEP(dstType, dstMem, 0, i));
        }
      }
    } else if (tsrc->ty == TY::Tarray || tsrc->ty == TY::Tsarray) {
      // Arrays: prefer a memcpy if the LL element types match, otherwise cast
      // and store element-wise.
      if (auto ts = tsrc->isTypeSArray()) {
        Logger::println("static array expression");
        (void)ts;
        assert(ts->dim->toInteger() == N &&
               "Static array vector initializer length mismatch, should have "
               "been handled in frontend.");
      } else {
        // TODO: bounds check?
        Logger::println("dynamic array expression, assume matching length");
      }

      DValue *e1 = toElem(e->e1);
      LLValue *arrayPtr = DtoArrayPtr(e1);
      Type *srcElementType = tsrc->nextOf();

      if (DtoMemType(elementType) == DtoMemType(srcElementType)) {
        DtoMemCpy(dstType, dstMem, arrayPtr);
      } else {
        for (unsigned i = 0; i < N; ++i) {
          LLValue *gep = DtoGEP1(DtoMemType(e1->type->nextOf()), arrayPtr, i);
          DLValue srcElement(srcElementType, gep);
          LLValue *llVal = getCastElement(&srcElement);
          DtoStore(llVal, DtoGEP(dstType, dstMem, 0, i));
        }
      }
    } else {
      // Try a splat vector constant, otherwise store element-wise.
      Logger::println("normal (splat) expression");
      DValue *val = toElem(e->e1);
      LLValue *llElement = getCastElement(val);
      if (auto llConstant = isaConstant(llElement)) {
        const auto elementCount = llvm::ElementCount::getFixed(N);
        auto vectorConstant =
            llvm::ConstantVector::getSplat(elementCount, llConstant);
        DtoStore(vectorConstant, dstMem);
      } else {
        for (unsigned int i = 0; i < N; ++i) {
          DtoStore(llElement, DtoGEP(dstType, dstMem, 0, i));
        }
      }
    }

    return new DLValue(e->to, dstMem);
  }

  void visit(VectorExp *e) override {
    IF_LOG Logger::print("VectorExp::toElem() %s\n", e->toChars());
    LOG_SCOPE;

    LLValue *vector = DtoAlloca(e->to);
    result = emitVector(e, vector);
  }

  void visit(VectorArrayExp* e) override {
    IF_LOG Logger::print("VectorArrayExp::toElem() %s\n", e->toChars());
    LOG_SCOPE;

    DValue *vector = toElem(e->e1);
    result = DtoCastVector(e->loc, vector, e->type);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(PowExp *e) override {
    IF_LOG Logger::print("PowExp::toElem() %s\n", e->toChars());
    LOG_SCOPE;

    error(e->loc, "must import `std.math` to use `^^` operator");
    result = new DNullValue(e->type, llvm::UndefValue::get(DtoType(e->type)));
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(TypeidExp *e) override {
    if (Type *t = isType(e->obj)) {
      result = DtoSymbolAddress(e->loc, e->type,
                                getOrCreateTypeInfoDeclaration(e->loc, t));
      return;
    }
    if (Expression *ex = isExpression(e->obj)) {
      const auto tc = ex->type->toBasetype()->isTypeClass();
      assert(tc);

      const auto irtc = getIrType(tc->sym->type, true)->isClass();
      const auto vtblType = irtc->getVtblType();
      LLValue *val = DtoRVal(ex);

      // Get and load vtbl pointer.
      llvm::Value *vtbl = DtoLoad(vtblType->getPointerTo(),
                                  DtoGEP(irtc->getMemoryLLType(), val, 0u, 0));

      // TypeInfo ptr is first vtbl entry.
      llvm::Value *typinf = DtoGEP(vtblType, vtbl, 0u, 0);

      Type *resultType;
      if (tc->sym->isInterfaceDeclaration()) {
        // For interfaces, the first entry in the vtbl is actually a pointer
        // to an Interface instance, which has the type info as its first
        // member, so we have to add an extra layer of indirection.
        resultType = getInterfaceTypeInfoType();
        LLType *pres = DtoType(pointerTo(resultType));
        typinf = DtoLoad(pres, typinf);
      } else {
        resultType = getClassInfoType();
      }

      result = new DLValue(resultType, typinf);
      return;
    }
    llvm_unreachable("Unknown TypeidExp argument kind");
  }

  ////////////////////////////////////////////////////////////////////////////////

#define STUB(x)                                                                \
  void visit(x *e) override {                                                  \
    error(e->loc,                                                              \
          "Internal compiler error: Type `" #x "` not implemented: `%s`",      \
          e->toChars());                                                       \
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

////////////////////////////////////////////////////////////////////////////////

namespace {
bool basetypesAreEqualWithoutModifiers(Type *l, Type *r) {
  l = stripModifiers(l->toBasetype(), true);
  r = stripModifiers(r->toBasetype(), true);
  return l->equals(r);
}

VarDeclaration *isTemporaryVar(Expression *e) {
  if (auto ce = e->isCommaExp())
    if (auto de = ce->getHead()->isDeclarationExp())
      if (auto vd = de->declaration->isVarDeclaration())
        if (vd->storage_class & STCtemp)
          if (auto ve = ce->getTail()->isVarExp())
            if (ve->var == vd)
              return vd;

  return nullptr;
}
}

bool toInPlaceConstruction(DLValue *lhs, Expression *rhs) {
  Logger::println("attempting in-place construction");
  LOG_SCOPE;

  // Is the rhs the init symbol of a zero-initialized struct?
  // Then aggressively zero-out the lhs, without any type checks, e.g., allowing
  // to initialize a `S[1]` lhs with a `S` rhs.
  if (auto ve = rhs->isVarExp()) {
    if (auto symdecl = ve->var->isSymbolDeclaration()) {
      // exclude void[]-typed `__traits(initSymbol)` (LDC extension)
      if (symdecl->type->toBasetype()->ty == TY::Tstruct) {
        auto sd = symdecl->dsym->isStructDeclaration();
        assert(sd);
        DtoResolveStruct(sd);
        if (sd->zeroInit()) {
          Logger::println("success, zeroing out");
          DtoMemSetZero(DtoType(lhs->type) ,DtoLVal(lhs));
          return true;
        }
      }
    }
  }

  if (!basetypesAreEqualWithoutModifiers(lhs->type, rhs->type)) {
    Logger::println("aborted due to different base types without modifiers");
    return false;
  }

  // skip over rhs casts only emitted because of differing constness
  if (auto ce = rhs->isCastExp()) {
    auto castSource = ce->e1;
    if (basetypesAreEqualWithoutModifiers(lhs->type, castSource->type))
      rhs = castSource;
  }

  if (auto ce = rhs->isCallExp()) {
    // Direct construction by rhs call via sret?
    // E.g., `T v = foo();` if the callee `T foo()` uses sret.
    // In this case, pass `&v` as hidden sret argument, i.e., let `foo()`
    // construct the return value directly into the lhs lvalue.
    if (DtoIsReturnInArg(ce)) {
      Logger::println("success, in-place-constructing sret return value");
      ToElemVisitor::call(gIR, ce, DtoLVal(lhs));
      return true;
    }

    // detect <structliteral | temporary>.ctor(args)
    if (auto dve = ce->e1->isDotVarExp()) {
      auto fd = dve->var->isFuncDeclaration();
      if (fd && fd->isCtorDeclaration()) {
        Logger::println("is a constructor call, checking lhs of DotVarExp");
        if (toInPlaceConstruction(lhs, dve->e1)) {
          Logger::println("success, calling ctor on in-place constructed lhs");
          auto fnval = new DFuncValue(fd, DtoCallee(fd), DtoLVal(lhs));
          DtoCallFunction(ce->loc, ce->type, fnval, ce->arguments);
          return true;
        }
      }
    }
  }
  // emit struct literals directly into the lhs lvalue
  else if (auto sle = rhs->isStructLiteralExp()) {
    Logger::println("success, in-place-constructing struct literal");
    ToElemVisitor::emitStructLiteral(sle, DtoLVal(lhs));
    return true;
  }
  // and static array literals
  else if (auto al = rhs->isArrayLiteralExp()) {
    if (lhs->type->toBasetype()->ty == TY::Tsarray) {
      Logger::println("success, in-place-constructing array literal");
      initializeArrayLiteral(gIR, al, DtoLVal(lhs), DtoMemType(lhs->type));
      return true;
    }
  }
  // and vector literals
  else if (auto ve = rhs->isVectorExp()) {
    Logger::println("success, in-place-constructing vector");
    ToElemVisitor::emitVector(ve, DtoLVal(lhs));
    return true;
  }
  // and temporaries
  else if (isTemporaryVar(rhs)) {
    Logger::println("success, in-place-constructing temporary");
    auto lhsLVal = DtoLVal(lhs);
    auto rhsLVal = DtoLVal(rhs);
    if (!llvm::isa<llvm::AllocaInst>(rhsLVal)) {
      error(rhs->loc, "lvalue of temporary is not an alloca, please "
                      "file an LDC issue");
      fatal();
    }
    if (lhsLVal != rhsLVal)
      rhsLVal->replaceAllUsesWith(lhsLVal);
    return true;
  }

  return false;
}
