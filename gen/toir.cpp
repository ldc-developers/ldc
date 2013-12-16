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
#include "gen/utils.h"
#include "gen/warnings.h"
#include "ir/irtypeclass.h"
#include "ir/irtypestruct.h"
#include "ir/irlandingpad.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include <fstream>
#include <math.h>
#include <stack>
#include <stdio.h>

// Needs other includes.
#include "ctfe.h"

llvm::cl::opt<bool> checkPrintf("check-printf-calls",
    llvm::cl::desc("Validate printf call format strings against arguments"),
    llvm::cl::ZeroOrMore);

//////////////////////////////////////////////////////////////////////////////////////////

void Expression::cacheLvalue(IRState* irs)
{
    error("expression %s does not mask any l-value", toChars());
    fatal();
}

/*******************************************
 * Evaluate Expression, then call destructors on any temporaries in it.
 */

DValue *Expression::toElemDtor(IRState *p)
{
    Logger::println("Expression::toElemDtor(): %s", toChars());
    LOG_SCOPE

    class CallDestructors : public IRLandingPadCatchFinallyInfo {
    public:
        CallDestructors(const std::vector<Expression*> &edtors_)
            : edtors(edtors_)
        {}

        const std::vector<Expression*> &edtors;

        void toIR(LLValue */*eh_ptr*/ = 0)
        {
            std::vector<Expression*>::const_reverse_iterator itr, end = edtors.rend();
            for (itr = edtors.rbegin(); itr != end; ++itr)
                (*itr)->toElem(gIR);
        }

        static int searchVarsWithDesctructors(Expression *exp, void *edtors)
        {
            if (exp->op == TOKdeclaration) {
                DeclarationExp *de = (DeclarationExp*)exp;
                if (VarDeclaration *vd = de->declaration->isVarDeclaration()) {
                    while (vd->aliassym) {
                        vd = vd->aliassym->isVarDeclaration();
                        if (!vd)
                            return 0;
                    }

                    if (vd->init) {
                        if (ExpInitializer *ex = vd->init->isExpInitializer())
                            ex->exp->apply(&searchVarsWithDesctructors, edtors);
                    }

                    if (!vd->isDataseg() && vd->edtor && !vd->noscope)
                        static_cast<std::vector<Expression*>*>(edtors)->push_back(vd->edtor);
                }
            }
            return 0;
        }
    };


    // find destructors that must be called
    std::vector<Expression*> edtors;
    apply(&CallDestructors::searchVarsWithDesctructors, &edtors);

    if (!edtors.empty()) {
        if (op == TOKcall) {
            // create finally block that calls destructors on temporaries
            CallDestructors *callDestructors = new CallDestructors(edtors);

            // create landing pad
            llvm::BasicBlock *oldend = p->scopeend();
            llvm::BasicBlock *landingpadbb = llvm::BasicBlock::Create(gIR->context(), "landingpad", p->topfunc(), oldend);

            // set up the landing pad
            IRLandingPad &pad = gIR->func()->gen->landingPadInfo;
            pad.addFinally(callDestructors);
            pad.push(landingpadbb);

            // evaluate the expression
            DValue *val = toElem(p);

            // build the landing pad
            llvm::BasicBlock *oldbb = p->scopebb();
            pad.pop();

            // call the destructors
            gIR->scope() = IRScope(oldbb, oldend);
            callDestructors->toIR();
            delete callDestructors;
            return val;
        } else {
            DValue *val = toElem(p);
            CallDestructors(edtors).toIR();
            return val;
        }
    }

    return toElem(p);

}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DeclarationExp::toElem(IRState* p)
{
    Logger::print("DeclarationExp::toElem: %s | T=%s\n", toChars(), type->toChars());
    LOG_SCOPE;

    return DtoDeclarationExp(declaration);
}

//////////////////////////////////////////////////////////////////////////////////////////

void VarExp::cacheLvalue(IRState* p)
{
    Logger::println("Caching l-value of %s", toChars());
    LOG_SCOPE;
    cachedLvalue = toElem(p)->getLVal();
}

DValue* VarExp::toElem(IRState* p)
{
    Logger::print("VarExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(var);

    if (cachedLvalue)
    {
        LLValue* V = cachedLvalue;
        return new DVarValue(type, V);
    }

    return DtoSymbolAddress(loc, type, var);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* VarExp::toConstElem(IRState* p)
{
    Logger::print("VarExp::toConstElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    if (SymbolDeclaration* sdecl = var->isSymbolDeclaration())
    {
        // this seems to be the static initialiser for structs
        Type* sdecltype = sdecl->type->toBasetype();
        Logger::print("Sym: type=%s\n", sdecltype->toChars());
        assert(sdecltype->ty == Tstruct);
        TypeStruct* ts = static_cast<TypeStruct*>(sdecltype);
        DtoResolveStruct(ts->sym);
        return ts->sym->ir.irAggr->getDefaultInit();
    }

    if (TypeInfoDeclaration* ti = var->isTypeInfoDeclaration())
    {
        LLType* vartype = DtoType(type);
        LLConstant* m = DtoTypeInfoOf(ti->tinfo, false);
        if (m->getType() != getPtrToType(vartype))
            m = llvm::ConstantExpr::getBitCast(m, vartype);
        return m;
    }

    VarDeclaration* vd = var->isVarDeclaration();
    if (vd && vd->isConst() && vd->init)
    {
        if (vd->inuse)
        {
            error("recursive reference %s", toChars());
            return llvm::UndefValue::get(DtoType(type));
        }
        vd->inuse++;
        LLConstant* ret = DtoConstInitializer(loc, type, vd->init);
        vd->inuse--;
        // return the initializer
        return ret;
    }

    // fail
    error("non-constant expression %s", toChars());
    return llvm::UndefValue::get(DtoType(type));
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* IntegerExp::toElem(IRState* p)
{
    Logger::print("IntegerExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    LLConstant* c = toConstElem(p);
    return new DConstValue(type, c);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* IntegerExp::toConstElem(IRState* p)
{
    Logger::print("IntegerExp::toConstElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    LLType* t = DtoType(type);
    if (isaPointer(t)) {
        Logger::println("pointer");
        LLConstant* i = LLConstantInt::get(DtoSize_t(),(uint64_t)value,false);
        return llvm::ConstantExpr::getIntToPtr(i, t);
    }
    assert(llvm::isa<LLIntegerType>(t));
    LLConstant* c = LLConstantInt::get(t,(uint64_t)value,!type->isunsigned());
    assert(c);
    if (Logger::enabled())
        Logger::cout() << "value = " << *c << '\n';
    return c;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* RealExp::toElem(IRState* p)
{
    Logger::print("RealExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    LLConstant* c = toConstElem(p);
    return new DConstValue(type, c);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* RealExp::toConstElem(IRState* p)
{
    Logger::print("RealExp::toConstElem: %s @ %s | %La\n", toChars(), type->toChars(), value);
    LOG_SCOPE;
    Type* t = type->toBasetype();
    return DtoConstFP(t, value);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* NullExp::toElem(IRState* p)
{
    Logger::print("NullExp::toElem(type=%s): %s\n", type->toChars(),toChars());
    LOG_SCOPE;
    LLConstant* c = toConstElem(p);
    return new DNullValue(type, c);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* NullExp::toConstElem(IRState* p)
{
    Logger::print("NullExp::toConstElem(type=%s): %s\n", type->toChars(),toChars());
    LOG_SCOPE;
    LLType* t = DtoType(type);
    if (type->ty == Tarray) {
        assert(isaStruct(t));
        return llvm::ConstantAggregateZero::get(t);
    }
    else {
        return LLConstant::getNullValue(t);
    }

    llvm_unreachable("Unknown type for null constant.");
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* ComplexExp::toElem(IRState* p)
{
    Logger::print("ComplexExp::toElem(): %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    LLConstant* c = toConstElem(p);
    LLValue* res;

    if (c->isNullValue()) {
        switch (type->toBasetype()->ty) {
        default: llvm_unreachable("Unexpected complex floating point type");
        case Tcomplex32: c = DtoConstFP(Type::tfloat32, ldouble(0)); break;
        case Tcomplex64: c = DtoConstFP(Type::tfloat64, ldouble(0)); break;
        case Tcomplex80: c = DtoConstFP(Type::tfloat80, ldouble(0)); break;
        }
        res = DtoAggrPair(DtoType(type), c, c);
    }
    else {
        res = DtoAggrPair(DtoType(type), c->getOperand(0), c->getOperand(1));
    }

    return new DImValue(type, res);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* ComplexExp::toConstElem(IRState* p)
{
    Logger::print("ComplexExp::toConstElem(): %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    return DtoConstComplex(type, value.re, value.im);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* StringExp::toElem(IRState* p)
{
    Logger::print("StringExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    Type* dtype = type->toBasetype();
    Type* cty = dtype->nextOf()->toBasetype();

    LLType* ct = voidToI8(DtoType(cty));
    //printf("ct = %s\n", type->nextOf()->toChars());
    LLArrayType* at = LLArrayType::get(ct,len+1);

    LLConstant* _init;
    switch (cty->size())
    {
    default:
        llvm_unreachable("Unknown char type");
    case 1:
        _init = toConstantArray(ct, at, static_cast<uint8_t *>(string), len);
        break;
    case 2:
        _init = toConstantArray(ct, at, static_cast<uint16_t *>(string), len);
        break;
    case 4:
        _init = toConstantArray(ct, at, static_cast<uint32_t *>(string), len);
        break;
    }

    llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::InternalLinkage;
    if (Logger::enabled())
    {
        Logger::cout() << "type: " << *at << '\n';
        Logger::cout() << "init: " << *_init << '\n';
    }
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(*gIR->module, at, true, _linkage, _init, ".str");
    gvar->setUnnamedAddr(true);

    llvm::ConstantInt* zero = LLConstantInt::get(LLType::getInt32Ty(gIR->context()), 0, false);
    LLConstant* idxs[2] = { zero, zero };
    LLConstant* arrptr = llvm::ConstantExpr::getGetElementPtr(gvar, idxs, true);

    if (dtype->ty == Tarray) {
        LLConstant* clen = LLConstantInt::get(DtoSize_t(),len,false);
        return new DImValue(type, DtoConstSlice(clen, arrptr, dtype));
    }
    else if (dtype->ty == Tsarray) {
        LLType* dstType = getPtrToType(LLArrayType::get(ct, len));
        LLValue* emem = (gvar->getType() == dstType) ? gvar : DtoBitCast(gvar, dstType);
        return new DVarValue(type, emem);
    }
    else if (dtype->ty == Tpointer) {
        return new DImValue(type, arrptr);
    }

    llvm_unreachable("Unknown type for StringExp.");
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* StringExp::toConstElem(IRState* p)
{
    Logger::print("StringExp::toConstElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    Type* t = type->toBasetype();
    Type* cty = t->nextOf()->toBasetype();

    bool nullterm = (t->ty != Tsarray);
    size_t endlen = nullterm ? len+1 : len;

    LLType* ct = voidToI8(DtoType(cty));
    LLArrayType* at = LLArrayType::get(ct,endlen);

    LLConstant* _init;
    switch (cty->size())
    {
    default:
        llvm_unreachable("Unknown char type");
    case 1:
        _init = toConstantArray(ct, at, static_cast<uint8_t *>(string), len, nullterm);
        break;
    case 2:
        _init = toConstantArray(ct, at, static_cast<uint16_t *>(string), len, nullterm);
        break;
    case 4:
        _init = toConstantArray(ct, at, static_cast<uint32_t *>(string), len, nullterm);
        break;
    }

    if (t->ty == Tsarray)
    {
        return _init;
    }

    llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::InternalLinkage;
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(*gIR->module, _init->getType(), true, _linkage, _init, ".str");
    gvar->setUnnamedAddr(true);

    llvm::ConstantInt* zero = LLConstantInt::get(LLType::getInt32Ty(gIR->context()), 0, false);
    LLConstant* idxs[2] = { zero, zero };
    LLConstant* arrptr = llvm::ConstantExpr::getGetElementPtr(gvar, idxs, true);

    if (t->ty == Tpointer) {
        return arrptr;
    }
    else if (t->ty == Tarray) {
        LLConstant* clen = LLConstantInt::get(DtoSize_t(),len,false);
        return DtoConstSlice(clen, arrptr, type);
    }

    llvm_unreachable("Unknown type for StringExp.");
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* AssignExp::toElem(IRState* p)
{
    Logger::print("AssignExp::toElem: %s | (%s)(%s = %s)\n", toChars(), type->toChars(), e1->type->toChars(), e2->type ? e2->type->toChars() : 0);
    LOG_SCOPE;

    if (e1->op == TOKarraylength)
    {
        Logger::println("performing array.length assignment");
        ArrayLengthExp *ale = static_cast<ArrayLengthExp *>(e1);
        DValue* arr = ale->e1->toElem(p);
        DVarValue arrval(ale->e1->type, arr->getLVal());
        DValue* newlen = e2->toElem(p);
        DSliceValue* slice = DtoResizeDynArray(arrval.getType(), &arrval, newlen->getRVal());
        DtoAssign(loc, &arrval, slice);
        return newlen;
    }

    // Can't just override ConstructExp::toElem because not all TOKconstruct
    // operations are actually instances of ConstructExp... Long live the DMD
    // coding style!
    if (op == TOKconstruct)
    {
        if (e1->op == TOKvar)
        {
            VarExp* ve = (VarExp*)e1;
            if (ve->var->storage_class & STCref)
            {
                Logger::println("performing ref variable initialization");
                // Note that the variable value is accessed directly (instead
                // of via getLVal(), which would perform a load from the
                // uninitialized location), and that rhs is stored as an l-value!
                DVarValue* lhs = e1->toElem(p)->isVar();
                assert(lhs);
                DValue* rhs = e2->toElem(p);

                // We shouldn't really need makeLValue() here, but the 2.063
                // frontend generates ref variables initialized from function
                // calls.
                DtoStore(makeLValue(loc, rhs), lhs->getRefStorage());

                return rhs;
            }
        }
    }

    if (e1->op == TOKslice)
    {
        // Check if this is an initialization of a static array with an array
        // literal that the frontend has foolishly rewritten into an
        // assignment of a dynamic array literal to a slice.
        Logger::println("performing static array literal assignment");
        SliceExp * const se = static_cast<SliceExp *>(e1);
        Type * const t2 = e2->type->toBasetype();
        Type * const ta = se->e1->type->toBasetype();

        if (se->lwr == NULL && ta->ty == Tsarray &&
            e2->op == TOKarrayliteral &&
            op == TOKconstruct &&   // DMD Bugzilla 11238: avoid aliasing issue
            t2->nextOf()->mutableOf()->implicitConvTo(ta->nextOf()))
        {
            ArrayLiteralExp * const ale = static_cast<ArrayLiteralExp *>(e2);
            initializeArrayLiteral(p, ale, se->e1->toElem(p)->getLVal());
            return e1->toElem(p);
        }
    }

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    if (e1->type->toBasetype()->ty == Tstruct && e2->op == TOKint64)
    {
        Logger::println("performing aggregate zero initialization");
        assert(e2->toInteger() == 0);
        DtoAggrZeroInit(l->getLVal());
        TypeStruct *ts = static_cast<TypeStruct*>(e1->type);
        if (ts->sym->isNested() && ts->sym->vthis)
            DtoResolveNestedContext(loc, ts->sym, l->getLVal());
        // Return value should be irrelevant.
        return r;
    }

    bool canSkipPostblit = false;
    if (!(e2->op == TOKslice && ((UnaExp *)e2)->e1->isLvalue()) &&
        !(e2->op == TOKcast && ((UnaExp *)e2)->e1->isLvalue()) &&
        (e2->op == TOKslice || !e2->isLvalue()))
    {
        canSkipPostblit = true;
    }

    Logger::println("performing normal assignment (canSkipPostblit = %d)", canSkipPostblit);
    DtoAssign(loc, l, r, op, canSkipPostblit);

    if (l->isSlice())
        return l;

    return r;
}

//////////////////////////////////////////////////////////////////////////////////////////

/// Finds the proper lvalue for a binassign expressions.
/// Makes sure the given LHS expression is only evaluated once.
static Expression* findLvalue(IRState* irs, Expression* exp)
{
    Expression* e = exp;

    // skip past any casts
    while(e->op == TOKcast)
        e = static_cast<CastExp*>(e)->e1;

    // cache lvalue and return
    e->cacheLvalue(irs);
    return e;
}

#define BIN_ASSIGN(X) \
DValue* X##AssignExp::toElem(IRState* p) \
{ \
    Logger::print(#X"AssignExp::toElem: %s @ %s\n", toChars(), type->toChars()); \
    LOG_SCOPE; \
    X##Exp e3(loc, e1, e2); \
    e3.type = e1->type; \
    DValue* dst = findLvalue(p, e1)->toElem(p); \
    DValue* res = e3.toElem(p); \
    /* Now that we are done with the expression, clear the cached lvalue. */ \
    Expression* e = e1; \
    while(e->op == TOKcast) \
        e = static_cast<CastExp*>(e)->e1; \
    e->cachedLvalue = NULL; \
    /* Assign the (casted) value and return it. */ \
    DValue* stval = DtoCast(loc, res, dst->getType()); \
    DtoAssign(loc, dst, stval); \
    return DtoCast(loc, res, type); \
}

BIN_ASSIGN(Add)
BIN_ASSIGN(Min)
BIN_ASSIGN(Mul)
BIN_ASSIGN(Div)
BIN_ASSIGN(Mod)
BIN_ASSIGN(And)
BIN_ASSIGN(Or)
BIN_ASSIGN(Xor)
BIN_ASSIGN(Shl)
BIN_ASSIGN(Shr)
BIN_ASSIGN(Ushr)

#undef BIN_ASSIGN

//////////////////////////////////////////////////////////////////////////////////////////

static void errorOnIllegalArrayOp(Expression* base, Expression* e1, Expression* e2)
{
    Type* t1 = e1->type->toBasetype();
    Type* t2 = e2->type->toBasetype();

    // valid array ops would have been transformed by optimize
    if ((t1->ty == Tarray || t1->ty == Tsarray) &&
        (t2->ty == Tarray || t2->ty == Tsarray)
       )
    {
        base->error("Array operation %s not recognized", base->toChars());
        fatal();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

static dinteger_t undoStrideMul(const Loc& loc, Type* t, dinteger_t offset)
{
    assert(t->ty == Tpointer);
    d_uns64 elemSize = t->nextOf()->size(loc);
    assert((offset % elemSize) == 0 &&
        "Expected offset by an integer amount of elements");

    return offset / elemSize;
}

LLConstant* AddExp::toConstElem(IRState* p)
{
    // add to pointer
    Type* t1b = e1->type->toBasetype();
    if (t1b->ty == Tpointer && e2->type->isintegral()) {
        llvm::Constant* ptr = e1->toConstElem(p);
        dinteger_t idx = undoStrideMul(loc, t1b, e2->toInteger());
        return llvm::ConstantExpr::getGetElementPtr(ptr, DtoConstSize_t(idx));
    }

    error("expression '%s' is not a constant", toChars());
    fatal();
    return NULL;
}

/// Tries to remove a MulExp by a constant value of baseSize from e. Returns
/// NULL if not possible.
static Expression* extractNoStrideInc(Expression* e, d_uns64 baseSize, bool& negate)
{
    MulExp* mul;
    while (true)
    {
        if (e->op == TOKneg)
        {
            negate = !negate;
            e = static_cast<NegExp*>(e)->e1;
            continue;
        }

        if (e->op == TOKmul)
        {
            mul = static_cast<MulExp*>(e);
            break;
        }

        return NULL;
    }

    if (!mul->e2->isConst()) return NULL;
    dinteger_t stride = mul->e2->toInteger();

    if (stride != baseSize) return NULL;

    return mul->e1;
}

static DValue* emitPointerOffset(IRState* p, Loc loc, DValue* base,
    Expression* offset, bool negateOffset, Type* resultType)
{
    // The operand emitted by the frontend is in units of bytes, and not
    // pointer elements. We try to undo this before resorting to
    // temporarily bitcasting the pointer to i8.

    llvm::Value* noStrideInc = NULL;
    if (offset->isConst())
    {
        dinteger_t byteOffset = offset->toInteger();
        if (byteOffset == 0)
        {
            Logger::println("offset is zero");
            return base;
        }
        noStrideInc = DtoConstSize_t(undoStrideMul(loc, base->type, byteOffset));
    }
    else if (Expression* inc = extractNoStrideInc(offset,
        base->type->nextOf()->size(loc), negateOffset))
    {
        noStrideInc = inc->toElem(p)->getRVal();
    }

    if (noStrideInc)
    {
        if (negateOffset) noStrideInc = p->ir->CreateNeg(noStrideInc);
        return new DImValue(base->type,
            DtoGEP1(base->getRVal(), noStrideInc, 0, p->scopebb()));
    }

    // This might not actually be generated by the frontend, just to be
    // safe.
    llvm::Value* inc = offset->toElem(p)->getRVal();
    if (negateOffset) inc = p->ir->CreateNeg(inc);
    llvm::Value* bytePtr = DtoBitCast(base->getRVal(), getVoidPtrType());
    DValue* result = new DImValue(Type::tvoidptr, DtoGEP1(bytePtr, inc));
    return DtoCast(loc, result, resultType);
}


DValue* AddExp::toElem(IRState* p)
{
    Logger::print("AddExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);

    Type* t = type->toBasetype();
    Type* e1type = e1->type->toBasetype();
    Type* e2type = e2->type->toBasetype();

    errorOnIllegalArrayOp(this, e1, e2);

    if (e1type != e2type && e1type->ty == Tpointer && e2type->isintegral())
    {
        Logger::println("Adding integer to pointer");
        return emitPointerOffset(p, loc, l, e2, false, type);
    }
    else if (t->iscomplex()) {
        return DtoComplexAdd(loc, type, l, e2->toElem(p));
    }
    else {
        return DtoBinAdd(l, e2->toElem(p));
    }
}

LLConstant* MinExp::toConstElem(IRState* p)
{
    Type* t1b = e1->type->toBasetype();
    if (t1b->ty == Tpointer && e2->type->isintegral()) {
        llvm::Constant* ptr = e1->toConstElem(p);
        dinteger_t idx = undoStrideMul(loc, t1b, e2->toInteger());

        llvm::Constant* negIdx = llvm::ConstantExpr::getNeg(DtoConstSize_t(idx));
        return llvm::ConstantExpr::getGetElementPtr(ptr, negIdx);
    }

    error("expression '%s' is not a constant", toChars());
    fatal();
    return NULL;
}

DValue* MinExp::toElem(IRState* p)
{
    Logger::print("MinExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);

    Type* t = type->toBasetype();
    Type* t1 = e1->type->toBasetype();
    Type* t2 = e2->type->toBasetype();

    errorOnIllegalArrayOp(this, e1, e2);

    if (t1->ty == Tpointer && t2->ty == Tpointer) {
        LLValue* lv = l->getRVal();
        LLValue* rv = e2->toElem(p)->getRVal();
        if (Logger::enabled())
            Logger::cout() << "lv: " << *lv << " rv: " << *rv << '\n';
        lv = p->ir->CreatePtrToInt(lv, DtoSize_t(), "tmp");
        rv = p->ir->CreatePtrToInt(rv, DtoSize_t(), "tmp");
        LLValue* diff = p->ir->CreateSub(lv,rv,"tmp");
        if (diff->getType() != DtoType(type))
            diff = p->ir->CreateIntToPtr(diff, DtoType(type), "tmp");
        return new DImValue(type, diff);
    }
    else if (t1->ty == Tpointer && t2->isintegral())
    {
        Logger::println("Subtracting integer from pointer");
        return emitPointerOffset(p, loc, l, e2, true, type);
    }
    else if (t->iscomplex()) {
        return DtoComplexSub(loc, type, l, e2->toElem(p));
    }
    else {
        return DtoBinSub(l, e2->toElem(p));
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* MulExp::toElem(IRState* p)
{
    Logger::print("MulExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    errorOnIllegalArrayOp(this, e1, e2);

    if (type->iscomplex()) {
        return DtoComplexMul(loc, type, l, r);
    }

    return DtoBinMul(type, l, r);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DivExp::toElem(IRState* p)
{
    Logger::print("DivExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    errorOnIllegalArrayOp(this, e1, e2);

    if (type->iscomplex()) {
        return DtoComplexDiv(loc, type, l, r);
    }

    return DtoBinDiv(type, l, r);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* ModExp::toElem(IRState* p)
{
    Logger::print("ModExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    errorOnIllegalArrayOp(this, e1, e2);

    if (type->iscomplex()) {
        return DtoComplexRem(loc, type, l, r);
    }

    return DtoBinRem(type, l, r);
}

//////////////////////////////////////////////////////////////////////////////////////////

void CallExp::cacheLvalue(IRState* p)
{
    Logger::println("Caching l-value of %s", toChars());
    LOG_SCOPE;
    cachedLvalue = toElem(p)->getLVal();
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CallExp::toElem(IRState* p)
{
    Logger::print("CallExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    if (cachedLvalue)
    {
        LLValue* V = cachedLvalue;
        return new DVarValue(type, V);
    }

    // handle magic inline asm
    if (e1->op == TOKvar)
    {
        VarExp* ve = static_cast<VarExp*>(e1);
        if (FuncDeclaration* fd = ve->var->isFuncDeclaration())
        {
            if (fd->llvmInternal == LLVMinline_asm)
            {
                return DtoInlineAsmExpr(loc, fd, arguments);
            }
        }
    }

    // get the callee value
    DValue* fnval = e1->toElem(p);

    // get func value if any
    DFuncValue* dfnval = fnval->isFunc();

    // handle magic intrinsics (mapping to instructions)
    if (dfnval && dfnval->func)
    {
        FuncDeclaration* fndecl = dfnval->func;

        // as requested by bearophile, see if it's a C printf call and that it's valid.
        if (global.params.warnings && checkPrintf)
        {
            if (fndecl->linkage == LINKc && strcmp(fndecl->ident->string, "printf") == 0)
            {
                warnInvalidPrintfCall(loc, static_cast<Expression*>(arguments->data[0]), arguments->dim);
            }
        }

        // va_start instruction
        if (fndecl->llvmInternal == LLVMva_start) {
            if (arguments->dim != 2) {
                error("va_start instruction expects 2 arguments");
                return NULL;
            }
            // llvm doesn't need the second param hence the override
            Expression* exp = static_cast<Expression*>(arguments->data[0]);
            LLValue* arg = exp->toElem(p)->getLVal();
            if (LLValue *argptr = gIR->func()->_argptr) {
                DtoStore(DtoLoad(argptr), DtoBitCast(arg, getPtrToType(getVoidPtrType())));
                return new DImValue(type, arg);
            } else if (global.params.targetTriple.getArch() == llvm::Triple::x86_64) {
                LLValue *va_list = DtoAlloca(exp->type->nextOf());
                DtoStore(va_list, arg);
                va_list = DtoBitCast(va_list, getVoidPtrType());
                return new DImValue(type, gIR->ir->CreateCall(GET_INTRINSIC_DECL(vastart), va_list, ""));
            } else
            {
                arg = DtoBitCast(arg, getVoidPtrType());
                return new DImValue(type, gIR->ir->CreateCall(GET_INTRINSIC_DECL(vastart), arg, ""));
            }
        }
        else if (fndecl->llvmInternal == LLVMva_copy &&
            global.params.targetTriple.getArch() == llvm::Triple::x86_64) {
            if (arguments->dim != 2) {
                error("va_copy instruction expects 2 arguments");
                return NULL;
            }
            Expression* exp1 = static_cast<Expression*>(arguments->data[0]);
            Expression* exp2 = static_cast<Expression*>(arguments->data[1]);
            LLValue* arg1 = exp1->toElem(p)->getLVal();
            LLValue* arg2 = exp2->toElem(p)->getLVal();

            LLValue *va_list = DtoAlloca(exp1->type->nextOf());
            DtoStore(va_list, arg1);

            DtoStore(DtoLoad(DtoLoad(arg2)), DtoLoad(arg1));
            return new DVarValue(type, arg1);
        }
        // va_arg instruction
        else if (fndecl->llvmInternal == LLVMva_arg) {
            if (arguments->dim != 1) {
                error("va_arg instruction expects 1 arguments");
                return NULL;
            }
            return DtoVaArg(loc, type, static_cast<Expression*>(arguments->data[0]));
        }
        // C alloca
        else if (fndecl->llvmInternal == LLVMalloca) {
            if (arguments->dim != 1) {
                error("alloca expects 1 arguments");
                return NULL;
            }
            Expression* exp = static_cast<Expression*>(arguments->data[0]);
            DValue* expv = exp->toElem(p);
            if (expv->getType()->toBasetype()->ty != Tint32)
                expv = DtoCast(loc, expv, Type::tint32);
            return new DImValue(type, p->ir->CreateAlloca(LLType::getInt8Ty(gIR->context()), expv->getRVal(), ".alloca"));
        }
        // fence instruction
        else if (fndecl->llvmInternal == LLVMfence) {
            if (arguments->dim != 1) {
                error("fence instruction expects 1 arguments");
                return NULL;
            }
            gIR->ir->CreateFence(llvm::AtomicOrdering(static_cast<Expression*>(arguments->data[0])->toInteger()));
            return NULL;
        // atomic store instruction
        } else if (fndecl->llvmInternal == LLVMatomic_store) {
            if (arguments->dim != 3) {
                error("atomic store instruction expects 3 arguments");
                return NULL;
            }
            Expression* exp1 = static_cast<Expression*>(arguments->data[0]);
            Expression* exp2 = static_cast<Expression*>(arguments->data[1]);
            int atomicOrdering = static_cast<Expression*>(arguments->data[2])->toInteger();
            LLValue* val = exp1->toElem(p)->getRVal();
            LLValue* ptr = exp2->toElem(p)->getRVal();
            llvm::StoreInst* ret = gIR->ir->CreateStore(val, ptr, "tmp");
            ret->setAtomic(llvm::AtomicOrdering(atomicOrdering));
            ret->setAlignment(getTypeAllocSize(val->getType()));
            return NULL;
        // atomic load instruction
        } else if (fndecl->llvmInternal == LLVMatomic_load) {
            if (arguments->dim != 2) {
                error("atomic load instruction expects 2 arguments");
                return NULL;
            }
            Expression* exp = static_cast<Expression*>(arguments->data[0]);
            int atomicOrdering = static_cast<Expression*>(arguments->data[1])->toInteger();
            LLValue* ptr = exp->toElem(p)->getRVal();
            Type* retType = exp->type->nextOf();
            llvm::LoadInst* val = gIR->ir->CreateLoad(ptr, "tmp");
            val->setAlignment(getTypeAllocSize(val->getType()));
            val->setAtomic(llvm::AtomicOrdering(atomicOrdering));
            return new DImValue(retType, val);
        // cmpxchg instruction
        } else if (fndecl->llvmInternal == LLVMatomic_cmp_xchg) {
            if (arguments->dim != 4) {
                error("cmpxchg instruction expects 4 arguments");
                return NULL;
            }
            Expression* exp1 = static_cast<Expression*>(arguments->data[0]);
            Expression* exp2 = static_cast<Expression*>(arguments->data[1]);
            Expression* exp3 = static_cast<Expression*>(arguments->data[2]);
            int atomicOrdering = static_cast<Expression*>(arguments->data[3])->toInteger();
            LLValue* ptr = exp1->toElem(p)->getRVal();
            LLValue* cmp = exp2->toElem(p)->getRVal();
            LLValue* val = exp3->toElem(p)->getRVal();
            LLValue* ret = gIR->ir->CreateAtomicCmpXchg(ptr, cmp, val, llvm::AtomicOrdering(atomicOrdering));
            return new DImValue(exp3->type, ret);
        // atomicrmw instruction
        } else if (fndecl->llvmInternal == LLVMatomic_rmw) {
            if (arguments->dim != 3) {
                error("atomic_rmw instruction expects 3 arguments");
                return NULL;
            }

            static const char *ops[] = {
                "xchg",
                "add",
                "sub",
                "and",
                "nand",
                "or",
                "xor",
                "max",
                "min",
                "umax",
                "umin",
                0
            };

            int op = 0;
            for (; ; ++op) {
                if (ops[op] == 0) {
                    error("unknown atomic_rmw operation %s", fndecl->intrinsicName.c_str());
                    return NULL;
                }
                if (fndecl->intrinsicName == ops[op])
                    break;
            }

            Expression* exp1 = static_cast<Expression*>(arguments->data[0]);
            Expression* exp2 = static_cast<Expression*>(arguments->data[1]);
            int atomicOrdering = static_cast<Expression*>(arguments->data[2])->toInteger();
            LLValue* ptr = exp1->toElem(p)->getRVal();
            LLValue* val = exp2->toElem(p)->getRVal();
            LLValue* ret = gIR->ir->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp(op), ptr, val,
                                                    llvm::AtomicOrdering(atomicOrdering));
            return new DImValue(exp2->type, ret);
        // bitop
        } else if (fndecl->llvmInternal == LLVMbitop_bt ||
                   fndecl->llvmInternal == LLVMbitop_btr ||
                   fndecl->llvmInternal == LLVMbitop_btc ||
                   fndecl->llvmInternal == LLVMbitop_bts)
        {
            if (arguments->dim != 2) {
                error("bitop intrinsic expects 2 arguments");
                return NULL;
            }

            Expression* exp1 = static_cast<Expression*>(arguments->data[0]);
            Expression* exp2 = static_cast<Expression*>(arguments->data[1]);
            LLValue* ptr = exp1->toElem(p)->getRVal();
            LLValue* bitnum = exp2->toElem(p)->getRVal();

            unsigned bitmask = DtoSize_t()->getBitWidth() - 1;
            assert(bitmask == 31 || bitmask == 63);
            // auto q = cast(size_t*)ptr + (bitnum >> (64bit ? 6 : 5));
            LLValue* q = DtoBitCast(ptr, DtoSize_t()->getPointerTo());
            q = DtoGEP1(q, p->ir->CreateLShr(bitnum, bitmask == 63 ? 6 : 5), "bitop.q");

            // auto mask = 1 << (bitnum & bitmask);
            LLValue* mask = p->ir->CreateAnd(bitnum, DtoConstSize_t(bitmask), "bitop.tmp");
            mask = p->ir->CreateShl(DtoConstSize_t(1), mask, "bitop.mask");

            // auto result = (*q & mask) ? -1 : 0;
            LLValue* val = p->ir->CreateZExt(DtoLoad(q, "bitop.tmp"), DtoSize_t(), "bitop.val");
            LLValue* result = p->ir->CreateAnd(val, mask, "bitop.tmp");
            result = p->ir->CreateICmpNE(result, DtoConstSize_t(0), "bitop.tmp");
            result = p->ir->CreateSelect(result, DtoConstInt(-1), DtoConstInt(0), "bitop.result");

            if (fndecl->llvmInternal != LLVMbitop_bt) {
                llvm::Instruction::BinaryOps op;
                if (fndecl->llvmInternal == LLVMbitop_btc) {
                    // *q ^= mask;
                    op = llvm::Instruction::Xor;
                } else if (fndecl->llvmInternal == LLVMbitop_btr) {
                    // *q &= ~mask;
                    mask = p->ir->CreateNot(mask);
                    op = llvm::Instruction::And;
                } else if (fndecl->llvmInternal == LLVMbitop_bts) {
                    // *q |= mask;
                    op = llvm::Instruction::Or;
                } else {
                    llvm_unreachable("Unrecognized bitop intrinsic.");
                }

                LLValue *newVal = p->ir->CreateBinOp(op, val, mask, "bitop.new_val");
                newVal = p->ir->CreateTrunc(newVal, DtoSize_t(), "bitop.tmp");
                DtoStore(newVal, q);
            }

            return new DImValue(type, result);
        }
    }
    return DtoCallFunction(loc, type, fnval, arguments);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CastExp::toElem(IRState* p)
{
    Logger::print("CastExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    // get the value to cast
    DValue* u = e1->toElem(p);

    // handle cast to void (usually created by frontend to avoid "has no effect" error)
    if (to == Type::tvoid)
        return new DImValue(Type::tvoid, llvm::UndefValue::get(voidToI8(DtoType(Type::tvoid))));

    // cast it to the 'to' type, if necessary
    DValue* v = u;
    if (!to->equals(e1->type))
        v = DtoCast(loc, u, to);

    // paint the type, if necessary
    if (!type->equals(to))
        v = DtoPaintType(loc, v, type);

    // return the new rvalue
    return v;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* CastExp::toConstElem(IRState* p)
{
    Logger::print("CastExp::toConstElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    LLConstant* res;
    LLType* lltype = DtoType(type);
    Type* tb = to->toBasetype();

    // string literal to dyn array:
    // reinterpret the string data as an array, calculate the length
    if (e1->op == TOKstring && tb->ty == Tarray) {
/*        StringExp *strexp = static_cast<StringExp*>(e1);
        size_t datalen = strexp->sz * strexp->len;
        Type* eltype = tb->nextOf()->toBasetype();
        if (datalen % eltype->size() != 0) {
            error("the sizes don't line up");
            return e1->toConstElem(p);
        }
        size_t arrlen = datalen / eltype->size();*/
        error("ct cast of string to dynamic array not fully implemented");
        return e1->toConstElem(p);
    }
    // pointer to pointer
    else if (tb->ty == Tpointer && e1->type->toBasetype()->ty == Tpointer) {
        res = llvm::ConstantExpr::getBitCast(e1->toConstElem(p), lltype);
    }
    // global variable to pointer
    else if (tb->ty == Tpointer && e1->op == TOKvar) {
        VarDeclaration *vd = static_cast<VarExp*>(e1)->var->isVarDeclaration();
        assert(vd);
        DtoResolveVariable(vd);
        LLConstant *value = vd->ir.irGlobal ? isaConstant(vd->ir.irGlobal->value) : 0;
        if (!value)
           goto Lerr;
        Type *type = vd->type->toBasetype();
        if (type->ty == Tarray || type->ty == Tdelegate) {
            LLConstant* idxs[2] = { DtoConstSize_t(0), DtoConstSize_t(1) };
            value = llvm::ConstantExpr::getGetElementPtr(value, idxs, true);
        }
        return DtoBitCast(value, DtoType(tb));
    }
    else if (tb->ty == Tclass && e1->type->ty == Tclass) {
        assert(e1->op == TOKclassreference);
        ClassDeclaration* cd = static_cast<ClassReferenceExp*>(e1)->originalClass();

        llvm::Constant* instance = e1->toConstElem(p);
        if (InterfaceDeclaration* it = static_cast<TypeClass*>(tb)->sym->isInterfaceDeclaration()) {
            assert(it->isBaseOf(cd, NULL));

            IrTypeClass* typeclass = cd->type->irtype->isClass();

            // find interface impl
            size_t i_index = typeclass->getInterfaceIndex(it);
            assert(i_index != ~0UL);

            // offset pointer
            instance = DtoGEPi(instance, 0, i_index);
        }
        return DtoBitCast(instance, DtoType(tb));
    }
    else {
        goto Lerr;
    }

    return res;

Lerr:
    error("cannot cast %s to %s at compile time", e1->type->toChars(), type->toChars());
    if (!global.gag)
        fatal();
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* SymOffExp::toElem(IRState* p)
{
    Logger::print("SymOffExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* base = DtoSymbolAddress(loc, var->type, var);

    // This weird setup is required to be able to handle both variables as
    // well as functions and TypeInfo references (which are not a DVarValue
    // as well due to the level-of-indirection hack in Type::getTypeInfo that
    // is unfortunately required by the frontend).
    llvm::Value* baseValue;
    if (base->isLVal())
        baseValue = base->getLVal();
    else
        baseValue = base->getRVal();
    assert(isaPointer(baseValue));

    llvm::Value* offsetValue;
    Type* offsetType;

    if (offset == 0)
    {
        offsetValue = baseValue;
        offsetType = base->type->pointerTo();
    }
    else
    {
        uint64_t elemSize = gDataLayout->getTypeStoreSize(
            baseValue->getType()->getContainedType(0));
        if (offset % elemSize == 0)
        {
            // We can turn this into a "nice" GEP.
            offsetValue = DtoGEPi1(baseValue, offset / elemSize);
            offsetType = base->type->pointerTo();
        }
        else
        {
            // Offset isn't a multiple of base type size, just cast to i8* and
            // apply the byte offset.
            offsetValue = DtoGEPi1(DtoBitCast(baseValue, getVoidPtrType()), offset);
            offsetType = Type::tvoidptr;
        }
    }

    // Casts are also "optimized into" SymOffExp by the frontend.
    return DtoCast(loc, new DImValue(offsetType, offsetValue), type);
}

llvm::Constant* SymOffExp::toConstElem(IRState* p)
{
    IF_LOG Logger::println("SymOffExp::toConstElem: %s @ %s", toChars(), type->toChars());
    LOG_SCOPE;

    // We might get null here due to the hackish implementation of
    // AssocArrayLiteralExp::toElem.
    llvm::Constant* base = DtoConstSymbolAddress(loc, var);
    if (!base) return 0;

    llvm::Constant* result;
    if (offset == 0)
    {
        result = base;
    }
    else
    {
        const unsigned elemSize = gDataLayout->getTypeStoreSize(
            base->getType()->getContainedType(0));

        Logger::println("adding offset: %u (elem size: %u)", offset, elemSize);

        if (offset % elemSize == 0)
        {
            // We can turn this into a "nice" GEP.
            result = llvm::ConstantExpr::getGetElementPtr(base,
                DtoConstSize_t(offset / elemSize));
        }
        else
        {
            // Offset isn't a multiple of base type size, just cast to i8* and
            // apply the byte offset.
            result = llvm::ConstantExpr::getGetElementPtr(
                DtoBitCast(base, getVoidPtrType()),
                DtoConstSize_t(offset));
        }
    }

    return DtoBitCast(result, DtoType(type));
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* AddrExp::toElem(IRState* p)
{
    IF_LOG Logger::println("AddrExp::toElem: %s @ %s", toChars(), type->toChars());
    LOG_SCOPE;

    // The address of a StructLiteralExp can in fact be a global variable, check
    // for that instead of re-codegening the literal.
    if (e1->op == TOKstructliteral)
    {
        IF_LOG Logger::println("is struct literal");
        StructLiteralExp* se = static_cast<StructLiteralExp*>(e1);

        // DMD uses origin here as well, necessary to handle messed-up AST on
        // forward references.
        if (se->origin->globalVar)
        {
            IF_LOG Logger::cout() << "returning address of global: " <<
                *se->globalVar << '\n';
            return new DImValue(type, DtoBitCast(se->origin->globalVar, DtoType(type)));
        }
    }

    DValue* v = e1->toElem(p);
    if (v->isField()) {
        Logger::println("is field");
        return v;
    }
    else if (DFuncValue* fv = v->isFunc()) {
        Logger::println("is func");
        //Logger::println("FuncDeclaration");
        FuncDeclaration* fd = fv->func;
        assert(fd);
        DtoResolveFunction(fd);
        return new DFuncValue(fd, fd->ir.irFunc->func);
    }
    else if (v->isIm()) {
        Logger::println("is immediate");
        return v;
    }
    Logger::println("is nothing special");

    // we special case here, since apparently taking the address of a slice is ok
    LLValue* lval;
    if (v->isLVal())
        lval = v->getLVal();
    else
    {
        assert(v->isSlice());
        LLValue* rval = v->getRVal();
        lval = DtoRawAlloca(rval->getType(), 0, ".tmp_slice_storage");
        DtoStore(rval, lval);
    }

    if (Logger::enabled())
        Logger::cout() << "lval: " << *lval << '\n';

    return new DImValue(type, DtoBitCast(lval, DtoType(type)));
}

LLConstant* AddrExp::toConstElem(IRState* p)
{
    IF_LOG Logger::println("AddrExp::toConstElem: %s @ %s", toChars(), type->toChars());
    LOG_SCOPE;
    // FIXME: this should probably be generalized more so we don't
    // need to have a case for each thing we can take the address of

    // address of global variable
    if (e1->op == TOKvar)
    {
        VarExp* vexp = static_cast<VarExp*>(e1);
        LLConstant *c = DtoConstSymbolAddress(loc, vexp->var);
        return c ? DtoBitCast(c, DtoType(type)) : 0;
    }
    // address of indexExp
    else if (e1->op == TOKindex)
    {
        IndexExp* iexp = static_cast<IndexExp*>(e1);

        // indexee must be global static array var
        assert(iexp->e1->op == TOKvar);
        VarExp* vexp = static_cast<VarExp*>(iexp->e1);
        VarDeclaration* vd = vexp->var->isVarDeclaration();
        assert(vd);
        assert(vd->type->toBasetype()->ty == Tsarray);
        DtoResolveVariable(vd);
        assert(vd->ir.irGlobal);

        // get index
        LLConstant* index = iexp->e2->toConstElem(p);
        assert(index->getType() == DtoSize_t());

        // gep
        LLConstant* idxs[2] = { DtoConstSize_t(0), index };
        LLConstant *val = isaConstant(vd->ir.irGlobal->value);
        val = DtoBitCast(val, DtoType(vd->type->pointerTo()));
        LLConstant* gep = llvm::ConstantExpr::getGetElementPtr(val, idxs, true);

        // bitcast to requested type
        assert(type->toBasetype()->ty == Tpointer);
        return DtoBitCast(gep, DtoType(type));
    }
    else if (e1->op == TOKstructliteral)
    {
        StructLiteralExp* se = static_cast<StructLiteralExp*>(e1);

        if (se->globalVar)
        {
            Logger::cout() << "Returning existing global: " << *se->globalVar << '\n';
            return se->globalVar;
        }

        se->globalVar = new llvm::GlobalVariable(*p->module,
            DtoType(e1->type), false, llvm::GlobalValue::InternalLinkage, 0,
            ".structliteral");

        llvm::Constant* constValue = se->toConstElem(p);
        if (constValue->getType() != se->globalVar->getType()->getContainedType(0))
        {
            llvm::GlobalVariable* finalGlobalVar = new llvm::GlobalVariable(
                *p->module, constValue->getType(), false,
                llvm::GlobalValue::InternalLinkage, 0, ".structliteral");
            se->globalVar->replaceAllUsesWith(
                DtoBitCast(finalGlobalVar, se->globalVar->getType()));
            se->globalVar->eraseFromParent();
            se->globalVar = finalGlobalVar;
        }
        se->globalVar->setInitializer(constValue);
        se->globalVar->setAlignment(e1->type->alignsize());

        return se->globalVar;
    }
    else if (e1->op == TOKslice)
    {
        error("non-constant expression '%s'", toChars());
        fatal();
    }
    // not yet supported
    else
    {
        error("constant expression '%s' not yet implemented", toChars());
        fatal();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void PtrExp::cacheLvalue(IRState* p)
{
    Logger::println("Caching l-value of %s", toChars());
    LOG_SCOPE;
    cachedLvalue = e1->toElem(p)->getRVal();
}

DValue* PtrExp::toElem(IRState* p)
{
    Logger::println("PtrExp::toElem: %s @ %s", toChars(), type->toChars());
    LOG_SCOPE;

    // function pointers are special
    if (type->toBasetype()->ty == Tfunction)
    {
        assert(!cachedLvalue);
        DValue *dv = e1->toElem(p);
        if (DFuncValue *dfv = dv->isFunc())
            return new DFuncValue(type, dfv->func, dfv->getRVal());
        else
            return new DImValue(type, dv->getRVal());
    }

    // get the rvalue and return it as an lvalue
    LLValue* V;
    if (cachedLvalue)
    {
        V = cachedLvalue;
    }
    else
    {
        V = e1->toElem(p)->getRVal();
    }

    // The frontend emits dereferences of class/interfaces types to access the
    // first member, which is the .classinfo property.
    Type* origType = e1->type->toBasetype();
    if (origType->ty == Tclass)
    {
        TypeClass* ct = static_cast<TypeClass*>(origType);

        Type* resultType;
        if (ct->sym->isInterfaceDeclaration())
        {
            // For interfaces, the first entry in the vtbl is actually a pointer
            // to an Interface instance, which has the type info as its first
            // member, so we have to add an extra layer of indirection.
            resultType = Type::typeinfointerface->type->pointerTo();
        }
        else
        {
            resultType = Type::typeinfointerface->type;
        }

        V = DtoBitCast(V, DtoType(resultType->pointerTo()->pointerTo()));
    }

    return new DVarValue(type, V);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DotVarExp::cacheLvalue(IRState* p)
{
    Logger::println("Caching l-value of %s", toChars());
    LOG_SCOPE;
    cachedLvalue = toElem(p)->getLVal();
}

DValue* DotVarExp::toElem(IRState* p)
{
    Logger::print("DotVarExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    if (cachedLvalue)
    {
        Logger::println("using cached lvalue");
        LLValue *V = cachedLvalue;
        VarDeclaration* vd = var->isVarDeclaration();
        assert(vd);
        return new DVarValue(type, vd, V);
    }

    DValue* l = e1->toElem(p);

    Type* e1type = e1->type->toBasetype();

    //Logger::println("e1type=%s", e1type->toChars());
    //Logger::cout() << *DtoType(e1type) << '\n';

    if (VarDeclaration* vd = var->isVarDeclaration()) {
        LLValue* arrptr;
        // indexing struct pointer
        if (e1type->ty == Tpointer) {
            assert(e1type->nextOf()->ty == Tstruct);
            TypeStruct* ts = static_cast<TypeStruct*>(e1type->nextOf());
            arrptr = DtoIndexStruct(l->getRVal(), ts->sym, vd);
        }
        // indexing normal struct
        else if (e1type->ty == Tstruct) {
            TypeStruct* ts = static_cast<TypeStruct*>(e1type);
            arrptr = DtoIndexStruct(l->getRVal(), ts->sym, vd);
        }
        // indexing class
        else if (e1type->ty == Tclass) {
            TypeClass* tc = static_cast<TypeClass*>(e1type);
            arrptr = DtoIndexClass(l->getRVal(), tc->sym, vd);
        }
        else
            llvm_unreachable("Unknown DotVarExp type for VarDeclaration.");

        //Logger::cout() << "mem: " << *arrptr << '\n';
        return new DVarValue(type, vd, arrptr);
    }
    else if (FuncDeclaration* fdecl = var->isFuncDeclaration())
    {
        DtoResolveFunction(fdecl);

        // This is a bit more convoluted than it would need to be, because it
        // has to take templated interface methods into account, for which
        // isFinalFunc is not necessarily true.
        const bool nonFinal = !fdecl->isFinalFunc() &&
            (fdecl->isAbstract() || fdecl->isVirtual());

        // If we are calling a non-final interface function, we need to get
        // the pointer to the underlying object instead of passing the
        // interface pointer directly.
        // Unless it is a cpp interface, in that case, we have to match
        // C++ behavior and pass the interface pointer.
        LLValue* passedThis = 0;
        if (e1type->ty == Tclass)
        {
            TypeClass* tc = static_cast<TypeClass*>(e1type);
            if (tc->sym->isInterfaceDeclaration() && nonFinal && !tc->sym->isCPPinterface())
                passedThis = DtoCastInterfaceToObject(l, NULL)->getRVal();
        }
        LLValue* vthis = l->getRVal();
        if (!passedThis) passedThis = vthis;

        // Decide whether this function needs to be looked up in the vtable.
        // Even virtual functions are looked up directly if super or DotTypeExp
        // are used, thus we need to walk through the this expression and check.
        bool vtbllookup = nonFinal;
        Expression* e = e1;
        while (e && vtbllookup)
        {
            if (e->op == TOKsuper || e->op == TOKdottype)
                vtbllookup = false;
            else if (e->op == TOKcast)
                e = static_cast<CastExp*>(e)->e1;
            else
                break;
        }

        // Get the actual function value to call.
        LLValue* funcval = 0;
        if (vtbllookup)
        {
            DImValue thisVal(e1type, vthis);
            funcval = DtoVirtualFunctionPointer(&thisVal, fdecl, toChars());
        }
        else
        {
            funcval = fdecl->ir.irFunc->func;
        }
        assert(funcval);

        return new DFuncValue(fdecl, funcval, passedThis);
    }

    llvm_unreachable("Unknown target for VarDeclaration.");
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* ThisExp::toElem(IRState* p)
{
    Logger::print("ThisExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    // regular this expr
    if (VarDeclaration* vd = var->isVarDeclaration()) {
        LLValue* v;
        Dsymbol* vdparent = vd->toParent2();
        Identifier *ident = p->func()->decl->ident;
        // In D1, contracts are treated as normal nested methods, 'this' is
        // just passed in the context struct along with any used parameters.
        if (ident == Id::ensure || ident == Id::require) {
            Logger::println("contract this exp");
            v = p->func()->nestArg;
            v = DtoBitCast(v, DtoType(type)->getPointerTo());
        } else
        if (vdparent != p->func()->decl) {
            Logger::println("nested this exp");
            return DtoNestedVariable(loc, type, vd, type->ty == Tstruct);
        }
        else {
            Logger::println("normal this exp");
            v = p->func()->thisArg;
        }
        return new DVarValue(type, vd, v);
    }

    llvm_unreachable("No VarDeclaration in ThisExp.");
}

//////////////////////////////////////////////////////////////////////////////////////////

void IndexExp::cacheLvalue(IRState* p)
{
    Logger::println("Caching l-value of %s", toChars());
    LOG_SCOPE;
    cachedLvalue = toElem(p)->getLVal();
}

DValue* IndexExp::toElem(IRState* p)
{
    Logger::print("IndexExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    if (cachedLvalue)
    {
        LLValue* V = cachedLvalue;
        return new DVarValue(type, V);
    }

    DValue* l = e1->toElem(p);

    Type* e1type = e1->type->toBasetype();

    p->arrays.push_back(l); // if $ is used it must be an array so this is fine.
    DValue* r = e2->toElem(p);
    p->arrays.pop_back();

    LLValue* zero = DtoConstUint(0);

    LLValue* arrptr = 0;
    if (e1type->ty == Tpointer) {
        arrptr = DtoGEP1(l->getRVal(),r->getRVal());
    }
    else if (e1type->ty == Tsarray) {
        if (gIR->emitArrayBoundsChecks() && !skipboundscheck)
            DtoArrayBoundsCheck(loc, l, r);
        arrptr = DtoGEP(l->getRVal(), zero, r->getRVal());
    }
    else if (e1type->ty == Tarray) {
        if (gIR->emitArrayBoundsChecks() && !skipboundscheck)
            DtoArrayBoundsCheck(loc, l, r);
        arrptr = DtoArrayPtr(l);
        arrptr = DtoGEP1(arrptr,r->getRVal());
    }
    else if (e1type->ty == Taarray) {
        return DtoAAIndex(loc, type, l, r, modifiable);
    }
    else {
        Logger::println("e1type: %s", e1type->toChars());
        llvm_unreachable("Unknown IndexExp target.");
    }
    return new DVarValue(type, arrptr);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* SliceExp::toElem(IRState* p)
{
    Logger::print("SliceExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    // this is the new slicing code, it's different in that a full slice will no longer retain the original pointer.
    // but this was broken if there *was* no original pointer, ie. a slice of a slice...
    // now all slices have *both* the 'len' and 'ptr' fields set to != null.

    // value being sliced
    LLValue* elen = 0;
    LLValue* eptr;
    DValue* e = e1->toElem(p);

    // handle pointer slicing
    Type* etype = e1->type->toBasetype();
    if (etype->ty == Tpointer)
    {
        assert(lwr);
        eptr = e->getRVal();
    }
    // array slice
    else
    {
        eptr = DtoArrayPtr(e);
    }

    // has lower bound, pointer needs adjustment
    if (lwr)
    {
        // must have upper bound too then
        assert(upr);

        // get bounds (make sure $ works)
        p->arrays.push_back(e);
        DValue* lo = lwr->toElem(p);
        DValue* up = upr->toElem(p);
        p->arrays.pop_back();
        LLValue* vlo = lo->getRVal();
        LLValue* vup = up->getRVal();

        if (gIR->emitArrayBoundsChecks())
            DtoArrayBoundsCheck(loc, e, up, lo);

        // offset by lower
        eptr = DtoGEP1(eptr, vlo);

        // adjust length
        elen = p->ir->CreateSub(vup, vlo, "tmp");
    }
    // no bounds or full slice -> just convert to slice
    else
    {
        assert(e1->type->toBasetype()->ty != Tpointer);
        // if the sliceee is a static array, we use the length of that as DMD seems
        // to give contrary inconsistent sizesin some multidimensional static array cases.
        // (namely default initialization, int[16][16] arr; -> int[256] arr = 0;)
        if (etype->ty == Tsarray)
        {
            TypeSArray* tsa = static_cast<TypeSArray*>(etype);
            elen = DtoConstSize_t(tsa->dim->toUInteger());

            // in this case, we also need to make sure the pointer is cast to the innermost element type
            eptr = DtoBitCast(eptr, DtoType(tsa->nextOf()->pointerTo()));
        }
    }

    // The frontend generates a SliceExp of static array type when assigning a
    // fixed-width slice to a static array.
    if (type->toBasetype()->ty == Tsarray)
    {
        return new DVarValue(type,
            DtoBitCast(eptr, DtoType(type->pointerTo())));
    }

    if (!elen) elen = DtoArrayLen(e);
    return new DSliceValue(type, elen, eptr);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CmpExp::toElem(IRState* p)
{
    Logger::print("CmpExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    Type* t = e1->type->toBasetype();

    LLValue* eval = 0;

    if (t->isintegral() || t->ty == Tpointer || t->ty == Tnull)
    {
        llvm::ICmpInst::Predicate icmpPred;
        tokToIcmpPred(op, isLLVMUnsigned(t), &icmpPred, &eval);

        if (!eval)
        {
            LLValue* a = l->getRVal();
            LLValue* b = r->getRVal();
            if (Logger::enabled())
            {
                Logger::cout() << "type 1: " << *a << '\n';
                Logger::cout() << "type 2: " << *b << '\n';
            }
            if (a->getType() != b->getType())
                b = DtoBitCast(b, a->getType());
            eval = p->ir->CreateICmp(icmpPred, a, b, "tmp");
        }
    }
    else if (t->isfloating())
    {
        llvm::FCmpInst::Predicate cmpop;
        switch(op)
        {
        case TOKlt:
            cmpop = llvm::FCmpInst::FCMP_OLT;break;
        case TOKle:
            cmpop = llvm::FCmpInst::FCMP_OLE;break;
        case TOKgt:
            cmpop = llvm::FCmpInst::FCMP_OGT;break;
        case TOKge:
            cmpop = llvm::FCmpInst::FCMP_OGE;break;
        case TOKunord:
            cmpop = llvm::FCmpInst::FCMP_UNO;break;
        case TOKule:
            cmpop = llvm::FCmpInst::FCMP_ULE;break;
        case TOKul:
            cmpop = llvm::FCmpInst::FCMP_ULT;break;
        case TOKuge:
            cmpop = llvm::FCmpInst::FCMP_UGE;break;
        case TOKug:
            cmpop = llvm::FCmpInst::FCMP_UGT;break;
        case TOKue:
            cmpop = llvm::FCmpInst::FCMP_UEQ;break;
        case TOKlg:
            cmpop = llvm::FCmpInst::FCMP_ONE;break;
        case TOKleg:
            cmpop = llvm::FCmpInst::FCMP_ORD;break;

        default:
            llvm_unreachable("Unsupported floating point comparison operator.");
        }
        eval = p->ir->CreateFCmp(cmpop, l->getRVal(), r->getRVal(), "tmp");
    }
    else if (t->ty == Tsarray || t->ty == Tarray)
    {
        Logger::println("static or dynamic array");
        eval = DtoArrayCompare(loc,op,l,r);
    }
    else if (t->ty == Taarray)
    {
        eval = LLConstantInt::getFalse(gIR->context());
    }
    else if (t->ty == Tdelegate)
    {
        llvm::ICmpInst::Predicate icmpPred;
        tokToIcmpPred(op, isLLVMUnsigned(t), &icmpPred, &eval);

        if (!eval)
        {
            // First compare the function pointers, then the context ones. This is
            // what DMD does.
            llvm::Value* lhs = l->getRVal();
            llvm::Value* rhs = r->getRVal();

            llvm::BasicBlock* oldend = p->scopeend();
            llvm::BasicBlock* fptreq = llvm::BasicBlock::Create(
                gIR->context(), "fptreq", gIR->topfunc(), oldend);
            llvm::BasicBlock* fptrneq = llvm::BasicBlock::Create(
                gIR->context(), "fptrneq", gIR->topfunc(), oldend);
            llvm::BasicBlock* dgcmpend = llvm::BasicBlock::Create(
                gIR->context(), "dgcmpend", gIR->topfunc(), oldend);

            llvm::Value* lfptr = p->ir->CreateExtractValue(lhs, 1, ".lfptr");
            llvm::Value* rfptr = p->ir->CreateExtractValue(rhs, 1, ".rfptr");

            llvm::Value* fptreqcmp = p->ir->CreateICmp(llvm::ICmpInst::ICMP_EQ,
                lfptr, rfptr, ".fptreqcmp");
            llvm::BranchInst::Create(fptreq, fptrneq, fptreqcmp, p->scopebb());

            p->scope() = IRScope(fptreq, fptrneq);
            llvm::Value* lctx = p->ir->CreateExtractValue(lhs, 0, ".lctx");
            llvm::Value* rctx = p->ir->CreateExtractValue(rhs, 0, ".rctx");
            llvm::Value* ctxcmp = p->ir->CreateICmp(icmpPred, lctx, rctx, ".ctxcmp");
            llvm::BranchInst::Create(dgcmpend,p->scopebb());

            p->scope() = IRScope(fptrneq, dgcmpend);
            llvm::Value* fptrcmp = p->ir->CreateICmp(icmpPred, lfptr, rfptr, ".fptrcmp");
            llvm::BranchInst::Create(dgcmpend,p->scopebb());

            p->scope() = IRScope(dgcmpend, oldend);
            llvm::PHINode* phi = p->ir->CreatePHI(ctxcmp->getType(), 2, ".dgcmp");
            phi->addIncoming(ctxcmp, fptreq);
            phi->addIncoming(fptrcmp, fptrneq);
            eval = phi;
        }
    }
    else
    {
        llvm_unreachable("Unsupported CmpExp type");
    }

    return new DImValue(type, eval);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* EqualExp::toElem(IRState* p)
{
    Logger::print("EqualExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);
    LLValue* lv = l->getRVal();
    LLValue* rv = r->getRVal();

    Type* t = e1->type->toBasetype();

    LLValue* eval = 0;

    // the Tclass catches interface comparisons, regular
    // class equality should be rewritten as a.opEquals(b) by this time
    if (t->isintegral() || t->ty == Tpointer || t->ty == Tclass || t->ty == Tnull)
    {
        Logger::println("integral or pointer or interface");
        llvm::ICmpInst::Predicate cmpop;
        switch(op)
        {
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
        if (Logger::enabled())
        {
            Logger::cout() << "lv: " << *lv << '\n';
            Logger::cout() << "rv: " << *rv << '\n';
        }
        eval = p->ir->CreateICmp(cmpop, lv, rv, "tmp");
    }
    else if (t->isfloating()) // includes iscomplex
    {
        eval = DtoBinNumericEquals(loc, l, r, op);
    }
    else if (t->ty == Tsarray || t->ty == Tarray)
    {
        Logger::println("static or dynamic array");
        eval = DtoArrayEquals(loc,op,l,r);
    }
    else if (t->ty == Taarray)
    {
        Logger::println("associative array");
        eval = DtoAAEquals(loc,op,l,r);
    }
    else if (t->ty == Tdelegate)
    {
        Logger::println("delegate");
        eval = DtoDelegateEquals(op,l->getRVal(),r->getRVal());
    }
    else if (t->ty == Tstruct)
    {
        Logger::println("struct");
        // when this is reached it means there is no opEquals overload.
        eval = DtoStructEquals(op,l,r);
    }
    else
    {
        llvm_unreachable("Unsupported EqualExp type.");
    }

    return new DImValue(type, eval);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* PostExp::toElem(IRState* p)
{
    Logger::print("PostExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    e2->toElem(p);

    LLValue* val = l->getRVal();
    LLValue* post = 0;

    Type* e1type = e1->type->toBasetype();
    Type* e2type = e2->type->toBasetype();

    if (e1type->isintegral())
    {
        assert(e2type->isintegral());
        LLValue* one = LLConstantInt::get(val->getType(), 1, !e2type->isunsigned());
        if (op == TOKplusplus) {
            post = llvm::BinaryOperator::CreateAdd(val,one,"tmp",p->scopebb());
        }
        else if (op == TOKminusminus) {
            post = llvm::BinaryOperator::CreateSub(val,one,"tmp",p->scopebb());
        }
    }
    else if (e1type->ty == Tpointer)
    {
        assert(e2->op == TOKint64);
        LLConstant* minusone = LLConstantInt::get(DtoSize_t(),static_cast<uint64_t>(-1),true);
        LLConstant* plusone = LLConstantInt::get(DtoSize_t(),static_cast<uint64_t>(1),false);
        LLConstant* whichone = (op == TOKplusplus) ? plusone : minusone;
        post = llvm::GetElementPtrInst::Create(val, whichone, "tmp", p->scopebb());
    }
    else if (e1type->isfloating())
    {
        assert(e2type->isfloating());
        LLValue* one = DtoConstFP(e1type, ldouble(1.0));
        if (op == TOKplusplus) {
            post = llvm::BinaryOperator::CreateFAdd(val,one,"tmp",p->scopebb());
        }
        else if (op == TOKminusminus) {
            post = llvm::BinaryOperator::CreateFSub(val,one,"tmp",p->scopebb());
        }
    }
    else
    assert(post);

    DtoStore(post,l->getLVal());

    return new DImValue(type,val);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* NewExp::toElem(IRState* p)
{
    Logger::print("NewExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(newtype);
    Type* ntype = newtype->toBasetype();

    // new class
    if (ntype->ty == Tclass) {
        Logger::println("new class");
        return DtoNewClass(loc, static_cast<TypeClass*>(ntype), this);
    }
    // new dynamic array
    else if (ntype->ty == Tarray)
    {
        Logger::println("new dynamic array: %s", newtype->toChars());
        // get dim
        assert(arguments);
        assert(arguments->dim >= 1);
        if (arguments->dim == 1)
        {
            DValue* sz = static_cast<Expression*>(arguments->data[0])->toElem(p);
            // allocate & init
            return DtoNewDynArray(loc, newtype, sz, true);
        }
        else
        {
            size_t ndims = arguments->dim;
            std::vector<DValue*> dims;
            dims.reserve(ndims);
            for (size_t i=0; i<ndims; ++i)
                dims.push_back(static_cast<Expression*>(arguments->data[i])->toElem(p));
            return DtoNewMulDimDynArray(loc, newtype, &dims[0], ndims, true);
        }
    }
    // new static array
    else if (ntype->ty == Tsarray)
    {
        llvm_unreachable("Static array new should decay to dynamic array.");
    }
    // new struct
    else if (ntype->ty == Tstruct)
    {
        Logger::println("new struct on heap: %s\n", newtype->toChars());
        // allocate
        LLValue* mem = 0;
        if (allocator)
        {
            // custom allocator
            DtoResolveFunction(allocator);
            DFuncValue dfn(allocator, allocator->ir.irFunc->func);
            DValue* res = DtoCallFunction(loc, NULL, &dfn, newargs);
            mem = DtoBitCast(res->getRVal(), DtoType(ntype->pointerTo()), ".newstruct_custom");
        } else
        {
            // default allocator
            mem = DtoNew(newtype);
        }
        // init
        TypeStruct* ts = static_cast<TypeStruct*>(ntype);
        if (ts->isZeroInit(ts->sym->loc)) {
            DtoAggrZeroInit(mem);
        }
        else {
            assert(ts->sym);
            DtoResolveStruct(ts->sym);
            DtoAggrCopy(mem, ts->sym->ir.irAggr->getInitSymbol());
        }
        if (ts->sym->isNested() && ts->sym->vthis)
            DtoResolveNestedContext(loc, ts->sym, mem);

        // call constructor
        if (member)
        {
            Logger::println("Calling constructor");
            assert(arguments != NULL);
            DtoResolveFunction(member);
            DFuncValue dfn(member, member->ir.irFunc->func, mem);
            DtoCallFunction(loc, ts, &dfn, arguments);
        }
        return new DImValue(type, mem);
    }
    // new basic type
    else
    {
        // allocate
        LLValue* mem = DtoNew(newtype);
        DVarValue tmpvar(newtype, mem);

        // default initialize
        // static arrays never appear here, so using the defaultInit is ok!
        Expression* exp = newtype->defaultInit(loc);
        DValue* iv = exp->toElem(gIR);
        DtoAssign(loc, &tmpvar, iv);

        // return as pointer-to
        return new DImValue(type, mem);
    }

    llvm_unreachable(0);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DeleteExp::toElem(IRState* p)
{
    Logger::print("DeleteExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* dval = e1->toElem(p);
    Type* et = e1->type->toBasetype();

    // simple pointer
    if (et->ty == Tpointer)
    {
        DtoDeleteMemory(dval->isLVal() ? dval->getLVal() : makeLValue(loc, dval));
    }
    // class
    else if (et->ty == Tclass)
    {
        bool onstack = false;
        TypeClass* tc = static_cast<TypeClass*>(et);
        if (tc->sym->isInterfaceDeclaration())
        {
            LLValue *val = dval->getLVal();
            DtoDeleteInterface(val);
            onstack = true;
        }
        else if (DVarValue* vv = dval->isVar()) {
            if (vv->var && vv->var->onstack) {
                DtoFinalizeClass(dval->getRVal());
                onstack = true;
            }
        }
        if (!onstack) {
            LLValue* rval = dval->getRVal();
            DtoDeleteClass(rval);
        }
        if (dval->isVar()) {
            LLValue* lval = dval->getLVal();
            DtoStore(LLConstant::getNullValue(lval->getType()->getContainedType(0)), lval);
        }
    }
    // dyn array
    else if (et->ty == Tarray)
    {
        DtoDeleteArray(dval);
        if (dval->isLVal())
            DtoSetArrayToNull(dval->getLVal());
    }
    // unknown/invalid
    else
    {
        llvm_unreachable("Unsupported DeleteExp target.");
    }

    // no value to return
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* ArrayLengthExp::toElem(IRState* p)
{
    Logger::print("ArrayLengthExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* u = e1->toElem(p);
    return new DImValue(type, DtoArrayLen(u));
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* AssertExp::toElem(IRState* p)
{
    Logger::print("AssertExp::toElem: %s\n", toChars());
    LOG_SCOPE;

    if(!global.params.useAssert)
        return NULL;

    // condition
    DValue* cond;
    Type* condty;

    // special case for dmd generated assert(this); when not in -release mode
    if (e1->op == TOKthis && static_cast<ThisExp*>(e1)->var == NULL)
    {
        LLValue* thisarg = p->func()->thisArg;
        assert(thisarg && "null thisarg, but we're in assert(this) exp;");
        LLValue* thisptr = DtoLoad(p->func()->thisArg);
        condty = e1->type->toBasetype();
        cond = new DImValue(condty, thisptr);
    }
    else
    {
        cond = e1->toElem(p);
        condty = e1->type->toBasetype();
    }

    // create basic blocks
    llvm::BasicBlock* oldend = p->scopeend();
    llvm::BasicBlock* assertbb = llvm::BasicBlock::Create(gIR->context(), "assert", p->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "noassert", p->topfunc(), oldend);

    // test condition
    LLValue* condval = DtoCast(loc, cond, Type::tbool)->getRVal();

    // branch
    llvm::BranchInst::Create(endbb, assertbb, condval, p->scopebb());

    // call assert runtime functions
    p->scope() = IRScope(assertbb,endbb);
    DtoAssert(p->func()->decl->getModule(), loc, msg ? msg->toElemDtor(p) : NULL);

    // rewrite the scope
    p->scope() = IRScope(endbb,oldend);


    FuncDeclaration* invdecl;
    // class invariants
    if(
        global.params.useInvariants &&
        condty->ty == Tclass &&
        !(static_cast<TypeClass*>(condty)->sym->isInterfaceDeclaration()))
    {
        Logger::println("calling class invariant");
        llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module,
            gABI->mangleForLLVM("_D9invariant12_d_invariantFC6ObjectZv", LINKd).c_str());
        LLValue* arg = DtoBitCast(cond->getRVal(), fn->getFunctionType()->getParamType(0));
        gIR->CreateCallOrInvoke(fn, arg);
    }
    // struct invariants
    else if(
        global.params.useInvariants &&
        condty->ty == Tpointer && condty->nextOf()->ty == Tstruct &&
        (invdecl = static_cast<TypeStruct*>(condty->nextOf())->sym->inv) != NULL)
    {
        Logger::print("calling struct invariant");
        DtoResolveFunction(invdecl);
        DFuncValue invfunc(invdecl, invdecl->ir.irFunc->func, cond->getRVal());
        DtoCallFunction(loc, NULL, &invfunc, NULL);
    }

    // DMD allows syntax like this:
    // f() == 0 || assert(false)
    return new DImValue(type, DtoConstBool(false));
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* NotExp::toElem(IRState* p)
{
    Logger::print("NotExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* u = e1->toElem(p);

    LLValue* b = DtoCast(loc, u, Type::tbool)->getRVal();

    LLConstant* zero = DtoConstBool(false);
    b = p->ir->CreateICmpEQ(b,zero);

    return new DImValue(type, b);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* AndAndExp::toElem(IRState* p)
{
    Logger::print("AndAndExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* u = e1->toElem(p);

    llvm::BasicBlock* oldend = p->scopeend();
    llvm::BasicBlock* andand = llvm::BasicBlock::Create(gIR->context(), "andand", gIR->topfunc(), oldend);
    llvm::BasicBlock* andandend = llvm::BasicBlock::Create(gIR->context(), "andandend", gIR->topfunc(), oldend);

    LLValue* ubool = DtoCast(loc, u, Type::tbool)->getRVal();

    llvm::BasicBlock* oldblock = p->scopebb();
    llvm::BranchInst::Create(andand,andandend,ubool,p->scopebb());

    p->scope() = IRScope(andand, andandend);
    DValue* v = e2->toElemDtor(p);

    LLValue* vbool = 0;
    if (!v->isFunc() && v->getType() != Type::tvoid)
    {
        vbool = DtoCast(loc, v, Type::tbool)->getRVal();
    }

    llvm::BasicBlock* newblock = p->scopebb();
    llvm::BranchInst::Create(andandend,p->scopebb());
    p->scope() = IRScope(andandend, oldend);

    LLValue* resval = 0;
    if (ubool == vbool || !vbool) {
        // No need to create a PHI node.
        resval = ubool;
    } else {
        llvm::PHINode* phi = p->ir->CreatePHI(LLType::getInt1Ty(gIR->context()), 2, "andandval");
        // If we jumped over evaluation of the right-hand side,
        // the result is false. Otherwise it's the value of the right-hand side.
        phi->addIncoming(LLConstantInt::getFalse(gIR->context()), oldblock);
        phi->addIncoming(vbool, newblock);
        resval = phi;
    }

    return new DImValue(type, resval);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* OrOrExp::toElem(IRState* p)
{
    Logger::print("OrOrExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* u = e1->toElem(p);

    llvm::BasicBlock* oldend = p->scopeend();
    llvm::BasicBlock* oror = llvm::BasicBlock::Create(gIR->context(), "oror", gIR->topfunc(), oldend);
    llvm::BasicBlock* ororend = llvm::BasicBlock::Create(gIR->context(), "ororend", gIR->topfunc(), oldend);

    LLValue* ubool = DtoCast(loc, u, Type::tbool)->getRVal();

    llvm::BasicBlock* oldblock = p->scopebb();
    llvm::BranchInst::Create(ororend,oror,ubool,p->scopebb());

    p->scope() = IRScope(oror, ororend);
    DValue* v = e2->toElemDtor(p);

    LLValue* vbool = 0;
    if (v && !v->isFunc() && v->getType() != Type::tvoid)
    {
        vbool = DtoCast(loc, v, Type::tbool)->getRVal();
    }

    llvm::BasicBlock* newblock = p->scopebb();
    llvm::BranchInst::Create(ororend,p->scopebb());
    p->scope() = IRScope(ororend, oldend);

    LLValue* resval = 0;
    if (ubool == vbool || !vbool) {
        // No need to create a PHI node.
        resval = ubool;
    } else {
        llvm::PHINode* phi = p->ir->CreatePHI(LLType::getInt1Ty(gIR->context()), 2, "ororval");
        // If we jumped over evaluation of the right-hand side,
        // the result is true. Otherwise, it's the value of the right-hand side.
        phi->addIncoming(LLConstantInt::getTrue(gIR->context()), oldblock);
        phi->addIncoming(vbool, newblock);
        resval = phi;
    }

    return new DImValue(type, resval);
}

//////////////////////////////////////////////////////////////////////////////////////////

#define BinBitExp(X,Y) \
DValue* X##Exp::toElem(IRState* p) \
{ \
    Logger::print("%sExp::toElem: %s @ %s\n", #X, toChars(), type->toChars()); \
    LOG_SCOPE; \
    DValue* u = e1->toElem(p); \
    DValue* v = e2->toElem(p); \
    errorOnIllegalArrayOp(this, e1, e2); \
    LLValue* x = llvm::BinaryOperator::Create(llvm::Instruction::Y, u->getRVal(), v->getRVal(), "tmp", p->scopebb()); \
    return new DImValue(type, x); \
}

BinBitExp(And,And)
BinBitExp(Or,Or)
BinBitExp(Xor,Xor)
BinBitExp(Shl,Shl)
BinBitExp(Ushr,LShr)

DValue* ShrExp::toElem(IRState* p)
{
    Logger::print("ShrExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    DValue* u = e1->toElem(p);
    DValue* v = e2->toElem(p);
    LLValue* x;
    if (isLLVMUnsigned(e1->type))
        x = p->ir->CreateLShr(u->getRVal(), v->getRVal(), "tmp");
    else
        x = p->ir->CreateAShr(u->getRVal(), v->getRVal(), "tmp");
    return new DImValue(type, x);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* HaltExp::toElem(IRState* p)
{
    Logger::print("HaltExp::toElem: %s\n", toChars());
    LOG_SCOPE;

    p->ir->CreateCall(GET_INTRINSIC_DECL(trap), "");
    p->ir->CreateUnreachable();

    // this terminated the basicblock, start a new one
    // this is sensible, since someone might goto behind the assert
    // and prevents compiler errors if a terminator follows the assert
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "afterhalt", p->topfunc(), oldend);
    p->scope() = IRScope(bb,oldend);

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DelegateExp::toElem(IRState* p)
{
    Logger::print("DelegateExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    if(func->isStatic())
        error("can't take delegate of static function %s, it does not require a context ptr", func->toChars());

    LLPointerType* int8ptrty = getPtrToType(LLType::getInt8Ty(gIR->context()));

    assert(type->toBasetype()->ty == Tdelegate);
    LLType* dgty = DtoType(type);

    DValue* u = e1->toElem(p);
    LLValue* uval;
    if (DFuncValue* f = u->isFunc()) {
        assert(f->func);
        LLValue* contextptr = DtoNestedContext(loc, f->func);
        uval = DtoBitCast(contextptr, getVoidPtrType());
    }
    else {
        DValue* src = u;
        if (ClassDeclaration* cd = u->getType()->isClassHandle())
        {
            Logger::println("context type is class handle");
            if (cd->isInterfaceDeclaration())
            {
                Logger::println("context type is interface");
                src = DtoCastInterfaceToObject(u, ClassDeclaration::object->type);
            }
        }
        uval = src->getRVal();
    }

    if (Logger::enabled())
        Logger::cout() << "context = " << *uval << '\n';

    LLValue* castcontext = DtoBitCast(uval, int8ptrty);

    Logger::println("func: '%s'", func->toPrettyChars());

    LLValue* castfptr;

    if (e1->op != TOKsuper && e1->op != TOKdottype && func->isVirtual() && !func->isFinalFunc())
        castfptr = DtoVirtualFunctionPointer(u, func, toChars());
    else if (func->isAbstract())
        llvm_unreachable("Delegate to abstract method not implemented.");
    else if (func->toParent()->isInterfaceDeclaration())
        llvm_unreachable("Delegate to interface method not implemented.");
    else
    {
        DtoResolveFunction(func);

        // We need to actually codegen the function here, as literals are not
        // added to the module member list.
        if (func->semanticRun == PASSsemantic3done)
        {
            Dsymbol *owner = func->toParent();
            while (!owner->isTemplateInstance() && owner->toParent())
                owner = owner->toParent();
            if (owner->isTemplateInstance() || owner == p->dmodule)
            {
                func->codegen(p);
            }
        }

        castfptr = func->ir.irFunc->func;
    }

    castfptr = DtoBitCast(castfptr, dgty->getContainedType(1));

    return new DImValue(type, DtoAggrPair(DtoType(type), castcontext, castfptr, ".dg"));
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* IdentityExp::toElem(IRState* p)
{
    Logger::print("IdentityExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);
    LLValue* lv = l->getRVal();
    LLValue* rv = r->getRVal();

    Type* t1 = e1->type->toBasetype();

    // handle dynarray specially
    if (t1->ty == Tarray)
        return new DImValue(type, DtoDynArrayIs(op,l,r));
    // also structs
    else if (t1->ty == Tstruct)
        return new DImValue(type, DtoStructEquals(op,l,r));

    // FIXME this stuff isn't pretty
    LLValue* eval = 0;

    if (t1->ty == Tdelegate) {
        if (r->isNull()) {
            rv = NULL;
        }
        else {
            assert(lv->getType() == rv->getType());
        }
        eval = DtoDelegateEquals(op,lv,rv);
    }
    else if (t1->isfloating()) // includes iscomplex
    {
       eval = DtoBinNumericEquals(loc, l, r, op);
    }
    else if (t1->ty == Tpointer || t1->ty == Tclass)
    {
        if (lv->getType() != rv->getType()) {
            if (r->isNull())
                rv = llvm::ConstantPointerNull::get(isaPointer(lv->getType()));
            else
                rv = DtoBitCast(rv, lv->getType());
        }
        eval = (op == TOKidentity)
        ?   p->ir->CreateICmpEQ(lv,rv,"tmp")
        :   p->ir->CreateICmpNE(lv,rv,"tmp");
    }
    else {
        assert(lv->getType() == rv->getType());
        eval = (op == TOKidentity)
        ?   p->ir->CreateICmpEQ(lv,rv,"tmp")
        :   p->ir->CreateICmpNE(lv,rv,"tmp");
    }
    return new DImValue(type, eval);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CommaExp::toElem(IRState* p)
{
    Logger::print("CommaExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    if (cachedLvalue)
    {
        LLValue* V = cachedLvalue;
        return new DVarValue(type, V);
    }

    e1->toElem(p);
    DValue* v = e2->toElem(p);

    // Actually, we can get qualifier mismatches in the 2.064 frontend:
    // assert(e2->type == type);

    return v;
}

void CommaExp::cacheLvalue(IRState* p)
{
    Logger::println("Caching l-value of %s", toChars());
    LOG_SCOPE;
    cachedLvalue = toElem(p)->getLVal();
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CondExp::toElem(IRState* p)
{
    Logger::print("CondExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    Type* dtype = type->toBasetype();

    DValue* dvv;
    // voids returns will need no storage
    if (dtype->ty != Tvoid) {
        // allocate a temporary for the final result. failed to come up with a better way :/
        LLValue* resval = DtoAlloca(dtype,"condtmp");
        dvv = new DVarValue(type, resval);
    } else {
        dvv = new DConstValue(type, getNullValue(voidToI8(DtoType(dtype))));
    }

    llvm::BasicBlock* oldend = p->scopeend();
    llvm::BasicBlock* condtrue = llvm::BasicBlock::Create(gIR->context(), "condtrue", gIR->topfunc(), oldend);
    llvm::BasicBlock* condfalse = llvm::BasicBlock::Create(gIR->context(), "condfalse", gIR->topfunc(), oldend);
    llvm::BasicBlock* condend = llvm::BasicBlock::Create(gIR->context(), "condend", gIR->topfunc(), oldend);

    DValue* c = econd->toElem(p);
    LLValue* cond_val = DtoCast(loc, c, Type::tbool)->getRVal();
    llvm::BranchInst::Create(condtrue,condfalse,cond_val,p->scopebb());

    p->scope() = IRScope(condtrue, condfalse);
    DValue* u = e1->toElemDtor(p);
    if (dtype->ty != Tvoid)
        DtoAssign(loc, dvv, u);
    llvm::BranchInst::Create(condend,p->scopebb());

    p->scope() = IRScope(condfalse, condend);
    DValue* v = e2->toElemDtor(p);
    if (dtype->ty != Tvoid)
        DtoAssign(loc, dvv, v);
    llvm::BranchInst::Create(condend,p->scopebb());

    p->scope() = IRScope(condend, oldend);
    return dvv;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* ComExp::toElem(IRState* p)
{
    Logger::print("ComExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* u = e1->toElem(p);

    LLValue* value = u->getRVal();
    LLValue* minusone = LLConstantInt::get(value->getType(), static_cast<uint64_t>(-1), true);
    value = llvm::BinaryOperator::Create(llvm::Instruction::Xor, value, minusone, "tmp", p->scopebb());

    return new DImValue(type, value);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* NegExp::toElem(IRState* p)
{
    Logger::print("NegExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);

    if (type->iscomplex()) {
        return DtoComplexNeg(loc, type, l);
    }

    LLValue* val = l->getRVal();

    if (type->isintegral())
        val = gIR->ir->CreateNeg(val,"negval");
    else
        val = gIR->ir->CreateFNeg(val,"negval");

    return new DImValue(type, val);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CatExp::toElem(IRState* p)
{
    Logger::print("CatExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    return DtoCatArrays(type, e1, e2);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CatAssignExp::toElem(IRState* p)
{
    Logger::print("CatAssignExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);

    Type* e1type = e1->type->toBasetype();
    assert(e1type->ty == Tarray);
    Type* elemtype = e1type->nextOf()->toBasetype();
    Type* e2type = e2->type->toBasetype();

    if (e1type->ty == Tarray && e2type->ty == Tdchar &&
        (elemtype->ty == Tchar || elemtype->ty == Twchar))
    {
        if (elemtype->ty == Tchar)
            // append dchar to char[]
            DtoAppendDCharToString(l, e2);
        else /*if (elemtype->ty == Twchar)*/
            // append dchar to wchar[]
            DtoAppendDCharToUnicodeString(l, e2);
    }
    else if (e1type->equals(e2type)) {
        // apeend array
        DSliceValue* slice = DtoCatAssignArray(l,e2);
        DtoAssign(loc, l, slice);
    }
    else {
        // append element
        DtoCatAssignElement(loc, e1type, l, e2);
    }

    return l;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* FuncExp::toElem(IRState* p)
{
    Logger::print("FuncExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(fd);

    if (fd->tok == TOKreserved && type->ty == Tpointer)
    {
        // This is a lambda that was inferred to be a function literal instead
        // of a delegate, so set tok here in order to get correct types/mangling.
        // Horrible hack, but DMD does the same thing.
        fd->tok = TOKfunction;
        fd->vthis = NULL;
    }

    if (fd->isNested()) Logger::println("nested");
    Logger::println("kind = %s", fd->kind());

    // We need to actually codegen the function here, as literals are not added
    // to the module member list.
    fd->codegen(p);
    assert(fd->ir.irFunc->func);

    if (fd->isNested()) {
        LLType* dgty = DtoType(type);

        LLValue* cval;
        IrFunction* irfn = p->func();
        if (irfn->nestedVar
            // We cannot use a frame allocated in one function
            // for a delegate created in another function
            // (that happens with anonymous functions)
            && fd->toParent2() == irfn->decl
            )
            cval = irfn->nestedVar;
        else if (irfn->nestArg)
            cval = DtoLoad(irfn->nestArg);
        // TODO: should we enable that for D1 as well?
        else if (irfn->thisArg)
        {
            AggregateDeclaration* ad = irfn->decl->isMember2();
            if (!ad || !ad->vthis) {
                cval = getNullPtr(getVoidPtrType());
            } else {
                cval = ad->isClassDeclaration() ? DtoLoad(irfn->thisArg) : irfn->thisArg;
                cval = DtoLoad(DtoGEPi(cval, 0,ad->vthis->ir.irField->index, ".vthis"));
            }
        }
        else
            cval = getNullPtr(getVoidPtrType());
        cval = DtoBitCast(cval, dgty->getContainedType(0));

        LLValue* castfptr = DtoBitCast(fd->ir.irFunc->func, dgty->getContainedType(1));

        return new DImValue(type, DtoAggrPair(cval, castfptr, ".func"));

    } else {
        return new DFuncValue(type, fd, fd->ir.irFunc->func);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* FuncExp::toConstElem(IRState* p)
{
    Logger::print("FuncExp::toConstElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(fd);

    if (fd->tok == TOKreserved && type->ty == Tpointer)
    {
        // This is a lambda that was inferred to be a function literal instead
        // of a delegate, so set tok here in order to get correct types/mangling.
        // Horrible hack, but DMD does the same thing in FuncExp::toElem and
        // other random places.
        fd->tok = TOKfunction;
        fd->vthis = NULL;
    }

    if (fd->tok != TOKfunction)
    {
        assert(fd->tok == TOKdelegate || fd->tok == TOKreserved);
        error("delegate literals as constant expressions are not yet allowed");
        return 0;
    }

    // We need to actually codegen the function here, as literals are not added
    // to the module member list.
    fd->codegen(p);
    assert(fd->ir.irFunc->func);

    return fd->ir.irFunc->func;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* ArrayLiteralExp::toElem(IRState* p)
{
    Logger::print("ArrayLiteralExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    // D types
    Type* arrayType = type->toBasetype();
    Type* elemType = arrayType->nextOf()->toBasetype();

    // is dynamic ?
    bool const dyn = (arrayType->ty == Tarray);
    // length
    size_t const len = elements->dim;

    // llvm target type
    LLType* llType = DtoType(arrayType);
    if (Logger::enabled())
        Logger::cout() << (dyn?"dynamic":"static") << " array literal with length " << len << " of D type: '" << arrayType->toChars() << "' has llvm type: '" << *llType << "'\n";

    // llvm storage type
    LLType* llElemType = voidToI8(DtoType(elemType));
    LLType* llStoType = LLArrayType::get(llElemType, len);
    if (Logger::enabled())
        Logger::cout() << "llvm storage type: '" << *llStoType << "'\n";

    // don't allocate storage for zero length dynamic array literals
    if (dyn && len == 0)
    {
        // dmd seems to just make them null...
        return new DSliceValue(type, DtoConstSize_t(0), getNullPtr(getPtrToType(llElemType)));
    }

    if (dyn)
    {
        if (arrayType->isImmutable() && isConstLiteral(this))
        {
            llvm::Constant* init = arrayLiteralToConst(p, this);
            llvm::GlobalVariable* global = new llvm::GlobalVariable(
                *gIR->module,
                init->getType(),
                true,
                llvm::GlobalValue::InternalLinkage,
                init,
                ".immutablearray"
            );
            return new DSliceValue(arrayType, DtoConstSize_t(elements->dim),
                DtoBitCast(global, getPtrToType(llElemType)));
        }

        DSliceValue* dynSlice = DtoNewDynArray(loc, arrayType,
            new DConstValue(Type::tsize_t, DtoConstSize_t(len)), false);
        initializeArrayLiteral(p, this, DtoBitCast(dynSlice->ptr, getPtrToType(llStoType)));
        return dynSlice;
    }
    else
    {
        llvm::Value* storage = DtoRawAlloca(llStoType, 0, "arrayliteral");
        initializeArrayLiteral(p, this, storage);
        return new DImValue(type, storage);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* ArrayLiteralExp::toConstElem(IRState* p)
{
    Logger::print("ArrayLiteralExp::toConstElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    // extract D types
    Type* bt = type->toBasetype();
    Type* elemt = bt->nextOf();

    // build llvm array type
    LLArrayType* arrtype = LLArrayType::get(i1ToI8(voidToI8(DtoType(elemt))), elements->dim);

    // dynamic arrays can occur here as well ...
    bool dyn = (bt->ty != Tsarray);

    llvm::Constant* initval = arrayLiteralToConst(p, this);

    // if static array, we're done
    if (!dyn)
        return initval;

    bool canBeConst = type->isConst() || type->isImmutable();
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(*gIR->module,
        initval->getType(), canBeConst, llvm::GlobalValue::InternalLinkage, initval,
        ".dynarrayStorage");
    gvar->setUnnamedAddr(canBeConst);
    llvm::Constant* store = DtoBitCast(gvar, getPtrToType(arrtype));

    if (bt->ty == Tpointer)
        // we need to return pointer to the static array.
        return store;

    // build a constant dynamic array reference with the .ptr field pointing into store
    LLConstant* idxs[2] = { DtoConstUint(0), DtoConstUint(0) };
    LLConstant* globalstorePtr = llvm::ConstantExpr::getGetElementPtr(store, idxs, true);

    return DtoConstSlice(DtoConstSize_t(elements->dim), globalstorePtr);
}

//////////////////////////////////////////////////////////////////////////////////////////

extern LLConstant* get_default_initializer(VarDeclaration* vd, Initializer* init);

static LLValue* write_zeroes(LLValue* mem, unsigned start, unsigned end) {
    mem = DtoBitCast(mem, getVoidPtrType());
    LLValue* gep = DtoGEPi1(mem, start, ".padding");
    DtoMemSetZero(gep, DtoConstSize_t(end - start));
    return mem;
}

DValue* StructLiteralExp::toElem(IRState* p)
{
    IF_LOG Logger::print("StructLiteralExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    if (sinit)
    {
        // Copied from VarExp::toElem, need to clean this mess up.
        Type* sdecltype = sinit->type->toBasetype();
        Logger::print("Sym: type = %s\n", sdecltype->toChars());
        assert(sdecltype->ty == Tstruct);
        TypeStruct* ts = static_cast<TypeStruct*>(sdecltype);
        assert(ts->sym);
        DtoResolveStruct(ts->sym);

        LLValue* initsym = ts->sym->ir.irAggr->getInitSymbol();
        initsym = DtoBitCast(initsym, DtoType(ts->pointerTo()));
        return new DVarValue(type, initsym);
    }

    if (inProgressMemory) return new DVarValue(type, inProgressMemory);

    // make sure the struct is fully resolved
    DtoResolveStruct(sd);

    // alloca a stack slot
    inProgressMemory = DtoRawAlloca(DtoType(type), 0, ".structliteral");

    // ready elements data
    assert(elements && "struct literal has null elements");
    size_t nexprs = elements->dim;
    Expression **exprs = reinterpret_cast<Expression **>(elements->data);

    // might be reset to an actual i8* value so only a single bitcast is emitted.
    LLValue* voidptr = inProgressMemory;
    unsigned offset = 0;

    // go through fields
    ArrayIter<VarDeclaration> it(sd->fields);
    for (; !it.done(); it.next())
    {
        VarDeclaration* vd = it.get();

        // get initializer expression
        Expression* expr = (it.index < nexprs) ? exprs[it.index] : NULL;
        if (!expr)
        {
            // In case of an union, we can't simply use the default initializer.
            // Consider the type union U7727A1 { int i; double d; } and
            // the declaration U7727A1 u = { d: 1.225 };
            // The loop will first visit variable i and then d. Since d has an
            // explicit initializer, we must use this one. The solution is to
            // peek at the next variables.
            ArrayIter<VarDeclaration> it2(it.array, it.index+1);
            for (; !it2.done(); it2.next())
            {
                VarDeclaration* vd2 = it2.get();
                if (vd->offset != vd2->offset) break;
                it.next(); // skip var
                Expression* expr2 = (it2.index < nexprs) ? exprs[it2.index] : NULL;
                if (expr2)
                {
                    vd = vd2;
                    expr = expr2;
                    break;
                }
            }
        }

        // don't re-initialize unions
        if (vd->offset < offset)
        {
            IF_LOG Logger::println("skipping field: %s %s (+%u)", vd->type->toChars(), vd->toChars(), vd->offset);
            continue;
        }

        // initialize any padding so struct comparisons work
        if (vd->offset != offset)
            voidptr = write_zeroes(voidptr, offset, vd->offset);
        offset = vd->offset + vd->type->size();

        IF_LOG Logger::println("initializing field: %s %s (+%u)", vd->type->toChars(), vd->toChars(), vd->offset);
        LOG_SCOPE

        // get initializer
        DValue* val;
        DConstValue cv(vd->type, NULL); // Only used in one branch; value is set beforehand
        if (expr)
        {
            IF_LOG Logger::println("expr %zu = %s", it.index, expr->toChars());
            val = expr->toElem(gIR);
        }
        else if (vd == sd->vthis) {
            IF_LOG Logger::println("initializing vthis");
            LOG_SCOPE
            val = new DImValue(vd->type, DtoBitCast(DtoNestedContext(loc, sd), DtoType(vd->type)));
        }
        else
        {
            if (vd->init && vd->init->isVoidInitializer())
                continue;
            IF_LOG Logger::println("using default initializer");
            LOG_SCOPE
            cv.c = get_default_initializer(vd, NULL);
            val = &cv;
        }

        // get a pointer to this field
        DVarValue field(vd->type, vd, DtoIndexStruct(inProgressMemory, sd, vd));

        // store the initializer there
        DtoAssign(loc, &field, val, TOKconstruct, true);

        if (expr)
            callPostblit(loc, expr, field.getLVal());

        // Also zero out padding bytes counted as being part of the type in DMD
        // but not in LLVM; e.g. real/x86_fp80.
        int implicitPadding =
            vd->type->size() - gDataLayout->getTypeStoreSize(DtoType(vd->type));
        assert(implicitPadding >= 0);
        if (implicitPadding > 0)
        {
            Logger::println("zeroing %d padding bytes", implicitPadding);
            voidptr = write_zeroes(voidptr, offset - implicitPadding, offset);
        }
    }
    // initialize trailing padding
    if (sd->structsize != offset)
        voidptr = write_zeroes(voidptr, offset, sd->structsize);

    // return as a var
    DValue* result = new DVarValue(type, inProgressMemory);
    inProgressMemory = 0;
    return result;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* StructLiteralExp::toConstElem(IRState* p)
{
    // type can legitimately be null for ClassReferenceExp::value.
    IF_LOG Logger::print("StructLiteralExp::toConstElem: %s @ %s\n",
        toChars(), type ? type->toChars() : "(null)");
    LOG_SCOPE;

    if (sinit)
    {
        // Copied from VarExp::toConstElem, need to clean this mess up.
        Type* sdecltype = sinit->type->toBasetype();
        Logger::print("Sym: type=%s\n", sdecltype->toChars());
        assert(sdecltype->ty == Tstruct);
        TypeStruct* ts = static_cast<TypeStruct*>(sdecltype);
        DtoResolveStruct(ts->sym);

        return ts->sym->ir.irAggr->getDefaultInit();
    }

    // make sure the struct is resolved
    DtoResolveStruct(sd);

    std::map<VarDeclaration*, llvm::Constant*> varInits;
    const size_t nexprs = elements->dim;
    for (size_t i = 0; i < nexprs; i++)
    {
        if ((*elements)[i])
        {
            varInits[sd->fields[i]] = (*elements)[i]->toConstElem(p);
        }
    }

    return sd->ir.irAggr->createInitializerConstant(varInits);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* ClassReferenceExp::toElem(IRState* p)
{
    IF_LOG Logger::print("ClassReferenceExp::toElem: %s @ %s\n",
        toChars(), type->toChars());
    LOG_SCOPE;

    return new DImValue(type, toConstElem(p));
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* ClassReferenceExp::toConstElem(IRState *p)
{
    IF_LOG Logger::print("ClassReferenceExp::toConstElem: %s @ %s\n",
        toChars(), type->toChars());
    LOG_SCOPE;

    ClassDeclaration* origClass = originalClass();
    DtoResolveClass(origClass);

    if (value->globalVar)
    {
        IF_LOG Logger::cout() << "Using existing global: " << *value->globalVar << '\n';
    }
    else
    {
        value->globalVar = new llvm::GlobalVariable(*p->module,
            origClass->type->irtype->isClass()->getMemoryLLType(),
            false, llvm::GlobalValue::InternalLinkage, 0, ".classref");

        std::map<VarDeclaration*, llvm::Constant*> varInits;

        // Unfortunately, ClassReferenceExp::getFieldAt is badly broken – it
        // places the base class fields _after_ those of the subclass.
        {
            const size_t nexprs = value->elements->dim;

            std::stack<ClassDeclaration*> classHierachy;
            ClassDeclaration* cur = origClass;
            while (cur)
            {
                classHierachy.push(cur);
                cur = cur->baseClass;
            }
            size_t i = 0;
            while (!classHierachy.empty())
            {
                cur = classHierachy.top();
                classHierachy.pop();
                for (size_t j = 0; j < cur->fields.dim; ++j)
                {
                    if ((*value->elements)[i])
                    {
                        VarDeclaration* field = cur->fields[j];
                        IF_LOG Logger::println("Getting initializer for: %s", field->toChars());
                        LOG_SCOPE;
                        varInits[field] = (*value->elements)[i]->toConstElem(p);
                    }
                    ++i;
                }
            }
            assert(i == nexprs);
        }

        llvm::Constant* constValue = origClass->ir.irAggr->createInitializerConstant(varInits);

        if (constValue->getType() != value->globalVar->getType()->getContainedType(0))
        {
            llvm::GlobalVariable* finalGlobalVar = new llvm::GlobalVariable(
                *p->module, constValue->getType(), false,
                llvm::GlobalValue::InternalLinkage, 0, ".classref");
            value->globalVar->replaceAllUsesWith(
                DtoBitCast(finalGlobalVar, value->globalVar->getType()));
            value->globalVar->eraseFromParent();
            value->globalVar = finalGlobalVar;
        }
        value->globalVar->setInitializer(constValue);
    }

    llvm::Constant* result = value->globalVar;

    if (type->ty == Tclass) {
        ClassDeclaration* targetClass = static_cast<TypeClass*>(type)->sym;
        if (InterfaceDeclaration* it = targetClass->isInterfaceDeclaration()) {
            assert(it->isBaseOf(origClass, NULL));

            IrTypeClass* typeclass = origClass->type->irtype->isClass();

            // find interface impl
            size_t i_index = typeclass->getInterfaceIndex(it);
            assert(i_index != ~0UL);

            // offset pointer
            result = DtoGEPi(result, 0, i_index);
        }
    }

    assert(type->ty == Tclass || type->ty == Tenum);
    return DtoBitCast(result, DtoType(type));
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* InExp::toElem(IRState* p)
{
    Logger::print("InExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* key = e1->toElem(p);
    DValue* aa = e2->toElem(p);

    return DtoAAIn(loc, type, aa, key);
}

DValue* RemoveExp::toElem(IRState* p)
{
    Logger::print("RemoveExp::toElem: %s\n", toChars());
    LOG_SCOPE;

    DValue* aa = e1->toElem(p);
    DValue* key = e2->toElem(p);

    return DtoAARemove(loc, aa, key);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* AssocArrayLiteralExp::toElem(IRState* p)
{
    Logger::print("AssocArrayLiteralExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(keys);
    assert(values);
    assert(keys->dim == values->dim);

    Type* basetype = type->toBasetype();
    Type* aatype = basetype;
    Type* vtype = aatype->nextOf();

    if (!keys->dim)
        goto LruntimeInit;

    if (aatype->ty != Taarray) {
        // It's the AssociativeArray type.
        // Turn it back into a TypeAArray
        vtype = values->tdata()[0]->type;
        aatype = new TypeAArray(vtype, keys->tdata()[0]->type);
        aatype = aatype->semantic(loc, NULL);
    }

    {
        std::vector<LLConstant*> keysInits, valuesInits;
        keysInits.reserve(keys->dim);
        valuesInits.reserve(keys->dim);
        for (size_t i = 0, n = keys->dim; i < n; ++i)
        {
            Expression* ekey = keys->tdata()[i];
            Expression* eval = values->tdata()[i];
            Logger::println("(%zu) aa[%s] = %s", i, ekey->toChars(), eval->toChars());
            unsigned errors = global.startGagging();
            LLConstant *ekeyConst = ekey->toConstElem(p);
            LLConstant *evalConst = eval->toConstElem(p);
            if (global.endGagging(errors))
                goto LruntimeInit;
            assert(ekeyConst && evalConst);
            keysInits.push_back(ekeyConst);
            valuesInits.push_back(evalConst);
        }

        assert(aatype->ty == Taarray);
        Type* indexType = static_cast<TypeAArray*>(aatype)->index;
        assert(indexType && vtype);

        llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_d_assocarrayliteralTX");
        LLFunctionType* funcTy = func->getFunctionType();
        LLValue* aaTypeInfo = DtoBitCast(DtoTypeInfoOf(stripModifiers(aatype)),
            DtoType(Type::typeinfoassociativearray->type));

        LLConstant* idxs[2] = { DtoConstUint(0), DtoConstUint(0) };

        LLArrayType* arrtype = LLArrayType::get(DtoType(indexType), keys->dim);
        LLConstant* initval = LLConstantArray::get(arrtype, keysInits);
        LLConstant* globalstore = new LLGlobalVariable(*gIR->module, arrtype, false, LLGlobalValue::InternalLinkage, initval, ".aaKeysStorage");
        LLConstant* slice = llvm::ConstantExpr::getGetElementPtr(globalstore, idxs, true);
        slice = DtoConstSlice(DtoConstSize_t(keys->dim), slice);
        LLValue* keysArray = DtoAggrPaint(slice, funcTy->getParamType(1));

        arrtype = LLArrayType::get(DtoType(vtype), values->dim);
        initval = LLConstantArray::get(arrtype, valuesInits);
        globalstore = new LLGlobalVariable(*gIR->module, arrtype, false, LLGlobalValue::InternalLinkage, initval, ".aaValuesStorage");
        slice = llvm::ConstantExpr::getGetElementPtr(globalstore, idxs, true);
        slice = DtoConstSlice(DtoConstSize_t(keys->dim), slice);
        LLValue* valuesArray = DtoAggrPaint(slice, funcTy->getParamType(2));

        LLValue* aa = gIR->CreateCallOrInvoke3(func, aaTypeInfo, keysArray, valuesArray, "aa").getInstruction();
        if (basetype->ty != Taarray) {
            LLValue *tmp = DtoAlloca(type, "aaliteral");
            DtoStore(aa, DtoGEPi(tmp, 0, 0));
            return new DVarValue(type, tmp);
        } else {
            return new DImValue(type, aa);
        }
    }

LruntimeInit:

    // it should be possible to avoid the temporary in some cases
    LLValue* tmp = DtoAlloca(type, "aaliteral");
    DValue* aa = new DVarValue(type, tmp);
    DtoStore(LLConstant::getNullValue(DtoType(type)), tmp);

    const size_t n = keys->dim;
    for (size_t i=0; i<n; ++i)
    {
        Expression* ekey = static_cast<Expression*>(keys->data[i]);
        Expression* eval = static_cast<Expression*>(values->data[i]);

        Logger::println("(%zu) aa[%s] = %s", i, ekey->toChars(), eval->toChars());

        // index
        DValue* key = ekey->toElem(p);
        DValue* mem = DtoAAIndex(loc, vtype, aa, key, true);

        // store
        DValue* val = eval->toElem(p);
        DtoAssign(loc, mem, val);
    }

    return aa;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* GEPExp::toElem(IRState* p)
{
    // (&a.foo).funcptr is a case where e1->toElem is genuinely not an l-value.
    LLValue* val = makeLValue(loc, e1->toElem(p));
    LLValue* v = DtoGEPi(val, 0, index);
    return new DVarValue(type, DtoBitCast(v, getPtrToType(DtoType(type))));
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* BoolExp::toElem(IRState* p)
{
    return new DImValue(type, DtoCast(loc, e1->toElem(p), Type::tbool)->getRVal());
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DotTypeExp::toElem(IRState* p)
{
    Type* t = sym->getType();
    assert(t);
    return e1->toElem(p);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* TypeExp::toElem(IRState *p)
{
    error("type %s is not an expression", toChars());
    //TODO: Improve error handling. DMD just returns some value here and hopes
    // some more sensible error messages will be triggered.
    fatal();
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* TupleExp::toElem(IRState *p)
{
    IF_LOG Logger::print("TupleExp::toElem() %s\n", toChars());
    LOG_SCOPE;

    // If there are any side effects, evaluate them first.
    if (e0) e0->toElem(p);

    std::vector<LLType*> types;
    types.reserve(exps->dim);
    for (size_t i = 0; i < exps->dim; i++)
    {
        Expression *el = static_cast<Expression *>(exps->data[i]);
        types.push_back(i1ToI8(voidToI8(DtoType(el->type))));
    }
    LLValue *val = DtoRawAlloca(LLStructType::get(gIR->context(), types),0, "tuple");
    for (size_t i = 0; i < exps->dim; i++)
    {
        Expression *el = static_cast<Expression *>(exps->data[i]);
        DValue* ep = el->toElem(p);
        LLValue *gep = DtoGEPi(val,0,i);
        if (el->type->ty == Tstruct)
            DtoStore(DtoLoad(ep->getRVal()), gep);
        else if (el->type->ty != Tvoid)
            DtoStoreZextI8(ep->getRVal(), gep);
        else
            DtoStore(LLConstantInt::get(LLType::getInt8Ty(gIR->context()), 0, false), gep);
    }
    return new DImValue(type, val);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* VectorExp::toElem(IRState* p)
{
    IF_LOG Logger::print("VectorExp::toElem() %s\n", toChars());
    LOG_SCOPE;

    TypeVector *type = static_cast<TypeVector*>(to->toBasetype());
    assert(type->ty == Tvector);

    LLValue *vector = DtoAlloca(to);

    // Array literals are assigned element-wise, other expressions are cast and
    // splat across the vector elements. This is what DMD does.
    if (e1->op == TOKarrayliteral) {
        Logger::println("array literal expression");
        ArrayLiteralExp *e = static_cast<ArrayLiteralExp*>(e1);
        assert(e->elements->dim == dim && "Array literal vector initializer "
            "length mismatch, should have been handled in frontend.");
        for (unsigned int i = 0; i < dim; ++i) {
            DValue *val = ((*e->elements)[i])->toElem(p);
            LLValue *llval = DtoCast(loc, val, type->elementType())->getRVal();
            DtoStore(llval, DtoGEPi(vector, 0, i));
        }
    } else {
        Logger::println("normal (splat) expression");
        DValue *val = e1->toElem(p);
        LLValue* llval = DtoCast(loc, val, type->elementType())->getRVal();
        for (unsigned int i = 0; i < dim; ++i) {
            DtoStore(llval, DtoGEPi(vector, 0, i));
        }
    }

    return new DVarValue(to, vector);
}

//////////////////////////////////////////////////////////////////////////////////////////

#define STUB(x) DValue *x::toElem(IRState * p) {error("Exp type "#x" not implemented: %s", toChars()); fatal(); return 0; }
STUB(Expression)
STUB(ScopeExp)
STUB(SymbolExp)
STUB(PowExp)
STUB(PowAssignExp)

llvm::Constant* Expression::toConstElem(IRState * p)
{
    error("expression '%s' is not a constant", toChars());
    if (!global.gag)
        fatal();
    return NULL;
}
