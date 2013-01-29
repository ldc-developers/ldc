//===-- toir.cpp ----------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <math.h>
#include <fstream>

#include "gen/llvm.h"
#include "llvm/Support/CommandLine.h"

#include "attrib.h"
#include "init.h"
#include "mtype.h"
#include "template.h"
#include "hdrgen.h"
#include "port.h"
#include "rmem.h"
#include "id.h"
#include "enum.h"

#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/runtime.h"
#include "gen/arrays.h"
#include "gen/structs.h"
#include "gen/classes.h"
#include "gen/typeinf.h"
#include "gen/complex.h"
#include "gen/dvalue.h"
#include "gen/aa.h"
#include "gen/functions.h"
#include "gen/todebug.h"
#include "gen/nested.h"
#include "gen/utils.h"
#include "gen/warnings.h"
#include "gen/optimizer.h"
#include "gen/pragma.h"

#include "llvm/Support/ManagedStatic.h"

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

DValue *Expression::toElemDtor(IRState *irs)
{
#if DMDV2
    Logger::println("Expression::toElemDtor(): %s", toChars());
    LOG_SCOPE

    size_t starti = irs->varsInScope().size();
    DValue *val = toElem(irs);
    size_t endi = irs->varsInScope().size();

    // Add destructors
    while (endi-- > starti)
    {
        VarDeclaration *vd = gIR->varsInScope().back();
        gIR->varsInScope().pop_back();
        vd->edtor->toElem(gIR);
    }
    return val;
#else
    return toElem(irs);
#endif
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

    if (VarDeclaration* vd = var->isVarDeclaration())
    {
        Logger::println("VarDeclaration ' %s ' of type ' %s '", vd->toChars(), vd->type->toChars());

#if DMDV2
        /* The magic variable __ctfe is always false at runtime
         */
        if (vd->ident == Id::ctfe) {
            return new DConstValue(type, DtoConstBool(false));
        }
#endif

        // this is an error! must be accessed with DotVarExp
        if (var->needThis())
        {
            error("need 'this' to access member %s", toChars());
            fatal();
        }

        // _arguments
        if (vd->ident == Id::_arguments && p->func()->_arguments)
        {
            Logger::println("Id::_arguments");
            LLValue* v = p->func()->_arguments;
            return new DVarValue(type, vd, v);
        }
        // _argptr
        else if (vd->ident == Id::_argptr && p->func()->_argptr)
        {
            Logger::println("Id::_argptr");
            LLValue* v = p->func()->_argptr;
            return new DVarValue(type, vd, v);
        }
        // _dollar
        else if (vd->ident == Id::dollar)
        {
            Logger::println("Id::dollar");
            LLValue* val = 0;
            if (vd->ir.isSet() && (val = vd->ir.getIrValue())) {
                // It must be length of a range
                return new DVarValue(type, vd, val);
            }
            assert(!p->arrays.empty());
            val = DtoArrayLen(p->arrays.back());
            return new DImValue(type, val);
        }
        // classinfo
        else if (ClassInfoDeclaration* cid = vd->isClassInfoDeclaration())
        {
            Logger::println("ClassInfoDeclaration: %s", cid->cd->toChars());
            cid->cd->codegen(Type::sir);;
            return new DVarValue(type, vd, cid->cd->ir.irStruct->getClassInfoSymbol());
        }
        // typeinfo
        else if (TypeInfoDeclaration* tid = vd->isTypeInfoDeclaration())
        {
            Logger::println("TypeInfoDeclaration");
            tid->codegen(Type::sir);
            assert(tid->ir.getIrValue());
            LLType* vartype = DtoType(type);
            LLValue* m = tid->ir.getIrValue();
            if (m->getType() != getPtrToType(vartype))
                m = p->ir->CreateBitCast(m, vartype, "tmp");
            return new DImValue(type, m);
        }
        // nested variable
    #if DMDV2
        else if (vd->nestedrefs.dim) {
    #else
        else if (vd->nestedref) {
    #endif
            Logger::println("nested variable");
            return DtoNestedVariable(loc, type, vd);
        }
        // function parameter
        else if (vd->isParameter()) {
            Logger::println("function param");
            Logger::println("type: %s", vd->type->toChars());
            FuncDeclaration* fd = vd->toParent2()->isFuncDeclaration();
            if (fd && fd != p->func()->decl) {
                Logger::println("nested parameter");
                return DtoNestedVariable(loc, type, vd);
            }
            else if (vd->storage_class & STClazy) {
                Logger::println("lazy parameter");
                assert(type->ty == Tdelegate);
                return new DVarValue(type, vd->ir.getIrValue());
            }
            else if (vd->isRef() || vd->isOut() || DtoIsPassedByRef(vd->type) || llvm::isa<llvm::AllocaInst>(vd->ir.getIrValue())) {
                return new DVarValue(type, vd, vd->ir.getIrValue());
            }
            else if (llvm::isa<llvm::Argument>(vd->ir.getIrValue())) {
                return new DImValue(type, vd->ir.getIrValue());
            }
            else assert(0);
        }
        else {
            Logger::println("a normal variable");

            // take care of forward references of global variables
            if (vd->isDataseg() || (vd->storage_class & STCextern))
                vd->codegen(Type::sir);

            LLValue* val;

            if (!vd->ir.isSet() || !(val = vd->ir.getIrValue())) {
                // FIXME: this error is bad!
                // We should be VERY careful about adding errors in general, as they have
                // a tendency to "mask" out the underlying problems ...
                error("variable %s not resolved", vd->toChars());
                if (Logger::enabled())
                    Logger::cout() << "unresolved variable had type: " << *DtoType(vd->type) << '\n';
                fatal();
            }

            if (vd->isDataseg() || (vd->storage_class & STCextern))
                val = DtoBitCast(val, DtoType(type->pointerTo()));

            return new DVarValue(type, vd, val);
        }
    }
    else if (FuncDeclaration* fdecl = var->isFuncDeclaration())
    {
        Logger::println("FuncDeclaration");
        LLValue* func = 0;
#if DMDV2
        fdecl = fdecl->toAliasFunc();
#endif
        if (fdecl->llvmInternal == LLVMinline_asm) {
            error("special ldc inline asm is not a normal function");
            fatal();
        }
        else if (fdecl->llvmInternal != LLVMva_arg) {
            fdecl->codegen(Type::sir);
            func = fdecl->ir.irFunc->func;
        }
        return new DFuncValue(fdecl, func);
    }
    else if (StaticStructInitDeclaration* sdecl = var->isStaticStructInitDeclaration())
    {
        // this seems to be the static initialiser for structs
        Type* sdecltype = sdecl->type->toBasetype();
        Logger::print("Sym: type=%s\n", sdecltype->toChars());
        assert(sdecltype->ty == Tstruct);
        TypeStruct* ts = static_cast<TypeStruct*>(sdecltype);
        assert(ts->sym);
        ts->sym->codegen(Type::sir);

        LLValue* initsym = ts->sym->ir.irStruct->getInitSymbol();
        initsym = DtoBitCast(initsym, DtoType(ts->pointerTo()));
        return new DVarValue(type, initsym);
    }
    else
    {
        assert(0 && "Unimplemented VarExp type");
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* VarExp::toConstElem(IRState* p)
{
    Logger::print("VarExp::toConstElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    if (StaticStructInitDeclaration* sdecl = var->isStaticStructInitDeclaration())
    {
        // this seems to be the static initialiser for structs
        Type* sdecltype = sdecl->type->toBasetype();
        Logger::print("Sym: type=%s\n", sdecltype->toChars());
        assert(sdecltype->ty == Tstruct);
        TypeStruct* ts = static_cast<TypeStruct*>(sdecltype);
        ts->sym->codegen(Type::sir);

        return ts->sym->ir.irStruct->getDefaultInit();
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
    assert(0);
    return NULL;
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

template <typename T>
static inline LLConstant* toConstantArray(LLType* ct, LLArrayType* at, T* str, size_t len, bool nullterm = true)
{
    std::vector<LLConstant*> vals;
    vals.reserve(len+1);
    for (size_t i = 0; i < len; ++i) {
        vals.push_back(LLConstantInt::get(ct, str[i], false));
    }
    if (nullterm)
        vals.push_back(LLConstantInt::get(ct, 0, false));
    return LLConstantArray::get(at, vals);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* StringExp::toElem(IRState* p)
{
    Logger::print("StringExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    Type* dtype = type->toBasetype();
    Type* cty = dtype->nextOf()->toBasetype();

    LLType* ct = DtoTypeNotVoid(cty);
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
        Logger::cout() << "type: " << *at << "\ninit: " << *_init << '\n';
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(*gIR->module,at,true,_linkage,_init,".str");

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

    assert(0);
    return 0;
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

    LLType* ct = DtoTypeNotVoid(cty);
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
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(*gIR->module,_init->getType(),true,_linkage,_init,".str");

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

    assert(0);
    return NULL;
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
                // of via getLValue(), which would perform a load from the
                // uninitialized location), and that rhs is stored as an l-value!

                IrLocal* const local = ve->var->ir.irLocal;
                assert(local && "ref var must be local and already initialized");

                DValue* rhs = e2->toElem(p);
                DtoStore(rhs->getLVal(), local->value);
                return rhs;
            }
        }
    }

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    if (e1->type->toBasetype()->ty == Tstruct && e2->op == TOKint64)
    {
        Logger::println("performing aggregate zero initialization");
        assert(e2->toInteger() == 0);
        DtoAggrZeroInit(l->getLVal());
#if DMDV2
        TypeStruct *ts = static_cast<TypeStruct*>(e1->type);
        if (ts->sym->isNested() && ts->sym->vthis)
            DtoResolveNestedContext(loc, ts->sym, l->getLVal());
#endif
        // Return value should be irrelevant.
        return r;
    }

    Logger::println("performing normal assignment");
    DtoAssign(loc, l, r, op);

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

LLConstant* AddExp::toConstElem(IRState* p)
{
    // add to pointer
    if (e1->type->ty == Tpointer && e2->type->isintegral()) {
        LLConstant *ptr = e1->toConstElem(p);
        LLConstant *index = e2->toConstElem(p);
        ptr = llvm::ConstantExpr::getGetElementPtr(ptr, llvm::makeArrayRef(&index, 1));
        return ptr;
    }

    error("expression '%s' is not a constant", toChars());
    fatal();
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* AddExp::toElem(IRState* p)
{
    Logger::print("AddExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    Type* t = type->toBasetype();
    Type* e1type = e1->type->toBasetype();
    Type* e2type = e2->type->toBasetype();

    errorOnIllegalArrayOp(this, e1, e2);

    if (e1type != e2type && e1type->ty == Tpointer) {
        Logger::println("add to pointer");
        if (DConstValue* cv = r->isConst()) {
            if (cv->c->isNullValue()) {
                Logger::println("is zero");
                return new DImValue(type, l->getRVal());
            }
        }
        LLValue* v = llvm::GetElementPtrInst::Create(l->getRVal(), r->getRVal(), "tmp", p->scopebb());
        return new DImValue(type, v);
    }
    else if (t->iscomplex()) {
        return DtoComplexAdd(loc, type, l, r);
    }
    else {
        return DtoBinAdd(l,r);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* MinExp::toConstElem(IRState* p)
{
    if (e1->type->ty == Tpointer && e2->type->isintegral()) {
        LLConstant *ptr = e1->toConstElem(p);
        LLConstant *index = e2->toConstElem(p);
        index = llvm::ConstantExpr::getNeg(index);
        ptr = llvm::ConstantExpr::getGetElementPtr(ptr, llvm::makeArrayRef(&index, 1));
        return ptr;
    }

    error("expression '%s' is not a constant", toChars());
    fatal();
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* MinExp::toElem(IRState* p)
{
    Logger::print("MinExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    Type* t = type->toBasetype();
    Type* t1 = e1->type->toBasetype();
    Type* t2 = e2->type->toBasetype();

    errorOnIllegalArrayOp(this, e1, e2);

    if (t1->ty == Tpointer && t2->ty == Tpointer) {
        LLValue* lv = l->getRVal();
        LLValue* rv = r->getRVal();
        if (Logger::enabled())
            Logger::cout() << "lv: " << *lv << " rv: " << *rv << '\n';
        lv = p->ir->CreatePtrToInt(lv, DtoSize_t(), "tmp");
        rv = p->ir->CreatePtrToInt(rv, DtoSize_t(), "tmp");
        LLValue* diff = p->ir->CreateSub(lv,rv,"tmp");
        if (diff->getType() != DtoType(type))
            diff = p->ir->CreateIntToPtr(diff, DtoType(type), "tmp");
        return new DImValue(type, diff);
    }
    else if (t1->ty == Tpointer) {
        LLValue* idx = p->ir->CreateNeg(r->getRVal(), "tmp");
        LLValue* v = llvm::GetElementPtrInst::Create(l->getRVal(), idx, "tmp", p->scopebb());
        return new DImValue(type, v);
    }
    else if (t->iscomplex()) {
        return DtoComplexSub(loc, type, l, r);
    }
    else {
        return DtoBinSub(l,r);
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
#if DMDV2
            if (LLValue *argptr = gIR->func()->_argptr) {
                DtoStore(DtoLoad(argptr), DtoBitCast(arg, getPtrToType(getVoidPtrType())));
                return new DImValue(type, arg);
            } else if (global.params.cpu == ARCHx86_64) {
                LLValue *va_list = DtoAlloca(exp->type->nextOf());
                DtoStore(va_list, arg);
                va_list = DtoBitCast(va_list, getVoidPtrType());
                return new DImValue(type, gIR->ir->CreateCall(GET_INTRINSIC_DECL(vastart), va_list, ""));
            } else
#endif
            {
                arg = DtoBitCast(arg, getVoidPtrType());
                return new DImValue(type, gIR->ir->CreateCall(GET_INTRINSIC_DECL(vastart), arg, ""));
            }
        }
#if DMDV2
        else if (fndecl->llvmInternal == LLVMva_copy && global.params.cpu == ARCHx86_64) {
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
#endif
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

            // auto q = cast(ubyte*)ptr + (bitnum >> 3);
            LLValue* q = DtoBitCast(ptr, DtoType(Type::tuns8->pointerTo()));
            q = DtoGEP1(q, p->ir->CreateLShr(bitnum, 3), "bitop.q");

            // auto mask = 1 << (bitnum & 7);
            LLValue* mask = p->ir->CreateAnd(bitnum, DtoConstSize_t(7), "bitop.tmp");
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
                    assert(false);
                }

                LLValue *newVal = p->ir->CreateBinOp(op, val, mask, "bitop.new_val");
                newVal = p->ir->CreateTrunc(newVal, DtoType(Type::tuns8), "bitop.tmp");
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
        vd->codegen(Type::sir);
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
    else {
        goto Lerr;
    }

    return res;

Lerr:
    error("can not cast %s to %s at compile time", e1->type->toChars(), type->toChars());
    if (!global.gag)
        fatal();
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* SymOffExp::toElem(IRState* p)
{
    Logger::print("SymOffExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(0 && "SymOffExp::toElem should no longer be called :/");
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* AddrExp::toElem(IRState* p)
{
    Logger::println("AddrExp::toElem: %s @ %s", toChars(), type->toChars());
    LOG_SCOPE;
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
        fd->codegen(Type::sir);
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
    // FIXME: this should probably be generalized more so we don't
    // need to have a case for each thing we can take the address of

    // address of global variable
    if (e1->op == TOKvar)
    {
        VarExp* vexp = static_cast<VarExp*>(e1);

        // make sure 'this' isn't needed
        if (vexp->var->needThis())
        {
            error("need 'this' to access %s", vexp->var->toChars());
            fatal();
        }

        // global variable
        if (VarDeclaration* vd = vexp->var->isVarDeclaration())
        {
            if (!vd->isDataseg())
            {
                // Not sure if this can be triggered from user code, but it is
                // needed for the current hacky implementation of
                // AssocArrayLiteralExp::toElem, which requires on error
                // gagging to check for constantness of the initializer.
                error("cannot use address of non-global variable '%s' "
                    "as constant initializer", vd->toChars());
                if (!global.gag) fatal();
                return NULL;
            }

            vd->codegen(Type::sir);
            LLConstant* llc = llvm::dyn_cast<LLConstant>(vd->ir.getIrValue());
            assert(llc);
            return DtoBitCast(llc, DtoType(type));
        }
        // static function
        else if (FuncDeclaration* fd = vexp->var->isFuncDeclaration())
        {
            fd->codegen(Type::sir);
            IrFunction* irfunc = fd->ir.irFunc;
            return irfunc->func;
        }
        // something else
        else
        {
            // fail
            goto Lerr;
        }
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
        vd->codegen(Type::sir);
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
    else if (
        e1->op == TOKstructliteral ||
        e1->op == TOKslice)
    {
        error("non-constant expression '%s'", toChars());
        fatal();
    }
    // not yet supported
    else
    {
    Lerr:
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
        return new DImValue(type, e1->toElem(p)->getRVal());
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
            assert(0);

        //Logger::cout() << "mem: " << *arrptr << '\n';
        return new DVarValue(type, vd, arrptr);
    }
    else if (FuncDeclaration* fdecl = var->isFuncDeclaration())
    {
        DtoResolveDsymbol(fdecl);

        // This is a bit more convoluted than it would need to be, because it
        // has to take templated interface methods into account, for which
        // isFinal is not necessarily true.
        const bool nonFinal = !fdecl->isFinal() &&
            (fdecl->isAbstract() || fdecl->isVirtual());

        // If we are calling a non-final interface function, we need to get
        // the pointer to the underlying object instead of passing the
        // interface pointer directly.
        LLValue* passedThis = 0;
        if (e1type->ty == Tclass)
        {
            TypeClass* tc = static_cast<TypeClass*>(e1type);
            if (tc->sym->isInterfaceDeclaration() && nonFinal)
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
            fdecl->codegen(Type::sir);
            funcval = fdecl->ir.irFunc->func;
        }
        assert(funcval);

        return new DFuncValue(fdecl, funcval, passedThis);
    }
    else {
        printf("unsupported dotvarexp: %s\n", var->toChars());
    }

    assert(0);
    return 0;
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
#if DMDV2
        // In D1, contracts are treated as normal nested methods, 'this' is
        // just passed in the context struct along with any used parameters.
        if (ident == Id::ensure || ident == Id::require) {
            Logger::println("contract this exp");
            v = p->func()->nestArg;
            v = DtoBitCast(v, DtoType(type)->getPointerTo());
        } else
#endif
        if (vdparent != p->func()->decl) {
            Logger::println("nested this exp");
#if STRUCTTHISREF
            return DtoNestedVariable(loc, type, vd, type->ty == Tstruct);
#else
            return DtoNestedVariable(loc, type, vd);
#endif
        }
        else {
            Logger::println("normal this exp");
            v = p->func()->thisArg;
        }
        return new DVarValue(type, vd, v);
    }

    // anything we're not yet handling ?
    assert(0 && "no var in ThisExp");
    return 0;
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
        if(global.params.useArrayBounds)
            DtoArrayBoundsCheck(loc, l, r);
        arrptr = DtoGEP(l->getRVal(), zero, r->getRVal());
    }
    else if (e1type->ty == Tarray) {
        if(global.params.useArrayBounds)
            DtoArrayBoundsCheck(loc, l, r);
        arrptr = DtoArrayPtr(l);
        arrptr = DtoGEP1(arrptr,r->getRVal());
    }
    else if (e1type->ty == Taarray) {
        return DtoAAIndex(loc, type, l, r, modifiable);
    }
    else {
        Logger::println("invalid index exp! e1type: %s", e1type->toChars());
        assert(0);
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
    LLValue* elen;
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

#if DMDV2
        if(global.params.useArrayBounds)
#else
        if(global.params.useArrayBounds && (etype->ty == Tsarray || etype->ty == Tarray))
#endif
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
        // for normal code the actual array length is what we want!
        else
        {
            elen = DtoArrayLen(e);
        }
    }

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
            assert(0);
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
        assert(0 && "Unsupported CmpExp type");
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
            assert(0);
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
        assert(0 && "Unsupported EqualExp type");
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
            std::vector<DValue*> dims(ndims);
            for (size_t i=0; i<ndims; ++i)
                dims[i] = static_cast<Expression*>(arguments->data[i])->toElem(p);
            return DtoNewMulDimDynArray(loc, newtype, &dims[0], ndims, true);
        }
    }
    // new static array
    else if (ntype->ty == Tsarray)
    {
        assert(0);
    }
    // new struct
    else if (ntype->ty == Tstruct)
    {
        Logger::println("new struct on heap: %s\n", newtype->toChars());
        // allocate
        LLValue* mem = 0;
#if DMDV2
        if (allocator)
        {
            // custom allocator
            allocator->codegen(Type::sir);
            DFuncValue dfn(allocator, allocator->ir.irFunc->func);
            DValue* res = DtoCallFunction(loc, NULL, &dfn, newargs);
            mem = DtoBitCast(res->getRVal(), DtoType(ntype->pointerTo()), ".newstruct_custom");
        } else
#endif
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
            ts->sym->codegen(Type::sir);
            DtoAggrCopy(mem, ts->sym->ir.irStruct->getInitSymbol());
        }
#if DMDV2
        if (ts->sym->isNested() && ts->sym->vthis)
            DtoResolveNestedContext(loc, ts->sym, mem);

        // call constructor
        if (member)
        {
            Logger::println("Calling constructor");
            assert(arguments != NULL);
            member->codegen(Type::sir);
            DFuncValue dfn(member, member->ir.irFunc->func, mem);
            DtoCallFunction(loc, ts, &dfn, arguments);
        }
#endif
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

    assert(0);
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
#if DMDV2
        DtoDeleteMemory(dval->isLVal() ? dval->getLVal() : makeLValue(loc, dval));
#else
        LLValue* rval = dval->getRVal();
        DtoDeleteMemory(rval);
        if (dval->isVar())
            DtoStore(LLConstant::getNullValue(rval->getType()), dval->getLVal());
#endif
    }
    // class
    else if (et->ty == Tclass)
    {
        bool onstack = false;
        TypeClass* tc = static_cast<TypeClass*>(et);
        if (tc->sym->isInterfaceDeclaration())
        {
#if DMDV2
            LLValue *val = dval->getLVal();
#else
            LLValue *val = dval->getRVal();
#endif
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
        assert(0 && "invalid delete");
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
    DtoAssert(p->func()->decl->getModule(), loc, msg ? msg->toElem(p) : NULL);

    // rewrite the scope
    p->scope() = IRScope(endbb,oldend);


    InvariantDeclaration* invdecl;
    // class invariants
    if(
        global.params.useInvariants &&
        condty->ty == Tclass &&
        !(static_cast<TypeClass*>(condty)->sym->isInterfaceDeclaration()))
    {
        Logger::println("calling class invariant");
        llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_invariant");
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
        static_cast<TypeStruct*>(condty->nextOf())->sym->codegen(Type::sir);
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

BinBitExp(And,And);
BinBitExp(Or,Or);
BinBitExp(Xor,Xor);
BinBitExp(Shl,Shl);
BinBitExp(Ushr,LShr);

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

    // FIXME: DMD inserts a trap here... we probably should as well !?!

#if 1
    DtoAssert(p->func()->decl->getModule(), loc, NULL);
#else
    // call the new (?) trap intrinsic
    p->ir->CreateCall(GET_INTRINSIC_DECL(trap),"");
    new llvm::UnreachableInst(p->scopebb());
#endif

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

    if (e1->op != TOKsuper && e1->op != TOKdottype && func->isVirtual() && !func->isFinal())
        castfptr = DtoVirtualFunctionPointer(u, func, toChars());
    else if (func->isAbstract())
        assert(0 && "TODO delegate to abstract method");
    else if (func->toParent()->isInterfaceDeclaration())
        assert(0 && "TODO delegate to interface method");
    else
    {
        func->codegen(Type::sir);
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
    assert(e2->type == type);
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
        dvv = new DConstValue(type, getNullValue(DtoTypeNotVoid(dtype)));
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

#if DMDV2
    return DtoCatArrays(type, e1, e2);
#else

    bool arrNarr = e1->type->toBasetype() == e2->type->toBasetype();
    // array ~ array
    if (arrNarr)
    {
        return DtoCatArrays(type, e1, e2);
    }
    // array ~ element
    // element ~ array
    else
    {
        return DtoCatArrayElement(type, e1, e2);
    }
#endif
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

    if (fd->isNested()) Logger::println("nested");
    Logger::println("kind = %s", fd->kind());

    fd->codegen(Type::sir);
    assert(fd->ir.irFunc->func);

    if(fd->isNested() && !(fd->tok == TOKreserved && type->ty == Tpointer && fd->vthis)) {
        LLType* dgty = DtoType(type);

        LLValue* cval;
        IrFunction* irfn = p->func();
        if (irfn->nestedVar
#if DMDV2
            // We cannot use a frame allocated in one function
            // for a delegate created in another function
            // (that happens with anonymous functions)
            && fd->toParent2() == irfn->decl
#endif
            )
            cval = irfn->nestedVar;
        else if (irfn->nestArg)
            cval = DtoLoad(irfn->nestArg);
#if DMDV2
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
#endif
        else
            cval = getNullPtr(getVoidPtrType());
        cval = DtoBitCast(cval, dgty->getContainedType(0));

        LLValue* castfptr = DtoBitCast(fd->ir.irFunc->func, dgty->getContainedType(1));

        return new DImValue(type, DtoAggrPair(cval, castfptr, ".func"));

    } else {
        return new DImValue(type, fd->ir.irFunc->func);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* FuncExp::toConstElem(IRState* p)
{
    Logger::print("FuncExp::toConstElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(fd);
    if (fd->tok != TOKfunction && !(fd->tok == TOKreserved && type->ty == Tpointer && fd->vthis))
    {
        assert(fd->tok == TOKdelegate || fd->tok == TOKreserved);
        error("delegate literals as constant expressions are not yet allowed");
        return 0;
    }

    fd->codegen(Type::sir);
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
    bool dyn = (arrayType->ty == Tarray);
    // length
    size_t len = elements->dim;

    // llvm target type
    LLType* llType = DtoType(arrayType);
    if (Logger::enabled())
        Logger::cout() << (dyn?"dynamic":"static") << " array literal with length " << len << " of D type: '" << arrayType->toChars() << "' has llvm type: '" << *llType << "'\n";

    // llvm storage type
    LLType* llElemType = DtoTypeNotVoid(elemType);
    LLType* llStoType = LLArrayType::get(llElemType, len);
    if (Logger::enabled())
        Logger::cout() << "llvm storage type: '" << *llStoType << "'\n";

    // don't allocate storage for zero length dynamic array literals
    if (dyn && len == 0)
    {
        // dmd seems to just make them null...
        return new DSliceValue(type, DtoConstSize_t(0), getNullPtr(getPtrToType(llElemType)));
    }

    // dst pointer
    LLValue* dstMem;
    DSliceValue* dynSlice = NULL;
    if(dyn)
    {
        dynSlice = DtoNewDynArray(loc, arrayType, new DConstValue(Type::tsize_t, DtoConstSize_t(len)), false);
        dstMem = dynSlice->ptr;
    }
    else
        dstMem = DtoRawAlloca(llStoType, 0, "arrayliteral");

    // store elements
    for (size_t i=0; i<len; ++i)
    {
        Expression* expr = static_cast<Expression*>(elements->data[i]);
        LLValue* elemAddr;
        if(dyn)
            elemAddr = DtoGEPi1(dstMem, i, "tmp", p->scopebb());
        else
            elemAddr = DtoGEPi(dstMem,0,i,"tmp",p->scopebb());

        // emulate assignment
        DVarValue* vv = new DVarValue(expr->type, elemAddr);
        DValue* e = expr->toElem(p);
        DtoAssign(loc, vv, e);
    }

    // return storage directly ?
    if (!dyn)
        return new DImValue(type, dstMem);

    // return slice
    return dynSlice;
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
    LLArrayType* arrtype = LLArrayType::get(DtoTypeNotVoid(elemt), elements->dim);

    // dynamic arrays can occur here as well ...
    bool dyn = (bt->ty != Tsarray);

    // build the initializer
    std::vector<LLConstant*> vals(elements->dim, NULL);
    for (unsigned i=0; i<elements->dim; ++i)
    {
        Expression* expr = static_cast<Expression*>(elements->data[i]);
        vals[i] = expr->toConstElem(p);
    }

    // build the constant array initialize
    LLArrayType *t = elements->dim == 0 ?
                           arrtype :
                           LLArrayType::get(vals.front()->getType(), elements->dim);
    LLConstant* initval = LLConstantArray::get(t, vals);

    // if static array, we're done
    if (!dyn)
        return initval;

    // we need to put the initializer in a global, and so we have a pointer to the array
    // Important: don't make the global constant, since this const initializer might
    // be used as an initializer for a static T[] - where modifying contents is allowed.
    LLConstant* globalstore = new LLGlobalVariable(*gIR->module, t, false, LLGlobalValue::InternalLinkage, initval, ".dynarrayStorage");
    globalstore = DtoBitCast(globalstore, getPtrToType(arrtype));

#if DMDV2
    if (bt->ty == Tpointer)
        // we need to return pointer to the static array.
        return globalstore;
#endif

    // build a constant dynamic array reference with the .ptr field pointing into globalstore
    LLConstant* idxs[2] = { DtoConstUint(0), DtoConstUint(0) };
    LLConstant* globalstorePtr = llvm::ConstantExpr::getGetElementPtr(globalstore, idxs, true);

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
    Logger::print("StructLiteralExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    // make sure the struct is fully resolved
    sd->codegen(Type::sir);

    // alloca a stack slot
    LLValue* mem = DtoRawAlloca(DtoType(type), 0, ".structliteral");

    // ready elements data
    assert(elements && "struct literal has null elements");
    size_t nexprs = elements->dim;
    Expression** exprs = (Expression**)elements->data;

    // might be reset to an actual i8* value so only a single bitcast is emitted.
    LLValue* voidptr = mem;
    unsigned offset = 0;

    // go through fields
    ArrayIter<VarDeclaration> it(sd->fields);
    for (; !it.done(); it.next())
    {
        VarDeclaration* vd = it.get();

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
        Expression* expr = (it.index < nexprs) ? exprs[it.index] : NULL;
        IF_LOG Logger::println("expr: %p", expr);
        DValue* val;
        DConstValue cv(vd->type, NULL); // Only used in one branch; value is set beforehand
        if (expr)
        {
            IF_LOG Logger::println("expr %zu = %s", it.index, expr->toChars());
            val = expr->toElem(gIR);
        }
#if DMDV2
        else if (vd == sd->vthis) {
            IF_LOG Logger::println("initializing vthis");
            LOG_SCOPE
            val = new DImValue(vd->type, DtoBitCast(DtoNestedContext(loc, sd), DtoType(vd->type)));
        }
#endif
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
        DVarValue field(vd->type, vd, DtoIndexStruct(mem, sd, vd));

        // store the initializer there
        DtoAssign(loc, &field, val, TOKconstruct);

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

#if DMDV2
        Type *tb = vd->type->toBasetype();
        if (tb->ty == Tstruct)
        {
            // Call postBlit()
            StructDeclaration *sd = static_cast<TypeStruct *>(tb)->sym;
            if (sd->postblit)
            {
                FuncDeclaration *fd = sd->postblit;
                fd->codegen(Type::sir);
                Expressions args;
                DFuncValue dfn(fd, fd->ir.irFunc->func, val->getLVal());
                DtoCallFunction(loc, Type::basic[Tvoid], &dfn, &args);
            }
        }
#endif

    }
    // initialize trailing padding
    if (sd->structsize != offset)
        voidptr = write_zeroes(voidptr, offset, sd->structsize);

    // return as a var
    return new DVarValue(type, mem);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* StructLiteralExp::toConstElem(IRState* p)
{
    Logger::print("StructLiteralExp::toConstElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    // make sure the struct is resolved
    sd->codegen(Type::sir);

    // get inits
    std::vector<LLConstant*> inits(sd->fields.dim, NULL);

    size_t nexprs = elements->dim;;
    Expression** exprs = (Expression**)elements->data;

    for (size_t i = 0; i < nexprs; i++)
        if (exprs[i])
            inits[i] = exprs[i]->toConstElem(p);

    // vector of values to build aggregate from
    std::vector<LLConstant*> values = DtoStructLiteralValues(sd, inits, true);

    // we know those values are constants.. cast them
    std::vector<LLConstant*> constvals(values.size(), NULL);
    std::vector<LLType*> types(values.size(), NULL);
    for (size_t i = 0; i < values.size(); ++i) {
        constvals[i] = llvm::cast<LLConstant>(values[i]);
        types[i] = values[i]->getType();
    }

    // return constant struct
    if (!constType)
    {
        if (type->ty == Ttypedef) // hack, see DtoConstInitializer.
            constType = LLStructType::get(gIR->context(), types);
        else
            constType = isaStruct(DtoType(type));
    }
    else if (constType->isOpaque())
        constType->setBody(types);
    return LLConstantStruct::get(constType, llvm::makeArrayRef(constvals));
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

#if DMDV2
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
        LLValue* aaTypeInfo = DtoTypeInfoOf(stripModifiers(aatype));

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
#endif

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
    Logger::print("TupleExp::toElem() %s\n", toChars());
    std::vector<LLType*> types(exps->dim, NULL);
    for (size_t i = 0; i < exps->dim; i++)
    {
        Expression *el = static_cast<Expression *>(exps->data[i]);
        types[i] = DtoTypeNotVoid(el->type);
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
            DtoStore(ep->getRVal(), gep);
        else
            DtoStore(LLConstantInt::get(LLType::getInt8Ty(gIR->context()), 0, false), gep);
    }
    return new DImValue(type, val);
}

//////////////////////////////////////////////////////////////////////////////////////////

#if DMDV2

DValue* VectorExp::toElem(IRState* p)
{
    Logger::print("VectorExp::toElem() %s\n", toChars());
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

#endif


//////////////////////////////////////////////////////////////////////////////////////////

#define STUB(x) DValue *x::toElem(IRState * p) {error("Exp type "#x" not implemented: %s", toChars()); fatal(); return 0; }
STUB(Expression);
STUB(ScopeExp);

#if DMDV2
STUB(SymbolExp);
STUB(PowExp);
STUB(PowAssignExp);
#endif

#define CONSTSTUB(x) LLConstant* x::toConstElem(IRState * p) { \
    error("expression '%s' is not a constant", toChars()); \
    if (!global.gag) \
        fatal(); \
    return NULL; \
}
CONSTSTUB(Expression);
CONSTSTUB(GEPExp);
CONSTSTUB(SliceExp);
CONSTSTUB(IndexExp);
CONSTSTUB(AssocArrayLiteralExp);

//////////////////////////////////////////////////////////////////////////////////////////

void obj_includelib(const char* lib)
{
    size_t n = strlen(lib)+3;
    char *arg = static_cast<char *>(mem.malloc(n));
    strcpy(arg, "-l");
    strncat(arg, lib, n);
    global.params.linkswitches->push(arg);
}

void backend_init()
{
    // LLVM_D_InitRuntime is done in Module::genLLVMModule
    // since it requires the semantic pass to be done
}

void backend_term()
{
    LLVM_D_FreeRuntime();
    llvm::llvm_shutdown();
}
