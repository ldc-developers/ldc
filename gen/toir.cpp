// Backend stubs

/* DMDFE backend stubs
 * This file contains the implementations of the backend routines.
 * For dmdfe these do nothing but print a message saying the module
 * has been parsed. Substitute your own behaviors for these routimes.
 */

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
        cachedLvalue = NULL;
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
            assert(!p->arrays.empty());
            LLValue* tmp = DtoArrayLen(p->arrays.back());
            return new DImValue(type, tmp);
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
            const LLType* vartype = DtoType(type);
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
            if (vd->isDataseg() || (vd->storage_class & STCextern)) {
                vd->codegen(Type::sir);
            }

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

            if (vd->isDataseg() || (vd->storage_class & STCextern)) {
                DtoConstInitGlobal(vd);
                val = DtoBitCast(val, DtoType(type->pointerTo()));
            }

            return new DVarValue(type, vd, val);
        }
    }
    else if (FuncDeclaration* fdecl = var->isFuncDeclaration())
    {
        Logger::println("FuncDeclaration");
        LLValue* func = 0;
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
        TypeStruct* ts = (TypeStruct*)sdecltype;
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
        TypeStruct* ts = (TypeStruct*)sdecltype;
        ts->sym->codegen(Type::sir);

        return ts->sym->ir.irStruct->getDefaultInit();
    }

    if (TypeInfoDeclaration* ti = var->isTypeInfoDeclaration())
    {
        const LLType* vartype = DtoType(type);
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
    const LLType* t = DtoType(type);
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
    const LLType* t = DtoType(type);
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
        Type* t = type->toBasetype();
        if (t->ty == Tcomplex32)
            c = DtoConstFP(Type::tfloat32, 0);
        else if (t->ty == Tcomplex64)
            c = DtoConstFP(Type::tfloat64, 0);
        else if (t->ty == Tcomplex80)
            c = DtoConstFP(Type::tfloat80, 0);
        else
            assert(0);
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

    const LLType* ct = DtoTypeNotVoid(cty);
    //printf("ct = %s\n", type->nextOf()->toChars());
    const LLArrayType* at = LLArrayType::get(ct,len+1);

    LLConstant* _init;
    if (cty->size() == 1) {
        uint8_t* str = (uint8_t*)string;
        llvm::StringRef cont((char*)str, len);
        _init = LLConstantArray::get(p->context(), cont, true);
    }
    else if (cty->size() == 2) {
        uint16_t* str = (uint16_t*)string;
        std::vector<LLConstant*> vals;
        vals.reserve(len+1);
        for(size_t i=0; i<len; ++i) {
            vals.push_back(LLConstantInt::get(ct, str[i], false));;
        }
        vals.push_back(LLConstantInt::get(ct, 0, false));
        _init = LLConstantArray::get(at,vals);
    }
    else if (cty->size() == 4) {
        uint32_t* str = (uint32_t*)string;
        std::vector<LLConstant*> vals;
        vals.reserve(len+1);
        for(size_t i=0; i<len; ++i) {
            vals.push_back(LLConstantInt::get(ct, str[i], false));;
        }
        vals.push_back(LLConstantInt::get(ct, 0, false));
        _init = LLConstantArray::get(at,vals);
    }
    else
    assert(0);

    llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::InternalLinkage;
    if (Logger::enabled())
        Logger::cout() << "type: " << *at << "\ninit: " << *_init << '\n';
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(*gIR->module,at,true,_linkage,_init,".str");

    llvm::ConstantInt* zero = LLConstantInt::get(LLType::getInt32Ty(gIR->context()), 0, false);
    LLConstant* idxs[2] = { zero, zero };
    LLConstant* arrptr = llvm::ConstantExpr::getGetElementPtr(gvar,idxs,2);

    if (dtype->ty == Tarray) {
        LLConstant* clen = LLConstantInt::get(DtoSize_t(),len,false);
        return new DImValue(type, DtoConstSlice(clen, arrptr));
    }
    else if (dtype->ty == Tsarray) {
        const LLType* dstType = getPtrToType(LLArrayType::get(ct, len));
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

    const LLType* ct = DtoTypeNotVoid(cty);
    const LLArrayType* at = LLArrayType::get(ct,endlen);

    LLConstant* _init;
    if (cty->size() == 1) {
        uint8_t* str = (uint8_t*)string;
        llvm::StringRef cont((char*)str, len);
        _init = LLConstantArray::get(p->context(), cont, nullterm);
    }
    else if (cty->size() == 2) {
        uint16_t* str = (uint16_t*)string;
        std::vector<LLConstant*> vals;
        vals.reserve(len+1);
        for(size_t i=0; i<len; ++i) {
            vals.push_back(LLConstantInt::get(ct, str[i], false));;
        }
        if (nullterm)
            vals.push_back(LLConstantInt::get(ct, 0, false));
        _init = LLConstantArray::get(at,vals);
    }
    else if (cty->size() == 4) {
        uint32_t* str = (uint32_t*)string;
        std::vector<LLConstant*> vals;
        vals.reserve(len+1);
        for(size_t i=0; i<len; ++i) {
            vals.push_back(LLConstantInt::get(ct, str[i], false));;
        }
        if (nullterm)
            vals.push_back(LLConstantInt::get(ct, 0, false));
        _init = LLConstantArray::get(at,vals);
    }
    else
    assert(0);

    if (t->ty == Tsarray)
    {
        return _init;
    }

    llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::InternalLinkage;
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(*gIR->module,_init->getType(),true,_linkage,_init,".str");

    llvm::ConstantInt* zero = LLConstantInt::get(LLType::getInt32Ty(gIR->context()), 0, false);
    LLConstant* idxs[2] = { zero, zero };
    LLConstant* arrptr = llvm::ConstantExpr::getGetElementPtr(gvar,idxs,2);

    if (t->ty == Tpointer) {
        return arrptr;
    }
    else if (t->ty == Tarray) {
        LLConstant* clen = LLConstantInt::get(DtoSize_t(),len,false);
        return DtoConstSlice(clen, arrptr);
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
        ArrayLengthExp *ale = (ArrayLengthExp *)e1;
        DValue* arr = ale->e1->toElem(p);
        DVarValue arrval(ale->e1->type, arr->getLVal());
        DValue* newlen = e2->toElem(p);
        DSliceValue* slice = DtoResizeDynArray(arrval.getType(), &arrval, newlen->getRVal());
        DtoAssign(loc, &arrval, slice);
        return newlen;
    }

    Logger::println("performing normal assignment");

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);
    DtoAssign(loc, l, r);

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
        e = ((CastExp*)e)->e1;

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

DValue* AddExp::toElem(IRState* p)
{
    Logger::print("AddExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    Type* t = type->toBasetype();
    Type* e1type = e1->type->toBasetype();
    Type* e1next = e1type->nextOf() ? e1type->nextOf()->toBasetype() : NULL;
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

    return DtoBinRem(type, l, r);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CallExp::toElem(IRState* p)
{
    Logger::print("CallExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    // handle magic inline asm
    if (e1->op == TOKvar)
    {
        VarExp* ve = (VarExp*)e1;
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
    bool va_intrinsic = false;
    if (dfnval && dfnval->func)
    {
        FuncDeclaration* fndecl = dfnval->func;

        // as requested by bearophile, see if it's a C printf call and that it's valid.
        if (global.params.warnings && checkPrintf)
        {
            if (fndecl->linkage == LINKc && strcmp(fndecl->ident->string, "printf") == 0)
            {
                warnInvalidPrintfCall(loc, (Expression*)arguments->data[0], arguments->dim);
            }
        }

        // va_start instruction
        if (fndecl->llvmInternal == LLVMva_start) {
            // llvm doesn't need the second param hence the override
            Expression* exp = (Expression*)arguments->data[0];
            DValue* expv = exp->toElem(p);
            LLValue* arg = DtoBitCast(expv->getLVal(), getVoidPtrType());
            return new DImValue(type, gIR->ir->CreateCall(GET_INTRINSIC_DECL(vastart), arg, ""));
        }
        // va_arg instruction
        else if (fndecl->llvmInternal == LLVMva_arg) {
            return DtoVaArg(loc, type, (Expression*)arguments->data[0]);
        }
        // C alloca
        else if (fndecl->llvmInternal == LLVMalloca) {
            Expression* exp = (Expression*)arguments->data[0];
            DValue* expv = exp->toElem(p);
            if (expv->getType()->toBasetype()->ty != Tint32)
                expv = DtoCast(loc, expv, Type::tint32);
            return new DImValue(type, p->ir->CreateAlloca(LLType::getInt8Ty(gIR->context()), expv->getRVal(), ".alloca"));
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
    const LLType* lltype = DtoType(type);
    Type* tb = to->toBasetype();

    // string literal to dyn array:
    // reinterpret the string data as an array, calculate the length
    if (e1->op == TOKstring && tb->ty == Tarray) {
/*        StringExp *strexp = (StringExp*)e1;
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
    else {
        error("can not cast %s to %s at compile time", e1->type->toChars(), type->toChars());
        return e1->toConstElem(p);
    }

    return res;
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
    else if (DImValue* im = v->isIm()) {
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
        VarExp* vexp = (VarExp*)e1;

        // make sure 'this' isn't needed
        if (vexp->var->needThis())
        {
            error("need 'this' to access %s", vexp->var->toChars());
            fatal();
        }

        // global variable
        if (VarDeclaration* vd = vexp->var->isVarDeclaration())
        {
            vd->codegen(Type::sir);
            LLConstant* llc = llvm::dyn_cast<LLConstant>(vd->ir.getIrValue());
            assert(llc);
            return llc;
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
        IndexExp* iexp = (IndexExp*)e1;

        // indexee must be global static array var
        assert(iexp->e1->op == TOKvar);
        VarExp* vexp = (VarExp*)iexp->e1;
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
        LLConstant* gep = llvm::ConstantExpr::getGetElementPtr(isaConstant(vd->ir.irGlobal->value), idxs, 2);

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
        cachedLvalue = NULL;
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
        cachedLvalue = NULL;
        VarDeclaration* vd = var->isVarDeclaration();
        assert(vd);
        return new DVarValue(type, vd, V);
    }

    DValue* l = e1->toElem(p);

    Type* t = type->toBasetype();
    Type* e1type = e1->type->toBasetype();

    //Logger::println("e1type=%s", e1type->toChars());
    //Logger::cout() << *DtoType(e1type) << '\n';

    if (VarDeclaration* vd = var->isVarDeclaration()) {
        LLValue* arrptr;
        // indexing struct pointer
        if (e1type->ty == Tpointer) {
            assert(e1type->nextOf()->ty == Tstruct);
            TypeStruct* ts = (TypeStruct*)e1type->nextOf();
            arrptr = DtoIndexStruct(l->getRVal(), ts->sym, vd);
        }
        // indexing normal struct
        else if (e1type->ty == Tstruct) {
            TypeStruct* ts = (TypeStruct*)e1type;
            arrptr = DtoIndexStruct(l->getRVal(), ts->sym, vd);
        }
        // indexing class
        else if (e1type->ty == Tclass) {
            TypeClass* tc = (TypeClass*)e1type;
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

        LLValue* funcval;
        LLValue* vthis2 = 0;
        if (e1type->ty == Tclass) {
            TypeClass* tc = (TypeClass*)e1type;
            if (tc->sym->isInterfaceDeclaration()) {
                vthis2 = DtoCastInterfaceToObject(l, NULL)->getRVal();
            }
        }
        LLValue* vthis = l->getRVal();
        if (!vthis2) vthis2 = vthis;

        //
        // decide whether this function needs to be looked up in the vtable
        //
        bool vtbllookup = fdecl->isAbstract() || (!fdecl->isFinal() && fdecl->isVirtual());

        // even virtual functions are looked up directly if super or DotTypeExp
        // are used, thus we need to walk through the this expression and check
        Expression* e = e1;
        while (e && vtbllookup) {
            if (e->op == TOKsuper || e->op == TOKdottype)
                vtbllookup = false;
            else if (e->op == TOKcast)
                e = ((CastExp*)e)->e1;
            else
                break;
        }

        //
        // look up function
        //
        if (!vtbllookup) {
            fdecl->codegen(Type::sir);
            funcval = fdecl->ir.irFunc->func;
            assert(funcval);
        }
        else {
            DImValue vthis3(e1type, vthis);
            funcval = DtoVirtualFunctionPointer(&vthis3, fdecl, toChars());
        }

        return new DFuncValue(fdecl, funcval, vthis2);
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
        if (vd->toParent2() != p->func()->decl) {
            Logger::println("nested this exp");
            return DtoNestedVariable(loc, type, vd);
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
        cachedLvalue = NULL;
        return new DVarValue(type, V);
    }

    DValue* l = e1->toElem(p);

    Type* e1type = e1->type->toBasetype();

    p->arrays.push_back(l); // if $ is used it must be an array so this is fine.
    DValue* r = e2->toElem(p);
    p->arrays.pop_back();

    LLValue* zero = DtoConstUint(0);
    LLValue* one = DtoConstUint(1);

    LLValue* arrptr = 0;
    if (e1type->ty == Tpointer) {
        arrptr = DtoGEP1(l->getRVal(),r->getRVal());
    }
    else if (e1type->ty == Tsarray) {
        if(global.params.useArrayBounds)
            DtoArrayBoundsCheck(loc, l, r, false);
        arrptr = DtoGEP(l->getRVal(), zero, r->getRVal());
    }
    else if (e1type->ty == Tarray) {
        if(global.params.useArrayBounds)
            DtoArrayBoundsCheck(loc, l, r, false);
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

        if(global.params.useArrayBounds && (etype->ty == Tsarray || etype->ty == Tarray))
            DtoArrayBoundsCheck(loc, e, up, true);

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
            TypeSArray* tsa = (TypeSArray*)etype;
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
    Type* e2t = e2->type->toBasetype();

    LLValue* eval = 0;

    if (t->isintegral() || t->ty == Tpointer)
    {
        llvm::ICmpInst::Predicate cmpop;
        bool skip = false;
        // pointers don't report as being unsigned
        bool uns = (t->isunsigned() || t->ty == Tpointer);
        switch(op)
        {
        case TOKlt:
        case TOKul:
            cmpop = uns ? llvm::ICmpInst::ICMP_ULT : llvm::ICmpInst::ICMP_SLT;
            break;
        case TOKle:
        case TOKule:
            cmpop = uns ? llvm::ICmpInst::ICMP_ULE : llvm::ICmpInst::ICMP_SLE;
            break;
        case TOKgt:
        case TOKug:
            cmpop = uns ? llvm::ICmpInst::ICMP_UGT : llvm::ICmpInst::ICMP_SGT;
            break;
        case TOKge:
        case TOKuge:
            cmpop = uns ? llvm::ICmpInst::ICMP_UGE : llvm::ICmpInst::ICMP_SGE;
            break;
        case TOKue:
            cmpop = llvm::ICmpInst::ICMP_EQ;
            break;
        case TOKlg:
            cmpop = llvm::ICmpInst::ICMP_NE;
            break;
        case TOKleg:
            skip = true;
            eval = LLConstantInt::getTrue(gIR->context());
            break;
        case TOKunord:
            skip = true;
            eval = LLConstantInt::getFalse(gIR->context());
            break;

        default:
            assert(0);
        }
        if (!skip)
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
            eval = p->ir->CreateICmp(cmpop, a, b, "tmp");
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
    Type* e2t = e2->type->toBasetype();
    //assert(t == e2t);

    LLValue* eval = 0;

    // the Tclass catches interface comparisons, regular
    // class equality should be rewritten as a.opEquals(b) by this time
    if (t->isintegral() || t->ty == Tpointer || t->ty == Tclass)
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
    DValue* r = e2->toElem(p);

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
        assert(e2type->isintegral());
        LLConstant* minusone = LLConstantInt::get(DtoSize_t(),(uint64_t)-1,true);
        LLConstant* plusone = LLConstantInt::get(DtoSize_t(),(uint64_t)1,false);
        LLConstant* whichone = (op == TOKplusplus) ? plusone : minusone;
        post = llvm::GetElementPtrInst::Create(val, whichone, "tmp", p->scopebb());
    }
    else if (e1type->isfloating())
    {
        assert(e2type->isfloating());
        LLValue* one = DtoConstFP(e1type, 1.0);
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
        return DtoNewClass(loc, (TypeClass*)ntype, this);
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
            DValue* sz = ((Expression*)arguments->data[0])->toElem(p);
            // allocate & init
            return DtoNewDynArray(loc, newtype, sz, true);
        }
        else
        {
            size_t ndims = arguments->dim;
            std::vector<DValue*> dims(ndims);
            for (size_t i=0; i<ndims; ++i)
                dims[i] = ((Expression*)arguments->data[i])->toElem(p);
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
        LLValue* mem = DtoNew(newtype);
        // init
        TypeStruct* ts = (TypeStruct*)ntype;
        if (ts->isZeroInit(ts->sym->loc)) {
            DtoAggrZeroInit(mem);
        }
        else {
            assert(ts->sym);
            ts->sym->codegen(Type::sir);
            DtoAggrCopy(mem, ts->sym->ir.irStruct->getInitSymbol());
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
        LLValue* rval = dval->getRVal();
        DtoDeleteMemory(rval);
        if (dval->isVar())
            DtoStore(LLConstant::getNullValue(rval->getType()), dval->getLVal());
    }
    // class
    else if (et->ty == Tclass)
    {
        bool onstack = false;
        TypeClass* tc = (TypeClass*)et;
        if (tc->sym->isInterfaceDeclaration())
        {
            DtoDeleteInterface(dval->getRVal());
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
    if (e1->op == TOKthis && ((ThisExp*)e1)->var == NULL)
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

    InvariantDeclaration* invdecl;

    // class invariants
    if(
        global.params.useInvariants &&
        condty->ty == Tclass &&
        !((TypeClass*)condty)->sym->isInterfaceDeclaration())
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
        (invdecl = ((TypeStruct*)condty->nextOf())->sym->inv) != NULL)
    {
        Logger::print("calling struct invariant");
        ((TypeStruct*)condty->nextOf())->sym->codegen(Type::sir);
        DFuncValue invfunc(invdecl, invdecl->ir.irFunc->func, cond->getRVal());
        DtoCallFunction(loc, NULL, &invfunc, NULL);
    }
    else
    {
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
    }

    // DMD allows syntax like this:
    // f() == 0 || assert(false)
    // TODO: or should we return true?
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
    DValue* v = e2->toElem(p);

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
        llvm::PHINode* phi = p->ir->CreatePHI(LLType::getInt1Ty(gIR->context()), "andandval");
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
    DValue* v = e2->toElem(p);

    LLValue* vbool = 0;
    if (!v->isFunc() && v->getType() != Type::tvoid)
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
        llvm::PHINode* phi = p->ir->CreatePHI(LLType::getInt1Ty(gIR->context()), "ororval");
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
    if (e1->type->isunsigned())
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

    const LLPointerType* int8ptrty = getPtrToType(LLType::getInt8Ty(gIR->context()));

    assert(type->toBasetype()->ty == Tdelegate);
    const LLType* dgty = DtoType(type);

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
    if (func->isVirtual() && !func->isFinal())
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

    return new DImValue(type, DtoAggrPair(castcontext, castfptr, ".dg"));
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

    DValue* u = e1->toElem(p);
    DValue* v = e2->toElem(p);
    assert(e2->type == type);
    return v;
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
    DValue* u = e1->toElem(p);
    if (dtype->ty != Tvoid)
        DtoAssign(loc, dvv, u);
    llvm::BranchInst::Create(condend,p->scopebb());

    p->scope() = IRScope(condfalse, condend);
    DValue* v = e2->toElem(p);
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
    LLValue* minusone = LLConstantInt::get(value->getType(), (uint64_t)-1, true);
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

    Type* t = type->toBasetype();

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
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CatAssignExp::toElem(IRState* p)
{
    Logger::print("CatAssignExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);

    Type* e1type = e1->type->toBasetype();
    Type* elemtype = e1type->nextOf()->toBasetype();
    Type* e2type = e2->type->toBasetype();

    if (e2type == elemtype) {
        DtoCatAssignElement(loc, e1type, l, e2);
    }
    else if (e1type == e2type) {
        DSliceValue* slice = DtoCatAssignArray(l,e2);
        DtoAssign(loc, l, slice);
    }
    else if (elemtype->ty == Tchar) {
        if (e2type->ty == Tdchar)
            DtoAppendDCharToString(l, e2);
        else
            assert(0 && "cannot append the element to a string");
    }
    else if (elemtype->ty == Twchar) {
        if (e2type->ty == Tdchar)
            DtoAppendDCharToUnicodeString(l, e2);
        else
            assert(0 && "cannot append the element to an unicode string");
    }
    else {
        assert(0 && "only one element at a time right now");
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
    Logger::println("kind = %s\n", fd->kind());

    fd->codegen(Type::sir);
    assert(fd->ir.irFunc->func);

    if(fd->tok == TOKdelegate) {
        const LLType* dgty = DtoType(type);

        LLValue* cval;
        IrFunction* irfn = p->func();
        if (irfn->nestedVar)
            cval = irfn->nestedVar;
        else if (irfn->nestArg)
            cval = irfn->nestArg;
        else
            cval = getNullPtr(getVoidPtrType());
        cval = DtoBitCast(cval, dgty->getContainedType(0));

        LLValue* castfptr = DtoBitCast(fd->ir.irFunc->func, dgty->getContainedType(1));

        return new DImValue(type, DtoAggrPair(cval, castfptr, ".func"));

    } else if(fd->tok == TOKfunction) {
        return new DImValue(type, fd->ir.irFunc->func);
    }

    assert(0 && "fd->tok must be TOKfunction or TOKdelegate");
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* FuncExp::toConstElem(IRState* p)
{
    Logger::print("FuncExp::toConstElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(fd);
    if (fd->tok != TOKfunction)
    {
        assert(fd->tok == TOKdelegate);
        error("delegate literals as constant expressions are not yet allowed");
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
    const LLType* llType = DtoType(arrayType);
    if (Logger::enabled())
        Logger::cout() << (dyn?"dynamic":"static") << " array literal with length " << len << " of D type: '" << arrayType->toChars() << "' has llvm type: '" << *llType << "'\n";

    // llvm storage type
    const LLType* llElemType = DtoTypeNotVoid(elemType);
    const LLType* llStoType = LLArrayType::get(llElemType, len);
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
        Expression* expr = (Expression*)elements->data[i];
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
    const LLArrayType* arrtype = LLArrayType::get(DtoTypeNotVoid(elemt), elements->dim);

    // dynamic arrays can occur here as well ...
    bool dyn = (bt->ty == Tarray);

    // build the initializer
    std::vector<LLConstant*> vals(elements->dim, NULL);
    for (unsigned i=0; i<elements->dim; ++i)
    {
        Expression* expr = (Expression*)elements->data[i];
        vals[i] = expr->toConstElem(p);
    }

    // build the constant array initializer
    LLConstant* initval = LLConstantArray::get(arrtype, vals);

    // if static array, we're done
    if (!dyn)
        return initval;

    // for dynamic arrays we need to put the initializer in a global, and build a constant dynamic array reference with the .ptr field pointing into this global
    // Important: don't make the global constant, since this const initializer might
    // be used as an initializer for a static T[] - where modifying contents is allowed.
    LLConstant* globalstore = new LLGlobalVariable(*gIR->module, arrtype, false, LLGlobalValue::InternalLinkage, initval, ".dynarrayStorage");
    LLConstant* idxs[2] = { DtoConstUint(0), DtoConstUint(0) };
    LLConstant* globalstorePtr = llvm::ConstantExpr::getGetElementPtr(globalstore, idxs, 2);

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
        else
        {
            IF_LOG Logger::println("using default initializer");
            cv.c = get_default_initializer(vd, NULL);
            val = &cv;
        }

        // get a pointer to this field
        DVarValue field(vd->type, vd, DtoIndexStruct(mem, sd, vd));

        // store the initializer there
        DtoAssign(loc, &field, val);
    }
    // initialize trailing padding
    if (sd->structsize != offset)
        write_zeroes(voidptr, offset, sd->structsize);

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
    std::vector<LLValue*> inits(sd->fields.dim, NULL);

    size_t nexprs = elements->dim;;
    Expression** exprs = (Expression**)elements->data;

    for (size_t i = 0; i < nexprs; i++)
        if (exprs[i])
            inits[i] = exprs[i]->toConstElem(p);

    // vector of values to build aggregate from
    std::vector<LLValue*> values = DtoStructLiteralValues(sd, inits);

    // we know those values are constants.. cast them
    std::vector<LLConstant*> constvals(values.size(), NULL);
    for (size_t i = 0; i < values.size(); ++i)
        constvals[i] = llvm::cast<LLConstant>(values[i]);

    // return constant struct
    return LLConstantStruct::get(gIR->context(), constvals, sd->ir.irStruct->packed);
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

    DtoAARemove(loc, aa, key);

    return NULL; // does not produce anything useful
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* AssocArrayLiteralExp::toElem(IRState* p)
{
    Logger::print("AssocArrayLiteralExp::toElem: %s @ %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(keys);
    assert(values);
    assert(keys->dim == values->dim);

    Type* aatype = type->toBasetype();
    Type* vtype = aatype->nextOf();

    // it should be possible to avoid the temporary in some cases
    LLValue* tmp = DtoAlloca(type,"aaliteral");
    DValue* aa = new DVarValue(type, tmp);
    DtoStore(LLConstant::getNullValue(DtoType(type)), tmp);

    const size_t n = keys->dim;
    for (size_t i=0; i<n; ++i)
    {
        Expression* ekey = (Expression*)keys->data[i];
        Expression* eval = (Expression*)values->data[i];

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
    // this should be good enough for now!
    DValue* val = e1->toElem(p);
    assert(val->isLVal());
    LLValue* v = DtoGEPi(val->getLVal(), 0, index);
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
    std::vector<const LLType*> types(exps->dim, NULL);
    for (size_t i = 0; i < exps->dim; i++)
    {
        Expression *el = (Expression *)exps->data[i];
        DValue* ep = el->toElem(p);
        types[i] = ep->getRVal()->getType();
    }
    LLValue *val = DtoRawAlloca(LLStructType::get(gIR->context(), types),0, "tuple");
    for (size_t i = 0; i < exps->dim; i++)
    {
        Expression *el = (Expression *)exps->data[i];
        DValue* ep = el->toElem(p);
        DtoStore(ep->getRVal(), DtoGEPi(val,0,i));
    }
    return new DImValue(type, val);
}

//////////////////////////////////////////////////////////////////////////////////////////

#define STUB(x) DValue *x::toElem(IRState * p) {error("Exp type "#x" not implemented: %s", toChars()); fatal(); return 0; }
STUB(Expression);
STUB(ScopeExp);

#if DMDV2
STUB(SymbolExp);
#endif

#define CONSTSTUB(x) LLConstant* x::toConstElem(IRState * p) { \
    error("expression '%s' is not a constant", toChars()); \
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
    char *arg = (char *)mem.malloc(n);
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
