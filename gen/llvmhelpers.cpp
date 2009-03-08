#include "gen/llvm.h"
#include "llvm/Target/TargetMachineRegistry.h"

#include "mars.h"
#include "init.h"
#include "id.h"
#include "expression.h"
#include "template.h"
#include "module.h"

#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/irstate.h"
#include "gen/runtime.h"
#include "gen/logger.h"
#include "gen/arrays.h"
#include "gen/dvalue.h"
#include "gen/complex.h"
#include "gen/classes.h"
#include "gen/functions.h"
#include "gen/typeinf.h"
#include "gen/todebug.h"
#include "gen/cl_options.h"
#include "ir/irmodule.h"

#include <stack>

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// DYNAMIC MEMORY HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/

LLValue* DtoNew(Type* newtype)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_allocmemoryT");
    // get type info
    LLConstant* ti = DtoTypeInfoOf(newtype);
    assert(isaPointer(ti));
    // call runtime allocator
    LLValue* mem = gIR->CreateCallOrInvoke(fn, ti, ".gc_mem").getInstruction();
    // cast
    return DtoBitCast(mem, getPtrToType(DtoType(newtype)), ".gc_mem");
}

void DtoDeleteMemory(LLValue* ptr)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delmemory");
    // build args
    LLSmallVector<LLValue*,1> arg;
    arg.push_back(DtoBitCast(ptr, getVoidPtrType(), ".tmp"));
    // call
    gIR->CreateCallOrInvoke(fn, arg.begin(), arg.end());
}

void DtoDeleteClass(LLValue* inst)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delclass");
    // build args
    LLSmallVector<LLValue*,1> arg;
    arg.push_back(DtoBitCast(inst, fn->getFunctionType()->getParamType(0), ".tmp"));
    // call
    gIR->CreateCallOrInvoke(fn, arg.begin(), arg.end());
}

void DtoDeleteInterface(LLValue* inst)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delinterface");
    // build args
    LLSmallVector<LLValue*,1> arg;
    arg.push_back(DtoBitCast(inst, fn->getFunctionType()->getParamType(0), ".tmp"));
    // call
    gIR->CreateCallOrInvoke(fn, arg.begin(), arg.end());
}

void DtoDeleteArray(DValue* arr)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delarray");
    // build args
    LLSmallVector<LLValue*,2> arg;
    arg.push_back(DtoArrayLen(arr));
    arg.push_back(DtoBitCast(DtoArrayPtr(arr), getVoidPtrType(), ".tmp"));
    // call
    gIR->CreateCallOrInvoke(fn, arg.begin(), arg.end());
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// ALLOCA HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/


llvm::AllocaInst* DtoAlloca(const LLType* lltype, const std::string& name)
{
    return new llvm::AllocaInst(lltype, name, gIR->topallocapoint());
}

llvm::AllocaInst* DtoAlloca(const LLType* lltype, LLValue* arraysize, const std::string& name)
{
    return new llvm::AllocaInst(lltype, arraysize, name, gIR->topallocapoint());
}


/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// ASSERT HELPER
////////////////////////////////////////////////////////////////////////////////////////*/

void DtoAssert(Module* M, Loc* loc, DValue* msg)
{
    std::vector<LLValue*> args;

    // func
    const char* fname = msg ? "_d_assert_msg" : "_d_assert";
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, fname);

    // msg param
    if (msg)
    {
        args.push_back(msg->getRVal());
    }

    // file param

    // we might be generating for an imported template function
    if (!M->ir.irModule)
        M->ir.irModule = new IrModule(M, M->srcfile->toChars());

    args.push_back(DtoLoad(M->ir.irModule->fileName));

    // line param
    LLConstant* c = DtoConstUint(loc->linnum);
    args.push_back(c);

    // call
    gIR->CreateCallOrInvoke(fn, args.begin(), args.end());

    // end debug info
    if (global.params.symdebug)
        DtoDwarfFuncEnd(gIR->func()->decl);

    // after assert is always unreachable
    gIR->ir->CreateUnreachable();
}


/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// LABEL HELPER
////////////////////////////////////////////////////////////////////////////////////////*/
LabelStatement* DtoLabelStatement(Identifier* ident)
{
    FuncDeclaration* fd = gIR->func()->decl;
    FuncDeclaration::LabelMap::iterator iter = fd->labmap.find(ident->toChars());
    if (iter == fd->labmap.end())
    {
        if (fd->returnLabel && fd->returnLabel->ident->equals(ident))
        {
            assert(fd->returnLabel->statement);
            return fd->returnLabel->statement;
        }
        return NULL;
    }
    return iter->second;
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// GOTO HELPER
////////////////////////////////////////////////////////////////////////////////////////*/
void DtoGoto(Loc* loc, Identifier* target, EnclosingHandler* enclosinghandler, TryFinallyStatement* sourcetf)
{
    assert(!gIR->scopereturned());

    LabelStatement* lblstmt = DtoLabelStatement(target);
    if(!lblstmt) {
        error(*loc, "the label %s does not exist", target->toChars());
        fatal();
    }

    // if the target label is inside inline asm, error
    if(lblstmt->asmLabel) {
        error(*loc, "cannot goto to label %s inside an inline asm block", target->toChars());
        fatal();
    }

    // find target basic block
    std::string labelname = gIR->func()->getScopedLabelName(target->toChars());
    llvm::BasicBlock*& targetBB = gIR->func()->labelToBB[labelname];
    if (targetBB == NULL)
        targetBB = llvm::BasicBlock::Create("label_" + labelname, gIR->topfunc());

    // find finallys between goto and label
    EnclosingHandler* endfinally = enclosinghandler;
    while(endfinally != NULL && endfinally != lblstmt->enclosinghandler) {
        endfinally = endfinally->getEnclosing();
    }

    // error if didn't find tf statement of label
    if(endfinally != lblstmt->enclosinghandler)
        error(*loc, "cannot goto into try block");

    // goto into finally blocks is forbidden by the spec
    // though it should not be problematic to implement
    if(lblstmt->tf != sourcetf) {
        error(*loc, "spec disallows goto into finally block");
        fatal();
    }

    // emit code for finallys between goto and label
    DtoEnclosingHandlers(enclosinghandler, endfinally);

    llvm::BranchInst::Create(targetBB, gIR->scopebb());
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// TRY-FINALLY, VOLATILE AND SYNCHRONIZED HELPER
////////////////////////////////////////////////////////////////////////////////////////*/

void EnclosingSynchro::emitCode(IRState * p)
{
    if (s->exp)
        DtoLeaveMonitor(s->llsync);
    else
        DtoLeaveCritical(s->llsync);
}

EnclosingHandler* EnclosingSynchro::getEnclosing()
{
    return s->enclosinghandler;
}

////////////////////////////////////////////////////////////////////////////////////////

void EnclosingVolatile::emitCode(IRState * p)
{
    // store-load barrier
    DtoMemoryBarrier(false, false, true, false);
}

EnclosingHandler* EnclosingVolatile::getEnclosing()
{
    return v->enclosinghandler;
}

////////////////////////////////////////////////////////////////////////////////////////

void EnclosingTryFinally::emitCode(IRState * p)
{
    if (tf->finalbody)
        tf->finalbody->toIR(p);
}

EnclosingHandler* EnclosingTryFinally::getEnclosing()
{
    return tf->enclosinghandler;
}

////////////////////////////////////////////////////////////////////////////////////////

void DtoEnclosingHandlers(EnclosingHandler* start, EnclosingHandler* end)
{
    // verify that end encloses start
    EnclosingHandler* endfinally = start;
    while(endfinally != NULL && endfinally != end) {
        endfinally = endfinally->getEnclosing();
    }
    assert(endfinally == end);


    //
    // emit code for finallys between start and end
    //

    // since the labelstatements possibly inside are private
    // and might already exist push a label scope
    gIR->func()->pushUniqueLabelScope("enclosing");
    EnclosingHandler* tf = start;
    while(tf != end) {
        tf->emitCode(gIR);
        tf = tf->getEnclosing();
    }
    gIR->func()->popLabelScope();
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// SYNCHRONIZED SECTION HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/

void DtoEnterCritical(LLValue* g)
{
    LLFunction* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_criticalenter");
    gIR->CreateCallOrInvoke(fn, g);
}

void DtoLeaveCritical(LLValue* g)
{
    LLFunction* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_criticalexit");
    gIR->CreateCallOrInvoke(fn, g);
}

void DtoEnterMonitor(LLValue* v)
{
    LLFunction* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_monitorenter");
    v = DtoBitCast(v, fn->getFunctionType()->getParamType(0));
    gIR->CreateCallOrInvoke(fn, v);
}

void DtoLeaveMonitor(LLValue* v)
{
    LLFunction* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_monitorexit");
    v = DtoBitCast(v, fn->getFunctionType()->getParamType(0));
    gIR->CreateCallOrInvoke(fn, v);
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// NESTED VARIABLE HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/

DValue* DtoNestedVariable(Loc loc, Type* astype, VarDeclaration* vd)
{
    Dsymbol* vdparent = vd->toParent2();
    assert(vdparent);
    
    IrFunction* irfunc = gIR->func();
    
    // is the nested variable in this scope?
    if (vdparent == irfunc->decl)
    {
        LLValue* val = vd->ir.getIrValue();
        return new DVarValue(astype, vd, val);
    }
    
    // get it from the nested context
    LLValue* ctx = 0;
    if (irfunc->decl->isMember2())
    {
        ClassDeclaration* cd = irfunc->decl->isMember2()->isClassDeclaration();
        LLValue* val = DtoLoad(irfunc->thisArg);
        ctx = DtoLoad(DtoGEPi(val, 0,cd->vthis->ir.irField->index, ".vthis"));
    }
    else
        ctx = irfunc->nestArg;
    assert(ctx);
    
    assert(vd->ir.irLocal);
    LLValue* val = DtoBitCast(ctx, getPtrToType(getVoidPtrType()));
    val = DtoGEPi1(val, vd->ir.irLocal->nestedIndex);
    val = DtoLoad(val);
    assert(vd->ir.irLocal->value);
    val = DtoBitCast(val, vd->ir.irLocal->value->getType(), vd->toChars());
    return new DVarValue(astype, vd, val);
}

LLValue* DtoNestedContext(Loc loc, Dsymbol* sym)
{
    Logger::println("DtoNestedContext for %s", sym->toPrettyChars());
    LOG_SCOPE;

    IrFunction* irfunc = gIR->func();

    // if this func has its own vars that are accessed by nested funcs
    // use its own context
    if (irfunc->nestedVar)
        return irfunc->nestedVar;
    // otherwise, it may have gotten a context from the caller
    else if (irfunc->nestArg)
        return irfunc->nestArg;
    // or just have a this argument
    else if (irfunc->thisArg)
    {
        ClassDeclaration* cd = irfunc->decl->isMember2()->isClassDeclaration();
        if (!cd || !cd->vthis)
            return getNullPtr(getVoidPtrType());
        LLValue* val = DtoLoad(irfunc->thisArg);
        return DtoLoad(DtoGEPi(val, 0,cd->vthis->ir.irField->index, ".vthis"));
    }
    else
    {
        return getNullPtr(getVoidPtrType());
    }
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// ASSIGNMENT HELPER (store this in that)
////////////////////////////////////////////////////////////////////////////////////////*/

// is this a good approach at all ?

void DtoAssign(Loc& loc, DValue* lhs, DValue* rhs)
{
    Logger::println("DtoAssign(...);\n");
    LOG_SCOPE;

    Type* t = lhs->getType()->toBasetype();
    Type* t2 = rhs->getType()->toBasetype();

    if (t->ty == Tstruct) {
        if (!t->equals(t2)) {
            // FIXME: use 'rhs' for something !?!
            DtoAggrZeroInit(lhs->getLVal());
        }
        else {
            DtoAggrCopy(lhs->getLVal(), rhs->getRVal());
        }
    }
    else if (t->ty == Tarray) {
        // lhs is slice
        if (DSliceValue* s = lhs->isSlice()) {
            if (DSliceValue* s2 = rhs->isSlice()) {
                DtoArrayCopySlices(s, s2);
            }
            else if (t->nextOf()->toBasetype()->equals(t2)) {
                DtoArrayInit(loc, s, rhs);
            }
            else {
                DtoArrayCopyToSlice(s, rhs);
            }
        }
        // rhs is slice
        else if (DSliceValue* s = rhs->isSlice()) {
            assert(s->getType()->toBasetype() == lhs->getType()->toBasetype());
            DtoSetArray(lhs->getLVal(),DtoArrayLen(s),DtoArrayPtr(s));
        }
        // null
        else if (rhs->isNull()) {
            DtoSetArrayToNull(lhs->getLVal());
        }
        // reference assignment
        else if (t2->ty == Tarray) {
            DtoStore(rhs->getRVal(), lhs->getLVal());
        }
        // some implicitly converting ref assignment
        else {
            DtoSetArray(lhs->getLVal(), DtoArrayLen(rhs), DtoArrayPtr(rhs));
        }
    }
    else if (t->ty == Tsarray) {
        // T[n] = T[n]
        if (DtoType(lhs->getType()) == DtoType(rhs->getType())) {
            DtoStaticArrayCopy(lhs->getLVal(), rhs->getRVal());
        }
        // T[n] = T
        else if (t->nextOf()->toBasetype()->equals(t2)) {
            DtoArrayInit(loc, lhs, rhs);
        }
        // T[n] = T[] - generally only generated by frontend in rare cases
        else if (t2->ty == Tarray && t->nextOf()->toBasetype()->equals(t2->nextOf()->toBasetype())) {
            DtoMemCpy(lhs->getLVal(), DtoArrayPtr(rhs), DtoArrayLen(rhs));
        } else {
            assert(0 && "Unimplemented static array assign!");
        }
    }
    else if (t->ty == Tdelegate) {
        LLValue* l = lhs->getLVal();
        LLValue* r = rhs->getRVal();
        if (Logger::enabled())
            Logger::cout() << "assign\nlhs: " << *l << "rhs: " << *r << '\n';
        DtoStore(r, l);
    }
    else if (t->ty == Tclass) {
        assert(t2->ty == Tclass);
        LLValue* l = lhs->getLVal();
        LLValue* r = rhs->getRVal();
        if (Logger::enabled())
        {
            Logger::cout() << "l : " << *l << '\n';
            Logger::cout() << "r : " << *r << '\n';
        }
        r = DtoBitCast(r, l->getType()->getContainedType(0));
        DtoStore(r, l);
    }
    else if (t->iscomplex()) {
        LLValue* dst;
        if (DLRValue* lr = lhs->isLRValue()) {
            dst = lr->getLVal();
            rhs = DtoCastComplex(loc, rhs, lr->getLType());
        }
        else {
            dst = lhs->getLVal();
        }
        DtoStore(rhs->getRVal(), dst);
    }
    else {
        LLValue* l = lhs->getLVal();
        LLValue* r = rhs->getRVal();
        if (Logger::enabled())
            Logger::cout() << "assign\nlhs: " << *l << "rhs: " << *r << '\n';
        const LLType* lit = l->getType()->getContainedType(0);
        if (r->getType() != lit) {
            // handle lvalue cast assignments
            if (DLRValue* lr = lhs->isLRValue()) {
                Logger::println("lvalue cast!");
                r = DtoCast(loc, rhs, lr->getLType())->getRVal();
            }
            else {
                r = DtoCast(loc, rhs, lhs->getType())->getRVal();
            }
            if (Logger::enabled())
                Logger::cout() << "really assign\nlhs: " << *l << "rhs: " << *r << '\n';
            assert(r->getType() == l->getType()->getContainedType(0));
        }
        gIR->ir->CreateStore(r, l);
    }
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
//      NULL VALUE HELPER
////////////////////////////////////////////////////////////////////////////////////////*/

DValue* DtoNullValue(Type* type)
{
    Type* basetype = type->toBasetype();
    TY basety = basetype->ty;
    const LLType* lltype = DtoType(basetype);

    // complex, needs to be first since complex are also floating
    if (basetype->iscomplex())
    {
        const LLType* basefp = DtoComplexBaseType(basetype);
        LLValue* res = DtoAggrPair(DtoType(type), LLConstant::getNullValue(basefp), LLConstant::getNullValue(basefp));
        return new DImValue(type, res);
    }
    // integer, floating, pointer and class have no special representation
    else if (basetype->isintegral() || basetype->isfloating() || basety == Tpointer || basety == Tclass)
    {
        return new DConstValue(type, LLConstant::getNullValue(lltype));
    }
    // dynamic array
    else if (basety == Tarray)
    {
        LLValue* len = DtoConstSize_t(0);
        LLValue* ptr = getNullPtr(getPtrToType(DtoType(basetype->nextOf())));
        return new DSliceValue(type, len, ptr);
    }
    // delegate
    else if (basety == Tdelegate)
    {
        return new DNullValue(type, LLConstant::getNullValue(lltype));
    }

    // unknown
    llvm::cout << "unsupported: null value for " << type->toChars() << '\n';
    assert(0);
    return 0;

}


/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
//      CASTING HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/

DValue* DtoCastInt(Loc& loc, DValue* val, Type* _to)
{
    const LLType* tolltype = DtoType(_to);

    Type* to = _to->toBasetype();
    Type* from = val->getType()->toBasetype();
    assert(from->isintegral());

    size_t fromsz = from->size();
    size_t tosz = to->size();

    LLValue* rval = val->getRVal();
    if (rval->getType() == tolltype) {
        return new DImValue(_to, rval);
    }

    if (to->ty == Tbool) {
        LLValue* zero = LLConstantInt::get(rval->getType(), 0, false);
        rval = gIR->ir->CreateICmpNE(rval, zero, "tmp");
    }
    else if (to->isintegral()) {
        if (fromsz < tosz || from->ty == Tbool) {
            if (Logger::enabled())
                Logger::cout() << "cast to: " << *tolltype << '\n';
            if (from->isunsigned() || from->ty == Tbool) {
                rval = new llvm::ZExtInst(rval, tolltype, "tmp", gIR->scopebb());
            } else {
                rval = new llvm::SExtInst(rval, tolltype, "tmp", gIR->scopebb());
            }
        }
        else if (fromsz > tosz) {
            rval = new llvm::TruncInst(rval, tolltype, "tmp", gIR->scopebb());
        }
        else {
            rval = DtoBitCast(rval, tolltype);
        }
    }
    else if (to->iscomplex()) {
        return DtoComplex(loc, to, val);
    }
    else if (to->isfloating()) {
        if (from->isunsigned()) {
            rval = new llvm::UIToFPInst(rval, tolltype, "tmp", gIR->scopebb());
        }
        else {
            rval = new llvm::SIToFPInst(rval, tolltype, "tmp", gIR->scopebb());
        }
    }
    else if (to->ty == Tpointer) {
        if (Logger::enabled())
            Logger::cout() << "cast pointer: " << *tolltype << '\n';
        rval = gIR->ir->CreateIntToPtr(rval, tolltype, "tmp");
    }
    else {
        error(loc, "invalid cast from '%s' to '%s'", val->getType()->toChars(), _to->toChars());
        fatal();
    }

    return new DImValue(_to, rval);
}

DValue* DtoCastPtr(Loc& loc, DValue* val, Type* to)
{
    const LLType* tolltype = DtoType(to);

    Type* totype = to->toBasetype();
    Type* fromtype = val->getType()->toBasetype();
    assert(fromtype->ty == Tpointer || fromtype->ty == Tfunction);

    LLValue* rval;

    if (totype->ty == Tpointer || totype->ty == Tclass) {
        LLValue* src = val->getRVal();
        if (Logger::enabled())
            Logger::cout() << "src: " << *src << "to type: " << *tolltype << '\n';
        rval = DtoBitCast(src, tolltype);
    }
    else if (totype->ty == Tbool) {
        LLValue* src = val->getRVal();
        LLValue* zero = LLConstant::getNullValue(src->getType());
        rval = gIR->ir->CreateICmpNE(src, zero, "tmp");
    }
    else if (totype->isintegral()) {
        rval = new llvm::PtrToIntInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
    }
    else {
        error(loc, "invalid cast from '%s' to '%s'", val->getType()->toChars(), to->toChars());
        fatal();
    }

    return new DImValue(to, rval);
}

DValue* DtoCastFloat(Loc& loc, DValue* val, Type* to)
{
    if (val->getType() == to)
        return val;

    const LLType* tolltype = DtoType(to);

    Type* totype = to->toBasetype();
    Type* fromtype = val->getType()->toBasetype();
    assert(fromtype->isfloating());

    size_t fromsz = fromtype->size();
    size_t tosz = totype->size();

    LLValue* rval;

    if (totype->ty == Tbool) {
        rval = val->getRVal();
        LLValue* zero = LLConstant::getNullValue(rval->getType());
        rval = gIR->ir->CreateFCmpUNE(rval, zero, "tmp");
    }
    else if (totype->iscomplex()) {
        return DtoComplex(loc, to, val);
    }
    else if (totype->isfloating()) {
        if (fromsz == tosz) {
            rval = val->getRVal();
            assert(rval->getType() == tolltype);
        }
        else if (fromsz < tosz) {
            rval = new llvm::FPExtInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
        }
        else if (fromsz > tosz) {
            rval = new llvm::FPTruncInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
        }
        else {
            error(loc, "invalid cast from '%s' to '%s'", val->getType()->toChars(), to->toChars());
            fatal();
        }
    }
    else if (totype->isintegral()) {
        if (totype->isunsigned()) {
            rval = new llvm::FPToUIInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
        }
        else {
            rval = new llvm::FPToSIInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
        }
    }
    else {
        error(loc, "invalid cast from '%s' to '%s'", val->getType()->toChars(), to->toChars());
        fatal();
    }

    return new DImValue(to, rval);
}

DValue* DtoCastDelegate(Loc& loc, DValue* val, Type* to)
{
    if (to->toBasetype()->ty == Tdelegate)
    {
        return DtoPaintType(loc, val, to);
    }
    else if (to->toBasetype()->ty == Tbool)
    {
        return new DImValue(to, DtoDelegateEquals(TOKnotequal, val->getRVal(), NULL));
    }
    else
    {
        error(loc, "invalid cast from '%s' to '%s'", val->getType()->toChars(), to->toChars());
        fatal();
    }
}

DValue* DtoCast(Loc& loc, DValue* val, Type* to)
{
    Type* fromtype = val->getType()->toBasetype();
    Logger::println("Casting from '%s' to '%s'", fromtype->toChars(), to->toChars());
    if (fromtype->isintegral()) {
        return DtoCastInt(loc, val, to);
    }
    else if (fromtype->iscomplex()) {
        return DtoCastComplex(loc, val, to);
    }
    else if (fromtype->isfloating()) {
        return DtoCastFloat(loc, val, to);
    }
    else if (fromtype->ty == Tclass) {
        return DtoCastClass(val, to);
    }
    else if (fromtype->ty == Tarray || fromtype->ty == Tsarray) {
        return DtoCastArray(loc, val, to);
    }
    else if (fromtype->ty == Tpointer || fromtype->ty == Tfunction) {
        return DtoCastPtr(loc, val, to);
    }
    else if (fromtype->ty == Tdelegate) {
        return DtoCastDelegate(loc, val, to);
    }
    else {
        error(loc, "invalid cast from '%s' to '%s'", val->getType()->toChars(), to->toChars());
        fatal();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoPaintType(Loc& loc, DValue* val, Type* to)
{
    Type* from = val->getType()->toBasetype();
    Logger::println("repainting from '%s' to '%s'", from->toChars(), to->toChars());

    if (from->ty == Tarray)
    {
        Type* at = to->toBasetype();
        assert(at->ty == Tarray);
        Type* elem = at->nextOf()->pointerTo();
        if (DSliceValue* slice = val->isSlice())
        {
            return new DSliceValue(to, slice->len, DtoBitCast(slice->ptr, DtoType(elem)));
        }
        else if (val->isLVal())
        {
            LLValue* ptr = val->getLVal();
            ptr = DtoBitCast(ptr, DtoType(at->pointerTo()));
            return new DVarValue(to, ptr);
        }
        else
        {
            LLValue *len, *ptr;
            len = DtoArrayLen(val);
            ptr = DtoArrayPtr(val);
            ptr = DtoBitCast(ptr, DtoType(elem));
            return new DImValue(to, DtoAggrPair(len, ptr, "tmp"));
        }
    }
    else if (from->ty == Tdelegate)
    {
        Type* dgty = to->toBasetype();
        assert(dgty->ty == Tdelegate);
        if (val->isLVal())
        {
            LLValue* ptr = val->getLVal();
            assert(isaPointer(ptr));
            ptr = DtoBitCast(ptr, getPtrToType(DtoType(dgty)));
            if (Logger::enabled())
                Logger::cout() << "dg ptr: " << *ptr << '\n';
            return new DVarValue(to, ptr);
        }
        else
        {
            LLValue* dg = val->getRVal();
            LLValue* context = gIR->ir->CreateExtractValue(dg, 0, ".context");
            LLValue* funcptr = gIR->ir->CreateExtractValue(dg, 1, ".funcptr");
            funcptr = DtoBitCast(funcptr, DtoType(dgty)->getContainedType(1));
            LLValue* aggr = DtoAggrPair(context, funcptr, "tmp");
            if (Logger::enabled())
                Logger::cout() << "dg: " << *aggr << '\n';
            return new DImValue(to, aggr);
        }
    }
    else if (from->ty == Tpointer || from->ty == Tclass || from->ty == Taarray)
    {
        Type* b = to->toBasetype();
        assert(b->ty == Tpointer || b->ty == Tclass || b->ty == Taarray);
        LLValue* ptr = DtoBitCast(val->getRVal(), DtoType(b));
        return new DImValue(to, ptr);
    }
    else
    {
        assert(!val->isLVal());
        assert(DtoType(to) == DtoType(to));
        return new DImValue(to, val->getRVal());
    }
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
//      TEMPLATE HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/

TemplateInstance* DtoIsTemplateInstance(Dsymbol* s)
{
    if (!s) return NULL;
    if (s->isTemplateInstance() && !s->isTemplateMixin())
        return s->isTemplateInstance();
    else if (s->parent)
        return DtoIsTemplateInstance(s->parent);
    return NULL;
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
//      PROCESSING QUEUE HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/

void DtoResolveDsymbol(Dsymbol* dsym)
{
    if (StructDeclaration* sd = dsym->isStructDeclaration()) {
        DtoResolveStruct(sd);
    }
    else if (ClassDeclaration* cd = dsym->isClassDeclaration()) {
        DtoResolveClass(cd);
    }
    else if (FuncDeclaration* fd = dsym->isFuncDeclaration()) {
        DtoResolveFunction(fd);
    }
    else if (TypeInfoDeclaration* fd = dsym->isTypeInfoDeclaration()) {
        DtoResolveTypeInfo(fd);
    }
    else {
    error(dsym->loc, "unsupported dsymbol: %s", dsym->toChars());
    assert(0 && "unsupported dsymbol for DtoResolveDsymbol");
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDeclareDsymbol(Dsymbol* dsym)
{
    if (StructDeclaration* sd = dsym->isStructDeclaration()) {
        DtoDeclareStruct(sd);
    }
    else if (ClassDeclaration* cd = dsym->isClassDeclaration()) {
        DtoDeclareClass(cd);
    }
    else if (FuncDeclaration* fd = dsym->isFuncDeclaration()) {
        DtoDeclareFunction(fd);
    }
    else if (TypeInfoDeclaration* fd = dsym->isTypeInfoDeclaration()) {
        DtoDeclareTypeInfo(fd);
    }
    else {
    error(dsym->loc, "unsupported dsymbol: %s", dsym->toChars());
    assert(0 && "unsupported dsymbol for DtoDeclareDsymbol");
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoConstInitDsymbol(Dsymbol* dsym)
{
    if (StructDeclaration* sd = dsym->isStructDeclaration()) {
        DtoConstInitStruct(sd);
    }
    else if (ClassDeclaration* cd = dsym->isClassDeclaration()) {
        DtoConstInitClass(cd);
    }
    else if (TypeInfoDeclaration* fd = dsym->isTypeInfoDeclaration()) {
        DtoConstInitTypeInfo(fd);
    }
    else if (VarDeclaration* vd = dsym->isVarDeclaration()) {
        DtoConstInitGlobal(vd);
    }
    else {
    error(dsym->loc, "unsupported dsymbol: %s", dsym->toChars());
    assert(0 && "unsupported dsymbol for DtoConstInitDsymbol");
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDefineDsymbol(Dsymbol* dsym)
{
    if (StructDeclaration* sd = dsym->isStructDeclaration()) {
        DtoDefineStruct(sd);
    }
    else if (ClassDeclaration* cd = dsym->isClassDeclaration()) {
        DtoDefineClass(cd);
    }
    else if (FuncDeclaration* fd = dsym->isFuncDeclaration()) {
        DtoDefineFunction(fd);
    }
    else if (TypeInfoDeclaration* fd = dsym->isTypeInfoDeclaration()) {
        DtoDefineTypeInfo(fd);
    }
    else {
    error(dsym->loc, "unsupported dsymbol: %s", dsym->toChars());
    assert(0 && "unsupported dsymbol for DtoDefineDsymbol");
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoConstInitGlobal(VarDeclaration* vd)
{
    if (vd->ir.initialized) return;
    vd->ir.initialized = gIR->dmodule;

    Logger::println("DtoConstInitGlobal(%s) @ %s", vd->toChars(), vd->locToChars());
    LOG_SCOPE;

    Dsymbol* par = vd->toParent();

    // build the initializer
    LLConstant* initVal = DtoConstInitializer(vd->loc, vd->type, vd->init);

    // set the initializer if appropriate
    IrGlobal* glob = vd->ir.irGlobal;
    llvm::GlobalVariable* gvar = llvm::cast<llvm::GlobalVariable>(glob->value);

    // refine the global's opaque type to the type of the initializer
    llvm::cast<LLOpaqueType>(glob->type.get())->refineAbstractTypeTo(initVal->getType());

    assert(!glob->constInit);
    glob->constInit = initVal;

    // assign the initializer
    llvm::GlobalVariable* globalvar = llvm::cast<llvm::GlobalVariable>(glob->value);

    if (!(vd->storage_class & STCextern) && mustDefineSymbol(vd))
    {
        if (Logger::enabled())
        {
            Logger::println("setting initializer");
            Logger::cout() << "global: " << *gvar << '\n';
            Logger::cout() << "init:   " << *initVal << '\n';
        }

        gvar->setInitializer(initVal);

        // do debug info
        if (global.params.symdebug)
        {
            LLGlobalVariable* gv = DtoDwarfGlobalVariable(gvar, vd).getGV();
            // keep a reference so GDCE doesn't delete it !
            gIR->usedArray.push_back(llvm::ConstantExpr::getBitCast(gv, getVoidPtrType()));
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoEmptyResolveList()
{
    //Logger::println("DtoEmptyResolveList()");
    Dsymbol* dsym;
    while (!gIR->resolveList.empty()) {
        dsym = gIR->resolveList.front();
        gIR->resolveList.pop_front();
        DtoResolveDsymbol(dsym);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoEmptyDeclareList()
{
    //Logger::println("DtoEmptyDeclareList()");
    Dsymbol* dsym;
    while (!gIR->declareList.empty()) {
        dsym = gIR->declareList.front();
        gIR->declareList.pop_front();
        DtoDeclareDsymbol(dsym);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoEmptyConstInitList()
{
    //Logger::println("DtoEmptyConstInitList()");
    Dsymbol* dsym;
    while (!gIR->constInitList.empty()) {
        dsym = gIR->constInitList.front();
        gIR->constInitList.pop_front();
        DtoConstInitDsymbol(dsym);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoEmptyDefineList()
{
    //Logger::println("DtoEmptyDefineList()");
    Dsymbol* dsym;
    while (!gIR->defineList.empty()) {
        dsym = gIR->defineList.front();
        gIR->defineList.pop_front();
        DtoDefineDsymbol(dsym);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
void DtoEmptyAllLists()
{
    for(;;)
    {
        Dsymbol* dsym;
        if (!gIR->resolveList.empty()) {
            dsym = gIR->resolveList.front();
            gIR->resolveList.pop_front();
            DtoResolveDsymbol(dsym);
        }
        else if (!gIR->declareList.empty()) {
            dsym = gIR->declareList.front();
            gIR->declareList.pop_front();
            DtoDeclareDsymbol(dsym);
        }
        else if (!gIR->constInitList.empty()) {
            dsym = gIR->constInitList.front();
            gIR->constInitList.pop_front();
            DtoConstInitDsymbol(dsym);
        }
        else if (!gIR->defineList.empty()) {
            dsym = gIR->defineList.front();
            gIR->defineList.pop_front();
            DtoDefineDsymbol(dsym);
        }
        else {
            break;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoForceDeclareDsymbol(Dsymbol* dsym)
{
    if (dsym->ir.declared) return;
    Logger::println("DtoForceDeclareDsymbol(%s)", dsym->toPrettyChars());
    LOG_SCOPE;
    DtoResolveDsymbol(dsym);

    DtoEmptyResolveList();

    DtoDeclareDsymbol(dsym);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoForceConstInitDsymbol(Dsymbol* dsym)
{
    if (dsym->ir.initialized) return;
    Logger::println("DtoForceConstInitDsymbol(%s)", dsym->toPrettyChars());
    LOG_SCOPE;
    DtoResolveDsymbol(dsym);

    DtoEmptyResolveList();
    DtoEmptyDeclareList();

    DtoConstInitDsymbol(dsym);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoForceDefineDsymbol(Dsymbol* dsym)
{
    if (dsym->ir.defined) return;
    Logger::println("DtoForceDefineDsymbol(%s)", dsym->toPrettyChars());
    LOG_SCOPE;
    DtoResolveDsymbol(dsym);

    DtoEmptyResolveList();
    DtoEmptyDeclareList();
    DtoEmptyConstInitList();

    DtoDefineDsymbol(dsym);
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
//      DECLARATION EXP HELPER
////////////////////////////////////////////////////////////////////////////////////////*/
DValue* DtoDeclarationExp(Dsymbol* declaration)
{
    Logger::print("DtoDeclarationExp: %s\n", declaration->toChars());
    LOG_SCOPE;

    // variable declaration
    if (VarDeclaration* vd = declaration->isVarDeclaration())
    {
        Logger::println("VarDeclaration");

        // if aliassym is set, this VarDecl is redone as an alias to another symbol
        // this seems to be done to rewrite Tuple!(...) v;
        // as a TupleDecl that contains a bunch of individual VarDecls
        if (vd->aliassym)
            return DtoDeclarationExp(vd->aliassym);

        // static
        if (vd->isDataseg())
        {
            vd->toObjFile(0); // TODO: multiobj
        }
        else
        {
            if (global.params.llvmAnnotate)
                DtoAnnotation(declaration->toChars());

            Logger::println("vdtype = %s", vd->type->toChars());

            // referenced by nested delegate?
        #if DMDV2
            if (vd->nestedrefs.dim) {
        #else
            if (vd->nestedref) {
        #endif
                Logger::println("has nestedref set");
                assert(vd->ir.irLocal);
                
                // alloca as usual if no value already
                if (!vd->ir.irLocal->value)
                {
                    vd->ir.irLocal->value = DtoAlloca(DtoType(vd->type), vd->toChars());
                }
                
                // store the address into the nested vars array
                
                assert(vd->ir.irLocal->nestedIndex >= 0);
                LLValue* gep = DtoGEPi(gIR->func()->decl->ir.irFunc->nestedVar, 0, vd->ir.irLocal->nestedIndex);
                
                assert(isaPointer(vd->ir.irLocal->value));
                LLValue* val = DtoBitCast(vd->ir.irLocal->value, getVoidPtrType());
                
                DtoStore(val, gep);
                
            }
            // normal stack variable, allocate storage on the stack if it has not already been done
            else if(!vd->ir.irLocal) {
                const LLType* lltype = DtoType(vd->type);

                llvm::Value* allocainst;
                if(gTargetData->getTypeSizeInBits(lltype) == 0) 
                    allocainst = llvm::ConstantPointerNull::get(getPtrToType(lltype));
                else
                    allocainst = DtoAlloca(lltype, vd->toChars());

                //allocainst->setAlignment(vd->type->alignsize()); // TODO
                vd->ir.irLocal = new IrLocal(vd);
                vd->ir.irLocal->value = allocainst;

                if (global.params.symdebug)
                {
                    DtoDwarfLocalVariable(allocainst, vd);
                }
            }
            else
            {
                assert(vd->ir.irLocal->value);
            }

            if (Logger::enabled())
                Logger::cout() << "llvm value for decl: " << *vd->ir.irLocal->value << '\n';
            DValue* ie = DtoInitializer(vd->ir.irLocal->value, vd->init);
        }

        return new DVarValue(vd->type, vd, vd->ir.getIrValue());
    }
    // struct declaration
    else if (StructDeclaration* s = declaration->isStructDeclaration())
    {
        Logger::println("StructDeclaration");
        DtoForceConstInitDsymbol(s);
    }
    // function declaration
    else if (FuncDeclaration* f = declaration->isFuncDeclaration())
    {
        Logger::println("FuncDeclaration");
        DtoForceDeclareDsymbol(f);
    }
    // alias declaration
    else if (AliasDeclaration* a = declaration->isAliasDeclaration())
    {
        Logger::println("AliasDeclaration - no work");
        // do nothing
    }
    // enum
    else if (EnumDeclaration* e = declaration->isEnumDeclaration())
    {
        Logger::println("EnumDeclaration - no work");
        // do nothing
    }
    // class
    else if (ClassDeclaration* e = declaration->isClassDeclaration())
    {
        Logger::println("ClassDeclaration");
        DtoForceConstInitDsymbol(e);
    }
    // typedef
    else if (TypedefDeclaration* tdef = declaration->isTypedefDeclaration())
    {
        Logger::println("TypedefDeclaration");
        DtoTypeInfoOf(tdef->type, false);
    }
    // attribute declaration
    else if (AttribDeclaration* a = declaration->isAttribDeclaration())
    {
        Logger::println("AttribDeclaration");
        for (int i=0; i < a->decl->dim; ++i)
        {
            DtoDeclarationExp((Dsymbol*)a->decl->data[i]);
        }
    }
    // mixin declaration
    else if (TemplateMixin* m = declaration->isTemplateMixin())
    {
        Logger::println("TemplateMixin");
        for (int i=0; i < m->members->dim; ++i)
        {
            Dsymbol* mdsym = (Dsymbol*)m->members->data[i];
            DtoDeclarationExp(mdsym);
        }
    }
    // tuple declaration
    else if (TupleDeclaration* tupled = declaration->isTupleDeclaration())
    {
        Logger::println("TupleDeclaration");
        if(!tupled->isexp) {
            error(declaration->loc, "don't know how to handle non-expression tuple decls yet");
            assert(0);
        }

        assert(tupled->objects);
        for (int i=0; i < tupled->objects->dim; ++i)
        {
            DsymbolExp* exp = (DsymbolExp*)tupled->objects->data[i];
            DtoDeclarationExp(exp->s);
        }
    }
    // unsupported declaration
    else
    {
        error(declaration->loc, "Unimplemented Declaration type for DeclarationExp. kind: %s", declaration->kind());
        assert(0);
    }
    return NULL;
}

// does pretty much the same as DtoDeclarationExp, except it doesn't initialize, and only handles var declarations
LLValue* DtoRawVarDeclaration(VarDeclaration* var)
{
    // we don't handle globals with this one
    assert(!var->isDataseg());

    // we don't handle aliases either
    assert(!var->aliassym);
        
    // if this already has storage, it must've been handled already
    if (var->ir.irLocal && var->ir.irLocal->value)
        return var->ir.irLocal->value;

    // referenced by nested function?
#if DMDV2
    if (var->nestedrefs.dim)
#else
    if (var->nestedref)
#endif
    {
        assert(var->ir.irLocal);
        assert(!var->ir.irLocal->value);

        // alloca
        var->ir.irLocal->value = DtoAlloca(DtoType(var->type), var->toChars());

        // store the address into the nested vars array
        assert(var->ir.irLocal->nestedIndex >= 0);
        LLValue* gep = DtoGEPi(gIR->func()->decl->ir.irFunc->nestedVar, 0, var->ir.irLocal->nestedIndex);
        assert(isaPointer(var->ir.irLocal->value));
        LLValue* val = DtoBitCast(var->ir.irLocal->value, getVoidPtrType());
        DtoStore(val, gep);
    }
    // normal local variable
    else
    {
        assert(!var->ir.isSet());
        var->ir.irLocal = new IrLocal(var);
        var->ir.irLocal->value = DtoAlloca(DtoType(var->type), var->toChars());
    }

    // add debug info
    if (global.params.symdebug)
        DtoDwarfLocalVariable(var->ir.irLocal->value, var);

    // return the alloca
    return var->ir.irLocal->value;
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
//      INITIALIZER HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/

LLConstant* DtoConstInitializer(Loc loc, Type* type, Initializer* init)
{
    LLConstant* _init = 0; // may return zero
    if (!init)
    {
        Logger::println("const default initializer for %s", type->toChars());
        _init = DtoConstExpInit(loc, type, type->defaultInit());
    }
    else if (ExpInitializer* ex = init->isExpInitializer())
    {
        Logger::println("const expression initializer");
        _init = DtoConstExpInit(loc, type, ex->exp);;
    }
    else if (StructInitializer* si = init->isStructInitializer())
    {
        Logger::println("const struct initializer");
        _init = DtoConstStructInitializer(si);
    }
    else if (ArrayInitializer* ai = init->isArrayInitializer())
    {
        Logger::println("const array initializer");
        _init = DtoConstArrayInitializer(ai);
    }
    else if (init->isVoidInitializer())
    {
        Logger::println("const void initializer");
        const LLType* ty = DtoType(type);
        _init = llvm::Constant::getNullValue(ty);
    }
    else {
        Logger::println("unsupported const initializer: %s", init->toChars());
    }
    return _init;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoInitializer(LLValue* target, Initializer* init)
{
    if (!init)
        return 0;
    else if (ExpInitializer* ex = init->isExpInitializer())
    {
        Logger::println("expression initializer");
        assert(ex->exp);
        return ex->exp->toElem(gIR);
    }
    else if (init->isVoidInitializer())
    {
        // do nothing
    }
    else {
        Logger::println("unsupported initializer: %s", init->toChars());
        assert(0);
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

static LLConstant* expand_to_sarray(Type *base, Expression* exp)
{
    Logger::println("building type %s from expression (%s) of type %s", base->toChars(), exp->toChars(), exp->type->toChars());
    const LLType* dstTy = DtoType(base);
    if (Logger::enabled())
        Logger::cout() << "final llvm type requested: " << *dstTy << '\n';

    LLConstant* val = exp->toConstElem(gIR);

    Type* expbase = exp->type->toBasetype();
    Logger::println("expbase: %s", expbase->toChars());
    Type* t = base->toBasetype();

    LLSmallVector<size_t, 4> dims;

    while(1)
    {
        Logger::println("t: %s", t->toChars());
        if (t->equals(expbase))
            break;
        assert(t->ty == Tsarray);
        TypeSArray* tsa = (TypeSArray*)t;
        dims.push_back(tsa->dim->toInteger());
        assert(t->nextOf());
        t = t->nextOf()->toBasetype();
    }

    size_t i = dims.size();
    assert(i);

    std::vector<LLConstant*> inits;
    while (i--)
    {
        const LLArrayType* arrty = LLArrayType::get(val->getType(), dims[i]);
        inits.clear();
        inits.insert(inits.end(), dims[i], val);
        val = LLConstantArray::get(arrty, inits);
    }

    return val;
}

LLConstant* DtoConstExpInit(Loc loc, Type* type, Expression* exp)
{
    Type* expbase = exp->type->toBasetype();
    Type* base = type->toBasetype();

    // if not the same basetypes, we won't get the same llvm types either
    if (!expbase->equals(base))
    {
        if (base->ty == Tsarray)
        {
            if (base->nextOf()->toBasetype()->ty == Tvoid) {
                error(loc, "static arrays of voids have no default initializer");
                fatal();
            }
            Logger::println("type is a static array, building constant array initializer to single value");
            return expand_to_sarray(base, exp);
        }
        else
        {
            error("cannot yet convert default initializer %s to type %s to %s", exp->toChars(), exp->type->toChars(), type->toChars());
            fatal();
        }
        assert(0);
    }

    return exp->toConstElem(gIR);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoAnnotation(const char* str)
{
    std::string s("CODE: ");
    s.append(str);
    char* p = &s[0];
    while (*p)
    {
        if (*p == '"')
            *p = '\'';
        ++p;
    }
    // create a noop with the code as the result name!
    // FIXME: this is const folded and eliminated immediately ... :/
    gIR->ir->CreateAnd(DtoConstSize_t(0),DtoConstSize_t(0),s.c_str());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoTypeInfoOf(Type* type, bool base)
{
    type = type->merge(); // needed.. getTypeInfo does the same
    type->getTypeInfo(NULL);
    TypeInfoDeclaration* tidecl = type->vtinfo;
    assert(tidecl);
    DtoForceDeclareDsymbol(tidecl);
    assert(tidecl->ir.irGlobal != NULL);
    LLConstant* c = isaConstant(tidecl->ir.irGlobal->value);
    assert(c != NULL);
    if (base)
        return llvm::ConstantExpr::getBitCast(c, DtoType(Type::typeinfo->type));
    return c;
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoOverloadedIntrinsicName(TemplateInstance* ti, TemplateDeclaration* td, std::string& name)
{
    Logger::println("DtoOverloadedIntrinsicName");
    LOG_SCOPE;

    Logger::println("template instance: %s", ti->toChars());
    Logger::println("template declaration: %s", td->toChars());
    Logger::println("intrinsic name: %s", td->intrinsicName.c_str());
    
    // for now use the size in bits of the first template param in the instance
    assert(ti->tdtypes.dim == 1);
    Type* T = (Type*)ti->tdtypes.data[0];

    char tmp[10];
    if (T->toBasetype()->ty == Tbool) // otherwise we'd get a mismatch
        sprintf(tmp, "1");
    else
        sprintf(tmp, "%lu", T->size()*8);
    
    // replace # in name with bitsize
    name = td->intrinsicName;

    std::string needle("#");
    size_t pos;
    while(std::string::npos != (pos = name.find(needle)))
        name.replace(pos, 1, tmp);
    
    Logger::println("final intrinsic name: %s", name.c_str());
}

//////////////////////////////////////////////////////////////////////////////////////////

bool mustDefineSymbol(Dsymbol* s)
{
    TemplateInstance* tinst = DtoIsTemplateInstance(s);
    if (tinst)
    {
        if (!opts::singleObj)
            return true;
    
        if (!tinst->emittedInModule)
            tinst->emittedInModule = gIR->dmodule;
        return tinst->emittedInModule == gIR->dmodule;
    }
    
    return s->getModule() == gIR->dmodule;
}

//////////////////////////////////////////////////////////////////////////////////////////

bool needsTemplateLinkage(Dsymbol* s)
{
    return DtoIsTemplateInstance(s) && mustDefineSymbol(s);
}

//////////////////////////////////////////////////////////////////////////////////////////

bool hasUnalignedFields(Type* t)
{
    t = t->toBasetype();
    if (t->ty == Tsarray) {
        assert(t->next->size() % t->next->alignsize() == 0);
        return hasUnalignedFields(t->next);
    } else if (t->ty != Tstruct)
        return false;

    TypeStruct* ts = (TypeStruct*)t;
    if (ts->unaligned)
        return (ts->unaligned == 2);

    StructDeclaration* sym = ts->sym;

    // go through all the fields and try to find something unaligned
    ts->unaligned = 2;
    for (int i = 0; i < sym->fields.dim; i++)
    {
        VarDeclaration* f = (VarDeclaration*)sym->fields.data[i];
        unsigned a = f->type->alignsize() - 1;
        if (((f->offset + a) & ~a) != f->offset)
            return true;
        else if (f->type->toBasetype()->ty == Tstruct && hasUnalignedFields(f->type))
            return true;
    }

    ts->unaligned = 1;
    return false;
}
