#include "gen/llvmhelpers.h"
#include "gen/llvm.h"
#include "llvm/Target/TargetMachineRegistry.h"

#include "mars.h"
#include "init.h"
#include "id.h"
#include "expression.h"
#include "template.h"
#include "module.h"

#include "gen/tollvm.h"
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
#include "gen/nested.h"
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


llvm::AllocaInst* DtoAlloca(Type* type, const char* name)
{
    const llvm::Type* lltype = DtoType(type);
    llvm::AllocaInst* ai = new llvm::AllocaInst(lltype, name, gIR->topallocapoint());
    ai->setAlignment(type->alignsize());
    return ai;
}

llvm::AllocaInst* DtoArrayAlloca(Type* type, unsigned arraysize, const char* name)
{
    const llvm::Type* lltype = DtoType(type);
    llvm::AllocaInst* ai = new llvm::AllocaInst(
        lltype, DtoConstUint(arraysize), name, gIR->topallocapoint());
    ai->setAlignment(type->alignsize());
    return ai;
}

llvm::AllocaInst* DtoRawAlloca(const llvm::Type* lltype, size_t alignment, const char* name)
{
    llvm::AllocaInst* ai = new llvm::AllocaInst(lltype, name, gIR->topallocapoint());
    if (alignment)
        ai->setAlignment(alignment);
    return ai;
}


/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// ASSERT HELPER
////////////////////////////////////////////////////////////////////////////////////////*/

void DtoAssert(Module* M, Loc loc, DValue* msg)
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
    const char* cur_file = M->srcfile->name->toChars();
    if (loc.filename && strcmp(loc.filename, cur_file) != 0)
    {
        args.push_back(DtoConstString(loc.filename));
    }
    else
    {
        IrModule* irmod = getIrModule(M);
        args.push_back(DtoLoad(irmod->fileName));
    }

    // line param
    LLConstant* c = DtoConstUint(loc.linnum);
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
void DtoGoto(Loc loc, Identifier* target, TryFinallyStatement* sourceFinally)
{
    assert(!gIR->scopereturned());

    LabelStatement* lblstmt = DtoLabelStatement(target);
    if(!lblstmt) {
        error(loc, "the label %s does not exist", target->toChars());
        fatal();
    }

    // if the target label is inside inline asm, error
    if(lblstmt->asmLabel) {
        error(loc, "cannot goto to label %s inside an inline asm block", target->toChars());
        fatal();
    }

    // find target basic block
    std::string labelname = gIR->func()->gen->getScopedLabelName(target->toChars());
    llvm::BasicBlock*& targetBB = gIR->func()->gen->labelToBB[labelname];
    if (targetBB == NULL)
        targetBB = llvm::BasicBlock::Create("label_" + labelname, gIR->topfunc());

    // emit code for finallys between goto and label
    DtoEnclosingHandlers(loc, lblstmt);

    // goto into finally blocks is forbidden by the spec
    // but should work fine
    if(lblstmt->enclosingFinally != sourceFinally) {
        error(loc, "spec disallows goto into or out of finally block");
        fatal();
    }

    llvm::BranchInst::Create(targetBB, gIR->scopebb());
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// TRY-FINALLY, VOLATILE AND SYNCHRONIZED HELPER
////////////////////////////////////////////////////////////////////////////////////////*/

void EnclosingSynchro::emitCode(IRState * p)
{
    if (s->exp)
        DtoLeaveMonitor(s->exp->toElem(p)->getRVal());
    else
        DtoLeaveCritical(s->llsync);
}

////////////////////////////////////////////////////////////////////////////////////////

void EnclosingVolatile::emitCode(IRState * p)
{
    // store-load barrier
    DtoMemoryBarrier(false, false, true, false);
}

////////////////////////////////////////////////////////////////////////////////////////

void EnclosingTryFinally::emitCode(IRState * p)
{
    if (tf->finalbody)
    {
        llvm::BasicBlock* oldpad = p->func()->gen->landingPad;
        p->func()->gen->landingPad = landingPad;
        tf->finalbody->toIR(p);
        p->func()->gen->landingPad = oldpad;
    }
}

////////////////////////////////////////////////////////////////////////////////////////

void DtoEnclosingHandlers(Loc loc, Statement* target)
{
    // labels are a special case: they are not required to enclose the current scope
    // for them we use the enclosing scope handler as a reference point
    LabelStatement* lblstmt = dynamic_cast<LabelStatement*>(target);
    if (lblstmt)
        target = lblstmt->enclosingScopeExit;

    // figure out up until what handler we need to emit
    FuncGen::TargetScopeVec::reverse_iterator targetit = gIR->func()->gen->targetScopes.rbegin();
    FuncGen::TargetScopeVec::reverse_iterator it_end = gIR->func()->gen->targetScopes.rend();
    while(targetit != it_end) {
        if (targetit->s == target) {
            break;
        }
        ++targetit;
    }

    if (target && targetit == it_end) {
        if (lblstmt)
            error(loc, "cannot goto into try, volatile or synchronized statement at %s", target->loc.toChars());
        else
            error(loc, "internal error, cannot find jump path to statement at %s", target->loc.toChars());
        return;
    }

    //
    // emit code for enclosing handlers
    //

    // since the labelstatements possibly inside are private
    // and might already exist push a label scope
    gIR->func()->gen->pushUniqueLabelScope("enclosing");
    FuncGen::TargetScopeVec::reverse_iterator it = gIR->func()->gen->targetScopes.rbegin();
    while (it != targetit) {
        if (it->enclosinghandler)
            it->enclosinghandler->emitCode(gIR);
        ++it;
    }
    gIR->func()->gen->popLabelScope();
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
        LLValue* dst = lhs->getLVal();
        LLValue* src = DtoCast(loc, rhs, lhs->getType())->getRVal();
        DtoStore(src, dst);
    }
    else {
        LLValue* l = lhs->getLVal();
        LLValue* r = rhs->getRVal();
        if (Logger::enabled())
            Logger::cout() << "assign\nlhs: " << *l << "rhs: " << *r << '\n';
        const LLType* lit = l->getType()->getContainedType(0);
        if (r->getType() != lit) {
            r = DtoCast(loc, rhs, lhs->getType())->getRVal();
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
        LLValue* res = DtoAggrPair(DtoType(type), gIR->context().getNullValue(basefp), gIR->context().getNullValue(basefp));
        return new DImValue(type, res);
    }
    // integer, floating, pointer and class have no special representation
    else if (basetype->isintegral() || basetype->isfloating() || basety == Tpointer || basety == Tclass)
    {
        return new DConstValue(type, gIR->context().getNullValue(lltype));
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
        return new DNullValue(type, gIR->context().getNullValue(lltype));
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
        LLValue* zero = gIR->context().getConstantInt(rval->getType(), 0, false);
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
        LLValue* zero = gIR->context().getNullValue(src->getType());
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
        LLValue* zero = gIR->context().getNullValue(rval->getType());
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
    Type* totype = to->toBasetype();
    if (fromtype->equals(totype))
        return val;

    Logger::println("Casting from '%s' to '%s'", fromtype->toChars(), to->toChars());
    LOG_SCOPE;

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

void DtoConstInitGlobal(VarDeclaration* vd)
{
    vd->codegen(Type::sir);

    if (vd->ir.initialized) return;
    vd->ir.initialized = gIR->dmodule;

    Logger::println("DtoConstInitGlobal(%s) @ %s", vd->toChars(), vd->loc.toChars());
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
#if 0
            Logger::cout() << "init:   " << *initVal << '\n';
#endif
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
            vd->codegen(Type::sir);
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
                
                DtoNestedInit(vd);
            }
            // normal stack variable, allocate storage on the stack if it has not already been done
            else if(!vd->ir.irLocal) {
                const LLType* lltype = DtoType(vd->type);

                llvm::Value* allocainst;
                if(gTargetData->getTypeSizeInBits(lltype) == 0) 
                    allocainst = llvm::ConstantPointerNull::get(getPtrToType(lltype));
                else
                    allocainst = DtoAlloca(vd->type, vd->toChars());

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
        s->codegen(Type::sir);
    }
    // function declaration
    else if (FuncDeclaration* f = declaration->isFuncDeclaration())
    {
        Logger::println("FuncDeclaration");
        f->codegen(Type::sir);
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
        e->codegen(Type::sir);
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
        // choose the right set in case this is a conditional declaration
        Array *d = a->include(NULL, NULL);
        if (d)
            for (int i=0; i < d->dim; ++i)
            {
                DtoDeclarationExp((Dsymbol*)d->data[i]);
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
LLValue* DtoRawVarDeclaration(VarDeclaration* var, LLValue* addr)
{
    // we don't handle globals with this one
    assert(!var->isDataseg());

    // we don't handle aliases either
    assert(!var->aliassym);
    
    // alloca if necessary
    LLValue* allocaval = NULL;
    if (!addr && (!var->ir.irLocal || !var->ir.irLocal->value))
    {
        addr = DtoAlloca(var->type, var->toChars());
        
        // add debug info
        if (global.params.symdebug)
            DtoDwarfLocalVariable(addr, var);
    }
        
    // referenced by nested function?
#if DMDV2
    if (var->nestedrefs.dim)
#else
    if (var->nestedref)
#endif
    {
        assert(var->ir.irLocal);
        if(!var->ir.irLocal->value)
        {
            assert(addr);
            var->ir.irLocal->value = addr;
        }
        else
            assert(!addr || addr == var->ir.irLocal->value);

        DtoNestedInit(var);
    }
    // normal local variable
    else
    {
        // if this already has storage, it must've been handled already
        if (var->ir.irLocal && var->ir.irLocal->value) {
            if (addr && addr != var->ir.irLocal->value) {
                // This can happen, for example, in scope(exit) blocks which
                // are translated to IR multiple times.
                // That *should* only happen after the first one is completely done
                // though, so just set the address.
                IF_LOG {
                    Logger::println("Replacing LLVM address of %s", var->toChars());
                    LOG_SCOPE;
                    Logger::cout() << "Old val: " << *var->ir.irLocal->value << '\n';
                    Logger::cout() << "New val: " << *addr << '\n';
                }
                var->ir.irLocal->value = addr;
            }
            return addr;
        }

        assert(!var->ir.isSet());
        assert(addr);
        var->ir.irLocal = new IrLocal(var);
        var->ir.irLocal->value = addr;
    }

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
        si->ad->codegen(Type::sir);
        return si->ad->ir.irStruct->createStructInitializer(si);
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
        _init = gIR->context().getNullValue(ty);
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
#if DMDV2
    // FIXME: this is probably wrong, but it makes druntime's genobj.d compile!
    type = type->mutableOf()->merge(); // needed.. getTypeInfo does the same
#else
    type = type->merge(); // needed.. getTypeInfo does the same
#endif
    type->getTypeInfo(NULL);
    TypeInfoDeclaration* tidecl = type->vtinfo;
    assert(tidecl);
    tidecl->codegen(Type::sir);
    assert(tidecl->ir.irGlobal != NULL);
    assert(tidecl->ir.irGlobal->value != NULL);
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

    char prefix = T->isreal() ? 'f' : T->isintegral() ? 'i' : 0;
    if (!prefix) {
        ti->error("has invalid template parameter for intrinsic: %s", T->toChars());
        fatal(); // or LLVM asserts
    }

    char tmp[21]; // probably excessive, but covers a uint64_t
    sprintf(tmp, "%lu", (unsigned long) gTargetData->getTypeSizeInBits(DtoType(T)));
    
    // replace # in name with bitsize
    name = td->intrinsicName;

    std::string needle("#");
    size_t pos;
    while(std::string::npos != (pos = name.find(needle))) {
        if (pos > 0 && name[pos-1] == prefix) {
            // Properly prefixed, insert bitwidth.
            name.replace(pos, 1, tmp);
        } else {
            if (pos && (name[pos-1] == 'i' || name[pos-1] == 'f')) {
                // Wrong type character.
                ti->error("has invalid parameter type for intrinsic %s: %s is not a%s type",
                    name.c_str(), T->toChars(),
                    (name[pos-1] == 'i' ? "n integral" : " floating-point"));
            } else {
                // Just plain wrong. (Error in declaration, not instantiation)
                td->error("has an invalid intrinsic name: %s", name.c_str());
            }
            fatal(); // or LLVM asserts
        }
    }
    
    Logger::println("final intrinsic name: %s", name.c_str());
}

//////////////////////////////////////////////////////////////////////////////////////////

bool mustDefineSymbol(Dsymbol* s)
{
    if (FuncDeclaration* fd = s->isFuncDeclaration())
    {
        // we can't (and probably shouldn't?) define functions 
        // that weren't semantic3'ed
        if (fd->semanticRun < 4)
            return false;

	if (fd->isArrayOp)
            return true;

        if (global.params.useAvailableExternally && fd->availableExternally) {
            // Emit extra functions if we're inlining.
            // These will get available_externally linkage,
            // so they shouldn't end up in object code.
            
            assert(fd->type->ty == Tfunction);
            TypeFunction* tf = (TypeFunction*) fd->type;
            // * If we define extra static constructors, static destructors
            //   and unittests they'll get registered to run, and we won't
            //   be calling them directly anyway.
            // * If it's a large function, don't emit it unnecessarily.
            //   Use DMD's canInline() to determine whether it's large.
            //   inlineCost() members have been changed to pay less attention
            //   to DMDs limitations, but still have some issues. The most glaring
            //   offenders are any kind of control flow statements other than
            //   'if' and 'return'.
            if (   !fd->isStaticCtorDeclaration()
                && !fd->isStaticDtorDeclaration()
                && !fd->isUnitTestDeclaration()
                && fd->canInline(true))
            {
                return true;
            }
            
            // This was only semantic'ed for inlining checks.
            // We won't be inlining this, so we only need to emit a declaration.
            return false;
        }
    }

    // Inlining checks may create some variable and class declarations
    // we don't need to emit.
    if (global.params.useAvailableExternally)
    {
        if (VarDeclaration* vd = s->isVarDeclaration())
            if (vd->availableExternally)
                return false;

        if (ClassDeclaration* cd = s->isClassDeclaration())
            if (cd->availableExternally)
                return false;
    }

    TemplateInstance* tinst = DtoIsTemplateInstance(s);
    if (tinst)
    {
        if (!opts::singleObj)
            return true;
    
        if (!tinst->emittedInModule)
        {
            gIR->seenTemplateInstances.insert(tinst);
            tinst->emittedInModule = gIR->dmodule;
        }
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
        assert(t->nextOf()->size() % t->nextOf()->alignsize() == 0);
        return hasUnalignedFields(t->nextOf());
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

//////////////////////////////////////////////////////////////////////////////////////////

IrModule * getIrModule(Module * M)
{
    if (M == NULL)
        M = gIR->func()->decl->getModule();
    assert(M && "null module");
    if (!M->ir.irModule)
        M->ir.irModule = new IrModule(M, M->srcfile->toChars());
    return M->ir.irModule;
}

//////////////////////////////////////////////////////////////////////////////////////////

size_t realignOffset(size_t offset, Type* type)
{
    size_t alignsize = type->alignsize();
    size_t alignedoffset = (offset + alignsize - 1) & ~(alignsize - 1);

    // if the aligned offset already matches the input offset
    // don't waste time checking things are ok!
    if (alignedoffset == offset)
        return alignedoffset;

    // we cannot get the llvm alignment if the type is still opaque, this can happen in some
    // forward reference situations, so when this happens we fall back to manual padding.
    const llvm::Type* T = DtoType(type);
    if (llvm::isa<llvm::OpaqueType>(T))
    {
        return offset;
    }

    // then we check against the llvm alignment
    size_t alignsize2 = gTargetData->getABITypeAlignment(T);

    // if it differs we need to insert manual padding as well
    if (alignsize != alignsize2)
    {
        assert(alignsize > alignsize2 && "this is not good, the D and LLVM "
            "type alignments differ, but LLVM's is bigger! This will break "
            "aggregate type mapping");
        // don't try and align the offset, and let the mappers pad 100% manually
        return offset;
    }

    // ok, we're good, llvm will align properly!
    return alignedoffset;
}

//////////////////////////////////////////////////////////////////////////////////////////

Type * stripModifiers( Type * type )
{
#if DMDV2
	Type *t = type;
	while (t->mod)
	{
		switch (t->mod)
		{
			case MODconst:
				t = type->cto;
				break;
			case MODshared:
				t = type->sto;
				break;
			case MODinvariant:
				t = type->ito;
				break;
			case MODshared | MODconst:
				t = type->scto;
				break;
			default:
				assert(0 && "Unhandled type modifier");
		}

		if (!t)
		{
			unsigned sz = type->sizeTy[type->ty];
			t = (Type *)malloc(sz);
			memcpy(t, type, sz);
			t->mod = 0;
			t->deco = NULL;
			t->arrayof = NULL;
			t->pto = NULL;
			t->rto = NULL;
			t->cto = NULL;
			t->ito = NULL;
			t->sto = NULL;
			t->scto = NULL;
			t->vtinfo = NULL;
			t = t->merge();

			t->fixTo(type);
			switch (type->mod)
			{
			    case MODconst:
				t->cto = type;
				break;

			    case MODinvariant:
				t->ito = type;
				break;

			    case MODshared:
				t->sto = type;
				break;

			    case MODshared | MODconst:
				t->scto = type;
				break;

			    default:
				assert(0);
			}
		}
	}
	return t;
#else
	return type;
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////
