//===-- llvmhelpers.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvmhelpers.h"
#include "gen/llvm.h"

#include "mars.h"
#include "init.h"
#include "id.h"
#include "expression.h"
#include "template.h"
#include "module.h"

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
#if LDC_LLVM_VER >= 301
#include "llvm/Transforms/Utils/ModuleUtils.h"
#endif

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
#include "gen/nested.h"
#include "ir/irmodule.h"
#include "gen/llvmcompat.h"

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
    gIR->CreateCallOrInvoke(fn, arg);
}

void DtoDeleteClass(LLValue* inst)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delclass");
    // build args
    LLSmallVector<LLValue*,1> arg;
#if DMDV2
    // druntime wants a pointer to object
    LLValue *ptr = DtoRawAlloca(inst->getType(), 0, "objectPtr");
    DtoStore(inst, ptr);
    inst = ptr;
#endif
    arg.push_back(DtoBitCast(inst, fn->getFunctionType()->getParamType(0), ".tmp"));
    // call
    gIR->CreateCallOrInvoke(fn, arg);
}

void DtoDeleteInterface(LLValue* inst)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delinterface");
    // build args
    LLSmallVector<LLValue*,1> arg;
    arg.push_back(DtoBitCast(inst, fn->getFunctionType()->getParamType(0), ".tmp"));
    // call
    gIR->CreateCallOrInvoke(fn, arg);
}

#if DMDV2

void DtoDeleteArray(DValue* arr)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delarray_t");

    // build args
    LLSmallVector<LLValue*,2> arg;
    arg.push_back(DtoBitCast(arr->getLVal(), fn->getFunctionType()->getParamType(0)));
    arg.push_back(DtoBitCast(DtoTypeInfoOf(arr->type->nextOf()), fn->getFunctionType()->getParamType(1)));

    // call
    gIR->CreateCallOrInvoke(fn, arg);
}

#else

void DtoDeleteArray(DValue* arr)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delarray");

    // build args
    LLSmallVector<LLValue*,2> arg;
    arg.push_back(DtoArrayLen(arr));
    arg.push_back(DtoBitCast(DtoArrayPtr(arr), getVoidPtrType(), ".tmp"));

    // call
    gIR->CreateCallOrInvoke(fn, arg);
}

#endif

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// ALLOCA HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/


llvm::AllocaInst* DtoAlloca(Type* type, const char* name)
{
    LLType* lltype = DtoType(type);
    llvm::AllocaInst* ai = new llvm::AllocaInst(lltype, name, gIR->topallocapoint());
    ai->setAlignment(type->alignsize());
    return ai;
}

llvm::AllocaInst* DtoArrayAlloca(Type* type, unsigned arraysize, const char* name)
{
    LLType* lltype = DtoType(type);
    llvm::AllocaInst* ai = new llvm::AllocaInst(
        lltype, DtoConstUint(arraysize), name, gIR->topallocapoint());
    ai->setAlignment(type->alignsize());
    return ai;
}

llvm::AllocaInst* DtoRawAlloca(LLType* lltype, size_t alignment, const char* name)
{
    llvm::AllocaInst* ai = new llvm::AllocaInst(lltype, name, gIR->topallocapoint());
    if (alignment)
        ai->setAlignment(alignment);
    return ai;
}

LLValue* DtoGcMalloc(LLType* lltype, const char* name)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_allocmemory");
    // parameters
    LLValue *size = DtoConstSize_t(getTypeAllocSize(lltype));
    // call runtime allocator
    LLValue* mem = gIR->CreateCallOrInvoke(fn, size, name).getInstruction();
    // cast
    return DtoBitCast(mem, getPtrToType(lltype), name);
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
    gIR->CreateCallOrInvoke(fn, args);

    // end debug info
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
        targetBB = llvm::BasicBlock::Create(gIR->context(), "label_" + labelname, gIR->topfunc());

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
    LabelStatement* lblstmt = target ? target->isLabelStatement() : 0;
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

void DtoAssign(Loc& loc, DValue* lhs, DValue* rhs, int op)
{
    Logger::println("DtoAssign()");
    LOG_SCOPE;

    Type* t = lhs->getType()->toBasetype();
    Type* t2 = rhs->getType()->toBasetype();

    if (t->ty == Tstruct) {
        DtoAggrCopy(lhs->getLVal(), rhs->getRVal());
    }
    else if (t->ty == Tarray) {
        // lhs is slice
        if (DSliceValue* s = lhs->isSlice()) {
            if (t->nextOf()->toBasetype()->equals(t2)) {
                DtoArrayInit(loc, lhs, rhs, op);
            }
#if DMDV2
            else if (DtoArrayElementType(t)->equals(stripModifiers(t2))) {
                DtoArrayInit(loc, s, rhs, op);
            }
            else if (op != -1 && op != TOKblit && arrayNeedsPostblit(t)) {
                DtoArrayAssign(s, rhs, op);
            }
#endif
            else if (DSliceValue *s2 = rhs->isSlice()) {
                DtoArrayCopySlices(s, s2);
            }
            else {
                DtoArrayCopyToSlice(s, rhs);
            }
        }
        // rhs is slice
        else if (DSliceValue* s = rhs->isSlice()) {
            //assert(s->getType()->toBasetype() == lhs->getType()->toBasetype());
            DtoSetArray(lhs,DtoArrayLen(s),DtoArrayPtr(s));
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
            DtoSetArray(lhs, DtoArrayLen(rhs), DtoArrayPtr(rhs));
        }
    }
    else if (t->ty == Tsarray) {
        // T[n] = T
        if (t->nextOf()->toBasetype()->equals(t2)) {
            DtoArrayInit(loc, lhs, rhs, op);
        }
#if DMDV2
        else if (DtoArrayElementType(t)->equals(stripModifiers(t2))) {
            DtoArrayInit(loc, lhs, rhs, op);
        }
        else if (op != -1 && op != TOKblit && arrayNeedsPostblit(t)) {
            DtoArrayAssign(lhs, rhs, op);
        }
#endif
        // T[n] = T[n]
        else if (DtoType(lhs->getType()) == DtoType(rhs->getType())) {
            DtoStaticArrayCopy(lhs->getLVal(), rhs->getRVal());
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
        if (Logger::enabled()) {
            Logger::cout() << "lhs: " << *l << '\n';
            Logger::cout() << "rhs: " << *r << '\n';
        }
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
        if (Logger::enabled()) {
            Logger::cout() << "lhs: " << *l << '\n';
            Logger::cout() << "rhs: " << *r << '\n';
        }
        LLType* lit = l->getType()->getContainedType(0);
        if (r->getType() != lit) {
            r = DtoCast(loc, rhs, lhs->getType())->getRVal();
            if (Logger::enabled()) {
                Logger::println("Type mismatch, really assigning:");
                LOG_SCOPE
                Logger::cout() << "lhs: " << *l << '\n';
                Logger::cout() << "rhs: " << *r << '\n';
            }
#if 1
            if(r->getType() != lit) // It's wierd but it happens. TODO: try to remove this hack
                r = DtoBitCast(r, lit);
#else
            assert(r->getType() == lit);
#endif
        }
        gIR->ir->CreateStore(r, l);
    }

    DVarValue *var = lhs->isVar();
    VarDeclaration *vd = var ? var->var : 0;
    if (vd)
        DtoDwarfValue(DtoLoad(var->getLVal()), vd);
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
//      NULL VALUE HELPER
////////////////////////////////////////////////////////////////////////////////////////*/

DValue* DtoNullValue(Type* type)
{
    Type* basetype = type->toBasetype();
    TY basety = basetype->ty;
    LLType* lltype = DtoType(basetype);

    // complex, needs to be first since complex are also floating
    if (basetype->iscomplex())
    {
        LLType* basefp = DtoComplexBaseType(basetype);
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
    error("unsupported: null value for %s", type->toChars());
    assert(0);
    return 0;

}


/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
//      CASTING HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/

DValue* DtoCastInt(Loc& loc, DValue* val, Type* _to)
{
    LLType* tolltype = DtoType(_to);

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
            if (isLLVMUnsigned(from) || from->ty == Tbool) {
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
    LLType* tolltype = DtoType(to);

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

    LLType* tolltype = DtoType(to);

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

DValue* DtoCastNull(Loc& loc, DValue* val, Type* to)
{
    Type* totype = to->toBasetype();
    LLType* tolltype = DtoType(to);

    if (totype->ty == Tpointer)
    {
        if (Logger::enabled())
            Logger::cout() << "cast null to pointer: " << *tolltype << '\n';
        LLValue *rval = DtoBitCast(val->getRVal(), tolltype);
        return new DImValue(to, rval);
    }
    if (totype->ty == Tarray)
    {
        if (Logger::enabled())
            Logger::cout() << "cast null to array: " << *tolltype << '\n';
        LLValue *rval = val->getRVal();
        rval = DtoBitCast(rval, DtoType(to->nextOf()->pointerTo()));
        rval = DtoAggrPair(DtoConstSize_t(0), rval, "null_array");
        return new DImValue(to, rval);
    }
    else
    {
        error(loc, "invalid cast from null to '%s'", to->toChars());
        fatal();
    }
}

#if DMDV2
DValue* DtoCastVector(Loc& loc, DValue* val, Type* to)
{
    assert(val->getType()->toBasetype()->ty == Tvector);
    Type* totype = to->toBasetype();
    LLType* tolltype = DtoType(to);
    TypeVector *type = static_cast<TypeVector *>(val->getType()->toBasetype());

    if (totype->ty == Tsarray)
    {
        // If possible, we need to cast only the address of the vector without
        // creating a copy, because, besides the fact that this seem to be the
        // language semantics, DMD rewrites e.g. float4.array to
        // cast(float[4])array.
        if (val->isLVal())
        {
            LLValue* vector = val->getLVal();
            if (Logger::enabled())
            {
                Logger::cout() << "src: " << *vector << "to type: " <<
                    *tolltype << " (casting address)\n";
            }
            return new DVarValue(to, DtoBitCast(vector, getPtrToType(tolltype)));
        }
        else
        {
            LLValue* vector = val->getRVal();
            if (Logger::enabled())
            {
                Logger::cout() << "src: " << *vector << "to type: " <<
                    *tolltype << " (creating temporary)\n";
            }
            LLValue *array = DtoAlloca(to);

            TypeSArray *st = static_cast<TypeSArray*>(totype);

            for (int i = 0, n = st->dim->toInteger(); i < n; ++i) {
                LLValue *lelem = DtoExtractElement(vector, i);
                DImValue elem(type->elementType(), lelem);
                lelem = DtoCast(loc, &elem, to->nextOf())->getRVal();
                DtoStore(lelem, DtoGEPi(array, 0, i));
            }

            return new DImValue(to, array);
        }
    }
    else if (totype->ty == Tvector && to->size() == val->getType()->size())
    {
        return new DImValue(to, DtoBitCast(val->getRVal(), tolltype));
    }
    else
    {
        error(loc, "invalid cast from '%s' to '%s'", val->getType()->toChars(), to->toChars());
        fatal();
    }
}
#endif

DValue* DtoCast(Loc& loc, DValue* val, Type* to)
{
    Type* fromtype = val->getType()->toBasetype();
    Type* totype = to->toBasetype();

#if DMDV2
    if (fromtype->ty == Taarray)
        fromtype = static_cast<TypeAArray*>(fromtype)->getImpl()->type;
    if (totype->ty == Taarray)
        totype = static_cast<TypeAArray*>(totype)->getImpl()->type;
#endif

    if (fromtype->equals(totype))
        return val;

    Logger::println("Casting from '%s' to '%s'", fromtype->toChars(), to->toChars());
    LOG_SCOPE;

#if DMDV2
    if (fromtype->ty == Tvector) {
        return DtoCastVector(loc, val, to);
    }
    else
#endif
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
    else if (fromtype->ty == Tnull) {
        return DtoCastNull(loc, val, to);
    }
    else if (fromtype->ty == totype->ty) {
        return val;
    } else {
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
        // assert(!val->isLVal()); TODO: what is it needed for?
        assert(DtoType(to) == DtoType(to));
        return new DImValue(to, val->getRVal());
    }
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
//      TEMPLATE HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/

TemplateInstance* DtoIsTemplateInstance(Dsymbol* s, bool checkLiteralOwner)
{
    if (!s) return NULL;
    if (s->isTemplateInstance() && !s->isTemplateMixin())
        return s->isTemplateInstance();
#if DMDV2
    if (FuncLiteralDeclaration* fld = s->isFuncLiteralDeclaration())
    {
        if (checkLiteralOwner && fld->owningTemplate)
            return fld->owningTemplate;
    }
#endif
    if (s->parent)
        return DtoIsTemplateInstance(s->parent, checkLiteralOwner);
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

    // build the initializer
    LLConstant* initVal = DtoConstInitializer(vd->loc, vd->type, vd->init);

    // set the initializer if appropriate
    IrGlobal* glob = vd->ir.irGlobal;
    llvm::GlobalVariable* gvar = llvm::cast<llvm::GlobalVariable>(glob->value);

    //if (LLStructType *st = isaStruct(glob->type)) {
    //    st->setBody(initVal);
    //}

    assert(!glob->constInit);
    glob->constInit = initVal;

    // assign the initializer
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
        DtoDwarfGlobalVariable(gvar, vd);
    }
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
//      DECLARATION EXP HELPER
////////////////////////////////////////////////////////////////////////////////////////*/

// TODO: Merge with DtoRawVarDeclaration!
void DtoVarDeclaration(VarDeclaration* vd)
{
    assert(!vd->isDataseg() && "Statics/globals are handled in DtoDeclarationExp.");
    assert(!vd->aliassym && "Aliases are handled in DtoDeclarationExp.");

    Logger::println("vdtype = %s", vd->type->toChars());

#if DMDV2
    if (vd->nestedrefs.dim)
#else
    if (vd->nestedref)
#endif
    {
        Logger::println("has nestedref set (referenced by nested function/delegate)");
        assert(vd->ir.irLocal && "irLocal is expected to be already set by DtoCreateNestedContext");
    }

    if(vd->ir.irLocal)
    {
        // Nothing to do if it has already been allocated.
    }
#if DMDV2
    /* Named Return Value Optimization (NRVO):
        T f(){
            T ret;        // &ret == hidden pointer
            ret = ...
            return ret;    // NRVO.
        }
    */
    else if (gIR->func()->retArg && gIR->func()->decl->nrvo_can && gIR->func()->decl->nrvo_var == vd) {
        assert(!isSpecialRefVar(vd) && "Can this happen?");
        vd->ir.irLocal = new IrLocal(vd);
        vd->ir.irLocal->value = gIR->func()->retArg;
    }
#endif
    // normal stack variable, allocate storage on the stack if it has not already been done
    else {
        vd->ir.irLocal = new IrLocal(vd);

#if DMDV2
        /* NRVO again:
            T t = f();    // t's memory address is taken hidden pointer
        */
        ExpInitializer *ei = 0;
        if (vd->type->toBasetype()->ty == Tstruct && vd->init &&
            !!(ei = vd->init->isExpInitializer()))
        {
            if (ei->exp->op == TOKconstruct) {
                AssignExp *ae = static_cast<AssignExp*>(ei->exp);
                if (ae->e2->op == TOKcall) {
                    CallExp *ce = static_cast<CallExp *>(ae->e2);
                    TypeFunction *tf = static_cast<TypeFunction *>(ce->e1->type->toBasetype());
                    if (tf->ty == Tfunction && tf->fty.arg_sret) {
                        LLValue* const val = ce->toElem(gIR)->getLVal();
                        if (isSpecialRefVar(vd))
                        {
                            vd->ir.irLocal->value = DtoAlloca(
                                vd->type->pointerTo(), vd->toChars());
                            DtoStore(val, vd->ir.irLocal->value);
                        }
                        else
                        {
                            vd->ir.irLocal->value = val;
                        }
                        goto Lexit;
                    }
                }
            }
        }
#endif

        Type* type = isSpecialRefVar(vd) ? vd->type->pointerTo() : vd->type;
        LLType* lltype = DtoType(type);

        llvm::Value* allocainst;
        if(gDataLayout->getTypeSizeInBits(lltype) == 0)
            allocainst = llvm::ConstantPointerNull::get(getPtrToType(lltype));
        else
            allocainst = DtoAlloca(type, vd->toChars());

        vd->ir.irLocal->value = allocainst;

        DtoDwarfLocalVariable(allocainst, vd);
    }

    if (Logger::enabled())
        Logger::cout() << "llvm value for decl: " << *vd->ir.irLocal->value << '\n';

    DtoInitializer(vd->ir.irLocal->value, vd->init); // TODO: Remove altogether?

#if DMDV2
Lexit:
    /* Mark the point of construction of a variable that needs to be destructed.
     */
    if (vd->edtor && !vd->noscope)
    {
        // Put vd on list of things needing destruction
        gIR->varsInScope().push_back(vd);
    }
#endif
}

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
            DtoVarDeclaration(vd);
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
    else if (declaration->isAliasDeclaration())
    {
        Logger::println("AliasDeclaration - no work");
        // do nothing
    }
    // enum
    else if (declaration->isEnumDeclaration())
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
            for (unsigned i=0; i < d->dim; ++i)
            {
                DtoDeclarationExp(static_cast<Dsymbol*>(d->data[i]));
            }
    }
    // mixin declaration
    else if (TemplateMixin* m = declaration->isTemplateMixin())
    {
        Logger::println("TemplateMixin");
        for (unsigned i=0; i < m->members->dim; ++i)
        {
            Dsymbol* mdsym = static_cast<Dsymbol*>(m->members->data[i]);
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
        for (unsigned i=0; i < tupled->objects->dim; ++i)
        {
            DsymbolExp* exp = static_cast<DsymbolExp*>(tupled->objects->data[i]);
            DtoDeclarationExp(exp->s);
        }
    }
    // template
    else if (declaration->isTemplateDeclaration())
    {
        Logger::println("TemplateDeclaration");
        // do nothing
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
    if (!addr && (!var->ir.irLocal || !var->ir.irLocal->value))
    {
        addr = DtoAlloca(var->type, var->toChars());
        // add debug info
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

LLType* DtoConstInitializerType(Type* type, Initializer* init)
{
    if (type->ty == Ttypedef) {
        TypeTypedef *td = static_cast<TypeTypedef*>(type);
        if (td->sym->init)
            return DtoConstInitializerType(td->sym->basetype, td->sym->init);
    }

    type = type->toBasetype();
    if (type->ty == Tsarray)
    {
        if (!init)
        {
            TypeSArray *tsa = static_cast<TypeSArray*>(type);
            LLType *llnext = DtoConstInitializerType(type->nextOf(), init);
            return LLArrayType::get(llnext, tsa->dim->toUInteger());
        }
        else if (ArrayInitializer* ai = init->isArrayInitializer())
        {
            return DtoConstArrayInitializerType(ai);
        }
    }
    else if (type->ty == Tstruct)
    {
        if (!init)
        {
        LdefaultInit:
            TypeStruct *ts = static_cast<TypeStruct*>(type);
            DtoResolveStruct(ts->sym);
            return ts->sym->ir.irStruct->getDefaultInit()->getType();
        }
        else if (ExpInitializer* ex = init->isExpInitializer())
        {
            if (ex->exp->op == TOKstructliteral) {
                StructLiteralExp* le = static_cast<StructLiteralExp*>(ex->exp);
                if (!le->constType)
                    le->constType = LLStructType::create(gIR->context(), std::string(type->toChars()) + "_init");
                return le->constType;
            } else if (ex->exp->op == TOKvar) {
                if (static_cast<VarExp*>(ex->exp)->var->isStaticStructInitDeclaration())
                    goto LdefaultInit;
            }
        }
        else if (StructInitializer* si = init->isStructInitializer())
        {
            if (!si->ltype)
                si->ltype = LLStructType::create(gIR->context(), std::string(type->toChars()) + "_init");
            return si->ltype;
        }
    }

    return DtoTypeNotVoid(type);
}

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
        _init = DtoConstExpInit(loc, type, ex->exp);
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
        LLType* ty = DtoTypeNotVoid(type);
        _init = LLConstant::getNullValue(ty);
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
    else if (init->isArrayInitializer())
    {
        // TODO: do nothing ?
    }
    else if (init->isVoidInitializer())
    {
        // do nothing
    }
    else if (init->isStructInitializer()) {
        // TODO: again nothing ?
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
    LLType* dstTy = DtoType(base);
    if (Logger::enabled())
        Logger::cout() << "final llvm type requested: " << *dstTy << '\n';

    LLConstant* val = exp->toConstElem(gIR);

    Type* expbase = stripModifiers(exp->type->toBasetype());
    Logger::println("expbase: %s", expbase->toChars());
    Type* t = base->toBasetype();

    LLSmallVector<size_t, 4> dims;

    while(1)
    {
        Logger::println("t: %s", t->toChars());
        if (t->equals(expbase))
            break;
        assert(t->ty == Tsarray);
        TypeSArray* tsa = static_cast<TypeSArray*>(t);
        dims.push_back(tsa->dim->toInteger());
        assert(t->nextOf());
        t = stripModifiers(t->nextOf()->toBasetype());
    }

    size_t i = dims.size();
    assert(i);

    std::vector<LLConstant*> inits;
    while (i--)
    {
        LLArrayType* arrty = LLArrayType::get(val->getType(), dims[i]);
        inits.clear();
        inits.insert(inits.end(), dims[i], val);
        val = LLConstantArray::get(arrty, inits);
    }

    return val;
}

LLConstant* DtoConstExpInit(Loc loc, Type* type, Expression* exp)
{
#if DMDV2
    Type* expbase = stripModifiers(exp->type->toBasetype())->merge();
    Type* base = stripModifiers(type->toBasetype())->merge();
#else
    Type* expbase = exp->type->toBasetype();
    Type* base = type->toBasetype();
#endif

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
#if DMDV2
        else if (base->ty == Tvector)
        {
            LLConstant* val = exp->toConstElem(gIR);
            TypeVector* tv = (TypeVector*)base;
#if LDC_LLVM_VER == 300
            std::vector<LLConstant*> Elts(tv->size(loc), val);
            return llvm::ConstantVector::get(Elts);
#else
            return llvm::ConstantVector::getSplat(tv->size(loc), val);
#endif
        }
#endif
        else
        {
            error(loc, "LDC internal error: cannot yet convert default initializer %s of type %s to %s",
                exp->toChars(), exp->type->toChars(), type->toChars());
            fatal();
        }
        assert(0);
    }

    return exp->toConstElem(gIR);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoTypeInfoOf(Type* type, bool base)
{
    type = type->merge2(); // needed.. getTypeInfo does the same
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
    Type* T = static_cast<Type*>(ti->tdtypes.data[0]);

    char prefix = T->isreal() ? 'f' : T->isintegral() ? 'i' : 0;
    if (!prefix) {
        ti->error("has invalid template parameter for intrinsic: %s", T->toChars());
        fatal(); // or LLVM asserts
    }

    llvm::Type *dtype(DtoType(T));
    char tmp[21]; // probably excessive, but covers a uint64_t
    sprintf(tmp, "%lu", static_cast<unsigned long>(gDataLayout->getTypeSizeInBits(dtype)));

    // replace # in name with bitsize
    name = td->intrinsicName;

    std::string needle("#");
    size_t pos;
    while(std::string::npos != (pos = name.find(needle))) {
        if (pos > 0 && name[pos-1] == prefix) {
            // Check for special PPC128 double
            if (dtype->isPPC_FP128Ty()) {
                name.insert(pos-1, "ppc");
                pos += 3;
            }
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
        if (fd->semanticRun < PASSsemantic3)
            return false;

        if (fd->isArrayOp == 1)
            return true;

        if (global.params.useAvailableExternally && fd->availableExternally) {
            // Emit extra functions if we're inlining.
            // These will get available_externally linkage,
            // so they shouldn't end up in object code.

            assert(fd->type->ty == Tfunction);
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
                // FIXME: Should be canInline(true, false, false), but this
                // causes a misoptimization in druntime on x86, likely due to
                // an LLVM bug.
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

    TemplateInstance* tinst = DtoIsTemplateInstance(s, true);
    if (tinst)
    {
        if (!global.params.singleObj)
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

    TypeStruct* ts = static_cast<TypeStruct*>(t);
    if (ts->unaligned)
        return (ts->unaligned == 2);

    StructDeclaration* sym = ts->sym;

    // go through all the fields and try to find something unaligned
    ts->unaligned = 2;
    for (unsigned i = 0; i < sym->fields.dim; i++)
    {
        VarDeclaration* f = static_cast<VarDeclaration*>(sym->fields.data[i]);
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
    // also handle arbitrary "by-value" opaques nested inside aggregates.
    LLType* T = DtoType(type);
    if (!T->isSized())
    {
        return offset;
    }

    // then we check against the llvm alignment
    size_t alignsize2 = gDataLayout->getABITypeAlignment(T);

    // if it differs we need to insert manual padding as well
    if (alignsize != alignsize2)
    {
        // FIXME: this assert fails on std.typecons
        //assert(alignsize > alignsize2 && "this is not good, the D and LLVM "
        //    "type alignments differ, but LLVM's is bigger! This will break "
        //    "aggregate type mapping");
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
    if (type->ty == Tfunction)
        return type;
    return type->castMod(0);
#else
    return type;
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* makeLValue(Loc& loc, DValue* value)
{
    Type* valueType = value->getType();
    bool needsMemory;
    LLValue* valuePointer;
    if (value->isIm()) {
        valuePointer = value->getRVal();
        needsMemory = !DtoIsPassedByRef(valueType);
    }
    else if (value->isVar()) {
        valuePointer = value->getLVal();
        needsMemory = false;
    }
    else if (value->isConst()) {
        valuePointer = value->getRVal();
        needsMemory = true;
    }
    else {
        valuePointer = DtoAlloca(valueType, ".makelvaluetmp");
        DVarValue var(valueType, valuePointer);
        DtoAssign(loc, &var, value);
        needsMemory = false;
    }

    if (needsMemory) {
        LLValue* tmp = DtoAlloca(valueType, ".makelvaluetmp");
        DtoStore(valuePointer, tmp);
        valuePointer = tmp;
    }

    return valuePointer;
}

//////////////////////////////////////////////////////////////////////////////////////////

#if DMDV2
void callPostblit(Loc &loc, Expression *exp, LLValue *val)
{

    Type *tb = exp->type->toBasetype();
    if ((exp->op == TOKvar || exp->op == TOKdotvar || exp->op == TOKstar || exp->op == TOKthis || exp->op == TOKindex) &&
        tb->ty == Tstruct)
    {   StructDeclaration *sd = static_cast<TypeStruct *>(tb)->sym;
        if (sd->postblit)
        {
            FuncDeclaration *fd = sd->postblit;
            if (fd->storage_class & STCdisable)
                fd->toParent()->error(loc, "is not copyable because it is annotated with @disable");
            fd->codegen(Type::sir);
            Expressions args;
            DFuncValue dfn(fd, fd->ir.irFunc->func, val);
            DtoCallFunction(loc, Type::basic[Tvoid], &dfn, &args);
        }
    }
}
#endif

//////////////////////////////////////////////////////////////////////////////////////////

bool isSpecialRefVar(VarDeclaration* vd)
{
    return (vd->storage_class & STCref) && (vd->storage_class & STCforeach);
}

//////////////////////////////////////////////////////////////////////////////////////////

bool isLLVMUnsigned(Type* t)
{
    return t->isunsigned() || t->ty == Tpointer;
}

//////////////////////////////////////////////////////////////////////////////////////////

void printLabelName(std::ostream& target, const char* func_mangle, const char* label_name)
{
    target << gTargetMachine->getMCAsmInfo()->getPrivateGlobalPrefix() <<
        func_mangle << "_" << label_name;
}

//////////////////////////////////////////////////////////////////////////////////////////

void AppendFunctionToLLVMGlobalCtorsDtors(llvm::Function* func, const uint32_t priority, const bool isCtor)
{
    if (isCtor)
        llvm::appendToGlobalCtors(*gIR->module, func, priority);
    else
        llvm::appendToGlobalDtors(*gIR->module, func, priority);
}

//////////////////////////////////////////////////////////////////////////////////////////

void tokToIcmpPred(TOK op, bool isUnsigned, llvm::ICmpInst::Predicate* outPred, llvm::Value** outConst)
{
    switch(op)
    {
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
