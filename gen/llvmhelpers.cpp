//===-- llvmhelpers.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvmhelpers.h"
#include "expression.h"
#include "id.h"
#include "init.h"
#include "mars.h"
#include "module.h"
#include "template.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/complex.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmcompat.h"
#include "gen/logger.h"
#include "gen/nested.h"
#include "gen/pragma.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "gen/typeinf.h"
#include "gen/abi.h"
#include "ir/irmodule.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <stack>

#if LDC_LLVM_VER >= 302
#include "llvm/Support/CommandLine.h"

llvm::cl::opt<llvm::GlobalVariable::ThreadLocalMode> clThreadModel("fthread-model",
    llvm::cl::desc("Thread model"),
    llvm::cl::init(llvm::GlobalVariable::GeneralDynamicTLSModel),
    llvm::cl::values(
        clEnumValN(llvm::GlobalVariable::GeneralDynamicTLSModel, "global-dynamic",
                   "Global dynamic TLS model (default)"),
        clEnumValN(llvm::GlobalVariable::LocalDynamicTLSModel, "local-dynamic",
                   "Local dynamic TLS model"),
        clEnumValN(llvm::GlobalVariable::InitialExecTLSModel, "initial-exec",
                   "Initial exec TLS model"),
        clEnumValN(llvm::GlobalVariable::LocalExecTLSModel, "local-exec",
                   "Local exec TLS model"),
        clEnumValEnd));
#endif

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
    return DtoBitCast(mem, getPtrToType(i1ToI8(DtoType(newtype))), ".gc_mem");
}

void DtoDeleteMemory(LLValue* ptr)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delmemory");
    // build args
    LLValue* arg[] = { DtoBitCast(ptr, getVoidPtrType(), ".tmp") };
    // call
    gIR->CreateCallOrInvoke(fn, arg);
}

void DtoDeleteClass(LLValue* inst)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delclass");
    // druntime wants a pointer to object
    LLValue *ptr = DtoRawAlloca(inst->getType(), 0, "objectPtr");
    DtoStore(inst, ptr);
    inst = ptr;
    // build args
    LLValue* arg[] = {
        DtoBitCast(inst, fn->getFunctionType()->getParamType(0), ".tmp")
    };
    // call
    gIR->CreateCallOrInvoke(fn, arg);
}

void DtoDeleteInterface(LLValue* inst)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delinterface");
    // build args
    LLValue* arg[] = {
        DtoBitCast(inst, fn->getFunctionType()->getParamType(0), ".tmp")
    };
    // call
    gIR->CreateCallOrInvoke(fn, arg);
}

void DtoDeleteArray(DValue* arr)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delarray_t");

    // build args
    LLValue* arg[] = {
        DtoBitCast(arr->getLVal(), fn->getFunctionType()->getParamType(0)),
        DtoBitCast(DtoTypeInfoOf(arr->type->nextOf()), fn->getFunctionType()->getParamType(1))
    };

    // call
    gIR->CreateCallOrInvoke(fn, arg);
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// ALLOCA HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/


llvm::AllocaInst* DtoAlloca(Type* type, const char* name)
{
    LLType* lltype = i1ToI8(DtoType(type));
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
    // func
    const char* fname = msg ? "_d_assert_msg" : "_d_assert";
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, fname);

    // Arguments
    llvm::SmallVector<LLValue*, 3> args;

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
    args.push_back(DtoConstUint(loc.linnum));

    // call
    gIR->CreateCallOrInvoke(fn, args);

    // end debug info
    gIR->DBuilder.EmitFuncEnd(gIR->func()->decl);

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
// TRY-FINALLY AND SYNCHRONIZED HELPER
////////////////////////////////////////////////////////////////////////////////////////*/

void EnclosingSynchro::emitCode(IRState * p)
{
    if (s->exp)
        DtoLeaveMonitor(s->exp->toElem(p)->getRVal());
    else
        DtoLeaveCritical(s->llsync);
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

void DtoAssign(Loc& loc, DValue* lhs, DValue* rhs, int op, bool canSkipPostblit)
{
    Logger::println("DtoAssign()");
    LOG_SCOPE;

    Type* t = lhs->getType()->toBasetype();
    Type* t2 = rhs->getType()->toBasetype();

    assert(t->ty != Tvoid && "Cannot assign values of type void.");

    if (t->ty == Tbool) {
        DtoStoreZextI8(rhs->getRVal(), lhs->getLVal());
    }
    else if (t->ty == Tstruct) {
        llvm::Value* src = rhs->getRVal();
        llvm::Value* dst = lhs->getLVal();

        // Check whether source and destination values are the same at compile
        // time as to not emit an invalid (overlapping) memcpy on trivial
        // struct self-assignments like 'A a; a = a;'.
        if (src != dst)
            DtoAggrCopy(dst, src);
    }
    else if (t->ty == Tarray) {
        // lhs is slice
        if (DSliceValue* s = lhs->isSlice()) {
            if (t->nextOf()->toBasetype()->equals(t2)) {
                DtoArrayInit(loc, lhs, rhs, op);
            }
            else if (DtoArrayElementType(t)->equals(stripModifiers(t2))) {
                DtoArrayInit(loc, s, rhs, op);
            }
            else if (op != -1 && op != TOKblit && !canSkipPostblit &&
                arrayNeedsPostblit(t)
            ) {
                DtoArrayAssign(s, rhs, op);
            }
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
        else if (DtoArrayElementType(t)->equals(stripModifiers(t2))) {
            DtoArrayInit(loc, lhs, rhs, op);
        }
        else if (op != -1 && op != TOKblit && !canSkipPostblit &&
            arrayNeedsPostblit(t)
        ) {
            DtoArrayAssign(lhs, rhs, op);
        }
        // T[n] = T[n]
        else if (DtoType(lhs->getType()) == DtoType(rhs->getType())) {
            DtoStaticArrayCopy(lhs->getLVal(), rhs->getRVal());
        }
        // T[n] = T[] - generally only generated by frontend in rare cases
        else if (t2->ty == Tarray && t->nextOf()->toBasetype()->equals(t2->nextOf()->toBasetype())) {
            DtoMemCpy(lhs->getLVal(), DtoArrayPtr(rhs), DtoArrayLen(rhs));
        } else llvm_unreachable("Unimplemented static array assign!");
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
        gIR->DBuilder.EmitValue(DtoLoad(var->getLVal()), vd);
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

    llvm_unreachable("null not known for this type.");
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

    LLValue* rval = val->getRVal();
    if (rval->getType() == tolltype) {
        return new DImValue(_to, rval);
    }

    size_t fromsz = from->size();
    size_t tosz = to->size();

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
        {
            Logger::cout() << "src: " << *src << '\n';
            Logger::cout() << "to type: " << *tolltype << '\n';
        }
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

    if (totype->ty == Tpointer || totype->ty == Tclass)
    {
        if (Logger::enabled())
            Logger::cout() << "cast null to pointer/class: " << *tolltype << '\n';
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
    else if (totype->ty == Tbool)
    {
        // In theory, we could return 'false' as a constant here, but DMD
        // treats non-null values casted to typeof(null) as true.
        LLValue* rval = val->getRVal();
        LLValue* zero = LLConstant::getNullValue(rval->getType());
        return new DImValue(to, gIR->ir->CreateICmpNE(rval, zero, "tmp"));
    }
    else
    {
        error(loc, "invalid cast from null to '%s'", to->toChars());
        fatal();
    }
}

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

DValue* DtoCast(Loc& loc, DValue* val, Type* to)
{
    Type* fromtype = val->getType()->toBasetype();
    Type* totype = to->toBasetype();

    if (fromtype->ty == Taarray)
    {
        // DMD allows casting AAs to void*, even if they are internally
        // implemented as structs.
        if (totype->ty == Tpointer)
        {
            IF_LOG Logger::println("Casting AA to pointer.");
            LLValue *rval = DtoBitCast(val->getRVal(), DtoType(to));
            return new DImValue(to, rval);
        }
        else if (totype->ty == Tbool)
        {
            IF_LOG Logger::println("Casting AA to bool.");
            LLValue* rval = val->getRVal();
            LLValue* zero = LLConstant::getNullValue(rval->getType());
            return new DImValue(to, gIR->ir->CreateICmpNE(rval, zero));
        }

        // Else try dealing with the rewritten (struct) type.
        fromtype = static_cast<TypeAArray*>(fromtype)->getImpl()->type;
    }

    if (totype->ty == Taarray)
        totype = static_cast<TypeAArray*>(totype)->getImpl()->type;

    if (fromtype->equals(totype))
        return val;

    Logger::println("Casting from '%s' to '%s'", fromtype->toChars(), to->toChars());
    LOG_SCOPE;

    if (fromtype->ty == Tvector) {
        return DtoCastVector(loc, val, to);
    }
    else if (fromtype->isintegral()) {
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
    if (FuncLiteralDeclaration* fld = s->isFuncLiteralDeclaration())
    {
        if (checkLiteralOwner && fld->owningTemplate)
            return fld->owningTemplate;
    }
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
    else if (VarDeclaration* vd = dsym->isVarDeclaration()) {
        DtoResolveVariable(vd);
    }
}

void DtoResolveVariable(VarDeclaration* vd)
{
    if (vd->isTypeInfoDeclaration())
        return DtoResolveTypeInfo(static_cast<TypeInfoDeclaration *>(vd));

    IF_LOG Logger::println("DtoResolveVariable(%s)", vd->toPrettyChars());
    LOG_SCOPE;

    // just forward aliases
    // TODO: Is this required here or is the check in VarDeclaration::codegen
    // sufficient?
    if (vd->aliassym)
    {
        Logger::println("alias sym");
        DtoResolveDsymbol(vd->aliassym);
        return;
    }

    if (AggregateDeclaration* ad = vd->isMember())
        DtoResolveDsymbol(ad);

    // global variable
    if (vd->isDataseg() || (vd->storage_class & (STCconst | STCimmutable) && vd->init))
    {
        Logger::println("data segment");

    #if 0 // TODO:
        assert(!(storage_class & STCmanifest) &&
            "manifest constant being codegen'd!");
    #endif

        // don't duplicate work
        if (vd->ir.resolved) return;
        vd->ir.resolved = true;
        vd->ir.declared = true;

        vd->ir.irGlobal = new IrGlobal(vd);

        IF_LOG {
            if (vd->parent)
                Logger::println("parent: %s (%s)", vd->parent->toChars(), vd->parent->kind());
            else
                Logger::println("parent: null");
        }

        const bool isLLConst = (vd->isConst() || vd->isImmutable()) && vd->init;

        assert(!vd->ir.initialized);
        vd->ir.initialized = gIR->dmodule;
        std::string llName(vd->mangle());

        // Since the type of a global must exactly match the type of its
        // initializer, we cannot know the type until after we have emitted the
        // latter (e.g. in case of unions, …). However, it is legal for the
        // initializer to refer to the address of the variable. Thus, we first
        // create a global with the generic type (note the assignment to
        // vd->ir.irGlobal->value!), and in case we also do an initializer
        // with a different type later, swap it out and replace any existing
        // uses with bitcasts to the previous type.
        //
        // We always start out with external linkage; any other type is set
        // when actually defining it in VarDeclaration::codegen.
        llvm::GlobalVariable* gvar = getOrCreateGlobal(vd->loc, *gIR->module,
            i1ToI8(DtoType(vd->type)), isLLConst, llvm::GlobalValue::ExternalLinkage,
            0, llName, vd->isThreadlocal());
        vd->ir.irGlobal->value = gvar;

        // Set the alignment (it is important not to use type->alignsize because
        // VarDeclarations can have an align() attribute independent of the type
        // as well).
        if (vd->alignment != STRUCTALIGN_DEFAULT)
            gvar->setAlignment(vd->alignment);

        if (Logger::enabled())
            Logger::cout() << *gvar << '\n';
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
    LOG_SCOPE

    if (vd->nestedrefs.dim)
    {
        Logger::println("has nestedref set (referenced by nested function/delegate)");
        assert(vd->ir.irLocal && "irLocal is expected to be already set by DtoCreateNestedContext");
    }

    if(vd->ir.irLocal)
    {
        // Nothing to do if it has already been allocated.
    }
    /* Named Return Value Optimization (NRVO):
        T f(){
            T ret;        // &ret == hidden pointer
            ret = ...
            return ret;    // NRVO.
        }
    */
    else if (gIR->func()->retArg && gIR->func()->decl->nrvo_can && gIR->func()->decl->nrvo_var == vd) {
        assert(!isSpecialRefVar(vd) && "Can this happen?");
        vd->ir.irLocal = new IrLocal(vd, gIR->func()->retArg);
    }
    // normal stack variable, allocate storage on the stack if it has not already been done
    else {
        vd->ir.irLocal = new IrLocal(vd);

        /* NRVO again:
            T t = f();    // t's memory address is taken hidden pointer
        */
        ExpInitializer *ei = 0;
        if ((vd->type->toBasetype()->ty == Tstruct ||
             vd->type->toBasetype()->ty == Tsarray /* new in 2.064*/) &&
            vd->init &&
            (ei = vd->init->isExpInitializer()))
        {
            if (ei->exp->op == TOKconstruct) {
                AssignExp *ae = static_cast<AssignExp*>(ei->exp);
                // The return value can be casted to a different type.
                // Just look at the original expression in this case.
                // Happens with runnable/sdtor, test10094().
                Expression *rhs = ae->e2;
                if (rhs->op == TOKcast)
                    rhs = static_cast<CastExp *>(rhs)->e1;
                if (rhs->op == TOKcall) {
                    CallExp *ce = static_cast<CallExp *>(rhs);
                    TypeFunction *tf = static_cast<TypeFunction *>(ce->e1->type->toBasetype());
                    if (tf->ty == Tfunction && tf->linkage != LINKintrinsic) {
                        gABI->newFunctionType(tf);
                        bool retInArg = gABI->returnInArg(tf);
                        gABI->doneWithFunctionType();
                        if (retInArg) {
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
                            return;
                        }
                    }
                }
            }
        }

        Type* type = isSpecialRefVar(vd) ? vd->type->pointerTo() : vd->type;

        llvm::Value* allocainst;
        LLType* lltype = DtoType(type);
        if(gDataLayout->getTypeSizeInBits(lltype) == 0)
            allocainst = llvm::ConstantPointerNull::get(getPtrToType(lltype));
        else
            allocainst = DtoAlloca(type, vd->toChars());

        vd->ir.irLocal->value = allocainst;

        gIR->DBuilder.EmitLocalVariable(allocainst, vd);
    }

    if (Logger::enabled())
        Logger::cout() << "llvm value for decl: " << *vd->ir.irLocal->value << '\n';

    if (vd->init)
    {
        if (ExpInitializer* ex = vd->init->isExpInitializer())
        {
            // TODO: Refactor this so that it doesn't look like toElem has no effect.
            Logger::println("expression initializer");
            ex->exp->toElem(gIR);
        }
    }
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
            vd->codegen(gIR);
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
        s->codegen(gIR);
    }
    // function declaration
    else if (FuncDeclaration* f = declaration->isFuncDeclaration())
    {
        Logger::println("FuncDeclaration");
        f->codegen(gIR);
    }
    // class
    else if (ClassDeclaration* e = declaration->isClassDeclaration())
    {
        Logger::println("ClassDeclaration");
        e->codegen(gIR);
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
        Dsymbols *d = a->include(NULL, NULL);
        if (d)
            for (unsigned i=0; i < d->dim; ++i)
            {
                DtoDeclarationExp((*d)[i]);
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
        assert(tupled->isexp && "Non-expression tuple decls not handled yet.");
        assert(tupled->objects);
        for (unsigned i=0; i < tupled->objects->dim; ++i)
        {
            DsymbolExp* exp = static_cast<DsymbolExp*>(tupled->objects->data[i]);
            DtoDeclarationExp(exp->s);
        }
    }
    else
    {
        // Do nothing for template/alias/enum declarations and static
        // assertions. We cannot detect StaticAssert without RTTI, so don't
        // even bother to check.
        IF_LOG Logger::println("Ignoring Symbol: %s", declaration->kind());
    }

    return 0;
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
        gIR->DBuilder.EmitLocalVariable(addr, var);
    }

    // referenced by nested function?
    if (var->nestedrefs.dim)
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
        var->ir.irLocal = new IrLocal(var, addr);
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
        Expression *initExp = type->defaultInit();
        if (type->ty == Ttypedef)
            initExp->type = type; // This carries the typedef type into toConstElem.
        _init = DtoConstExpInit(loc, type, initExp);
    }
    else if (ExpInitializer* ex = init->isExpInitializer())
    {
        Logger::println("const expression initializer");
        _init = DtoConstExpInit(loc, type, ex->exp);
    }
    else if (ArrayInitializer* ai = init->isArrayInitializer())
    {
        Logger::println("const array initializer");
        _init = DtoConstArrayInitializer(ai);
    }
    else if (init->isVoidInitializer())
    {
        Logger::println("const void initializer");
        LLType* ty = voidToI8(DtoType(type));
        _init = LLConstant::getNullValue(ty);
    }
    else
    {
        // StructInitializer is no longer suposed to make it to the glue layer
        // in DMD 2.064.
        Logger::println("unsupported const initializer: %s", init->toChars());
    }
    return _init;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoConstExpInit(Loc loc, Type* targetType, Expression* exp)
{
    IF_LOG Logger::println("DtoConstExpInit(targetType = %s, exp = %s)",
        targetType->toChars(), exp->toChars());
    LOG_SCOPE

    LLConstant* val = exp->toConstElem(gIR);

    // The situation here is a bit tricky: In an ideal world, we would always
    // have val->getType() == DtoType(targetType). But there are two reasons
    // why this is not true. One is that the LLVM type system cannot represent
    // all the C types, leading to differences in types being necessary e.g. for
    // union initializers. The second is that the frontend actually does not
    // explicitly lowers things like initializing an array/vector with a scalar
    // constant, or since 2.061 sometimes does not get implicit conversions for
    // integers right. However, we cannot just rely on the actual Types being
    // equal if there are no rewrites to do because of – as usual – AST
    // inconsistency bugs.

    Type* expBase = stripModifiers(exp->type->toBasetype())->merge();
    Type* targetBase = stripModifiers(targetType->toBasetype())->merge();

    if (expBase->equals(targetBase) && targetBase->ty != Tbool)
    {
        return val;
    }

    llvm::Type* llType = val->getType();
    llvm::Type* targetLLType = i1ToI8(DtoType(targetBase));
    if (llType == targetLLType)
    {
        Logger::println("Matching LLVM types, ignoring frontend glitch.");
        return val;
    }

    if (targetBase->ty == Tsarray)
    {
        if (targetBase->nextOf()->toBasetype()->ty == Tvoid) {
           error(loc, "static arrays of voids have no default initializer");
           fatal();
        }
        Logger::println("Building constant array initializer to single value.");

        d_uns64 elemCount = targetBase->size() / expBase->size();
        assert(targetBase->size() % expBase->size() == 0);

        std::vector<llvm::Constant*> initVals(elemCount, val);
        return llvm::ConstantArray::get(llvm::ArrayType::get(llType, elemCount), initVals);
    }

    if (targetBase->ty == Tvector)
    {
        Logger::println("Building vector initializer from scalar.");

        TypeVector* tv = static_cast<TypeVector*>(targetBase);
        assert(tv->basetype->ty == Tsarray);
        dinteger_t elemCount =
            static_cast<TypeSArray *>(tv->basetype)->dim->toInteger();
        return llvm::ConstantVector::getSplat(elemCount, val);
    }

    if (llType->isIntegerTy() && targetLLType->isIntegerTy())
    {
        // This should really be fixed in the frontend.
        Logger::println("Fixing up unresolved implicit integer conversion.");

        llvm::IntegerType* source = llvm::cast<llvm::IntegerType>(llType);
        llvm::IntegerType* target = llvm::cast<llvm::IntegerType>(targetLLType);

        assert(target->getBitWidth() > source->getBitWidth() && "On initializer "
            "integer type mismatch, the target should be wider than the source.");
        return llvm::ConstantExpr::getZExtOrBitCast(val, target);
    }

    Logger::println("Unhandled type mismatch, giving up.");
    return val;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoTypeInfoOf(Type* type, bool base)
{
    IF_LOG Logger::println("DtoTypeInfoOf(type = '%s', base='%d')", type->toChars(), base);
    LOG_SCOPE

    type = type->merge2(); // needed.. getTypeInfo does the same
    type->getTypeInfo(NULL);
    TypeInfoDeclaration* tidecl = type->vtinfo;
    assert(tidecl);
    tidecl->codegen(gIR);
    assert(tidecl->ir.irGlobal != NULL);
    assert(tidecl->ir.irGlobal->value != NULL);
    LLConstant* c = isaConstant(tidecl->ir.irGlobal->value);
    assert(c != NULL);
    if (base)
        return llvm::ConstantExpr::getBitCast(c, DtoType(Type::dtypeinfo->type));
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
    if (type->ty == Tfunction)
        return type;
    return type->castMod(0);
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
        DtoStoreZextI8(valuePointer, tmp);
        valuePointer = tmp;
    }

    return valuePointer;
}

//////////////////////////////////////////////////////////////////////////////////////////

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
            DtoResolveFunction(fd);
            Expressions args;
            DFuncValue dfn(fd, fd->ir.irFunc->func, val);
            DtoCallFunction(loc, Type::basic[Tvoid], &dfn, &args);
        }
    }
}

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

///////////////////////////////////////////////////////////////////////////////
DValue* DtoSymbolAddress(const Loc& loc, Type* type, Declaration* decl)
{
    IF_LOG Logger::println("DtoSymbolAddress ('%s' of type '%s')",
        decl->toChars(), decl->type->toChars());
    LOG_SCOPE

    if (VarDeclaration* vd = decl->isVarDeclaration())
    {
        // The magic variable __ctfe is always false at runtime
        if (vd->ident == Id::ctfe)
        {
            return new DConstValue(type, DtoConstBool(false));
        }

        // this is an error! must be accessed with DotVarExp
        if (vd->needThis())
        {
            error("need 'this' to access member %s", vd->toChars());
            fatal();
        }

        // _arguments
        if (vd->ident == Id::_arguments && gIR->func()->_arguments)
        {
            Logger::println("Id::_arguments");
            LLValue* v = gIR->func()->_arguments;
            return new DVarValue(type, vd, v);
        }
        // _argptr
        else if (vd->ident == Id::_argptr && gIR->func()->_argptr)
        {
            Logger::println("Id::_argptr");
            LLValue* v = gIR->func()->_argptr;
            return new DVarValue(type, vd, v);
        }
        // _dollar
        else if (vd->ident == Id::dollar)
        {
            Logger::println("Id::dollar");
            LLValue* val = 0;
            if (vd->ir.isSet() && (val = vd->ir.getIrValue()))
            {
                // It must be length of a range
                return new DVarValue(type, vd, val);
            }
            assert(!gIR->arrays.empty());
            val = DtoArrayLen(gIR->arrays.back());
            return new DImValue(type, val);
        }
        // classinfo
        else if (ClassInfoDeclaration* cid = vd->isClassInfoDeclaration())
        {
            Logger::println("ClassInfoDeclaration: %s", cid->cd->toChars());
            DtoResolveClass(cid->cd);
            return new DVarValue(type, vd, cid->cd->ir.irAggr->getClassInfoSymbol());
        }
        // typeinfo
        else if (TypeInfoDeclaration* tid = vd->isTypeInfoDeclaration())
        {
            Logger::println("TypeInfoDeclaration");
            DtoResolveTypeInfo(tid);
            assert(tid->ir.getIrValue());
            LLType* vartype = DtoType(type);
            LLValue* m = tid->ir.getIrValue();
            if (m->getType() != getPtrToType(vartype))
                m = gIR->ir->CreateBitCast(m, vartype, "tmp");
            return new DImValue(type, m);
        }
        // nested variable
        else if (vd->nestedrefs.dim)
        {
            Logger::println("nested variable");
            return DtoNestedVariable(loc, type, vd);
        }
        // function parameter
        else if (vd->isParameter())
        {
            Logger::println("function param");
            Logger::println("type: %s", vd->type->toChars());
            FuncDeclaration* fd = vd->toParent2()->isFuncDeclaration();
            if (fd && fd != gIR->func()->decl)
            {
                Logger::println("nested parameter");
                return DtoNestedVariable(loc, type, vd);
            }
            else if (vd->storage_class & STClazy)
            {
                Logger::println("lazy parameter");
                assert(type->ty == Tdelegate);
                return new DVarValue(type, vd->ir.getIrValue());
            }
            else if (vd->isRef() || vd->isOut() || DtoIsPassedByRef(vd->type) ||
                llvm::isa<llvm::AllocaInst>(vd->ir.getIrValue()))
            {
                return new DVarValue(type, vd, vd->ir.getIrValue());
            }
            else if (llvm::isa<llvm::Argument>(vd->ir.getIrValue()))
            {
                return new DImValue(type, vd->ir.getIrValue());
            }
            else llvm_unreachable("Unexpected parameter value.");
        }
        else
        {
            Logger::println("a normal variable");

            // take care of forward references of global variables
            const bool isGlobal = vd->isDataseg() || (vd->storage_class & STCextern);
            if (isGlobal)
                DtoResolveVariable(vd);

            assert(vd->ir.isSet() && "Variable not resolved.");

            llvm::Value* val = vd->ir.getIrValue();
            assert(val && "Variable value not set yet.");

            if (isGlobal)
            {
                llvm::Type* expectedType = llvm::PointerType::getUnqual(i1ToI8(DtoType(type)));
                // The type of globals is determined by their initializer, so
                // we might need to cast. Make sure that the type sizes fit -
                // '==' instead of '<=' should probably work as well.
                if (val->getType() != expectedType)
                {
                    llvm::Type* t = llvm::cast<llvm::PointerType>(val->getType())->getElementType();
                    assert(getTypeStoreSize(DtoType(type)) <= getTypeStoreSize(t) &&
                        "Global type mismatch, encountered type too small.");
                    val = DtoBitCast(val, expectedType);
                }
            }

            return new DVarValue(type, vd, val);
        }
    }

    if (FuncDeclaration* fdecl = decl->isFuncDeclaration())
    {
        Logger::println("FuncDeclaration");
        fdecl = fdecl->toAliasFunc();
        if (fdecl->llvmInternal == LLVMinline_asm)
        {
            // TODO: Is this needed? If so, what about other intrinsics?
            error("special ldc inline asm is not a normal function");
            fatal();
        }
        DtoResolveFunction(fdecl);
        assert(fdecl->llvmInternal == LLVMva_arg || fdecl->ir.irFunc);
        return new DFuncValue(fdecl, fdecl->ir.irFunc ? fdecl->ir.irFunc->func : 0);
    }

    if (SymbolDeclaration* sdecl = decl->isSymbolDeclaration())
    {
        // this seems to be the static initialiser for structs
        Type* sdecltype = sdecl->type->toBasetype();
        Logger::print("Sym: type=%s\n", sdecltype->toChars());
        assert(sdecltype->ty == Tstruct);
        TypeStruct* ts = static_cast<TypeStruct*>(sdecltype);
        assert(ts->sym);
        DtoResolveStruct(ts->sym);

        LLValue* initsym = ts->sym->ir.irAggr->getInitSymbol();
        initsym = DtoBitCast(initsym, DtoType(ts->pointerTo()));
        return new DVarValue(type, initsym);
    }

    llvm_unreachable("Unimplemented VarExp type");
}

llvm::Constant* DtoConstSymbolAddress(const Loc& loc, Declaration* decl)
{
    // Make sure 'this' isn't needed.
    // TODO: This check really does not belong here, should be moved to
    // semantic analysis in the frontend.
    if (decl->needThis())
    {
        error("need 'this' to access %s", decl->toChars());
        fatal();
    }

    // global variable
    if (VarDeclaration* vd = decl->isVarDeclaration())
    {
        if (!vd->isDataseg())
        {
            // Not sure if this can be triggered from user code, but it is
            // needed for the current hacky implementation of
            // AssocArrayLiteralExp::toElem, which requires on error
            // gagging to check for constantness of the initializer.
            error(loc, "cannot use address of non-global variable '%s' "
                "as constant initializer", vd->toChars());
            if (!global.gag) fatal();
            return NULL;
        }

        DtoResolveVariable(vd);
        LLConstant* llc = llvm::dyn_cast<LLConstant>(vd->ir.getIrValue());
        assert(llc);
        return llc;
    }
    // static function
    else if (FuncDeclaration* fd = decl->isFuncDeclaration())
    {
        DtoResolveFunction(fd);
        IrFunction* irfunc = fd->ir.irFunc;
        return irfunc->func;
    }

    llvm_unreachable("Taking constant address not implemented.");
}

llvm::GlobalVariable* getOrCreateGlobal(Loc loc, llvm::Module& module,
    llvm::Type* type, bool isConstant, llvm::GlobalValue::LinkageTypes linkage,
    llvm::Constant* init, llvm::StringRef name, bool isThreadLocal)
{
    llvm::GlobalVariable* existing = module.getGlobalVariable(name, true);
    if (existing)
    {
        if (existing->getType()->getElementType() != type)
        {
            error(loc, "Global variable type does not match previous "
                "declaration with same mangled name: %s", name.str().c_str());
            fatal();
        }
        return existing;
    }

#if LDC_LLVM_VER >= 302
    // Use a command line option for the thread model.
    // On PPC there is only local-exec available - in this case just ignore the
    // command line.
    const llvm::GlobalVariable::ThreadLocalMode tlsModel =
        isThreadLocal
            ? (global.params.targetTriple.getArch() == llvm::Triple::ppc
                  ? llvm::GlobalVariable::LocalExecTLSModel
                  : clThreadModel.getValue())
            : llvm::GlobalVariable::NotThreadLocal;
    return new llvm::GlobalVariable(module, type, isConstant, linkage,
                                    init, name, 0, tlsModel);
#else
    return new llvm::GlobalVariable(module, type, isConstant, linkage,
                                    init, name, 0, isThreadLocal);
#endif
}

FuncDeclaration* getParentFunc(Dsymbol* sym, bool stopOnStatic) {
    if (!sym)
        return NULL;
    Dsymbol* parent = sym->parent;
    assert(parent);
    while (parent && !parent->isFuncDeclaration()) {
        if (stopOnStatic) {
            Declaration* decl = sym->isDeclaration();
            if (decl && decl->isStatic())
                return NULL;
        }
        parent = parent->parent;
    }

    return (parent ? parent->isFuncDeclaration() : NULL);
}
