#include "gen/llvm.h"
#include "llvm/Target/TargetMachineRegistry.h"

#include "mars.h"
#include "init.h"
#include "id.h"
#include "expression.h"
#include "template.h"

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
    LLValue* mem = gIR->CreateCallOrInvoke(fn, ti, ".gc_mem")->get();
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

void DtoAssert(Loc* loc, DValue* msg)
{
    std::vector<LLValue*> args;
    LLConstant* c;

    // func
    const char* fname = msg ? "_d_assert_msg" : "_d_assert";
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, fname);

    // param attrs
    llvm::PAListPtr palist;
    int idx = 1;

    // FIXME: every assert creates a global for the filename !!!
    c = DtoConstString(loc->filename);

    // msg param
    if (msg)
    {
        if (DSliceValue* s = msg->isSlice())
        {
            llvm::AllocaInst* alloc = gIR->func()->msgArg;
            if (!alloc)
            {
                alloc = DtoAlloca(c->getType(), ".assertmsg");
                DtoSetArray(alloc, DtoArrayLen(s), DtoArrayPtr(s));
                gIR->func()->msgArg = alloc;
            }
            args.push_back(alloc);
        }
        else
        {
            args.push_back(msg->getRVal());
        }
        palist = palist.addAttr(idx++, llvm::ParamAttr::ByVal);
    }

    // file param
    llvm::AllocaInst* alloc = gIR->func()->srcfileArg;
    if (!alloc)
    {
        alloc = DtoAlloca(c->getType(), ".srcfile");
        gIR->func()->srcfileArg = alloc;
    }
    LLValue* ptr = DtoGEPi(alloc, 0,0, "tmp");
    DtoStore(c->getOperand(0), ptr);
    ptr = DtoGEPi(alloc, 0,1, "tmp");
    DtoStore(c->getOperand(1), ptr);

    args.push_back(alloc);
    palist = palist.addAttr(idx++, llvm::ParamAttr::ByVal);


    // line param
    c = DtoConstUint(loc->linnum);
    args.push_back(c);

    // call
    CallOrInvoke* call = gIR->CreateCallOrInvoke(fn, args.begin(), args.end());
    call->setParamAttrs(palist);

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
        targetBB = llvm::BasicBlock::Create("label", gIR->topfunc());

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
    assert(tf->finalbody);
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
        ctx = DtoLoad(DtoGEPi(val, 0,2+cd->vthis->ir.irField->index, ".vthis"));
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
    
    if (irfunc->nestedVar)
        return irfunc->nestedVar;
    else if (irfunc->nestArg)
        return irfunc->nestArg;
    else if (irfunc->thisArg)
    {
        ClassDeclaration* cd = irfunc->decl->isMember2()->isClassDeclaration();
        if (!cd || !cd->vthis)
            return getNullPtr(getVoidPtrType());
        LLValue* val = DtoLoad(irfunc->thisArg);
        return DtoLoad(DtoGEPi(val, 0,2+cd->vthis->ir.irField->index, ".vthis"));
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

void DtoAssign(Loc& loc, DValue* lhs, DValue* rhs)
{
    Logger::cout() << "DtoAssign(...);\n";
    LOG_SCOPE;

    Type* t = lhs->getType()->toBasetype();
    Type* t2 = rhs->getType()->toBasetype();

    if (t->ty == Tstruct) {
        if (!t->equals(t2)) {
            // TODO: fix this, use 'rhs' for something
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
            else if (t->next->toBasetype()->equals(t2)) {
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
        else {
            DtoArrayAssign(lhs->getLVal(), rhs->getRVal());
        }
    }
    else if (t->ty == Tsarray) {
        if (DtoType(lhs->getType()) == DtoType(rhs->getType())) {
            DtoStaticArrayCopy(lhs->getLVal(), rhs->getRVal());
        }
        else {
            DtoArrayInit(loc, lhs, rhs);
        }
    }
    else if (t->ty == Tdelegate) {
        if (rhs->isNull())
            DtoAggrZeroInit(lhs->getLVal());
        else {
            LLValue* l = lhs->getLVal();
            LLValue* r = rhs->getRVal();
            Logger::cout() << "assign\nlhs: " << *l << "rhs: " << *r << '\n';
            DtoAggrCopy(l, r);
        }
    }
    else if (t->ty == Tclass) {
        assert(t2->ty == Tclass);
        LLValue* l = lhs->getLVal();
        LLValue* r = rhs->getRVal();
        Logger::cout() << "l : " << *l << '\n';
        Logger::cout() << "r : " << *r << '\n';
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
        LLValue* ptr = getNullPtr(getPtrToType(DtoType(basetype->next)));
        return new DSliceValue(type, len, ptr);
    }
    // delegate
    else if (basety == Tdelegate)
    {
        return new DNullValue(type, LLConstant::getNullValue(lltype));
    }

    // unknown
    std::cout << "unsupported: null value for " << type->toChars() << '\n';
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

    if (to->isintegral()) {
        if (fromsz < tosz) {
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
        Logger::cout() << "src: " << *src << "to type: " << *tolltype << '\n';
        rval = DtoBitCast(src, tolltype);
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

    if (totype->iscomplex()) {
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
    LLValue* res = 0;
    to = to->toBasetype();

    if (to->ty == Tdelegate)
    {
        const LLType* toll = getPtrToType(DtoType(to));
        res = DtoBitCast(val->getRVal(), toll);
    }
    else
    {
        error(loc, "invalid cast from '%s' to '%s'", val->getType()->toChars(), to->toChars());
        fatal();
    }

    return new DImValue(to, res);
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

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
//      TEMPLATE HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/

bool DtoIsTemplateInstance(Dsymbol* s)
{
    if (!s) return false;
    if (s->isTemplateInstance() && !s->isTemplateMixin())
        return true;
    else if (s->parent)
        return DtoIsTemplateInstance(s->parent);
    return false;
}

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
//      LAZY STATIC INIT HELPER
////////////////////////////////////////////////////////////////////////////////////////*/

void DtoLazyStaticInit(bool istempl, LLValue* gvar, Initializer* init, Type* t)
{
    // create a flag to make sure initialization only happens once
    llvm::GlobalValue::LinkageTypes gflaglink = istempl ? llvm::GlobalValue::WeakLinkage : llvm::GlobalValue::InternalLinkage;
    std::string gflagname(gvar->getName());
    gflagname.append("__initflag");
    llvm::GlobalVariable* gflag = new llvm::GlobalVariable(LLType::Int1Ty,false,gflaglink,DtoConstBool(false),gflagname,gIR->module);

    // check flag and do init if not already done
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* initbb = llvm::BasicBlock::Create("ifnotinit",gIR->topfunc(),oldend);
    llvm::BasicBlock* endinitbb = llvm::BasicBlock::Create("ifnotinitend",gIR->topfunc(),oldend);
    LLValue* cond = gIR->ir->CreateICmpEQ(gIR->ir->CreateLoad(gflag,"tmp"),DtoConstBool(false));
    gIR->ir->CreateCondBr(cond, initbb, endinitbb);
    gIR->scope() = IRScope(initbb,endinitbb);
    DValue* ie = DtoInitializer(gvar, init);
    
    DVarValue dst(t, gvar);
    DtoAssign(init->loc, &dst, ie);
    
    gIR->ir->CreateStore(DtoConstBool(true), gflag);
    gIR->ir->CreateBr(endinitbb);
    gIR->scope() = IRScope(endinitbb,oldend);
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
        DtoDefineFunc(fd);
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

    Logger::println("* DtoConstInitGlobal(%s)", vd->toChars());
    LOG_SCOPE;

    bool emitRTstaticInit = false;

    LLConstant* _init = 0;
    if (vd->parent && vd->parent->isFuncDeclaration() && vd->init && vd->init->isExpInitializer()) {
        _init = DtoConstInitializer(vd->type, NULL);
        emitRTstaticInit = true;
    }
    else {
        _init = DtoConstInitializer(vd->type, vd->init);
    }

    const LLType* _type = DtoType(vd->type);
    Type* t = vd->type->toBasetype();

    //Logger::cout() << "initializer: " << *_init << '\n';
    if (_type != _init->getType()) {
        Logger::cout() << "got type '" << *_init->getType() << "' expected '" << *_type << "'\n";
        // zero initalizer
        if (_init->isNullValue())
            _init = llvm::Constant::getNullValue(_type);
        // pointer to global constant (struct.init)
        else if (llvm::isa<llvm::GlobalVariable>(_init))
        {
            assert(_init->getType()->getContainedType(0) == _type);
            llvm::GlobalVariable* gv = llvm::cast<llvm::GlobalVariable>(_init);
            assert(t->ty == Tstruct);
            TypeStruct* ts = (TypeStruct*)t;
            assert(ts->sym->ir.irStruct->constInit);
            _init = ts->sym->ir.irStruct->constInit;
        }
        // array single value init
        else if (isaArray(_type))
        {
            _init = DtoConstStaticArray(_type, _init);
        }
        else {
            Logger::cout() << "Unexpected initializer type: " << *_type << '\n';
            //assert(0);
        }
    }

    bool istempl = false;
    if ((vd->storage_class & STCcomdat) || (vd->parent && DtoIsTemplateInstance(vd->parent))) {
        istempl = true;
    }

    if (_init && _init->getType() != _type)
        _type = _init->getType();
    llvm::cast<LLOpaqueType>(vd->ir.irGlobal->type.get())->refineAbstractTypeTo(_type);
    _type = vd->ir.irGlobal->type.get();
    //_type->dump();
    assert(!_type->isAbstract());

    llvm::GlobalVariable* gvar = llvm::cast<llvm::GlobalVariable>(vd->ir.irGlobal->value);
    if (!(vd->storage_class & STCextern) && (vd->getModule() == gIR->dmodule || istempl))
    {
        Logger::println("setting initializer");
        Logger::cout() << "global: " << *gvar << '\n';
        Logger::cout() << "init:   " << *_init << '\n';
        gvar->setInitializer(_init);
        // do debug info
        if (global.params.symdebug)
        {
            LLGlobalVariable* gv = DtoDwarfGlobalVariable(gvar, vd);
            // keep a reference so GDCE doesn't delete it !
            gIR->usedArray.push_back(llvm::ConstantExpr::getBitCast(gv, getVoidPtrType()));
        }
    }

    if (emitRTstaticInit)
        DtoLazyStaticInit(istempl, gvar, vd->init, t);
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
            if (vd->nestedref) {
                Logger::println("has nestedref set");
                assert(vd->ir.irLocal);
                
                // alloca as usual is no value already
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
            DtoForceDeclareDsymbol((Dsymbol*)a->decl->data[i]);
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
    // unsupported declaration
    else
    {
        error(declaration->loc, "Unimplemented Declaration type for DeclarationExp. kind: %s", declaration->kind());
        assert(0);
    }
    return NULL;
}


/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
//      INITIALIZER HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/

LLConstant* DtoConstInitializer(Type* type, Initializer* init)
{
    LLConstant* _init = 0; // may return zero
    if (!init)
    {
        Logger::println("const default initializer for %s", type->toChars());
        _init = DtoDefaultInit(type);
    }
    else if (ExpInitializer* ex = init->isExpInitializer())
    {
        Logger::println("const expression initializer");
        _init = ex->exp->toConstElem(gIR);
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

LLConstant* DtoConstFieldInitializer(Type* t, Initializer* init)
{
    Logger::println("DtoConstFieldInitializer");
    LOG_SCOPE;

    const LLType* _type = DtoType(t);

    LLConstant* _init = DtoConstInitializer(t, init);
    assert(_init);
    if (_type != _init->getType())
    {
        Logger::cout() << "field init is: " << *_init << " type should be " << *_type << '\n';
        if (t->ty == Tsarray)
        {
            const LLArrayType* arrty = isaArray(_type);
            uint64_t n = arrty->getNumElements();
            std::vector<LLConstant*> vals(n,_init);
            _init = llvm::ConstantArray::get(arrty, vals);
        }
        else if (t->ty == Tarray)
        {
            assert(isaStruct(_type));
            _init = llvm::ConstantAggregateZero::get(_type);
        }
        else if (t->ty == Tstruct)
        {
            const LLStructType* structty = isaStruct(_type);
            TypeStruct* ts = (TypeStruct*)t;
            assert(ts);
            assert(ts->sym);
            assert(ts->sym->ir.irStruct->constInit);
            _init = ts->sym->ir.irStruct->constInit;
        }
        else if (t->ty == Tclass)
        {
            _init = llvm::Constant::getNullValue(_type);
        }
        else {
            Logger::println("failed for type %s", t->toChars());
            assert(0);
        }
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
        DValue* res = ex->exp->toElem(gIR);

        assert(llvm::isa<llvm::PointerType>(target->getType()) && "init target must be ptr");
        const LLType* targetty = target->getType()->getContainedType(0);
        if(targetty == LLType::X86_FP80Ty)
        {
            Logger::println("setting fp80 padding to zero");

            LLValue* castv = DtoBitCast(target, getPtrToType(LLType::Int16Ty));
            LLValue* padding = DtoGEPi1(castv, 5);
            DtoStore(llvm::Constant::getNullValue(LLType::Int16Ty), padding);
        }
        else if(targetty == DtoComplexType(Type::tcomplex80))
        {
            Logger::println("setting complex fp80 padding to zero");

            LLValue* castv = DtoBitCast(target, getPtrToType(LLType::Int16Ty));
            LLValue* padding = DtoGEPi1(castv, 5);
            DtoStore(llvm::Constant::getNullValue(LLType::Int16Ty), padding);
            padding = DtoGEPi1(castv, 11);
            DtoStore(llvm::Constant::getNullValue(LLType::Int16Ty), padding);
        }

        return res;
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
    Logger::cout() << "final llvm type requested: " << *dstTy << '\n';
    
    LLConstant* val = exp->toConstElem(gIR);
    
    Type* expbase = exp->type->toBasetype();
    Type* t = base;
    
    LLSmallVector<size_t, 4> dims;

    while(1)
    {
        if (t->equals(expbase))
            break;
        assert(t->ty == Tsarray);
        TypeSArray* tsa = (TypeSArray*)t;
        dims.push_back(tsa->dim->toInteger());
        assert(t->next);
        t = t->next->toBasetype();
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

LLConstant* DtoDefaultInit(Type* type)
{
    Expression* exp = type->defaultInit();
    
    Type* expbase = exp->type->toBasetype();
    Type* base = type->toBasetype();
    
    // if not the same basetypes, we won't get the same llvm types either
    if (!expbase->equals(base))
    {
        if (base->ty == Tsarray)
        {
            Logger::println("type is a static array, building constant array initializer from single value");
            return expand_to_sarray(base, exp);
        }
        else
        {
            error("cannot yet convert default initializer %s from type %s to %s", exp->toChars(), exp->type->toChars(), type->toChars());
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

LLValue* DtoBoolean(Loc& loc, DValue* dval)
{
    Type* dtype = dval->getType()->toBasetype();
    TY ty = dtype->ty;

    // integer
    if (dtype->isintegral())
    {
        LLValue* val = dval->getRVal();
        if (val->getType() == LLType::Int1Ty)
            return val;
        else {
            LLValue* zero = LLConstantInt::get(val->getType(), 0, false);
            return gIR->ir->CreateICmpNE(val, zero, "tmp");
        }
    }
    // complex
    else if (dtype->iscomplex())
    {
        return DtoComplexEquals(loc, TOKnotequal, dval, DtoNullValue(dtype));
    }
    // floating point
    else if (dtype->isfloating())
    {
        LLValue* val = dval->getRVal();
        LLValue* zero = LLConstant::getNullValue(val->getType());
        return gIR->ir->CreateFCmpONE(val, zero, "tmp");
    }
    // pointer/class
    else if (ty == Tpointer || ty == Tclass) {
        LLValue* val = dval->getRVal();
        LLValue* zero = LLConstant::getNullValue(val->getType());
        Logger::cout() << "val:  " << *val << '\n';
        Logger::cout() << "zero: " << *zero << '\n';
        return gIR->ir->CreateICmpNE(val, zero, "tmp");
    }
    // dynamic array
    else if (ty == Tarray)
    {
        // return (arr.length != 0)
        return gIR->ir->CreateICmpNE(DtoArrayLen(dval), DtoConstSize_t(0), "tmp");
    }
    // delegate
    else if (ty == Tdelegate)
    {
        // return (dg !is null)
        return DtoDelegateEquals(TOKnotequal, dval->getRVal(), NULL);
    }
    // unknown
    std::cout << "unsupported -> bool : " << dtype->toChars() << '\n';
    assert(0);
    return 0;
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
        sprintf(tmp, "%d", T->size()*8);
    
    // replace # in name with bitsize
    name = td->intrinsicName;

    std::string needle("#");
    size_t pos;
    while(std::string::npos != (pos = name.find(needle)))
        name.replace(pos, 1, tmp);
    
    Logger::println("final intrinsic name: %s", name.c_str());
}
