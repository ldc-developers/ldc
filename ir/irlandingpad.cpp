#include "gen/llvm.h"
#include "gen/tollvm.h"
#include "gen/irstate.h"
#include "gen/runtime.h"
#include "gen/logger.h"
#include "gen/classes.h"
#include "gen/llvmhelpers.h"
#include "ir/irlandingpad.h"

IRLandingPadInfo::IRLandingPadInfo(Catch* catchstmt, llvm::BasicBlock* end)
: finallyBody(NULL)
{
    target = llvm::BasicBlock::Create("catch", gIR->topfunc(), end);
    gIR->scope() = IRScope(target,end);

    // assign storage to catch var
    if(catchstmt->var) {
        assert(!catchstmt->var->ir.irLocal);
        catchstmt->var->ir.irLocal = new IrLocal(catchstmt->var);
        LLValue* catch_var = gIR->func()->landingPad.getExceptionStorage();
        catchstmt->var->ir.irLocal->value = gIR->ir->CreateBitCast(catch_var, getPtrToType(DtoType(catchstmt->var->type)));
    }

    // emit handler, if there is one
    // handler is zero for instance for 'catch { debug foo(); }'
    if(catchstmt->handler);
        catchstmt->handler->toIR(gIR);

    if (!gIR->scopereturned())
        gIR->ir->CreateBr(end);

    assert(catchstmt->type);
    //TODO: Is toBasetype correct here? Should catch handlers with typedefed
    // classes behave differently?
    catchType = catchstmt->type->toBasetype()->isClassHandle();
    assert(catchType);
    DtoForceDeclareDsymbol(catchType);
}

IRLandingPadInfo::IRLandingPadInfo(Statement* finallystmt)
: target(NULL), finallyBody(finallystmt), catchType(NULL)
{

}


void IRLandingPad::addCatch(Catch* catchstmt, llvm::BasicBlock* end)
{
    unpushed_infos.push_front(IRLandingPadInfo(catchstmt, end));
}

void IRLandingPad::addFinally(Statement* finallystmt)
{
    unpushed_infos.push_front(IRLandingPadInfo(finallystmt));
}

void IRLandingPad::push(llvm::BasicBlock* inBB)
{
    // store infos such that matches are right to left
    nInfos.push(infos.size());
    infos.insert(infos.end(), unpushed_infos.begin(), unpushed_infos.end());
    unpushed_infos.clear();

    constructLandingPad(inBB);

    // store as invoke target
    padBBs.push(inBB);
}

void IRLandingPad::pop()
{
    padBBs.pop();

    size_t n = nInfos.top();
    infos.resize(n);
    nInfos.pop();
}

llvm::BasicBlock* IRLandingPad::get()
{
    if(padBBs.size() == 0)
        return NULL;
    else
        return padBBs.top();
}

void IRLandingPad::constructLandingPad(llvm::BasicBlock* inBB)
{
    // save and rewrite scope
    IRScope savedscope = gIR->scope();
    gIR->scope() = IRScope(inBB,savedscope.end);

    // eh_ptr = llvm.eh.exception();
    llvm::Function* eh_exception_fn = GET_INTRINSIC_DECL(eh_exception);
    LLValue* eh_ptr = gIR->ir->CreateCall(eh_exception_fn);

    // build selector arguments
    LLSmallVector<LLValue*, 6> selectorargs;
    selectorargs.push_back(eh_ptr);

    llvm::Function* personality_fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_eh_personality");
    LLValue* personality_fn_arg = gIR->ir->CreateBitCast(personality_fn, getPtrToType(LLType::Int8Ty));
    selectorargs.push_back(personality_fn_arg);

    bool hasFinally = false;
    // associate increasing ints with each unique classdecl encountered
    std::map<ClassDeclaration*, int> catchToInt;
    std::deque<IRLandingPadInfo>::reverse_iterator it = infos.rbegin(), end = infos.rend();
    for(size_t i = infos.size() - 1; it != end; ++it, --i)
    {
        if(it->finallyBody)
            hasFinally = true;
        else
        {
            if(catchToInt.find(it->catchType) == catchToInt.end())
            {
                int newval = catchToInt.size();
                catchToInt[it->catchType] = newval;
            }
            assert(it->catchType);
            assert(it->catchType->ir.irStruct);
            selectorargs.push_back(it->catchType->ir.irStruct->classInfo);
        }
    }
    // if there's a finally, the eh table has to have a 0 action
    if(hasFinally)
        selectorargs.push_back(llvm::ConstantInt::get(LLType::Int32Ty, 0));
    // if there is a catch and some catch allocated storage, store exception object
    if(catchToInt.size() && catch_var) 
    {
        const LLType* objectTy = DtoType(ClassDeclaration::object->type);
        gIR->ir->CreateStore(gIR->ir->CreateBitCast(eh_ptr, objectTy), catch_var);
    }

    // eh_sel = llvm.eh.selector(eh_ptr, cast(byte*)&_d_eh_personality, <selectorargs>);
    llvm::Function* eh_selector_fn;
    if (global.params.is64bit)
        eh_selector_fn = GET_INTRINSIC_DECL(eh_selector_i64);
    else
        eh_selector_fn = GET_INTRINSIC_DECL(eh_selector_i32);
    LLValue* eh_sel = gIR->ir->CreateCall(eh_selector_fn, selectorargs.begin(), selectorargs.end());

    // emit finallys and switches that branch to catches until there are no more catches
    // then simply branch to the finally chain
    llvm::SwitchInst* switchinst = NULL;
    for(it = infos.rbegin(); it != end; ++it)
    {
        // if it's a finally, emit its code
        if(it->finallyBody)
        {
            if(switchinst)
                switchinst = NULL;
            it->finallyBody->toIR(gIR);
        }
        // otherwise it's a catch and we'll add a switch case
        else
        {
            if(!switchinst)
            {
                switchinst = gIR->ir->CreateSwitch(eh_sel, llvm::BasicBlock::Create("switchdefault", gIR->topfunc(), gIR->scopeend()), infos.size());
                gIR->scope() = IRScope(switchinst->getDefaultDest(), gIR->scopeend());
            }
            // catches matched first get the largest switchval, so do size - unique int
            llvm::ConstantInt* switchval = llvm::ConstantInt::get(LLType::Int32Ty, catchToInt.size() - catchToInt[it->catchType]);
            // and make sure we don't add the same switchval twice, may happen with nested trys
            if(!switchinst->findCaseValue(switchval))
                switchinst->addCase(switchval, it->target);
        }
    }

    // no catch matched and all finallys executed - resume unwind
    llvm::Function* unwind_resume_fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_eh_resume_unwind");
    gIR->ir->CreateCall(unwind_resume_fn, eh_ptr);
    gIR->ir->CreateUnreachable();

    gIR->scope() = savedscope;
}

LLValue* IRLandingPad::getExceptionStorage()
{
    if(!catch_var)
    {
        Logger::println("Making new catch var");
        const LLType* objectTy = DtoType(ClassDeclaration::object->type);
        catch_var = new llvm::AllocaInst(objectTy,"catchvar",gIR->topallocapoint());
    }
    return catch_var;
}
