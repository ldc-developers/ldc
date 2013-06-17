//===-- irlandingpad.cpp --------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"
#include "gen/classes.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/runtime.h"
#include "gen/todebug.h"
#include "gen/tollvm.h"
#include "ir/irlandingpad.h"

IRLandingPadInfo::IRLandingPadInfo(Catch* catchstmt_, llvm::BasicBlock* end_) :
    finallyBody(NULL), catchstmt(catchstmt_), end(end_)
{
    target = llvm::BasicBlock::Create(gIR->context(), "catch", gIR->topfunc(), end);

    assert(catchstmt->type);
    catchType = catchstmt->type->toBasetype()->isClassHandle();
    assert(catchType);
    catchType->codegen(Type::sir);

    if(catchstmt->var) {
            if(!catchstmt->var->nestedrefs.dim) {
                gIR->func()->gen->landingPadInfo.getExceptionStorage();
            }
    }
}

IRLandingPadInfo::IRLandingPadInfo(Statement* finallystmt) :
    target(NULL), finallyBody(finallystmt), catchstmt(NULL)
{

}

void IRLandingPadInfo::toIR()
{
    if (!catchstmt)
        return;

    gIR->scope() = IRScope(target, target);
    DtoDwarfBlockStart(catchstmt->loc);

    // assign storage to catch var
    if(catchstmt->var) {
        // use the same storage for all exceptions that are not accessed in
        // nested functions
        if(!catchstmt->var->nestedrefs.dim) {
            assert(!catchstmt->var->ir.irLocal);
            catchstmt->var->ir.irLocal = new IrLocal(catchstmt->var);
            LLValue* catch_var = gIR->func()->gen->landingPadInfo.getExceptionStorage();
            catchstmt->var->ir.irLocal->value = gIR->ir->CreateBitCast(catch_var, getPtrToType(DtoType(catchstmt->var->type)));
        }

        // this will alloca if we haven't already and take care of nested refs
        DtoDeclarationExp(catchstmt->var);

        // the exception will only be stored in catch_var. copy it over if necessary
        if(catchstmt->var->ir.irLocal->value != gIR->func()->gen->landingPadInfo.getExceptionStorage()) {
            LLValue* exc = gIR->ir->CreateBitCast(DtoLoad(gIR->func()->gen->landingPadInfo.getExceptionStorage()), DtoType(catchstmt->var->type));
            DtoStore(exc, catchstmt->var->ir.irLocal->value);
        }
    }

    // emit handler, if there is one
    // handler is zero for instance for 'catch { debug foo(); }'
    if(catchstmt->handler)
        catchstmt->handler->toIR(gIR);

    if (!gIR->scopereturned())
        gIR->ir->CreateBr(end);

    DtoDwarfBlockEnd();
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

    // store as invoke target
    padBBs.push(inBB);
    gIR->func()->gen->landingPad = get();
}

void IRLandingPad::pop()
{
    llvm::BasicBlock *inBB = padBBs.top();
    padBBs.pop();
    gIR->func()->gen->landingPad = get();

    size_t n = nInfos.top();
    for (int i = n, c = infos.size(); i < c; ++i)
        infos.at(i).toIR();
    constructLandingPad(inBB);

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

    // personality fn
    llvm::Function* personality_fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_eh_personality");
    // create landingpad
    LLType *retType = LLStructType::get(LLType::getInt8PtrTy(gIR->context()), LLType::getInt32Ty(gIR->context()), NULL);
    llvm::LandingPadInst *landingPad = gIR->ir->CreateLandingPad(retType, personality_fn, 0);
    LLValue* eh_ptr = DtoExtractValue(landingPad, 0);
    LLValue* eh_sel = DtoExtractValue(landingPad, 1);

    // add landingpad clauses, emit finallys and 'if' chain to catch the exception
    llvm::Function* eh_typeid_for_fn = GET_INTRINSIC_DECL(eh_typeid_for);
    std::deque<IRLandingPadInfo> infos = this->infos;
    std::stack<size_t> nInfos = this->nInfos;
    std::deque<IRLandingPadInfo>::reverse_iterator rit, rend = infos.rend();
    bool isFirstCatch = true;
    for(rit = infos.rbegin(); rit != rend; ++rit)
    {
        // if it's a finally, emit its code
        if(rit->finallyBody)
        {
            size_t n = this->nInfos.top();
            this->infos.resize(n);
            this->nInfos.pop();
            rit->finallyBody->toIR(gIR);
            landingPad->setCleanup(true);
        }
        // otherwise it's a catch and we'll add a if-statement
        else
        {
            // if it is a first catch and some catch allocated storage, store exception object
            if(isFirstCatch && catch_var)
            {
                // eh_ptr is a pointer to _d_exception, which has a reference
                // to the Throwable object at offset 0.
                LLType *objectPtrTy = DtoType(ClassDeclaration::object->type->pointerTo());
                LLValue *objectPtr = gIR->ir->CreateBitCast(eh_ptr, objectPtrTy);
                gIR->ir->CreateStore(gIR->ir->CreateLoad(objectPtr), catch_var);
                isFirstCatch = false;
            }
            // create next block
            llvm::BasicBlock *next = llvm::BasicBlock::Create(gIR->context(), "eh.next", gIR->topfunc(), gIR->scopeend());
            // get class info symbol
            LLValue *classInfo = rit->catchType->ir.irAggr->getClassInfoSymbol();
            // add that symbol as landing pad clause
            landingPad->addClause(classInfo);
            // call llvm.eh.typeid.for to get class info index in the exception table
            classInfo = DtoBitCast(classInfo, getPtrToType(DtoType(Type::tint8)));
            LLValue *eh_id = gIR->ir->CreateCall(eh_typeid_for_fn, classInfo);
            // check exception selector (eh_sel) against the class info index
            gIR->ir->CreateCondBr(gIR->ir->CreateICmpEQ(eh_sel, eh_id), rit->target, next);
            gIR->scope() = IRScope(next, gIR->scopeend());
        }
    }

    // restore landing pad infos
    this->infos = infos;
    this->nInfos = nInfos;

    // no catch matched and all finallys executed - resume unwind
    llvm::Function* unwind_resume_fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_eh_resume_unwind");
    gIR->ir->CreateCall(unwind_resume_fn, eh_ptr);
    gIR->ir->CreateUnreachable();

    // restore scope
    gIR->scope() = savedscope;
}

LLValue* IRLandingPad::getExceptionStorage()
{
    if(!catch_var)
    {
        Logger::println("Making new catch var");
        catch_var = DtoAlloca(ClassDeclaration::object->type, "catchvar");
    }
    return catch_var;
}
