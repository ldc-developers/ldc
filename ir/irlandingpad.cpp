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
#include "gen/tollvm.h"
#include "ir/irlandingpad.h"

// creates new landing pad
static llvm::LandingPadInst *createLandingPadInst()
{
    LLType* retType = LLStructType::get(LLType::getInt8PtrTy(gIR->context()),
                                        LLType::getInt32Ty(gIR->context()),
                                        NULL);
#if LDC_LLVM_VER >= 307
    LLFunction* currentFunction = gIR->func()->func;
    if (!currentFunction->hasPersonalityFn()) {
        LLFunction* personalityFn = LLVM_D_GetRuntimeFunction(Loc(), gIR->module, "_d_eh_personality");
        currentFunction->setPersonalityFn(personalityFn);
    }
    return gIR->ir->CreateLandingPad(retType, 0, "landing_pad");
#else
    LLFunction* personalityFn = LLVM_D_GetRuntimeFunction(Loc(), gIR->module, "_d_eh_personality");
    return gIR->ir->CreateLandingPad(retType, personalityFn, 0, "landing_pad");
#endif
}

IRLandingPadCatchInfo::IRLandingPadCatchInfo(Catch* catchstmt_, llvm::BasicBlock* end_) :
    end(end_), catchStmt(catchstmt_)
{
    target = llvm::BasicBlock::Create(gIR->context(), "catch", gIR->topfunc(), end);

    assert(catchStmt->type);
    catchType = catchStmt->type->toBasetype()->isClassHandle();
    assert(catchType);
    DtoResolveClass(catchType);

    if (catchStmt->var) {
        if (!catchStmt->var->nestedrefs.dim) {
            gIR->func()->gen->landingPadInfo.getExceptionStorage();
        }
    }
}

void IRLandingPadCatchInfo::toIR()
{
    if (!catchStmt)
        return;

    gIR->scope() = IRScope(target, target);
    gIR->DBuilder.EmitBlockStart(catchStmt->loc);

    // assign storage to catch var
    if (catchStmt->var) {
        // use the same storage for all exceptions that are not accessed in
        // nested functions
        if (!catchStmt->var->nestedrefs.dim) {
            assert(!isIrLocalCreated(catchStmt->var));
            IrLocal *irLocal = getIrLocal(catchStmt->var, true);
            LLValue* catch_var = gIR->func()->gen->landingPadInfo.getExceptionStorage();
            irLocal->value = gIR->ir->CreateBitCast(catch_var, getPtrToType(DtoType(catchStmt->var->type)));
        } else {
            // this will alloca if we haven't already and take care of nested refs
            DtoDeclarationExp(catchStmt->var);

            // the exception will only be stored in catch_var. copy it over if necessary
            LLValue* exc = gIR->ir->CreateBitCast(DtoLoad(gIR->func()->gen->landingPadInfo.getExceptionStorage()), DtoType(catchStmt->var->type));
            DtoStore(exc, getIrLocal(catchStmt->var)->value);
        }
    }

    // emit handler, if there is one
    // handler is zero for instance for 'catch { debug foo(); }'
    if (catchStmt->handler)
        Statement_toIR(catchStmt->handler, gIR);

    if (!gIR->scopereturned())
        gIR->ir->CreateBr(end);

    gIR->DBuilder.EmitBlockEnd();
}

IRLandingPadFinallyStatementInfo::IRLandingPadFinallyStatementInfo(Statement *finallyBody_) :
    finallyBody(finallyBody_)
{
}

void IRLandingPadFinallyStatementInfo::toIR(LLValue *eh_ptr)
{
    IRLandingPad &padInfo = gIR->func()->gen->landingPadInfo;
    llvm::BasicBlock* &pad = gIR->func()->gen->landingPad;

    // create collision landing pad that handles exceptions thrown inside the finally block
    llvm::BasicBlock *collision = llvm::BasicBlock::Create(gIR->context(), "eh.collision", gIR->topfunc(), gIR->scopeend());
    llvm::BasicBlock *bb = gIR->scopebb();
    gIR->scope() = IRScope(collision, gIR->scopeend());
    llvm::LandingPadInst *collisionLandingPad = createLandingPadInst();
    LLValue* collision_eh_ptr = DtoExtractValue(collisionLandingPad, 0);
    collisionLandingPad->setCleanup(true);
    llvm::Function* collision_fn = LLVM_D_GetRuntimeFunction(Loc(), gIR->module, "_d_eh_handle_collision");
    gIR->CreateCallOrInvoke2(collision_fn, collision_eh_ptr, eh_ptr);
    gIR->ir->CreateUnreachable();
    gIR->scope() = IRScope(bb, gIR->scopeend());

    // set collision landing pad as unwind target and emit the body of the finally
    gIR->DBuilder.EmitBlockStart(finallyBody->loc);
    padInfo.scopeStack.push(IRLandingPadScope(collision));
    pad = collision;
    Statement_toIR(finallyBody, gIR);
    padInfo.scopeStack.pop();
    pad = padInfo.get();
    gIR->DBuilder.EmitBlockEnd();
}

void IRLandingPad::addCatch(Catch* catchstmt, llvm::BasicBlock* end)
{
    unpushedScope.catches.push_back(IRLandingPadCatchInfo(catchstmt, end));
}

void IRLandingPad::addFinally(Statement* finallyStmt)
{
    assert(unpushedScope.finally == NULL && "only one finally per try-finally block");
    unpushedScope.finally = new IRLandingPadFinallyStatementInfo(finallyStmt);
    unpushedScope.isFinallyCreatedInternally = true;
}

void IRLandingPad::addFinally(IRLandingPadCatchFinallyInfo *finallyInfo)
{
    assert(unpushedScope.finally == NULL && "only one finally per try-finally block");
    unpushedScope.finally = finallyInfo;
}

void IRLandingPad::push(llvm::BasicBlock* inBB)
{
    unpushedScope.target = inBB;
    scopeStack.push(unpushedScope);
    unpushedScope = IRLandingPadScope();
    gIR->func()->gen->landingPad = get();
}

void IRLandingPad::pop()
{
    IRLandingPadScope scope = scopeStack.top();
    scopeStack.pop();
    gIR->func()->gen->landingPad = get();

    std::deque<IRLandingPadCatchInfo>::iterator itr, end = scope.catches.end();
    for (itr = scope.catches.begin(); itr != end; ++itr)
        itr->toIR();
    constructLandingPad(scope);
    if (scope.finally && scope.isFinallyCreatedInternally)
        delete scope.finally;
}

llvm::BasicBlock* IRLandingPad::get()
{
    if (scopeStack.size() == 0)
        return NULL;
    else
        return scopeStack.top().target;
}

void IRLandingPad::constructLandingPad(IRLandingPadScope scope)
{
    // save and rewrite scope
    IRScope savedIRScope = gIR->scope();
    gIR->scope() = IRScope(scope.target, savedIRScope.end);

    // create landingpad
    llvm::LandingPadInst *landingPad = createLandingPadInst();
    LLValue* eh_ptr = DtoExtractValue(landingPad, 0);
    LLValue* eh_sel = DtoExtractValue(landingPad, 1);

    // add landingpad clauses, emit finallys and 'if' chain to catch the exception
    llvm::Function* eh_typeid_for_fn = GET_INTRINSIC_DECL(eh_typeid_for);
    bool isFirstCatch = true;
    std::stack<IRLandingPadScope> savedScopeStack = scopeStack;
    std::deque<IRLandingPadCatchInfo>::iterator catchItr, catchItrEnd;
    while (true) {
        catchItr = scope.catches.begin();
        catchItrEnd = scope.catches.end();
        for (; catchItr != catchItrEnd; ++catchItr) {
            // if it is a first catch and some catch allocated storage, store exception object
            if (isFirstCatch && catch_var) {
#if LDC_LLVM_VER >= 305
                if (global.params.targetTriple.isWindowsMSVCEnvironment())
                {
                    // eh_ptr is a pointer to the Throwable object.
                    LLType *objectTy = DtoType(ClassDeclaration::object->type);
                    LLValue *object = gIR->ir->CreateBitCast(eh_ptr, objectTy);
                    gIR->ir->CreateStore(object, catch_var);
                }
                else
#endif
                {
                    // eh_ptr is a pointer to _d_exception, which has a reference
                    // to the Throwable object at offset 0.
                    LLType *objectPtrTy = DtoType(ClassDeclaration::object->type->pointerTo());
                    LLValue *objectPtr = gIR->ir->CreateBitCast(eh_ptr, objectPtrTy);
                    gIR->ir->CreateStore(gIR->ir->CreateLoad(objectPtr), catch_var);
                }
                isFirstCatch = false;
            }

            // create next block
            llvm::BasicBlock *next = llvm::BasicBlock::Create(gIR->context(), "eh.next", gIR->topfunc(), gIR->scopeend());
            // get class info symbol
            LLValue *classInfo = getIrAggr(catchItr->catchType)->getClassInfoSymbol();
            // add that symbol as landing pad clause
            landingPad->addClause(llvm::cast<llvm::Constant>(classInfo));
            // call llvm.eh.typeid.for to get class info index in the exception table
            classInfo = DtoBitCast(classInfo, getPtrToType(DtoType(Type::tint8)));
            LLValue *eh_id = gIR->ir->CreateCall(eh_typeid_for_fn, classInfo);
            // check exception selector (eh_sel) against the class info index
            gIR->ir->CreateCondBr(gIR->ir->CreateICmpEQ(eh_sel, eh_id), catchItr->target, next);
            gIR->scope() = IRScope(next, gIR->scopeend());
        }

        if (scope.finally) {
           scope.finally->toIR(eh_ptr);
           landingPad->setCleanup(true);
        }

        if (scopeStack.empty())
            break;
        scope = scopeStack.top();
        scopeStack.pop();
        gIR->func()->gen->landingPad = get();
    }

    // restore landing pad infos
    scopeStack = savedScopeStack;
    gIR->func()->gen->landingPad = get();

    // no catch matched and all finallys executed - resume unwind
    llvm::Function* unwind_resume_fn = LLVM_D_GetRuntimeFunction(Loc(), gIR->module, "_d_eh_resume_unwind");
    gIR->ir->CreateCall(unwind_resume_fn, eh_ptr);
    gIR->ir->CreateUnreachable();

    // restore scope
    gIR->scope() = savedIRScope;
}

LLValue* IRLandingPad::getExceptionStorage()
{
    if (!catch_var) {
        Logger::println("Making new catch var");
        catch_var = DtoAlloca(ClassDeclaration::object->type, "catchvar");
    }
    return catch_var;
}
