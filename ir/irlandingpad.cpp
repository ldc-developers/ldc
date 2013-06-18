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

IRLandingPadCatchInfo::IRLandingPadCatchInfo(Catch* catchstmt_, llvm::BasicBlock* end_) :
    catchStmt(catchstmt_), end(end_)
{
    target = llvm::BasicBlock::Create(gIR->context(), "catch", gIR->topfunc(), end);

    assert(catchStmt->type);
    catchType = catchStmt->type->toBasetype()->isClassHandle();
    assert(catchType);
    catchType->codegen(Type::sir);

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
    DtoDwarfBlockStart(catchStmt->loc);

    // assign storage to catch var
    if (catchStmt->var) {
        // use the same storage for all exceptions that are not accessed in
        // nested functions
        if (!catchStmt->var->nestedrefs.dim) {
            assert(!catchStmt->var->ir.irLocal);
            catchStmt->var->ir.irLocal = new IrLocal(catchStmt->var);
            LLValue* catch_var = gIR->func()->gen->landingPadInfo.getExceptionStorage();
            catchStmt->var->ir.irLocal->value = gIR->ir->CreateBitCast(catch_var, getPtrToType(DtoType(catchStmt->var->type)));
        }

        // this will alloca if we haven't already and take care of nested refs
        DtoDeclarationExp(catchStmt->var);

        // the exception will only be stored in catch_var. copy it over if necessary
        if (catchStmt->var->ir.irLocal->value != gIR->func()->gen->landingPadInfo.getExceptionStorage()) {
            LLValue* exc = gIR->ir->CreateBitCast(DtoLoad(gIR->func()->gen->landingPadInfo.getExceptionStorage()), DtoType(catchStmt->var->type));
            DtoStore(exc, catchStmt->var->ir.irLocal->value);
        }
    }

    // emit handler, if there is one
    // handler is zero for instance for 'catch { debug foo(); }'
    if (catchStmt->handler)
        catchStmt->handler->toIR(gIR);

    if (!gIR->scopereturned())
        gIR->ir->CreateBr(end);

    DtoDwarfBlockEnd();
}


void IRLandingPad::addCatch(Catch* catchstmt, llvm::BasicBlock* end)
{
    unpushedScope.catches.push_back(IRLandingPadCatchInfo(catchstmt, end));
}

void IRLandingPad::addFinally(Statement* finallystmt)
{
    assert(unpushedScope.finallyBody == NULL && "only one finally per try-finally block");
    unpushedScope.finallyBody = finallystmt;
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

    // personality fn
    llvm::Function* personality_fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_eh_personality");
    // create landingpad
    LLType *retType = LLStructType::get(LLType::getInt8PtrTy(gIR->context()), LLType::getInt32Ty(gIR->context()), NULL);
    llvm::LandingPadInst *landingPad = gIR->ir->CreateLandingPad(retType, personality_fn, 0);
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
            LLValue *classInfo = catchItr->catchType->ir.irAggr->getClassInfoSymbol();
            // add that symbol as landing pad clause
            landingPad->addClause(classInfo);
            // call llvm.eh.typeid.for to get class info index in the exception table
            classInfo = DtoBitCast(classInfo, getPtrToType(DtoType(Type::tint8)));
            LLValue *eh_id = gIR->ir->CreateCall(eh_typeid_for_fn, classInfo);
            // check exception selector (eh_sel) against the class info index
            gIR->ir->CreateCondBr(gIR->ir->CreateICmpEQ(eh_sel, eh_id), catchItr->target, next);
            gIR->scope() = IRScope(next, gIR->scopeend());
        }

        if (scope.finallyBody) {
            scope.finallyBody->toIR(gIR);
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
    llvm::Function* unwind_resume_fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_eh_resume_unwind");
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
