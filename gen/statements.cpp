//===-- statements.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "init.h"
#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "port.h"
#include "gen/abi.h"
#include "gen/arrays.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "ir/irlandingpad.h"
#include "ir/irmodule.h"
#include "llvm/Support/CFG.h"
#if LDC_LLVM_VER >= 303
#include "llvm/IR/InlineAsm.h"
#else
#include "llvm/InlineAsm.h"
#endif
#include <fstream>
#include <math.h>
#include <stdio.h>

// Need to include this after the other DMD includes because of missing
// dependencies.
#include "hdrgen.h"

//////////////////////////////////////////////////////////////////////////////

void CompoundStatement::toIR(IRState* p)
{
    Logger::println("CompoundStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    for (unsigned i=0; i<statements->dim; i++)
    {
        Statement* s = static_cast<Statement*>(statements->data[i]);
        if (s) {
            s->toIR(p);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

void ReturnStatement::toIR(IRState* p)
{
    Logger::println("ReturnStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // emit dwarf stop point
    gIR->DBuilder.EmitStopPoint(loc.linnum);

    // is there a return value expression?
    if (exp || (!exp && (p->topfunc() == p->mainFunc)) )
    {
        // if the functions return type is void this means that
        // we are returning through a pointer argument
        if (p->topfunc()->getReturnType() == LLType::getVoidTy(gIR->context()))
        {
            // sanity check
            IrFunction* f = p->func();
            assert(f->decl->ir.irFunc->retArg);

            // FIXME: is there ever a case where a sret return needs to be rewritten for the ABI?

            // get return pointer
            DValue* rvar = new DVarValue(f->type->next, f->decl->ir.irFunc->retArg);
            DValue* e = exp->toElemDtor(p);
            // store return value
            if (rvar->getLVal() != e->getRVal())
                DtoAssign(loc, rvar, e);

            // call postblit if necessary
            if (!p->func()->type->isref && !(f->decl->nrvo_can && f->decl->nrvo_var))
                callPostblit(loc, exp, rvar->getLVal());

            // emit scopes
            DtoEnclosingHandlers(loc, NULL);

            // emit dbg end function
            gIR->DBuilder.EmitFuncEnd(f->decl);

            // emit ret
            llvm::ReturnInst::Create(gIR->context(), p->scopebb());

        }
        // the return type is not void, so this is a normal "register" return
        else
        {
            LLValue* v = 0;
            if (!exp && (p->topfunc() == p->mainFunc)) {
                v = LLConstant::getNullValue(p->mainFunc->getReturnType());
            } else {
                if (exp->op == TOKnull)
                    exp->type = p->func()->type->next;
                DValue* dval = 0;
                // call postblit if necessary
                if (!p->func()->type->isref) {
                    dval = exp->toElemDtor(p);
                    callPostblit(loc, exp, dval->getRVal());
                } else {
                    Expression *ae = exp->addressOf(NULL);
                    dval = ae->toElemDtor(p);
                }
                // do abi specific transformations on the return value
                v = p->func()->type->fty.putRet(exp->type, dval);
            }

            if (Logger::enabled())
                Logger::cout() << "return value is '" <<*v << "'\n";

            IrFunction* f = p->func();
            // Hack around LDC assuming structs and static arrays are in memory:
            // If the function returns a struct or a static array, and the return
            // value is a pointer to a struct or a static array, load from it
            // before returning.
            int ty = f->type->next->toBasetype()->ty;
            if (v->getType() != p->topfunc()->getReturnType() &&
                (ty == Tstruct
                 || ty == Tsarray
                 ) && isaPointer(v->getType()))
            {
                Logger::println("Loading value for return");
                v = DtoLoad(v);
            }

            // can happen for classes and void main
            if (v->getType() != p->topfunc()->getReturnType())
            {
                // for the main function this only happens if it is declared as void
                // and then contains a return (exp); statement. Since the actual
                // return type remains i32, we just throw away the exp value
                // and return 0 instead
                // if we're not in main, just bitcast
                if (p->topfunc() == p->mainFunc)
                    v = LLConstant::getNullValue(p->mainFunc->getReturnType());
                else
                    v = gIR->ir->CreateBitCast(v, p->topfunc()->getReturnType(), "tmp");

                if (Logger::enabled())
                    Logger::cout() << "return value after cast: " << *v << '\n';
            }

            // emit scopes
            DtoEnclosingHandlers(loc, NULL);

            gIR->DBuilder.EmitFuncEnd(p->func()->decl);
            llvm::ReturnInst::Create(gIR->context(), v, p->scopebb());
        }
    }
    // no return value expression means it's a void function
    else
    {
        assert(p->topfunc()->getReturnType() == LLType::getVoidTy(gIR->context()));
        DtoEnclosingHandlers(loc, NULL);
        gIR->DBuilder.EmitFuncEnd(p->func()->decl);
        llvm::ReturnInst::Create(gIR->context(), p->scopebb());
    }

    // the return terminated this basicblock, start a new one
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "afterreturn", p->topfunc(), oldend);
    p->scope() = IRScope(bb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void ExpStatement::toIR(IRState* p)
{
    Logger::println("ExpStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // emit dwarf stop point
    gIR->DBuilder.EmitStopPoint(loc.linnum);

    if (exp) {
        elem* e;
        // a cast(void) around the expression is allowed, but doesn't require any code
        if(exp->op == TOKcast && exp->type == Type::tvoid) {
            CastExp* cexp = static_cast<CastExp*>(exp);
            e = cexp->e1->toElemDtor(p);
        }
        else
            e = exp->toElemDtor(p);
        delete e;
    }
    /*elem* e = exp->toElem(p);
    p->buf.printf("%s", e->toChars());
    delete e;
    p->buf.writenl();*/
}

//////////////////////////////////////////////////////////////////////////////

void DtorExpStatement::toIR(IRState *irs)
{
    assert(irs->func());
    FuncDeclaration *fd = irs->func()->decl;
    assert(fd);
    if (fd->nrvo_can && fd->nrvo_var == var)
        /* Do not call destructor, because var is returned as the nrvo variable.
         * This is done at this stage because nrvo can be turned off at a
         * very late stage in semantic analysis.
         */
        ;
    else
    {
        ExpStatement::toIR(irs);
    }
}

//////////////////////////////////////////////////////////////////////////////

void IfStatement::toIR(IRState* p)
{
    Logger::println("IfStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // start a dwarf lexical block
    gIR->DBuilder.EmitBlockStart(loc);
    if (match)
        DtoRawVarDeclaration(match);

    DValue* cond_e = condition->toElemDtor(p);
    LLValue* cond_val = cond_e->getRVal();

    llvm::BasicBlock* oldend = gIR->scopeend();

    llvm::BasicBlock* ifbb = llvm::BasicBlock::Create(gIR->context(), "if", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "endif", gIR->topfunc(), oldend);
    llvm::BasicBlock* elsebb = elsebody ? llvm::BasicBlock::Create(gIR->context(), "else", gIR->topfunc(), endbb) : endbb;

    if (cond_val->getType() != LLType::getInt1Ty(gIR->context())) {
        if (Logger::enabled())
            Logger::cout() << "if conditional: " << *cond_val << '\n';
        cond_val = DtoCast(loc, cond_e, Type::tbool)->getRVal();
    }
    llvm::BranchInst::Create(ifbb, elsebb, cond_val, gIR->scopebb());

    // replace current scope
    gIR->scope() = IRScope(ifbb,elsebb);

    // do scoped statements

    if (ifbody) {
        gIR->DBuilder.EmitBlockStart(ifbody->loc);
        ifbody->toIR(p);
        gIR->DBuilder.EmitBlockEnd();
    }
    if (!gIR->scopereturned()) {
        llvm::BranchInst::Create(endbb,gIR->scopebb());
    }

    if (elsebody) {
        //assert(0);
        gIR->scope() = IRScope(elsebb,endbb);
        gIR->DBuilder.EmitBlockStart(elsebody->loc);
        elsebody->toIR(p);
        if (!gIR->scopereturned()) {
            llvm::BranchInst::Create(endbb,gIR->scopebb());
        }
        gIR->DBuilder.EmitBlockEnd();
    }

    // end the dwarf lexical block
    gIR->DBuilder.EmitBlockEnd();

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void ScopeStatement::toIR(IRState* p)
{
    Logger::println("ScopeStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    /*llvm::BasicBlock* oldend = p->scopeend();

    llvm::BasicBlock* beginbb = 0;

    // remove useless branches by clearing and reusing the current basicblock
    llvm::BasicBlock* bb = p->scopebb();
    if (bb->empty()) {
        beginbb = bb;
    }
    else {
        beginbb = llvm::BasicBlock::Create(gIR->context(), "scope", p->topfunc(), oldend);
        if (!p->scopereturned())
            llvm::BranchInst::Create(beginbb, bb);
    }

    llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "endscope", p->topfunc(), oldend);
    if (beginbb != bb)
        p->scope() = IRScope(beginbb, endbb);
    else
        p->scope().end = endbb;*/

    if (statement) {
        gIR->DBuilder.EmitBlockStart(statement->loc);
        statement->toIR(p);
        gIR->DBuilder.EmitBlockEnd();
    }

    /*p->scope().end = oldend;
    Logger::println("Erasing scope endbb");
    endbb->eraseFromParent();*/
}

//////////////////////////////////////////////////////////////////////////////

void WhileStatement::toIR(IRState* p)
{
    Logger::println("WhileStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // start a dwarf lexical block
    gIR->DBuilder.EmitBlockStart(loc);

    // create while blocks
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* whilebb = llvm::BasicBlock::Create(gIR->context(), "whilecond", gIR->topfunc(), oldend);
    llvm::BasicBlock* whilebodybb = llvm::BasicBlock::Create(gIR->context(), "whilebody", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "endwhile", gIR->topfunc(), oldend);

    // move into the while block
    p->ir->CreateBr(whilebb);
    //llvm::BranchInst::Create(whilebb, gIR->scopebb());

    // replace current scope
    gIR->scope() = IRScope(whilebb,endbb);

    // create the condition
    DValue* cond_e = condition->toElemDtor(p);
    LLValue* cond_val = DtoCast(loc, cond_e, Type::tbool)->getRVal();
    delete cond_e;

    // conditional branch
    llvm::BranchInst::Create(whilebodybb, endbb, cond_val, p->scopebb());

    // rewrite scope
    gIR->scope() = IRScope(whilebodybb,endbb);

    // while body code
    p->func()->gen->targetScopes.push_back(IRTargetScope(this,NULL,whilebb,endbb));
    if (body)
        body->toIR(p);
    p->func()->gen->targetScopes.pop_back();

    // loop
    if (!gIR->scopereturned())
        llvm::BranchInst::Create(whilebb, gIR->scopebb());

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);

    // end the dwarf lexical block
    gIR->DBuilder.EmitBlockEnd();
}

//////////////////////////////////////////////////////////////////////////////

void DoStatement::toIR(IRState* p)
{
    Logger::println("DoStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // start a dwarf lexical block
    gIR->DBuilder.EmitBlockStart(loc);

    // create while blocks
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* dowhilebb = llvm::BasicBlock::Create(gIR->context(), "dowhile", gIR->topfunc(), oldend);
    llvm::BasicBlock* condbb = llvm::BasicBlock::Create(gIR->context(), "dowhilecond", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "enddowhile", gIR->topfunc(), oldend);

    // move into the while block
    assert(!gIR->scopereturned());
    llvm::BranchInst::Create(dowhilebb, gIR->scopebb());

    // replace current scope
    gIR->scope() = IRScope(dowhilebb,condbb);

    // do-while body code
    p->func()->gen->targetScopes.push_back(IRTargetScope(this,NULL,condbb,endbb));
    if (body)
        body->toIR(p);
    p->func()->gen->targetScopes.pop_back();

    // branch to condition block
    llvm::BranchInst::Create(condbb, gIR->scopebb());
    gIR->scope() = IRScope(condbb,endbb);

    // create the condition
    DValue* cond_e = condition->toElemDtor(p);
    LLValue* cond_val = DtoCast(loc, cond_e, Type::tbool)->getRVal();
    delete cond_e;

    // conditional branch
    llvm::BranchInst::Create(dowhilebb, endbb, cond_val, gIR->scopebb());

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);

    // end the dwarf lexical block
    gIR->DBuilder.EmitBlockEnd();
}

//////////////////////////////////////////////////////////////////////////////

void ForStatement::toIR(IRState* p)
{
    Logger::println("ForStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // start new dwarf lexical block
    gIR->DBuilder.EmitBlockStart(loc);

    // create for blocks
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* forbb = llvm::BasicBlock::Create(gIR->context(), "forcond", gIR->topfunc(), oldend);
    llvm::BasicBlock* forbodybb = llvm::BasicBlock::Create(gIR->context(), "forbody", gIR->topfunc(), oldend);
    llvm::BasicBlock* forincbb = llvm::BasicBlock::Create(gIR->context(), "forinc", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "endfor", gIR->topfunc(), oldend);

    // init
    if (init != 0)
        init->toIR(p);

    // move into the for condition block, ie. start the loop
    assert(!gIR->scopereturned());
    llvm::BranchInst::Create(forbb, gIR->scopebb());

    p->func()->gen->targetScopes.push_back(IRTargetScope(this,NULL,forincbb,endbb));

    // replace current scope
    gIR->scope() = IRScope(forbb,forbodybb);

    // create the condition
    LLValue* cond_val;
    if (condition)
    {
        DValue* cond_e = condition->toElemDtor(p);
        cond_val = DtoCast(loc, cond_e, Type::tbool)->getRVal();
        delete cond_e;
    }
    else
    {
        cond_val = DtoConstBool(true);
    }

    // conditional branch
    assert(!gIR->scopereturned());
    llvm::BranchInst::Create(forbodybb, endbb, cond_val, gIR->scopebb());

    // rewrite scope
    gIR->scope() = IRScope(forbodybb,forincbb);

    // do for body code
    if (body)
        body->toIR(p);

    // move into the for increment block
    if (!gIR->scopereturned())
        llvm::BranchInst::Create(forincbb, gIR->scopebb());
    gIR->scope() = IRScope(forincbb, endbb);

    // increment
    if (increment) {
        DValue* inc = increment->toElemDtor(p);
        delete inc;
    }

    // loop
    if (!gIR->scopereturned())
        llvm::BranchInst::Create(forbb, gIR->scopebb());

    p->func()->gen->targetScopes.pop_back();

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);

    // end the dwarf lexical block
    gIR->DBuilder.EmitBlockEnd();
}

//////////////////////////////////////////////////////////////////////////////

void BreakStatement::toIR(IRState* p)
{
    Logger::println("BreakStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // don't emit two terminators in a row
    // happens just before DMD generated default statements if the last case terminates
    if (p->scopereturned())
        return;

    // emit dwarf stop point
    gIR->DBuilder.EmitStopPoint(loc.linnum);

    if (ident != 0) {
        Logger::println("ident = %s", ident->toChars());

        DtoEnclosingHandlers(loc, target);

        // get the loop statement the label refers to
        Statement* targetLoopStatement = target->statement;
        ScopeStatement* tmp;
        while((tmp = targetLoopStatement->isScopeStatement()))
            targetLoopStatement = tmp->statement;

        // find the right break block and jump there
        bool found = false;
        FuncGen::TargetScopeVec::reverse_iterator it = p->func()->gen->targetScopes.rbegin();
        FuncGen::TargetScopeVec::reverse_iterator it_end = p->func()->gen->targetScopes.rend();
        while(it != it_end) {
            if(it->s == targetLoopStatement) {
                llvm::BranchInst::Create(it->breakTarget, p->scopebb());
                found = true;
                break;
            }
            ++it;
        }
        assert(found);
    }
    else {
        // find closest scope with a break target
        FuncGen::TargetScopeVec::reverse_iterator it = p->func()->gen->targetScopes.rbegin();
        FuncGen::TargetScopeVec::reverse_iterator it_end = p->func()->gen->targetScopes.rend();
        while(it != it_end) {
            if(it->breakTarget && !it->onlyLabeledBreak) {
                break;
            }
            ++it;
        }
        DtoEnclosingHandlers(loc, it->s);
        llvm::BranchInst::Create(it->breakTarget, gIR->scopebb());
    }

    // the break terminated this basicblock, start a new one
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "afterbreak", p->topfunc(), oldend);
    p->scope() = IRScope(bb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void ContinueStatement::toIR(IRState* p)
{
    Logger::println("ContinueStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // emit dwarf stop point
    gIR->DBuilder.EmitStopPoint(loc.linnum);

    if (ident != 0) {
        Logger::println("ident = %s", ident->toChars());

        DtoEnclosingHandlers(loc, target);

        // get the loop statement the label refers to
        Statement* targetLoopStatement = target->statement;
        ScopeStatement* tmp;
        while((tmp = targetLoopStatement->isScopeStatement()))
            targetLoopStatement = tmp->statement;

        // find the right continue block and jump there
        bool found = false;
        FuncGen::TargetScopeVec::reverse_iterator it = p->func()->gen->targetScopes.rbegin();
        FuncGen::TargetScopeVec::reverse_iterator it_end = p->func()->gen->targetScopes.rend();
        while(it != it_end) {
            if(it->s == targetLoopStatement) {
                llvm::BranchInst::Create(it->continueTarget, gIR->scopebb());
                found = true;
                break;
            }
            ++it;
        }
        assert(found);
    }
    else {
        // find closest scope with a continue target
        FuncGen::TargetScopeVec::reverse_iterator it = p->func()->gen->targetScopes.rbegin();
        FuncGen::TargetScopeVec::reverse_iterator it_end = p->func()->gen->targetScopes.rend();
        while(it != it_end) {
            if(it->continueTarget) {
                break;
            }
            ++it;
        }
        DtoEnclosingHandlers(loc, it->s);
        llvm::BranchInst::Create(it->continueTarget, gIR->scopebb());
    }

    // the continue terminated this basicblock, start a new one
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "aftercontinue", p->topfunc(), oldend);
    p->scope() = IRScope(bb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void OnScopeStatement::toIR(IRState* p)
{
    Logger::println("OnScopeStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    assert(statement);
    //statement->toIR(p); // this seems to be redundant
}

//////////////////////////////////////////////////////////////////////////////

void TryFinallyStatement::toIR(IRState* p)
{
    Logger::println("TryFinallyStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // emit dwarf stop point
    gIR->DBuilder.EmitStopPoint(loc.linnum);

    // if there's no finalbody or no body, things are simple
    if (!finalbody) {
        if (body) {
            gIR->DBuilder.EmitBlockStart(body->loc);
            body->toIR(p);
            gIR->DBuilder.EmitBlockEnd();
        }
        return;
    }
    if (!body) {
        gIR->DBuilder.EmitBlockStart(finalbody->loc);
        finalbody->toIR(p);
        gIR->DBuilder.EmitBlockEnd();
        return;
    }

    // create basic blocks
    llvm::BasicBlock* oldend = p->scopeend();

    llvm::BasicBlock* trybb = llvm::BasicBlock::Create(gIR->context(), "try", p->topfunc(), oldend);
    llvm::BasicBlock* finallybb = llvm::BasicBlock::Create(gIR->context(), "finally", p->topfunc(), oldend);
    // the landing pad for statements in the try block
    llvm::BasicBlock* landingpadbb = llvm::BasicBlock::Create(gIR->context(), "landingpad", p->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "endtryfinally", p->topfunc(), oldend);

    // pass the previous BB into this
    assert(!gIR->scopereturned());
    llvm::BranchInst::Create(trybb, p->scopebb());

    //
    // set up the landing pad
    //
    p->scope() = IRScope(landingpadbb, endbb);

    assert(finalbody);
    IRLandingPad& pad = gIR->func()->gen->landingPadInfo;
    pad.addFinally(finalbody);
    pad.push(landingpadbb);
    gIR->func()->gen->targetScopes.push_back(
        IRTargetScope(
            this,
            new EnclosingTryFinally(this, gIR->func()->gen->landingPad),
            NULL,
            endbb,
            true
        )
    );

    //
    // do the try block
    //
    p->scope() = IRScope(trybb,finallybb);

    assert(body);
    gIR->DBuilder.EmitBlockStart(body->loc);
    body->toIR(p);
    gIR->DBuilder.EmitBlockEnd();

    // terminate try BB
    if (!p->scopereturned())
        llvm::BranchInst::Create(finallybb, p->scopebb());

    pad.pop();
    gIR->func()->gen->targetScopes.pop_back();

    //
    // do finally block
    //
    p->scope() = IRScope(finallybb,landingpadbb);
    gIR->DBuilder.EmitBlockStart(finalbody->loc);
    finalbody->toIR(p);
    gIR->DBuilder.EmitBlockEnd();

    // terminate finally
    //TODO: isn't it an error to have a 'returned' finally block?
    if (!gIR->scopereturned()) {
        llvm::BranchInst::Create(endbb, p->scopebb());
    }

    // rewrite the scope
    p->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void TryCatchStatement::toIR(IRState* p)
{
    Logger::println("TryCatchStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // emit dwarf stop point
    gIR->DBuilder.EmitStopPoint(loc.linnum);

    // create basic blocks
    llvm::BasicBlock* oldend = p->scopeend();

    llvm::BasicBlock* trybb = llvm::BasicBlock::Create(gIR->context(), "try", p->topfunc(), oldend);
    // the landing pad will be responsible for branching to the correct catch block
    llvm::BasicBlock* landingpadbb = llvm::BasicBlock::Create(gIR->context(), "landingpad", p->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "endtrycatch", p->topfunc(), oldend);

    // pass the previous BB into this
    assert(!gIR->scopereturned());
    llvm::BranchInst::Create(trybb, p->scopebb());

    //
    // set up the landing pad
    //
    assert(catches);
    gIR->scope() = IRScope(landingpadbb, endbb);

    IRLandingPad& pad = gIR->func()->gen->landingPadInfo;
    for (unsigned i = 0; i < catches->dim; i++)
    {
        Catch *c = static_cast<Catch *>(catches->data[i]);
        pad.addCatch(c, endbb);
    }

    pad.push(landingpadbb);

    //
    // do the try block
    //
    p->scope() = IRScope(trybb,landingpadbb);

    assert(body);
    gIR->DBuilder.EmitBlockStart(body->loc);
    body->toIR(p);
    gIR->DBuilder.EmitBlockEnd();

    if (!gIR->scopereturned())
        llvm::BranchInst::Create(endbb, p->scopebb());

    pad.pop();

    // rewrite the scope
    p->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void ThrowStatement::toIR(IRState* p)
{
    Logger::println("ThrowStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // emit dwarf stop point
    gIR->DBuilder.EmitStopPoint(loc.linnum);

    assert(exp);
    DValue* e = exp->toElemDtor(p);

    gIR->DBuilder.EmitFuncEnd(gIR->func()->decl);

    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_throw_exception");
    //Logger::cout() << "calling: " << *fn << '\n';
    LLValue* arg = DtoBitCast(e->getRVal(), fn->getFunctionType()->getParamType(0));
    //Logger::cout() << "arg: " << *arg << '\n';
    gIR->CreateCallOrInvoke(fn, arg);
    gIR->ir->CreateUnreachable();

    // need a block after the throw for now
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "afterthrow", p->topfunc(), oldend);
    p->scope() = IRScope(bb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

// used to build the sorted list of cases
struct Case : Object
{
    StringExp* str;
    size_t index;

    Case(StringExp* s, size_t i) {
        str = s;
        index = i;
    }

    int compare(Object *obj) {
        Case* c2 = static_cast<Case*>(obj);
        return str->compare(c2->str);
    }
};

static LLValue* call_string_switch_runtime(llvm::Value* table, Expression* e)
{
    Type* dt = e->type->toBasetype();
    Type* dtnext = dt->nextOf()->toBasetype();
    TY ty = dtnext->ty;
    const char* fname;
    if (ty == Tchar) {
        fname = "_d_switch_string";
    }
    else if (ty == Twchar) {
        fname = "_d_switch_ustring";
    }
    else if (ty == Tdchar) {
        fname = "_d_switch_dstring";
    }
    else {
        llvm_unreachable("not char/wchar/dchar");
    }

    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, fname);

    if (Logger::enabled())
    {
        Logger::cout() << *table->getType() << '\n';
        Logger::cout() << *fn->getFunctionType()->getParamType(0) << '\n';
    }
    assert(table->getType() == fn->getFunctionType()->getParamType(0));

    DValue* val = e->toElemDtor(gIR);
    LLValue* llval = val->getRVal();
    assert(llval->getType() == fn->getFunctionType()->getParamType(1));

    LLCallSite call = gIR->CreateCallOrInvoke2(fn, table, llval, "tmp");

    return call.getInstruction();
}

void SwitchStatement::toIR(IRState* p)
{
    Logger::println("SwitchStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // emit dwarf stop point
    gIR->DBuilder.EmitStopPoint(loc.linnum);

    llvm::BasicBlock* oldbb = gIR->scopebb();
    llvm::BasicBlock* oldend = gIR->scopeend();

    // clear data from previous passes... :/
    for (unsigned i=0; i<cases->dim; ++i)
    {
        CaseStatement* cs = static_cast<CaseStatement*>(cases->data[i]);
        cs->bodyBB = NULL;
        cs->llvmIdx = NULL;
    }

    // If one of the case expressions is non-constant, we can't use
    // 'switch' instruction (that can happen because D2 allows to
    // initialize a global variable in a static constructor).
    bool useSwitchInst = true;
    for (unsigned i=0; i<cases->dim; ++i)
    {
        CaseStatement* cs = static_cast<CaseStatement*>(cases->data[i]);
        VarDeclaration* vd = 0;
        if (cs->exp->op == TOKvar)
            vd = static_cast<VarExp*>(cs->exp)->var->isVarDeclaration();
        if (vd && (!vd->init || !vd->isConst())) {
            cs->llvmIdx = cs->exp->toElemDtor(p)->getRVal();
            useSwitchInst = false;
        }
    }

    // body block.
    // FIXME: that block is never used
    llvm::BasicBlock* bodybb = llvm::BasicBlock::Create(gIR->context(), "switchbody", p->topfunc(), oldend);

    // default
    llvm::BasicBlock* defbb = 0;
    if (sdefault) {
        Logger::println("has default");
        defbb = llvm::BasicBlock::Create(gIR->context(), "default", p->topfunc(), oldend);
        sdefault->bodyBB = defbb;
    }

    // end (break point)
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "switchend", p->topfunc(), oldend);

    // do switch body
    assert(body);
    p->scope() = IRScope(bodybb, endbb);
    p->func()->gen->targetScopes.push_back(IRTargetScope(this,NULL,NULL,endbb));
    body->toIR(p);
    p->func()->gen->targetScopes.pop_back();
    if (!p->scopereturned())
        llvm::BranchInst::Create(endbb, p->scopebb());

    gIR->scope() = IRScope(oldbb,oldend);
    if (useSwitchInst)
    {
        // string switch?
        llvm::Value* switchTable = 0;
        Array caseArray;
        if (!condition->type->isintegral())
        {
            Logger::println("is string switch");
            // build array of the stringexpS
            caseArray.reserve(cases->dim);
            for (unsigned i=0; i<cases->dim; ++i)
            {
                CaseStatement* cs = static_cast<CaseStatement*>(cases->data[i]);

                assert(cs->exp->op == TOKstring);
                caseArray.push(new Case(static_cast<StringExp*>(cs->exp), i));
            }
            // first sort it
            caseArray.sort();
            // iterate and add indices to cases
            std::vector<LLConstant*> inits(caseArray.dim, 0);
            for (size_t i=0; i<caseArray.dim; ++i)
            {
                Case* c = static_cast<Case*>(caseArray.data[i]);
                CaseStatement* cs = static_cast<CaseStatement*>(cases->data[c->index]);
                cs->llvmIdx = DtoConstUint(i);
                inits[i] = c->str->toConstElem(p);
            }
            // build static array for ptr or final array
            LLType* elemTy = DtoType(condition->type);
            LLArrayType* arrTy = llvm::ArrayType::get(elemTy, inits.size());
            LLConstant* arrInit = LLConstantArray::get(arrTy, inits);
            LLGlobalVariable* arr = new llvm::GlobalVariable(*gIR->module, arrTy, true, llvm::GlobalValue::InternalLinkage, arrInit, ".string_switch_table_data");

            LLType* elemPtrTy = getPtrToType(elemTy);
            LLConstant* arrPtr = llvm::ConstantExpr::getBitCast(arr, elemPtrTy);

            // build the static table
            LLType* types[] = { DtoSize_t(), elemPtrTy };
            LLStructType* sTy = llvm::StructType::get(gIR->context(), types, false);
            LLConstant* sinits[] = { DtoConstSize_t(inits.size()), arrPtr };
            switchTable = llvm::ConstantStruct::get(sTy, llvm::ArrayRef<LLConstant*>(sinits));
        }

        // condition var
        LLValue* condVal;
        // integral switch
        if (condition->type->isintegral()) {
            DValue* cond = condition->toElemDtor(p);
            condVal = cond->getRVal();
        }
        // string switch
        else {
            condVal = call_string_switch_runtime(switchTable, condition);
        }

        // create switch and add the cases
        llvm::SwitchInst* si = llvm::SwitchInst::Create(condVal, defbb ? defbb : endbb, cases->dim, p->scopebb());
        for (unsigned i=0; i<cases->dim; ++i)
        {
            CaseStatement* cs = static_cast<CaseStatement*>(cases->data[i]);
            si->addCase(isaConstantInt(cs->llvmIdx), cs->bodyBB);
        }
    }
    else
    { // we can't use switch, so we will use a bunch of br instructions instead
        DValue* cond = condition->toElemDtor(p);
        LLValue *condVal = cond->getRVal();

        llvm::BasicBlock* nextbb = llvm::BasicBlock::Create(gIR->context(), "checkcase", p->topfunc(), oldend);
        llvm::BranchInst::Create(nextbb, p->scopebb());

        p->scope() = IRScope(nextbb, endbb);
        for (unsigned i=0; i<cases->dim; ++i)
        {
            CaseStatement* cs = static_cast<CaseStatement*>(cases->data[i]);

            LLValue* cmp = p->ir->CreateICmp(llvm::ICmpInst::ICMP_EQ, cs->llvmIdx, condVal, "checkcase");
            nextbb = llvm::BasicBlock::Create(gIR->context(), "checkcase", p->topfunc(), oldend);
            llvm::BranchInst::Create(cs->bodyBB, nextbb, cmp, p->scopebb());
            p->scope() = IRScope(nextbb, endbb);
        }

        if (sdefault) {
            llvm::BranchInst::Create(sdefault->bodyBB, p->scopebb());
        } else {
            llvm::BranchInst::Create(endbb, p->scopebb());
        }
        endbb->moveAfter(nextbb);
    }

    gIR->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////
void CaseStatement::toIR(IRState* p)
{
    Logger::println("CaseStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    llvm::BasicBlock* nbb = llvm::BasicBlock::Create(gIR->context(), "case", p->topfunc(), p->scopeend());

    if (bodyBB && !bodyBB->getTerminator())
    {
        llvm::BranchInst::Create(nbb, bodyBB);
    }
    bodyBB = nbb;

    if (llvmIdx == NULL) {
        LLConstant* c = exp->toConstElem(p);
        llvmIdx = isaConstantInt(c);
    }

    if (!p->scopereturned())
        llvm::BranchInst::Create(bodyBB, p->scopebb());

    p->scope() = IRScope(bodyBB, p->scopeend());

    assert(statement);
    gIR->DBuilder.EmitBlockStart(statement->loc);
    statement->toIR(p);
    gIR->DBuilder.EmitBlockEnd();
}

//////////////////////////////////////////////////////////////////////////////
void DefaultStatement::toIR(IRState* p)
{
    Logger::println("DefaultStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    assert(bodyBB);

    llvm::BasicBlock* nbb = llvm::BasicBlock::Create(gIR->context(), "default", p->topfunc(), p->scopeend());

    if (!bodyBB->getTerminator())
    {
        llvm::BranchInst::Create(nbb, bodyBB);
    }
    bodyBB = nbb;

    if (!p->scopereturned())
        llvm::BranchInst::Create(bodyBB, p->scopebb());

    p->scope() = IRScope(bodyBB, p->scopeend());

    assert(statement);
    gIR->DBuilder.EmitBlockStart(statement->loc);
    statement->toIR(p);
    gIR->DBuilder.EmitBlockEnd();
}

//////////////////////////////////////////////////////////////////////////////

void UnrolledLoopStatement::toIR(IRState* p)
{
    Logger::println("UnrolledLoopStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // if no statements, there's nothing to do
    if (!statements || !statements->dim)
        return;

    // start a dwarf lexical block
    gIR->DBuilder.EmitBlockStart(loc);

    // DMD doesn't fold stuff like continue/break, and since this isn't really a loop
    // we have to keep track of each statement and jump to the next/end on continue/break

    llvm::BasicBlock* oldend = gIR->scopeend();

    // create a block for each statement
    size_t nstmt = statements->dim;
    LLSmallVector<llvm::BasicBlock*, 4> blocks(nstmt, NULL);

    for (size_t i=0; i<nstmt; i++)
    {
        blocks[i] = llvm::BasicBlock::Create(gIR->context(), "unrolledstmt", p->topfunc(), oldend);
    }

    // create end block
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "unrolledend", p->topfunc(), oldend);

    // enter first stmt
    if (!p->scopereturned())
        p->ir->CreateBr(blocks[0]);

    // do statements
    Statement** stmts = (Statement**)statements->data;

    for (size_t i=0; i<nstmt; i++)
    {
        Statement* s = stmts[i];

        // get blocks
        llvm::BasicBlock* thisbb = blocks[i];
        llvm::BasicBlock* nextbb = (i+1 == nstmt) ? endbb : blocks[i+1];

        // update scope
        p->scope() = IRScope(thisbb,nextbb);

        // push loop scope
        // continue goes to next statement, break goes to end
        p->func()->gen->targetScopes.push_back(IRTargetScope(this,NULL,nextbb,endbb));

        // do statement
        s->toIR(p);

        // pop loop scope
        p->func()->gen->targetScopes.pop_back();

        // next stmt
        if (!p->scopereturned())
            p->ir->CreateBr(nextbb);
    }

    // finish scope
    if (!p->scopereturned())
        p->ir->CreateBr(endbb);
    p->scope() = IRScope(endbb,oldend);

    // end the dwarf lexical block
    gIR->DBuilder.EmitBlockEnd();
}

//////////////////////////////////////////////////////////////////////////////

void ForeachStatement::toIR(IRState* p)
{
    Logger::println("ForeachStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // start a dwarf lexical block
    gIR->DBuilder.EmitBlockStart(loc);

    //assert(arguments->dim == 1);
    assert(value != 0);
    assert(aggr != 0);
    assert(func != 0);

    //Argument* arg = static_cast<Argument*>(arguments->data[0]);
    //Logger::println("Argument is %s", arg->toChars());

    Logger::println("aggr = %s", aggr->toChars());

    // key
    LLType* keytype = key ? DtoType(key->type) : DtoSize_t();
    LLValue* keyvar;
    if (key)
        keyvar = DtoRawVarDeclaration(key);
    else
        keyvar = DtoRawAlloca(keytype, 0, "foreachkey"); // FIXME: align?
    LLValue* zerokey = LLConstantInt::get(keytype,0,false);

    // value
    Logger::println("value = %s", value->toPrettyChars());
    LLValue* valvar = NULL;
    if (!value->isRef() && !value->isOut()) {
        // Create a local variable to serve as the value.
        DtoRawVarDeclaration(value);
        valvar = value->ir.irLocal->value;
    }

    // what to iterate
    DValue* aggrval = aggr->toElemDtor(p);

    // get length and pointer
    LLValue* niters = DtoArrayLen(aggrval);
    LLValue* val = DtoArrayPtr(aggrval);

    if (niters->getType() != keytype)
    {
        size_t sz1 = getTypeBitSize(niters->getType());
        size_t sz2 = getTypeBitSize(keytype);
        if (sz1 < sz2)
            niters = gIR->ir->CreateZExt(niters, keytype, "foreachtrunckey");
        else if (sz1 > sz2)
            niters = gIR->ir->CreateTrunc(niters, keytype, "foreachtrunckey");
        else
            niters = gIR->ir->CreateBitCast(niters, keytype, "foreachtrunckey");
    }

    if (op == TOKforeach) {
        new llvm::StoreInst(zerokey, keyvar, p->scopebb());
    }
    else {
        new llvm::StoreInst(niters, keyvar, p->scopebb());
    }

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* condbb = llvm::BasicBlock::Create(gIR->context(), "foreachcond", p->topfunc(), oldend);
    llvm::BasicBlock* bodybb = llvm::BasicBlock::Create(gIR->context(), "foreachbody", p->topfunc(), oldend);
    llvm::BasicBlock* nextbb = llvm::BasicBlock::Create(gIR->context(), "foreachnext", p->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "foreachend", p->topfunc(), oldend);

    llvm::BranchInst::Create(condbb, p->scopebb());

    // condition
    p->scope() = IRScope(condbb,bodybb);

    LLValue* done = 0;
    LLValue* load = DtoLoad(keyvar);
    if (op == TOKforeach) {
        done = p->ir->CreateICmpULT(load, niters, "tmp");
    }
    else if (op == TOKforeach_reverse) {
        done = p->ir->CreateICmpUGT(load, zerokey, "tmp");
        load = p->ir->CreateSub(load, LLConstantInt::get(keytype, 1, false), "tmp");
        DtoStore(load, keyvar);
    }
    llvm::BranchInst::Create(bodybb, endbb, done, p->scopebb());

    // init body
    p->scope() = IRScope(bodybb,nextbb);

    // get value for this iteration
    LLValue* loadedKey = p->ir->CreateLoad(keyvar,"tmp");
    LLValue* gep = DtoGEP1(val,loadedKey);

    if (!value->isRef() && !value->isOut()) {
        // Copy value to local variable, and use it as the value variable.
        DVarValue dst(value->type, valvar);
        DVarValue src(value->type, gep);
        DtoAssign(loc, &dst, &src);
        value->ir.irLocal->value = valvar;
    } else {
        // Use the GEP as the address of the value variable.
        DtoRawVarDeclaration(value, gep);
    }

    // emit body
    p->func()->gen->targetScopes.push_back(IRTargetScope(this,NULL,nextbb,endbb));
    if(body)
        body->toIR(p);
    p->func()->gen->targetScopes.pop_back();

    if (!p->scopereturned())
        llvm::BranchInst::Create(nextbb, p->scopebb());

    // next
    p->scope() = IRScope(nextbb,endbb);
    if (op == TOKforeach) {
        LLValue* load = DtoLoad(keyvar);
        load = p->ir->CreateAdd(load, LLConstantInt::get(keytype, 1, false), "tmp");
        DtoStore(load, keyvar);
    }
    llvm::BranchInst::Create(condbb, p->scopebb());

    // end the dwarf lexical block
    gIR->DBuilder.EmitBlockEnd();

    // end
    p->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void ForeachRangeStatement::toIR(IRState* p)
{
    Logger::println("ForeachRangeStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // start a dwarf lexical block
    gIR->DBuilder.EmitBlockStart(loc);

    // evaluate lwr/upr
    assert(lwr->type->isintegral());
    LLValue* lower = lwr->toElemDtor(p)->getRVal();
    assert(upr->type->isintegral());
    LLValue* upper = upr->toElemDtor(p)->getRVal();

    // handle key
    assert(key->type->isintegral());
    LLValue* keyval = DtoRawVarDeclaration(key);

    // store initial value in key
    if (op == TOKforeach)
        DtoStore(lower, keyval);
    else
        DtoStore(upper, keyval);

    // set up the block we'll need
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* condbb = llvm::BasicBlock::Create(gIR->context(), "foreachrange_cond", p->topfunc(), oldend);
    llvm::BasicBlock* bodybb = llvm::BasicBlock::Create(gIR->context(), "foreachrange_body", p->topfunc(), oldend);
    llvm::BasicBlock* nextbb = llvm::BasicBlock::Create(gIR->context(), "foreachrange_next", p->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "foreachrange_end", p->topfunc(), oldend);

    // jump to condition
    llvm::BranchInst::Create(condbb, p->scopebb());

    // CONDITION
    p->scope() = IRScope(condbb,bodybb);

    // first we test that lwr < upr
    lower = DtoLoad(keyval);
    assert(lower->getType() == upper->getType());
    llvm::ICmpInst::Predicate cmpop;
    if (isLLVMUnsigned(key->type))
    {
        cmpop = (op == TOKforeach)
        ? llvm::ICmpInst::ICMP_ULT
        : llvm::ICmpInst::ICMP_UGT;
    }
    else
    {
        cmpop = (op == TOKforeach)
        ? llvm::ICmpInst::ICMP_SLT
        : llvm::ICmpInst::ICMP_SGT;
    }
    LLValue* cond = p->ir->CreateICmp(cmpop, lower, upper);

    // jump to the body if range is ok, to the end if not
    llvm::BranchInst::Create(bodybb, endbb, cond, p->scopebb());

    // BODY
    p->scope() = IRScope(bodybb,nextbb);

    // reverse foreach decrements here
    if (op == TOKforeach_reverse)
    {
        LLValue* v = DtoLoad(keyval);
        LLValue* one = LLConstantInt::get(v->getType(), 1, false);
        v = p->ir->CreateSub(v, one);
        DtoStore(v, keyval);
    }

    // emit body
    p->func()->gen->targetScopes.push_back(IRTargetScope(this,NULL,nextbb,endbb));
    if (body)
        body->toIR(p);
    p->func()->gen->targetScopes.pop_back();

    // jump to next iteration
    if (!p->scopereturned())
        llvm::BranchInst::Create(nextbb, p->scopebb());

    // NEXT
    p->scope() = IRScope(nextbb,endbb);

    // forward foreach increments here
    if (op == TOKforeach)
    {
        LLValue* v = DtoLoad(keyval);
        LLValue* one = LLConstantInt::get(v->getType(), 1, false);
        v = p->ir->CreateAdd(v, one);
        DtoStore(v, keyval);
    }

    // jump to condition
    llvm::BranchInst::Create(condbb, p->scopebb());

    // end the dwarf lexical block
    gIR->DBuilder.EmitBlockEnd();

    // END
    p->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void LabelStatement::toIR(IRState* p)
{
    Logger::println("LabelStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // if it's an inline asm label, we don't create a basicblock, just emit it in the asm
    if (p->asmBlock)
    {
        IRAsmStmt* a = new IRAsmStmt;
        std::stringstream label;
        printLabelName(label, p->func()->decl->mangle(), ident->toChars());
        label << ":";
        a->code = label.str();
        p->asmBlock->s.push_back(a);
        p->asmBlock->internalLabels.push_back(ident);

        // disable inlining
        gIR->func()->setNeverInline();
    }
    else
    {
        std::string labelname = p->func()->gen->getScopedLabelName(ident->toChars());
        llvm::BasicBlock*& labelBB = p->func()->gen->labelToBB[labelname];

        llvm::BasicBlock* oldend = gIR->scopeend();
        if (labelBB != NULL) {
            labelBB->moveBefore(oldend);
        } else {
            labelBB = llvm::BasicBlock::Create(gIR->context(), "label_" + labelname, p->topfunc(), oldend);
        }

        if (!p->scopereturned())
            llvm::BranchInst::Create(labelBB, p->scopebb());

        p->scope() = IRScope(labelBB,oldend);
    }

    if (statement) {
        p->func()->gen->targetScopes.push_back(IRTargetScope(this,NULL,NULL,NULL));
        statement->toIR(p);
        p->func()->gen->targetScopes.pop_back();
    }
}

//////////////////////////////////////////////////////////////////////////////

void GotoStatement::toIR(IRState* p)
{
    Logger::println("GotoStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    gIR->DBuilder.EmitStopPoint(loc.linnum);

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "aftergoto", p->topfunc(), oldend);

    DtoGoto(loc, label->ident, enclosingFinally);

    p->scope() = IRScope(bb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void GotoDefaultStatement::toIR(IRState* p)
{
    Logger::println("GotoDefaultStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    gIR->DBuilder.EmitStopPoint(loc.linnum);

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "aftergotodefault", p->topfunc(), oldend);

    assert(!p->scopereturned());
    assert(sw->sdefault->bodyBB);

    DtoEnclosingHandlers(loc, sw);

    llvm::BranchInst::Create(sw->sdefault->bodyBB, p->scopebb());
    p->scope() = IRScope(bb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void GotoCaseStatement::toIR(IRState* p)
{
    Logger::println("GotoCaseStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    gIR->DBuilder.EmitStopPoint(loc.linnum);

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "aftergotocase", p->topfunc(), oldend);

    assert(!p->scopereturned());
    if (!cs->bodyBB)
    {
        cs->bodyBB = llvm::BasicBlock::Create(gIR->context(), "goto_case", p->topfunc(), p->scopeend());
    }

    DtoEnclosingHandlers(loc, sw);

    llvm::BranchInst::Create(cs->bodyBB, p->scopebb());
    p->scope() = IRScope(bb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void WithStatement::toIR(IRState* p)
{
    Logger::println("WithStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    gIR->DBuilder.EmitBlockStart(loc);

    assert(exp);

    // with(..) can either be used with expressions or with symbols
    // wthis == null indicates the symbol form
    if (wthis) {
        DValue* e = exp->toElemDtor(p);
        LLValue* mem = DtoRawVarDeclaration(wthis);
        DtoStore(e->getRVal(), mem);
    }

    if (body)
        body->toIR(p);

    gIR->DBuilder.EmitBlockEnd();
}

//////////////////////////////////////////////////////////////////////////////

static LLConstant* generate_unique_critical_section()
{
    LLType* Mty = DtoMutexType();
    return new llvm::GlobalVariable(*gIR->module, Mty, false, llvm::GlobalValue::InternalLinkage, LLConstant::getNullValue(Mty), ".uniqueCS");
}

void SynchronizedStatement::toIR(IRState* p)
{
    Logger::println("SynchronizedStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // emit dwarf stop point
    gIR->DBuilder.EmitStopPoint(loc.linnum);

    // enter lock
    if (exp)
    {
        llsync = exp->toElem(p)->getRVal();
        DtoEnterMonitor(llsync);
    }
    else
    {
        llsync = generate_unique_critical_section();
        DtoEnterCritical(llsync);
    }

    // emit body
    p->func()->gen->targetScopes.push_back(IRTargetScope(this,new EnclosingSynchro(this),NULL,NULL));
    gIR->DBuilder.EmitBlockStart(body->loc);
    body->toIR(p);
    gIR->DBuilder.EmitBlockEnd();
    p->func()->gen->targetScopes.pop_back();

    // exit lock
    // no point in a unreachable unlock, terminating statements must insert this themselves.
    if (p->scopereturned())
        return;
    else if (exp)
        DtoLeaveMonitor(llsync);
    else
        DtoLeaveCritical(llsync);
}

//////////////////////////////////////////////////////////////////////////////

void SwitchErrorStatement::toIR(IRState* p)
{
    Logger::println("SwitchErrorStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_switch_error");

    LLValue *moduleInfoSymbol = gIR->func()->decl->getModule()->moduleInfoSymbol();
    LLType *moduleInfoType = DtoType(Module::moduleinfo->type);

    LLValue* args[] = {
        // module param
        DtoBitCast(moduleInfoSymbol, getPtrToType(moduleInfoType)),
        // line param
        DtoConstUint(loc.linnum)
    };

    // call
    LLCallSite call = gIR->CreateCallOrInvoke(fn, args);
    call.setDoesNotReturn();
}

//////////////////////////////////////////////////////////////////////////////

void ImportStatement::toIR(IRState *irs)
{
}

//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////

#define STUBST(x) void x::toIR(IRState * p) {error("Statement type "#x" not implemented: %s", toChars());fatal();}
//STUBST(BreakStatement);
//STUBST(ForStatement);
//STUBST(WithStatement);
//STUBST(SynchronizedStatement);
//STUBST(ReturnStatement);
//STUBST(ContinueStatement);
//STUBST(DefaultStatement);
//STUBST(CaseStatement);
//STUBST(SwitchStatement);
//STUBST(SwitchErrorStatement);
STUBST(Statement);
//STUBST(IfStatement);
//STUBST(ForeachStatement);
//STUBST(DoStatement);
//STUBST(WhileStatement);
//STUBST(ExpStatement);
//STUBST(CompoundStatement);
//STUBST(ScopeStatement);
//STUBST(AsmStatement);
//STUBST(TryCatchStatement);
//STUBST(TryFinallyStatement);
//STUBST(LabelStatement);
//STUBST(ThrowStatement);
//STUBST(GotoCaseStatement);
//STUBST(GotoDefaultStatement);
//STUBST(GotoStatement);
//STUBST(UnrolledLoopStatement);
//STUBST(OnScopeStatement);
STUBST(PragmaStatement);

//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////

AsmBlockStatement* Statement::endsWithAsm()
{
    // does not end with inline asm
    return NULL;
}

AsmBlockStatement* CompoundStatement::endsWithAsm()
{
    // make the last inner statement decide
    if (statements && statements->dim)
    {
        unsigned last = statements->dim - 1;
        Statement* s = static_cast<Statement*>(statements->data[last]);
        if (s) return s->endsWithAsm();
    }
    return NULL;
}

AsmBlockStatement* AsmBlockStatement::endsWithAsm()
{
    // yes this is inline asm
    return this;
}
