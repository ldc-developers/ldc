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
#if LDC_LLVM_VER >= 305
#include "llvm/IR/CFG.h"
#else
#include "llvm/Support/CFG.h"
#endif
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
// FIXME: Integrate these functions
void AsmStatement_toIR(AsmStatement *stmt, IRState * irs);
void AsmBlockStatement_toIR(AsmBlockStatement *stmt, IRState* p);

//////////////////////////////////////////////////////////////////////////////

// used to build the sorted list of cases
struct Case : RootObject
{
    StringExp* str;
    size_t index;

    Case(StringExp* s, size_t i) {
        str = s;
        index = i;
    }

    int compare(RootObject *obj) {
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

    llvm::Function* fn = LLVM_D_GetRuntimeFunction(e->loc, gIR->module, fname);

    IF_LOG {
        Logger::cout() << *table->getType() << '\n';
        Logger::cout() << *fn->getFunctionType()->getParamType(0) << '\n';
    }
    assert(table->getType() == fn->getFunctionType()->getParamType(0));

    DValue* val = toElemDtor(e);
    LLValue* llval = val->getRVal();
    assert(llval->getType() == fn->getFunctionType()->getParamType(1));

    LLCallSite call = gIR->CreateCallOrInvoke2(fn, table, llval);

    return call.getInstruction();
}

//////////////////////////////////////////////////////////////////////////////

/* A visitor to walk entire tree of statements.
 */
class StatementVisitor : public Visitor
{
    void visitStmt(Statement *s) { s->accept(this); }
public:
    // Import all functions from class Visitor
    using Visitor::visit;

    void visit(ErrorStatement *s) {  }
    void visit(PeelStatement *s)
    {
        if (s->s)
            visitStmt(s->s);
    }
    void visit(ExpStatement *s) {  }
    void visit(DtorExpStatement *s) {  }
    void visit(CompileStatement *s) {  }
    void visit(CompoundStatement *s)
    {
        if (s->statements && s->statements->dim)
        {
            for (size_t i = 0; i < s->statements->dim; i++)
            {
                if ((*s->statements)[i])
                    visitStmt((*s->statements)[i]);
            }
        }
    }
    void visit(CompoundDeclarationStatement *s) { visit((CompoundStatement *)s); }
    void visit(UnrolledLoopStatement *s)
    {
        if (s->statements && s->statements->dim)
        {
            for (size_t i = 0; i < s->statements->dim; i++)
            {
                if ((*s->statements)[i])
                    visitStmt((*s->statements)[i]);
            }
        }
    }
    void visit(ScopeStatement *s)
    {
        if (s->statement)
            visitStmt(s->statement);
    }
    void visit(WhileStatement *s)
    {
        if (s->body)
            visitStmt(s->body);
    }
    void visit(DoStatement *s)
    {
        if (s->body)
            visitStmt(s->body);
    }
    void visit(ForStatement *s)
    {
        if (s->init)
            visitStmt(s->init);
        if (s->body)
            visitStmt(s->body);
    }
    void visit(ForeachStatement *s)
    {
        if (s->body)
            visitStmt(s->body);
    }
    void visit(ForeachRangeStatement *s)
    {
        if (s->body)
            visitStmt(s->body);
    }
    void visit(IfStatement *s)
    {
        if (s->ifbody)
            visitStmt(s->ifbody);
        if (s->elsebody)
            visitStmt(s->elsebody);
    }
    void visit(ConditionalStatement *s) {  }
    void visit(PragmaStatement *s) {  }
    void visit(StaticAssertStatement *s) {  }
    void visit(SwitchStatement *s)
    {
        if (s->body)
            visitStmt(s->body);
    }
    void visit(CaseStatement *s)
    {
        if (s->statement)
            visitStmt(s->statement);
    }
    void visit(CaseRangeStatement *s)
    {
        if (s->statement)
            visitStmt(s->statement);
    }
    void visit(DefaultStatement *s)
    {
        if (s->statement)
            visitStmt(s->statement);
    }
    void visit(GotoDefaultStatement *s) {  }
    void visit(GotoCaseStatement *s) {  }
    void visit(SwitchErrorStatement *s) {  }
    void visit(ReturnStatement *s) {  }
    void visit(BreakStatement *s) {  }
    void visit(ContinueStatement *s) {  }
    void visit(SynchronizedStatement *s)
    {
        if (s->body)
            visitStmt(s->body);
    }
    void visit(WithStatement *s)
    {
        if (s->body)
            visitStmt(s->body);
    }
    void visit(TryCatchStatement *s)
    {
        if (s->body)
            visitStmt(s->body);
        if (s->catches && s->catches->dim)
        {
            for (size_t i = 0; i < s->catches->dim; i++)
            {
                Catch *c = (*s->catches)[i];
                if (c && c->handler)
                    visitStmt(c->handler);
            }
        }
    }
    void visit(TryFinallyStatement *s)
    {
        if (s->body)
            visitStmt(s->body);
        if (s->finalbody)
            visitStmt(s->finalbody);
    }
    void visit(OnScopeStatement *s) {  }
    void visit(ThrowStatement *s) {  }
    void visit(DebugStatement *s)
    {
        if (s->statement)
            visitStmt(s->statement);
    }
    void visit(GotoStatement *s) {  }
    void visit(LabelStatement *s)
    {
        if (s->statement)
            visitStmt(s->statement);
    }
    void visit(AsmStatement *s) {  }
    void visit(ImportStatement *s) {  }
    void visit(AsmBlockStatement *s) {  }
};

//////////////////////////////////////////////////////////////////////////////

class FindEnclosingTryFinally : public StatementVisitor {
    std::stack<TryFinallyStatement*> m_tryFinally;
    std::stack<SwitchStatement*> m_switches;
public:
    // Import all functions from class StatementVisitor
    using StatementVisitor::visit;

    TryFinallyStatement *enclosingTryFinally() const
    {
        return m_tryFinally.empty() ? 0 : m_tryFinally.top();
    }

    SwitchStatement *enclosingSwitch() const
    {
        return m_switches.empty() ? 0 : m_switches.top();
    }

    void visit(SwitchStatement *s)
    {
        m_switches.push(s);
        s->enclosingScopeExit = enclosingTryFinally();
        StatementVisitor::visit(s);
        m_switches.pop();
    }

    void visit(CaseStatement *s)
    {
        s->enclosingScopeExit = enclosingTryFinally();
        if (s->enclosingScopeExit != enclosingSwitch()->enclosingScopeExit)
            s->error("switch and case are in different try blocks");
        StatementVisitor::visit(s);
    }

    void visit(DefaultStatement *s)
    {
        s->enclosingScopeExit = enclosingTryFinally();
        if (s->enclosingScopeExit != enclosingSwitch()->enclosingScopeExit)
            s->error("switch and default case are in different try blocks");
        StatementVisitor::visit(s);
    }

    void visit(TryFinallyStatement *s)
    {
        m_tryFinally.push(s);
        s->body->accept(this);
        m_tryFinally.pop();
        s->finalbody->accept(this);
    }

    void visit(LabelStatement *s)
    {
        s->enclosingScopeExit = enclosingTryFinally();
        StatementVisitor::visit(s);
    }

    void visit(GotoStatement *s)
    {
        s->enclosingScopeExit = enclosingTryFinally();
        StatementVisitor::visit(s);
    }

    void visit(AsmBlockStatement *s)
    {
        s->enclosingScopeExit = enclosingTryFinally();
        StatementVisitor::visit(s);
    }
};

//////////////////////////////////////////////////////////////////////////////


class ToIRVisitor : public Visitor {
    IRState *irs;
public:

    ToIRVisitor(IRState *irs) : irs(irs) { }

    //////////////////////////////////////////////////////////////////////////

    // Import all functions from class Visitor
    using Visitor::visit;

    //////////////////////////////////////////////////////////////////////////

    void visit(CompoundStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("CompoundStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        for (Statements::iterator I = stmt->statements->begin(),
                                  E = stmt->statements->end();
                                  I != E; ++I)
        {
            Statement *s = *I;
            if (s) {
                s->accept(this);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ReturnStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ReturnStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // emit dwarf stop point
        gIR->DBuilder.EmitStopPoint(stmt->loc.linnum);

        // is there a return value expression?
        if (stmt->exp || (!stmt->exp && (irs->topfunc() == irs->mainFunc)) )
        {
            // if the functions return type is void this means that
            // we are returning through a pointer argument
            if (irs->topfunc()->getReturnType() == LLType::getVoidTy(gIR->context()))
            {
                // sanity check
                IrFunction* f = irs->func();
                assert(getIrFunc(f->decl)->retArg);

                // FIXME: is there ever a case where a sret return needs to be rewritten for the ABI?

                // get return pointer
                DValue* rvar = new DVarValue(f->type->next, getIrFunc(f->decl)->retArg);
                DValue* e = toElemDtor(stmt->exp);
                // store return value
                if (rvar->getLVal() != e->getRVal())
                    DtoAssign(stmt->loc, rvar, e);

                // call postblit if necessary
                if (!irs->func()->type->isref && !(f->decl->nrvo_can && f->decl->nrvo_var))
                    callPostblit(stmt->loc, stmt->exp, rvar->getLVal());

                // emit scopes
                DtoEnclosingHandlers(stmt->loc, NULL);

                // emit dbg end function
                gIR->DBuilder.EmitFuncEnd(f->decl);

                // emit ret
                llvm::ReturnInst::Create(gIR->context(), irs->scopebb());
            }
            // the return type is not void, so this is a normal "register" return
            else
            {
                LLValue* v = 0;
                if (!stmt->exp && (irs->topfunc() == irs->mainFunc)) {
                    v = LLConstant::getNullValue(irs->mainFunc->getReturnType());
                } else {
                    if (stmt->exp->op == TOKnull)
                        stmt->exp->type = irs->func()->type->next;
                    DValue* dval = 0;
                    // call postblit if necessary
                    if (!irs->func()->type->isref) {
                        dval = toElemDtor(stmt->exp);
                        callPostblit(stmt->loc, stmt->exp, dval->getRVal());
                    } else {
                        Expression *ae = stmt->exp->addressOf();
                        dval = toElemDtor(ae);
                    }
                    // do abi specific transformations on the return value
                    v = getIrFunc(irs->func()->decl)->irFty.putRet(stmt->exp->type, dval);
                }

                IF_LOG Logger::cout() << "return value is '" <<*v << "'\n";

                IrFunction* f = irs->func();
                // Hack around LDC assuming structs and static arrays are in memory:
                // If the function returns a struct or a static array, and the return
                // value is a pointer to a struct or a static array, load from it
                // before returning.
                int ty = f->type->next->toBasetype()->ty;
                if (v->getType() != irs->topfunc()->getReturnType() &&
                    (ty == Tstruct
                     || ty == Tsarray
                     ) && isaPointer(v->getType()))
                {
                    Logger::println("Loading value for return");
                    v = DtoLoad(v);
                }

                // can happen for classes and void main
                if (v->getType() != irs->topfunc()->getReturnType())
                {
                    // for the main function this only happens if it is declared as void
                    // and then contains a return (exp); statement. Since the actual
                    // return type remains i32, we just throw away the exp value
                    // and return 0 instead
                    // if we're not in main, just bitcast
                    if (irs->topfunc() == irs->mainFunc)
                        v = LLConstant::getNullValue(irs->mainFunc->getReturnType());
                    else
                        v = gIR->ir->CreateBitCast(v, irs->topfunc()->getReturnType());

                    IF_LOG Logger::cout() << "return value after cast: " << *v << '\n';
                }

                // emit scopes
                DtoEnclosingHandlers(stmt->loc, NULL);

                gIR->DBuilder.EmitFuncEnd(irs->func()->decl);
                llvm::ReturnInst::Create(gIR->context(), v, irs->scopebb());
            }
        }
        // no return value expression means it's a void function
        else
        {
            assert(irs->topfunc()->getReturnType() == LLType::getVoidTy(gIR->context()));
            DtoEnclosingHandlers(stmt->loc, NULL);
            gIR->DBuilder.EmitFuncEnd(irs->func()->decl);
            llvm::ReturnInst::Create(gIR->context(), irs->scopebb());
        }

        // the return terminated this basicblock, start a new one
        llvm::BasicBlock* oldend = gIR->scopeend();
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "afterreturn", irs->topfunc(), oldend);
        irs->scope() = IRScope(bb, oldend);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ExpStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ExpStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // emit dwarf stop point
        gIR->DBuilder.EmitStopPoint(stmt->loc.linnum);

        if (stmt->exp) {
            elem* e;
            // a cast(void) around the expression is allowed, but doesn't require any code
            if (stmt->exp->op == TOKcast && stmt->exp->type == Type::tvoid) {
                CastExp* cexp = static_cast<CastExp*>(stmt->exp);
                e = toElemDtor(cexp->e1);
            }
            else
                e = toElemDtor(stmt->exp);
            delete e;
        }
        /*elem* e = exp->toElem(irs);
        p->buf.printf("%s", e->toChars());
        delete e;
        p->buf.writenl();*/
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(IfStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("IfStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // start a dwarf lexical block
        gIR->DBuilder.EmitBlockStart(stmt->loc);
        if (stmt->match)
            DtoRawVarDeclaration(stmt->match);

        DValue* cond_e = toElemDtor(stmt->condition);
        LLValue* cond_val = cond_e->getRVal();

        llvm::BasicBlock* oldend = gIR->scopeend();

        llvm::BasicBlock* ifbb = llvm::BasicBlock::Create(gIR->context(), "if", gIR->topfunc(), oldend);
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "endif", gIR->topfunc(), oldend);
        llvm::BasicBlock* elsebb = stmt->elsebody ? llvm::BasicBlock::Create(gIR->context(), "else", gIR->topfunc(), endbb) : endbb;

        if (cond_val->getType() != LLType::getInt1Ty(gIR->context())) {
            IF_LOG Logger::cout() << "if conditional: " << *cond_val << '\n';
            cond_val = DtoCast(stmt->loc, cond_e, Type::tbool)->getRVal();
        }
        llvm::BranchInst::Create(ifbb, elsebb, cond_val, gIR->scopebb());

        // replace current scope
        gIR->scope() = IRScope(ifbb, elsebb);

        // do scoped statements

        if (stmt->ifbody) {
            gIR->DBuilder.EmitBlockStart(stmt->ifbody->loc);
            stmt->ifbody->accept(this);
            gIR->DBuilder.EmitBlockEnd();
        }
        if (!gIR->scopereturned()) {
            llvm::BranchInst::Create(endbb, gIR->scopebb());
        }

        if (stmt->elsebody) {
            gIR->scope() = IRScope(elsebb, endbb);
            gIR->DBuilder.EmitBlockStart(stmt->elsebody->loc);
            stmt->elsebody->accept(this);
            if (!gIR->scopereturned()) {
                llvm::BranchInst::Create(endbb, gIR->scopebb());
            }
            gIR->DBuilder.EmitBlockEnd();
        }

        // end the dwarf lexical block
        gIR->DBuilder.EmitBlockEnd();

        // rewrite the scope
        gIR->scope() = IRScope(endbb, oldend);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ScopeStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ScopeStatement::toIR(): %s", stmt->loc.toChars());
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

        if (stmt->statement) {
            gIR->DBuilder.EmitBlockStart(stmt->statement->loc);
            stmt->statement->accept(this);
            gIR->DBuilder.EmitBlockEnd();
        }

        /*p->scope().end = oldend;
        Logger::println("Erasing scope endbb");
        endbb->eraseFromParent();*/
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(WhileStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("WhileStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // start a dwarf lexical block
        gIR->DBuilder.EmitBlockStart(stmt->loc);

        // create while blocks
        llvm::BasicBlock* oldend = gIR->scopeend();
        llvm::BasicBlock* whilebb = llvm::BasicBlock::Create(gIR->context(), "whilecond", gIR->topfunc(), oldend);
        llvm::BasicBlock* whilebodybb = llvm::BasicBlock::Create(gIR->context(), "whilebody", gIR->topfunc(), oldend);
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "endwhile", gIR->topfunc(), oldend);

        // move into the while block
        irs->ir->CreateBr(whilebb);
        //llvm::BranchInst::Create(whilebb, gIR->scopebb());

        // replace current scope
        gIR->scope() = IRScope(whilebb, endbb);

        // create the condition
        DValue* cond_e = toElemDtor(stmt->condition);
        LLValue* cond_val = DtoCast(stmt->loc, cond_e, Type::tbool)->getRVal();
        delete cond_e;

        // conditional branch
        llvm::BranchInst::Create(whilebodybb, endbb, cond_val, irs->scopebb());

        // rewrite scope
        gIR->scope() = IRScope(whilebodybb, endbb);

        // while body code
        irs->func()->gen->targetScopes.push_back(IRTargetScope(stmt, NULL, whilebb, endbb));
        if (stmt->body)
            stmt->body->accept(this);
        irs->func()->gen->targetScopes.pop_back();

        // loop
        if (!gIR->scopereturned())
            llvm::BranchInst::Create(whilebb, gIR->scopebb());

        // rewrite the scope
        gIR->scope() = IRScope(endbb, oldend);

        // end the dwarf lexical block
        gIR->DBuilder.EmitBlockEnd();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(DoStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("DoStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // start a dwarf lexical block
        gIR->DBuilder.EmitBlockStart(stmt->loc);

        // create while blocks
        llvm::BasicBlock* oldend = gIR->scopeend();
        llvm::BasicBlock* dowhilebb = llvm::BasicBlock::Create(gIR->context(), "dowhile", gIR->topfunc(), oldend);
        llvm::BasicBlock* condbb = llvm::BasicBlock::Create(gIR->context(), "dowhilecond", gIR->topfunc(), oldend);
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "enddowhile", gIR->topfunc(), oldend);

        // move into the while block
        assert(!gIR->scopereturned());
        llvm::BranchInst::Create(dowhilebb, gIR->scopebb());

        // replace current scope
        gIR->scope() = IRScope(dowhilebb, condbb);

        // do-while body code
        irs->func()->gen->targetScopes.push_back(IRTargetScope(stmt, NULL, condbb, endbb));
        if (stmt->body)
            stmt->body->accept(this);
        irs->func()->gen->targetScopes.pop_back();

        // branch to condition block
        llvm::BranchInst::Create(condbb, gIR->scopebb());
        gIR->scope() = IRScope(condbb,endbb);

        // create the condition
        DValue* cond_e = toElemDtor(stmt->condition);
        LLValue* cond_val = DtoCast(stmt->loc, cond_e, Type::tbool)->getRVal();
        delete cond_e;

        // conditional branch
        llvm::BranchInst::Create(dowhilebb, endbb, cond_val, gIR->scopebb());

        // rewrite the scope
        gIR->scope() = IRScope(endbb, oldend);

        // end the dwarf lexical block
        gIR->DBuilder.EmitBlockEnd();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ForStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ForStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // start new dwarf lexical block
        gIR->DBuilder.EmitBlockStart(stmt->loc);

        // create for blocks
        llvm::BasicBlock* oldend = gIR->scopeend();
        llvm::BasicBlock* forbb = llvm::BasicBlock::Create(gIR->context(), "forcond", gIR->topfunc(), oldend);
        llvm::BasicBlock* forbodybb = llvm::BasicBlock::Create(gIR->context(), "forbody", gIR->topfunc(), oldend);
        llvm::BasicBlock* forincbb = llvm::BasicBlock::Create(gIR->context(), "forinc", gIR->topfunc(), oldend);
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "endfor", gIR->topfunc(), oldend);

        // init
        if (stmt->init != 0)
            stmt->init->accept(this);

        // move into the for condition block, ie. start the loop
        assert(!gIR->scopereturned());
        llvm::BranchInst::Create(forbb, gIR->scopebb());

        // In case of loops that have been rewritten to a composite statement
        // containing the initializers and then the actual loop, we need to
        // register the former as target scope start.
        Statement* scopeStart = stmt->getRelatedLabeled();
        while (ScopeStatement* scope = scopeStart->isScopeStatement())
        {
            scopeStart = scope->statement;
        }
        irs->func()->gen->targetScopes.push_back(IRTargetScope(
            scopeStart, NULL, forincbb, endbb));

        // replace current scope
        gIR->scope() = IRScope(forbb, forbodybb);

        // create the condition
        llvm::Value* cond_val;
        if (stmt->condition)
        {
            DValue* cond_e = toElemDtor(stmt->condition);
            cond_val = DtoCast(stmt->loc, cond_e, Type::tbool)->getRVal();
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
        gIR->scope() = IRScope(forbodybb, forincbb);

        // do for body code
        if (stmt->body)
            stmt->body->accept(this);

        // move into the for increment block
        if (!gIR->scopereturned())
            llvm::BranchInst::Create(forincbb, gIR->scopebb());
        gIR->scope() = IRScope(forincbb, endbb);

        // increment
        if (stmt->increment) {
            DValue* inc = toElemDtor(stmt->increment);
            delete inc;
        }

        // loop
        if (!gIR->scopereturned())
            llvm::BranchInst::Create(forbb, gIR->scopebb());

        irs->func()->gen->targetScopes.pop_back();

        // rewrite the scope
        gIR->scope() = IRScope(endbb, oldend);

        // end the dwarf lexical block
        gIR->DBuilder.EmitBlockEnd();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(BreakStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("BreakStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // don't emit two terminators in a row
        // happens just before DMD generated default statements if the last case terminates
        if (irs->scopereturned())
            return;

        // emit dwarf stop point
        gIR->DBuilder.EmitStopPoint(stmt->loc.linnum);

        if (stmt->ident != 0) {
            IF_LOG Logger::println("ident = %s", stmt->ident->toChars());

            DtoEnclosingHandlers(stmt->loc, stmt->target);

            // get the loop statement the label refers to
            Statement* targetLoopStatement = stmt->target->statement;
            ScopeStatement* tmp;
            while((tmp = targetLoopStatement->isScopeStatement()))
                targetLoopStatement = tmp->statement;

            // find the right break block and jump there
            // the right break block is found in the nearest scope to the LabelStatement
            // with onlyLabelBreak == true. Therefore the search starts at the outer
            // scope (in contract to most other searches, which start with the inner
            // scope). This code is tested by test runnable/foreach5.d, test9068().
            bool found = false;
            FuncGen::TargetScopeVec::iterator it = irs->func()->gen->targetScopes.begin();
            FuncGen::TargetScopeVec::iterator it_end = irs->func()->gen->targetScopes.end();
            while (it != it_end && it->s != stmt->target)
                ++it;
            assert(it != it_end && "Labeled break but no label found");
            while (it != it_end) {
                if (it->onlyLabeledBreak || it->s == targetLoopStatement) {
                    llvm::BranchInst::Create(it->breakTarget, irs->scopebb());
                    found = true;
                    break;
                }
                ++it;
            }
            assert(found && "Labeled break but no jump target found");
        }
        else {
            // find closest scope with a break target
            FuncGen::TargetScopeVec::reverse_iterator it = irs->func()->gen->targetScopes.rbegin();
            FuncGen::TargetScopeVec::reverse_iterator it_end = irs->func()->gen->targetScopes.rend();
            while (it != it_end) {
                if (it->breakTarget && !it->onlyLabeledBreak) {
                    break;
                }
                ++it;
            }
            DtoEnclosingHandlers(stmt->loc, it->s);
            llvm::BranchInst::Create(it->breakTarget, gIR->scopebb());
        }

        // the break terminated this basicblock, start a new one
        llvm::BasicBlock* oldend = gIR->scopeend();
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "afterbreak", irs->topfunc(), oldend);
        irs->scope() = IRScope(bb, oldend);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ContinueStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ContinueStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // emit dwarf stop point
        gIR->DBuilder.EmitStopPoint(stmt->loc.linnum);

        if (stmt->ident != 0) {
            IF_LOG Logger::println("ident = %s", stmt->ident->toChars());

            // get the loop statement the label refers to
            Statement* targetLoopStatement = stmt->target->statement;
            ScopeStatement* tmp;
            while((tmp = targetLoopStatement->isScopeStatement()))
                targetLoopStatement = tmp->statement;

            // find the right continue block
            bool found = false;
            FuncGen::TargetScopeVec::reverse_iterator it = irs->func()->gen->targetScopes.rbegin();
            FuncGen::TargetScopeVec::reverse_iterator it_end = irs->func()->gen->targetScopes.rend();
            while (it != it_end) {
                if (it->s == targetLoopStatement) {
                    found = true;
                    break;
                }
                ++it;
            }

            assert(found);
            // emit destructors and finally statements
            DtoEnclosingHandlers(stmt->loc, it->s);
            // jump to the continue block
            llvm::BranchInst::Create(it->continueTarget, gIR->scopebb());
        }
        else {
            // find closest scope with a continue target
            FuncGen::TargetScopeVec::reverse_iterator it = irs->func()->gen->targetScopes.rbegin();
            FuncGen::TargetScopeVec::reverse_iterator it_end = irs->func()->gen->targetScopes.rend();
            while (it != it_end) {
                if (it->continueTarget) {
                    break;
                }
                ++it;
            }
            DtoEnclosingHandlers(stmt->loc, it->s);
            llvm::BranchInst::Create(it->continueTarget, gIR->scopebb());
        }

        // the continue terminated this basicblock, start a new one
        llvm::BasicBlock* oldend = gIR->scopeend();
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "aftercontinue", irs->topfunc(), oldend);
        irs->scope() = IRScope(bb, oldend);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(OnScopeStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("OnScopeStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        assert(stmt->statement);
        //statement->toIR(p); // this seems to be redundant
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(TryFinallyStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("TryFinallyStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // emit dwarf stop point
        gIR->DBuilder.EmitStopPoint(stmt->loc.linnum);

        // if there's no finalbody or no body, things are simple
        if (!stmt->finalbody) {
            if (stmt->body) {
                gIR->DBuilder.EmitBlockStart(stmt->body->loc);
                stmt->body->accept(this);
                gIR->DBuilder.EmitBlockEnd();
            }
            return;
        }
        if (!stmt->body) {
            gIR->DBuilder.EmitBlockStart(stmt->finalbody->loc);
            stmt->finalbody->accept(this);
            gIR->DBuilder.EmitBlockEnd();
            return;
        }

        // create basic blocks
        llvm::BasicBlock* oldend = irs->scopeend();

        llvm::BasicBlock* trybb = llvm::BasicBlock::Create(gIR->context(), "try", irs->topfunc(), oldend);
        llvm::BasicBlock* finallybb = llvm::BasicBlock::Create(gIR->context(), "finally", irs->topfunc(), oldend);
        // the landing pad for statements in the try block
        llvm::BasicBlock* landingpadbb = llvm::BasicBlock::Create(gIR->context(), "landingpad", irs->topfunc(), oldend);
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "endtryfinally", irs->topfunc(), oldend);

        // pass the previous BB into this
        assert(!gIR->scopereturned());
        llvm::BranchInst::Create(trybb, irs->scopebb());

        //
        // set up the landing pad
        //
        irs->scope() = IRScope(landingpadbb, endbb);

        assert(stmt->finalbody);
        IRLandingPad& pad = gIR->func()->gen->landingPadInfo;
        pad.addFinally(stmt->finalbody);
        pad.push(landingpadbb);
        gIR->func()->gen->targetScopes.push_back(
            IRTargetScope(
                stmt,
                new EnclosingTryFinally(stmt, gIR->func()->gen->landingPad),
                NULL,
                endbb,
                true
            )
        );

        //
        // do the try block
        //
        irs->scope() = IRScope(trybb, finallybb);

        assert(stmt->body);
        gIR->DBuilder.EmitBlockStart(stmt->body->loc);
        stmt->body->accept(this);
        gIR->DBuilder.EmitBlockEnd();

        // terminate try BB
        if (!irs->scopereturned())
            llvm::BranchInst::Create(finallybb, irs->scopebb());

        pad.pop();
        gIR->func()->gen->targetScopes.pop_back();

        //
        // do finally block
        //
        irs->scope() = IRScope(finallybb, landingpadbb);
        gIR->DBuilder.EmitBlockStart(stmt->finalbody->loc);
        stmt->finalbody->accept(this);
        gIR->DBuilder.EmitBlockEnd();

        // terminate finally
        //TODO: isn't it an error to have a 'returned' finally block?
        if (!gIR->scopereturned()) {
            llvm::BranchInst::Create(endbb, irs->scopebb());
        }

        // rewrite the scope
        irs->scope() = IRScope(endbb, oldend);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(TryCatchStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("TryCatchStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // emit dwarf stop point
        gIR->DBuilder.EmitStopPoint(stmt->loc.linnum);

        // create basic blocks
        llvm::BasicBlock* oldend = irs->scopeend();

        llvm::BasicBlock* trybb = llvm::BasicBlock::Create(gIR->context(), "try", irs->topfunc(), oldend);
        // the landing pad will be responsible for branching to the correct catch block
        llvm::BasicBlock* landingpadbb = llvm::BasicBlock::Create(gIR->context(), "landingpad", irs->topfunc(), oldend);
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "endtrycatch", irs->topfunc(), oldend);

        // pass the previous BB into this
        assert(!gIR->scopereturned());
        llvm::BranchInst::Create(trybb, irs->scopebb());

        //
        // set up the landing pad
        //
        assert(stmt->catches);
        gIR->scope() = IRScope(landingpadbb, endbb);

        IRLandingPad& pad = gIR->func()->gen->landingPadInfo;
        for (Catches::iterator I = stmt->catches->begin(),
                               E = stmt->catches->end();
                               I != E; ++I)
        {
            Catch *c = *I;
            pad.addCatch(c, endbb);
        }

        pad.push(landingpadbb);

        //
        // do the try block
        //
        irs->scope() = IRScope(trybb, landingpadbb);

        assert(stmt->body);
        gIR->DBuilder.EmitBlockStart(stmt->body->loc);
        stmt->body->accept(this);
        gIR->DBuilder.EmitBlockEnd();

        if (!gIR->scopereturned())
            llvm::BranchInst::Create(endbb, irs->scopebb());

        pad.pop();

        // rewrite the scope
        irs->scope() = IRScope(endbb, oldend);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ThrowStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ThrowStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // emit dwarf stop point
        gIR->DBuilder.EmitStopPoint(stmt->loc.linnum);

        assert(stmt->exp);
        DValue* e = toElemDtor(stmt->exp);

        gIR->DBuilder.EmitFuncEnd(gIR->func()->decl);

        llvm::Function* fn = LLVM_D_GetRuntimeFunction(stmt->loc, gIR->module, "_d_throw_exception");
        //Logger::cout() << "calling: " << *fn << '\n';
        LLValue* arg = DtoBitCast(e->getRVal(), fn->getFunctionType()->getParamType(0));
        //Logger::cout() << "arg: " << *arg << '\n';
        gIR->CreateCallOrInvoke(fn, arg);
        gIR->ir->CreateUnreachable();

        // need a block after the throw for now
        llvm::BasicBlock* oldend = gIR->scopeend();
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "afterthrow", irs->topfunc(), oldend);
        irs->scope() = IRScope(bb, oldend);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(SwitchStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("SwitchStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // emit dwarf stop point
        gIR->DBuilder.EmitStopPoint(stmt->loc.linnum);

        llvm::BasicBlock* oldbb = gIR->scopebb();
        llvm::BasicBlock* oldend = gIR->scopeend();

        // clear data from previous passes... :/
        for (CaseStatements::iterator I = stmt->cases->begin(),
                                      E = stmt->cases->end();
                                      I != E; ++I)
        {
            CaseStatement *cs = *I;
            cs->bodyBB = NULL;
            cs->llvmIdx = NULL;
        }

        // If one of the case expressions is non-constant, we can't use
        // 'switch' instruction (that can happen because D2 allows to
        // initialize a global variable in a static constructor).
        bool useSwitchInst = true;
        for (CaseStatements::iterator I = stmt->cases->begin(),
                                      E = stmt->cases->end();
                                      I != E; ++I)
        {
            CaseStatement *cs = *I;
            VarDeclaration* vd = 0;
            if (cs->exp->op == TOKvar)
                vd = static_cast<VarExp*>(cs->exp)->var->isVarDeclaration();
            if (vd && (!vd->init || !vd->isConst())) {
                cs->llvmIdx = toElemDtor(cs->exp)->getRVal();
                useSwitchInst = false;
            }
        }

        // body block.
        // FIXME: that block is never used
        llvm::BasicBlock* bodybb = llvm::BasicBlock::Create(gIR->context(), "switchbody", irs->topfunc(), oldend);

        // default
        llvm::BasicBlock* defbb = 0;
        if (stmt->sdefault) {
            Logger::println("has default");
            defbb = llvm::BasicBlock::Create(gIR->context(), "default", irs->topfunc(), oldend);
            stmt->sdefault->bodyBB = defbb;
        }

        // end (break point)
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "switchend", irs->topfunc(), oldend);

        // do switch body
        assert(stmt->body);
        irs->scope() = IRScope(bodybb, endbb);
        irs->func()->gen->targetScopes.push_back(IRTargetScope(stmt, NULL, NULL, endbb));
        stmt->body->accept(this);
        irs->func()->gen->targetScopes.pop_back();
        if (!irs->scopereturned())
            llvm::BranchInst::Create(endbb, irs->scopebb());

        gIR->scope() = IRScope(oldbb, oldend);
        if (useSwitchInst)
        {
            // string switch?
            llvm::Value* switchTable = 0;
            Objects caseArray;
            if (!stmt->condition->type->isintegral())
            {
                Logger::println("is string switch");
                // build array of the stringexpS
                caseArray.reserve(stmt->cases->dim);
                for (unsigned i=0; i < stmt->cases->dim; ++i)
                {
                    CaseStatement* cs = static_cast<CaseStatement*>(stmt->cases->data[i]);

                    assert(cs->exp->op == TOKstring);
                    caseArray.push(new Case(static_cast<StringExp*>(cs->exp), i));
                }
                // first sort it
                caseArray.sort();
                // iterate and add indices to cases
                std::vector<llvm::Constant*> inits(caseArray.dim, 0);
                for (size_t i=0; i < caseArray.dim; ++i)
                {
                    Case* c = static_cast<Case*>(caseArray.data[i]);
                    CaseStatement* cs = static_cast<CaseStatement*>(stmt->cases->data[c->index]);
                    cs->llvmIdx = DtoConstUint(i);
                    inits[i] = toConstElem(c->str, irs);
                }
                // build static array for ptr or final array
                LLType* elemTy = DtoType(stmt->condition->type);
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
            if (stmt->condition->type->isintegral()) {
                DValue* cond = toElemDtor(stmt->condition);
                condVal = cond->getRVal();
            }
            // string switch
            else {
                condVal = call_string_switch_runtime(switchTable, stmt->condition);
            }

            // create switch and add the cases
            llvm::SwitchInst* si = llvm::SwitchInst::Create(condVal, defbb ? defbb : endbb, stmt->cases->dim, irs->scopebb());
            for (CaseStatements::iterator I = stmt->cases->begin(),
                                          E = stmt->cases->end();
                                          I != E; ++I)
            {
                CaseStatement *cs = *I;
                si->addCase(isaConstantInt(cs->llvmIdx), cs->bodyBB);
            }
        }
        else
        { // we can't use switch, so we will use a bunch of br instructions instead
            DValue* cond = toElemDtor(stmt->condition);
            LLValue *condVal = cond->getRVal();

            llvm::BasicBlock* nextbb = llvm::BasicBlock::Create(gIR->context(), "checkcase", irs->topfunc(), oldend);
            llvm::BranchInst::Create(nextbb, irs->scopebb());

            irs->scope() = IRScope(nextbb, endbb);
            for (CaseStatements::iterator I = stmt->cases->begin(),
                                          E = stmt->cases->end();
                                          I != E; ++I)
            {
                CaseStatement *cs = *I;

                LLValue *cmp = irs->ir->CreateICmp(llvm::ICmpInst::ICMP_EQ, cs->llvmIdx, condVal, "checkcase");
                nextbb = llvm::BasicBlock::Create(gIR->context(), "checkcase", irs->topfunc(), oldend);
                llvm::BranchInst::Create(cs->bodyBB, nextbb, cmp, irs->scopebb());
                irs->scope() = IRScope(nextbb, endbb);
            }

            if (stmt->sdefault) {
                llvm::BranchInst::Create(stmt->sdefault->bodyBB, irs->scopebb());
            } else {
                llvm::BranchInst::Create(endbb, irs->scopebb());
            }
            endbb->moveAfter(nextbb);
        }

        gIR->scope() = IRScope(endbb, oldend);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(CaseStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("CaseStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        llvm::BasicBlock* nbb = llvm::BasicBlock::Create(gIR->context(), "case", irs->topfunc(), irs->scopeend());

        if (stmt->bodyBB && !stmt->bodyBB->getTerminator())
        {
            llvm::BranchInst::Create(nbb, stmt->bodyBB);
        }
        stmt->bodyBB = nbb;

        if (stmt->llvmIdx == NULL) {
            llvm::Constant *c = toConstElem(stmt->exp, irs);
            stmt->llvmIdx = isaConstantInt(c);
        }

        if (!irs->scopereturned())
            llvm::BranchInst::Create(stmt->bodyBB, irs->scopebb());

        irs->scope() = IRScope(stmt->bodyBB, irs->scopeend());

        assert(stmt->statement);
        gIR->DBuilder.EmitBlockStart(stmt->statement->loc);
        stmt->statement->accept(this);
        gIR->DBuilder.EmitBlockEnd();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(DefaultStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("DefaultStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        assert(stmt->bodyBB);

        llvm::BasicBlock* nbb = llvm::BasicBlock::Create(gIR->context(), "default", irs->topfunc(), irs->scopeend());

        if (!stmt->bodyBB->getTerminator())
        {
            llvm::BranchInst::Create(nbb, stmt->bodyBB);
        }
        stmt->bodyBB = nbb;

        if (!irs->scopereturned())
            llvm::BranchInst::Create(stmt->bodyBB, irs->scopebb());

        irs->scope() = IRScope(stmt->bodyBB, irs->scopeend());

        assert(stmt->statement);
        gIR->DBuilder.EmitBlockStart(stmt->statement->loc);
        stmt->statement->accept(this);
        gIR->DBuilder.EmitBlockEnd();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(UnrolledLoopStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("UnrolledLoopStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // if no statements, there's nothing to do
        if (!stmt->statements || !stmt->statements->dim)
            return;

        // start a dwarf lexical block
        gIR->DBuilder.EmitBlockStart(stmt->loc);

        // DMD doesn't fold stuff like continue/break, and since this isn't really a loop
        // we have to keep track of each statement and jump to the next/end on continue/break

        llvm::BasicBlock* oldend = gIR->scopeend();

        // create a block for each statement
        size_t nstmt = stmt->statements->dim;
        llvm::SmallVector<llvm::BasicBlock*, 4> blocks(nstmt, NULL);

        for (size_t i=0; i < nstmt; i++)
        {
            blocks[i] = llvm::BasicBlock::Create(gIR->context(), "unrolledstmt", irs->topfunc(), oldend);
        }

        // create end block
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "unrolledend", irs->topfunc(), oldend);

        // enter first stmt
        if (!irs->scopereturned())
            irs->ir->CreateBr(blocks[0]);

        // do statements
        Statement** stmts = static_cast<Statement **>(stmt->statements->data);

        for (size_t i=0; i < nstmt; i++)
        {
            Statement *s = stmts[i];

            // get blocks
            llvm::BasicBlock* thisbb = blocks[i];
            llvm::BasicBlock* nextbb = (i+1 == nstmt) ? endbb : blocks[i+1];

            // update scope
            irs->scope() = IRScope(thisbb, nextbb);

            // push loop scope
            // continue goes to next statement, break goes to end
            irs->func()->gen->targetScopes.push_back(IRTargetScope(stmt, NULL, nextbb, endbb));

            // do statement
            s->accept(this);

            // pop loop scope
            irs->func()->gen->targetScopes.pop_back();

            // next stmt
            if (!irs->scopereturned())
                irs->ir->CreateBr(nextbb);
        }

        // finish scope
        if (!irs->scopereturned())
            irs->ir->CreateBr(endbb);
        irs->scope() = IRScope(endbb, oldend);

        // end the dwarf lexical block
        gIR->DBuilder.EmitBlockEnd();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ForeachStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ForeachStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // start a dwarf lexical block
        gIR->DBuilder.EmitBlockStart(stmt->loc);

        //assert(arguments->dim == 1);
        assert(stmt->value != 0);
        assert(stmt->aggr != 0);
        assert(stmt->func != 0);

        //Argument* arg = static_cast<Argument*>(arguments->data[0]);
        //Logger::println("Argument is %s", arg->toChars());

        IF_LOG Logger::println("aggr = %s", stmt->aggr->toChars());

        // key
        LLType* keytype = stmt->key ? DtoType(stmt->key->type) : DtoSize_t();
        LLValue* keyvar;
        if (stmt->key)
            keyvar = DtoRawVarDeclaration(stmt->key);
        else
            keyvar = DtoRawAlloca(keytype, 0, "foreachkey"); // FIXME: align?
        LLValue* zerokey = LLConstantInt::get(keytype, 0, false);

        // value
        IF_LOG Logger::println("value = %s", stmt->value->toPrettyChars());
        LLValue* valvar = NULL;
        if (!stmt->value->isRef() && !stmt->value->isOut()) {
            // Create a local variable to serve as the value.
            DtoRawVarDeclaration(stmt->value);
            valvar = getIrLocal(stmt->value)->value;
        }

        // what to iterate
        DValue* aggrval = toElemDtor(stmt->aggr);

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

        if (stmt->op == TOKforeach) {
            new llvm::StoreInst(zerokey, keyvar, irs->scopebb());
        }
        else {
            new llvm::StoreInst(niters, keyvar, irs->scopebb());
        }

        llvm::BasicBlock* oldend = gIR->scopeend();
        llvm::BasicBlock* condbb = llvm::BasicBlock::Create(gIR->context(), "foreachcond", irs->topfunc(), oldend);
        llvm::BasicBlock* bodybb = llvm::BasicBlock::Create(gIR->context(), "foreachbody", irs->topfunc(), oldend);
        llvm::BasicBlock* nextbb = llvm::BasicBlock::Create(gIR->context(), "foreachnext", irs->topfunc(), oldend);
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "foreachend", irs->topfunc(), oldend);

        llvm::BranchInst::Create(condbb, irs->scopebb());

        // condition
        irs->scope() = IRScope(condbb, bodybb);

        LLValue* done = 0;
        LLValue* load = DtoLoad(keyvar);
        if (stmt->op == TOKforeach) {
            done = irs->ir->CreateICmpULT(load, niters);
        }
        else if (stmt->op == TOKforeach_reverse) {
            done = irs->ir->CreateICmpUGT(load, zerokey);
            load = irs->ir->CreateSub(load, LLConstantInt::get(keytype, 1, false));
            DtoStore(load, keyvar);
        }
        llvm::BranchInst::Create(bodybb, endbb, done, irs->scopebb());

        // init body
        irs->scope() = IRScope(bodybb, nextbb);

        // get value for this iteration
        LLValue* loadedKey = irs->ir->CreateLoad(keyvar);
        LLValue* gep = DtoGEP1(val, loadedKey);

        if (!stmt->value->isRef() && !stmt->value->isOut()) {
            // Copy value to local variable, and use it as the value variable.
            DVarValue dst(stmt->value->type, valvar);
            DVarValue src(stmt->value->type, gep);
            DtoAssign(stmt->loc, &dst, &src);
            getIrLocal(stmt->value)->value = valvar;
        } else {
            // Use the GEP as the address of the value variable.
            DtoRawVarDeclaration(stmt->value, gep);
        }

        // emit body
        irs->func()->gen->targetScopes.push_back(IRTargetScope(stmt, NULL, nextbb, endbb));
        if (stmt->body)
            stmt->body->accept(this);
        irs->func()->gen->targetScopes.pop_back();

        if (!irs->scopereturned())
            llvm::BranchInst::Create(nextbb, irs->scopebb());

        // next
        irs->scope() = IRScope(nextbb, endbb);
        if (stmt->op == TOKforeach) {
            LLValue* load = DtoLoad(keyvar);
            load = irs->ir->CreateAdd(load, LLConstantInt::get(keytype, 1, false));
            DtoStore(load, keyvar);
        }
        llvm::BranchInst::Create(condbb, irs->scopebb());

        // end the dwarf lexical block
        gIR->DBuilder.EmitBlockEnd();

        // end
        irs->scope() = IRScope(endbb, oldend);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ForeachRangeStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ForeachRangeStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // start a dwarf lexical block
        gIR->DBuilder.EmitBlockStart(stmt->loc);

        // evaluate lwr/upr
        assert(stmt->lwr->type->isintegral());
        LLValue* lower = toElemDtor(stmt->lwr)->getRVal();
        assert(stmt->upr->type->isintegral());
        LLValue* upper = toElemDtor(stmt->upr)->getRVal();

        // handle key
        assert(stmt->key->type->isintegral());
        LLValue* keyval = DtoRawVarDeclaration(stmt->key);

        // store initial value in key
        if (stmt->op == TOKforeach)
            DtoStore(lower, keyval);
        else
            DtoStore(upper, keyval);

        // set up the block we'll need
        llvm::BasicBlock* oldend = gIR->scopeend();
        llvm::BasicBlock* condbb = llvm::BasicBlock::Create(gIR->context(), "foreachrange_cond", irs->topfunc(), oldend);
        llvm::BasicBlock* bodybb = llvm::BasicBlock::Create(gIR->context(), "foreachrange_body", irs->topfunc(), oldend);
        llvm::BasicBlock* nextbb = llvm::BasicBlock::Create(gIR->context(), "foreachrange_next", irs->topfunc(), oldend);
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "foreachrange_end", irs->topfunc(), oldend);

        // jump to condition
        llvm::BranchInst::Create(condbb, irs->scopebb());

        // CONDITION
        irs->scope() = IRScope(condbb, bodybb);

        // first we test that lwr < upr
        lower = DtoLoad(keyval);
        assert(lower->getType() == upper->getType());
        llvm::ICmpInst::Predicate cmpop;
        if (isLLVMUnsigned(stmt->key->type))
        {
            cmpop = (stmt->op == TOKforeach)
            ? llvm::ICmpInst::ICMP_ULT
            : llvm::ICmpInst::ICMP_UGT;
        }
        else
        {
            cmpop = (stmt->op == TOKforeach)
            ? llvm::ICmpInst::ICMP_SLT
            : llvm::ICmpInst::ICMP_SGT;
        }
        LLValue* cond = irs->ir->CreateICmp(cmpop, lower, upper);

        // jump to the body if range is ok, to the end if not
        llvm::BranchInst::Create(bodybb, endbb, cond, irs->scopebb());

        // BODY
        irs->scope() = IRScope(bodybb, nextbb);

        // reverse foreach decrements here
        if (stmt->op == TOKforeach_reverse)
        {
            LLValue* v = DtoLoad(keyval);
            LLValue* one = LLConstantInt::get(v->getType(), 1, false);
            v = irs->ir->CreateSub(v, one);
            DtoStore(v, keyval);
        }

        // emit body
        irs->func()->gen->targetScopes.push_back(IRTargetScope(stmt, NULL, nextbb, endbb));
        if (stmt->body)
            stmt->body->accept(this);
        irs->func()->gen->targetScopes.pop_back();

        // jump to next iteration
        if (!irs->scopereturned())
            llvm::BranchInst::Create(nextbb, irs->scopebb());

        // NEXT
        irs->scope() = IRScope(nextbb, endbb);

        // forward foreach increments here
        if (stmt->op == TOKforeach)
        {
            LLValue* v = DtoLoad(keyval);
            LLValue* one = LLConstantInt::get(v->getType(), 1, false);
            v = irs->ir->CreateAdd(v, one);
            DtoStore(v, keyval);
        }

        // jump to condition
        llvm::BranchInst::Create(condbb, irs->scopebb());

        // end the dwarf lexical block
        gIR->DBuilder.EmitBlockEnd();

        // END
        irs->scope() = IRScope(endbb, oldend);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(LabelStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("LabelStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // if it's an inline asm label, we don't create a basicblock, just emit it in the asm
        if (irs->asmBlock)
        {
            IRAsmStmt* a = new IRAsmStmt;
            std::stringstream label;
            printLabelName(label, mangleExact(irs->func()->decl), stmt->ident->toChars());
            label << ":";
            a->code = label.str();
            irs->asmBlock->s.push_back(a);
            irs->asmBlock->internalLabels.push_back(stmt->ident);

            // disable inlining
            gIR->func()->setNeverInline();
        }
        else
        {
            std::string labelname = irs->func()->gen->getScopedLabelName(stmt->ident->toChars());
            llvm::BasicBlock*& labelBB = irs->func()->gen->labelToBB[labelname];

            llvm::BasicBlock* oldend = gIR->scopeend();
            if (labelBB != NULL) {
                labelBB->moveBefore(oldend);
            } else {
                labelBB = llvm::BasicBlock::Create(gIR->context(), "label_" + labelname, irs->topfunc(), oldend);
            }

            if (!irs->scopereturned())
                llvm::BranchInst::Create(labelBB, irs->scopebb());

            irs->scope() = IRScope(labelBB, oldend);
        }

        if (stmt->statement) {
            irs->func()->gen->targetScopes.push_back(IRTargetScope(stmt, NULL, NULL, NULL));
            stmt->statement->accept(this);
            irs->func()->gen->targetScopes.pop_back();
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(GotoStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("GotoStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        gIR->DBuilder.EmitStopPoint(stmt->loc.linnum);

        llvm::BasicBlock* oldend = gIR->scopeend();
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "aftergoto", irs->topfunc(), oldend);

        DtoGoto(stmt->loc, stmt->label, stmt->tf);

        irs->scope() = IRScope(bb, oldend);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(GotoDefaultStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("GotoDefaultStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        gIR->DBuilder.EmitStopPoint(stmt->loc.linnum);

        llvm::BasicBlock* oldend = gIR->scopeend();
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "aftergotodefault", irs->topfunc(), oldend);

        assert(!irs->scopereturned());
        assert(stmt->sw->sdefault->bodyBB);

        DtoEnclosingHandlers(stmt->loc, stmt->sw);

        llvm::BranchInst::Create(stmt->sw->sdefault->bodyBB, irs->scopebb());
        irs->scope() = IRScope(bb,oldend);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(GotoCaseStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("GotoCaseStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        gIR->DBuilder.EmitStopPoint(stmt->loc.linnum);

        llvm::BasicBlock* oldend = gIR->scopeend();
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "aftergotocase", irs->topfunc(), oldend);

        assert(!irs->scopereturned());
        if (!stmt->cs->bodyBB)
        {
            stmt->cs->bodyBB = llvm::BasicBlock::Create(gIR->context(), "goto_case", irs->topfunc(), irs->scopeend());
        }

        DtoEnclosingHandlers(stmt->loc, stmt->sw);

        llvm::BranchInst::Create(stmt->cs->bodyBB, irs->scopebb());
        irs->scope() = IRScope(bb, oldend);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(WithStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("WithStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        gIR->DBuilder.EmitBlockStart(stmt->loc);

        assert(stmt->exp);

        // with(..) can either be used with expressions or with symbols
        // wthis == null indicates the symbol form
        if (stmt->wthis) {
            DValue* e = toElemDtor(stmt->exp);
            LLValue* mem = DtoRawVarDeclaration(stmt->wthis);
            DtoStore(e->getRVal(), mem);
        }

        if (stmt->body)
            stmt->body->accept(this);

        gIR->DBuilder.EmitBlockEnd();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(SwitchErrorStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("SwitchErrorStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        llvm::Function* fn = LLVM_D_GetRuntimeFunction(stmt->loc, gIR->module, "_d_switch_error");

        LLValue *moduleInfoSymbol = gIR->func()->decl->getModule()->moduleInfoSymbol();
        LLType *moduleInfoType = DtoType(Module::moduleinfo->type);

        LLValue* args[] = {
            // module param
            DtoBitCast(moduleInfoSymbol, getPtrToType(moduleInfoType)),
            // line param
            DtoConstUint(stmt->loc.linnum)
        };

        // call
        LLCallSite call = gIR->CreateCallOrInvoke(fn, args);
        call.setDoesNotReturn();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(AsmStatement *stmt) LLVM_OVERRIDE {
        AsmStatement_toIR(stmt, irs);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(AsmBlockStatement *stmt) LLVM_OVERRIDE {
        AsmBlockStatement_toIR(stmt, irs);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ImportStatement *stmt) LLVM_OVERRIDE {
        // Empty.
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(Statement *stmt) LLVM_OVERRIDE {
        error(stmt->loc, "Statement type Statement not implemented: %s", stmt->toChars());
        fatal();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(PragmaStatement *stmt) LLVM_OVERRIDE {
        error(stmt->loc, "Statement type PragmaStatement not implemented: %s", stmt->toChars());
        fatal();
    }
};

//////////////////////////////////////////////////////////////////////////////

void codegenFunction(Statement *s, IRState *irs)
{
    FindEnclosingTryFinally v;
    s->accept(&v);
    Statement_toIR(s, irs);
}

void Statement_toIR(Statement *s, IRState *irs)
{
    ToIRVisitor v(irs);
    s->accept(&v);
}
