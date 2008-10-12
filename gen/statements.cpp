// Statements: D -> LLVM glue

#include <stdio.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <iostream>

#include "gen/llvm.h"
#include "llvm/InlineAsm.h"
#include "llvm/Support/CFG.h"

#include "mars.h"
#include "total.h"
#include "init.h"
#include "mtype.h"
#include "hdrgen.h"
#include "port.h"
#include "module.h"

#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/runtime.h"
#include "gen/arrays.h"
#include "gen/todebug.h"
#include "gen/dvalue.h"

#include "ir/irfunction.h"
#include "ir/irmodule.h"
#include "ir/irlandingpad.h"

//////////////////////////////////////////////////////////////////////////////

void CompoundStatement::toIR(IRState* p)
{
    Logger::println("CompoundStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    for (int i=0; i<statements->dim; i++)
    {
        Statement* s = (Statement*)statements->data[i];
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

    if (exp)
    {
        if (p->topfunc()->getReturnType() == LLType::VoidTy) {
            IrFunction* f = p->func();
            assert(f->type->retInPtr);
            assert(f->decl->ir.irFunc->retArg);

            if (global.params.symdebug) DtoDwarfStopPoint(loc.linnum);

            DValue* rvar = new DVarValue(f->type->next, f->decl->ir.irFunc->retArg);

            DValue* e = exp->toElem(p);

            DtoAssign(loc, rvar, e);

            DtoEnclosingHandlers(enclosinghandler, NULL);

            if (global.params.symdebug) DtoDwarfFuncEnd(f->decl);
            llvm::ReturnInst::Create(p->scopebb());

        }
        else {
            if (global.params.symdebug) DtoDwarfStopPoint(loc.linnum);
            DValue* e = exp->toElem(p);
            LLValue* v = e->getRVal();
            delete e;

            if (Logger::enabled())
                Logger::cout() << "return value is '" <<*v << "'\n";

            // can happen for classes
            if (v->getType() != p->topfunc()->getReturnType())
            {
                v = gIR->ir->CreateBitCast(v, p->topfunc()->getReturnType(), "tmp");
                if (Logger::enabled())
                    Logger::cout() << "return value after cast: " << *v << '\n';
            }

            DtoEnclosingHandlers(enclosinghandler, NULL);

            if (global.params.symdebug) DtoDwarfFuncEnd(p->func()->decl);
            llvm::ReturnInst::Create(v, p->scopebb());
        }
    }
    else
    {
        assert(p->topfunc()->getReturnType() == LLType::VoidTy);
        DtoEnclosingHandlers(enclosinghandler, NULL);

        if (global.params.symdebug) DtoDwarfFuncEnd(p->func()->decl);
        llvm::ReturnInst::Create(p->scopebb());
    }

    // the return terminated this basicblock, start a new one
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create("afterreturn", p->topfunc(), oldend);
    p->scope() = IRScope(bb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void ExpStatement::toIR(IRState* p)
{
    Logger::println("ExpStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    if (exp) {
        if (global.params.llvmAnnotate)
            DtoAnnotation(exp->toChars());
        elem* e;
        // a cast(void) around the expression is allowed, but doesn't require any code
        if(exp->op == TOKcast && exp->type == Type::tvoid) {
            CastExp* cexp = (CastExp*)exp;
            e = cexp->e1->toElem(p);
        }
        else
            e = exp->toElem(p);
        delete e;
    }
    /*elem* e = exp->toElem(p);
    p->buf.printf("%s", e->toChars());
    delete e;
    p->buf.writenl();*/
}

//////////////////////////////////////////////////////////////////////////////

void IfStatement::toIR(IRState* p)
{
    Logger::println("IfStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    if (match)
    {
        LLValue* allocainst = DtoAlloca(DtoType(match->type), "._tmp_if_var");
        match->ir.irLocal = new IrLocal(match);
        match->ir.irLocal->value = allocainst;
    }

    DValue* cond_e = condition->toElem(p);
    LLValue* cond_val = cond_e->getRVal();

    llvm::BasicBlock* oldend = gIR->scopeend();

    llvm::BasicBlock* ifbb = llvm::BasicBlock::Create("if", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("endif", gIR->topfunc(), oldend);
    llvm::BasicBlock* elsebb = elsebody ? llvm::BasicBlock::Create("else", gIR->topfunc(), endbb) : endbb;

    if (cond_val->getType() != LLType::Int1Ty) {
        if (Logger::enabled())
            Logger::cout() << "if conditional: " << *cond_val << '\n';
        cond_val = DtoBoolean(loc, cond_e);
    }
    LLValue* ifgoback = llvm::BranchInst::Create(ifbb, elsebb, cond_val, gIR->scopebb());

    // replace current scope
    gIR->scope() = IRScope(ifbb,elsebb);

    // do scoped statements
    if (ifbody)
        ifbody->toIR(p);
    if (!gIR->scopereturned()) {
        llvm::BranchInst::Create(endbb,gIR->scopebb());
    }

    if (elsebody) {
        //assert(0);
        gIR->scope() = IRScope(elsebb,endbb);
        elsebody->toIR(p);
        if (!gIR->scopereturned()) {
            llvm::BranchInst::Create(endbb,gIR->scopebb());
        }
    }

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
        beginbb = llvm::BasicBlock::Create("scope", p->topfunc(), oldend);
        if (!p->scopereturned())
            llvm::BranchInst::Create(beginbb, bb);
    }

    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("endscope", p->topfunc(), oldend);
    if (beginbb != bb)
        p->scope() = IRScope(beginbb, endbb);
    else
        p->scope().end = endbb;*/

    if (statement)
        statement->toIR(p);

    /*p->scope().end = oldend;
    Logger::println("Erasing scope endbb");
    endbb->eraseFromParent();*/
}

//////////////////////////////////////////////////////////////////////////////

void WhileStatement::toIR(IRState* p)
{
    Logger::println("WhileStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    // create while blocks
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* whilebb = llvm::BasicBlock::Create("whilecond", gIR->topfunc(), oldend);
    llvm::BasicBlock* whilebodybb = llvm::BasicBlock::Create("whilebody", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("endwhile", gIR->topfunc(), oldend);

    // move into the while block
    p->ir->CreateBr(whilebb);
    //llvm::BranchInst::Create(whilebb, gIR->scopebb());

    // replace current scope
    gIR->scope() = IRScope(whilebb,endbb);

    // create the condition
    DValue* cond_e = condition->toElem(p);
    LLValue* cond_val = DtoBoolean(loc, cond_e);
    delete cond_e;

    // conditional branch
    LLValue* ifbreak = llvm::BranchInst::Create(whilebodybb, endbb, cond_val, p->scopebb());

    // rewrite scope
    gIR->scope() = IRScope(whilebodybb,endbb);

    // while body code
    p->loopbbs.push_back(IRLoopScope(this,enclosinghandler,whilebb,endbb));
    body->toIR(p);
    p->loopbbs.pop_back();

    // loop
    if (!gIR->scopereturned())
        llvm::BranchInst::Create(whilebb, gIR->scopebb());

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void DoStatement::toIR(IRState* p)
{
    Logger::println("DoStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    // create while blocks
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* dowhilebb = llvm::BasicBlock::Create("dowhile", gIR->topfunc(), oldend);
    llvm::BasicBlock* condbb = llvm::BasicBlock::Create("dowhilecond", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("enddowhile", gIR->topfunc(), oldend);

    // move into the while block
    assert(!gIR->scopereturned());
    llvm::BranchInst::Create(dowhilebb, gIR->scopebb());

    // replace current scope
    gIR->scope() = IRScope(dowhilebb,condbb);

    // do-while body code
    p->loopbbs.push_back(IRLoopScope(this,enclosinghandler,condbb,endbb));
    body->toIR(p);
    p->loopbbs.pop_back();

    // branch to condition block
    llvm::BranchInst::Create(condbb, gIR->scopebb());
    gIR->scope() = IRScope(condbb,endbb);

    // create the condition
    DValue* cond_e = condition->toElem(p);
    LLValue* cond_val = DtoBoolean(loc, cond_e);
    delete cond_e;

    // conditional branch
    LLValue* ifbreak = llvm::BranchInst::Create(dowhilebb, endbb, cond_val, gIR->scopebb());

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void ForStatement::toIR(IRState* p)
{
    Logger::println("ForStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    // create for blocks
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* forbb = llvm::BasicBlock::Create("forcond", gIR->topfunc(), oldend);
    llvm::BasicBlock* forbodybb = llvm::BasicBlock::Create("forbody", gIR->topfunc(), oldend);
    llvm::BasicBlock* forincbb = llvm::BasicBlock::Create("forinc", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("endfor", gIR->topfunc(), oldend);

    // init
    if (init != 0)
    init->toIR(p);

    // move into the for condition block, ie. start the loop
    assert(!gIR->scopereturned());
    llvm::BranchInst::Create(forbb, gIR->scopebb());

    p->loopbbs.push_back(IRLoopScope(this,enclosinghandler,forincbb,endbb));

    // replace current scope
    gIR->scope() = IRScope(forbb,forbodybb);

    // create the condition
    DValue* cond_e = condition->toElem(p);
    LLValue* cond_val = DtoBoolean(loc, cond_e);
    delete cond_e;

    // conditional branch
    assert(!gIR->scopereturned());
    llvm::BranchInst::Create(forbodybb, endbb, cond_val, gIR->scopebb());

    // rewrite scope
    gIR->scope() = IRScope(forbodybb,forincbb);

    // do for body code
    body->toIR(p);

    // move into the for increment block
    if (!gIR->scopereturned())
        llvm::BranchInst::Create(forincbb, gIR->scopebb());
    gIR->scope() = IRScope(forincbb, endbb);

    // increment
    if (increment) {
        DValue* inc = increment->toElem(p);
        delete inc;
    }

    // loop
    if (!gIR->scopereturned())
        llvm::BranchInst::Create(forbb, gIR->scopebb());

    p->loopbbs.pop_back();

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);
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

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    if (ident != 0) {
        Logger::println("ident = %s", ident->toChars());

        DtoEnclosingHandlers(enclosinghandler, target->enclosinghandler);

        // get the loop statement the label refers to
        Statement* targetLoopStatement = target->statement;
        ScopeStatement* tmp;
        while(tmp = targetLoopStatement->isScopeStatement())
            targetLoopStatement = tmp->statement;

        // find the right break block and jump there
        bool found = false;
        IRState::LoopScopeVec::reverse_iterator it;
        for(it = p->loopbbs.rbegin(); it != p->loopbbs.rend(); ++it) {
            if(it->s == targetLoopStatement) {
                llvm::BranchInst::Create(it->end, p->scopebb());
                found = true;
                break;
            }
        }
        assert(found);
    }
    else {
        DtoEnclosingHandlers(enclosinghandler, p->loopbbs.back().enclosinghandler);
        llvm::BranchInst::Create(p->loopbbs.back().end, p->scopebb());
    }

    // the break terminated this basicblock, start a new one
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create("afterbreak", p->topfunc(), oldend);
    p->scope() = IRScope(bb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void ContinueStatement::toIR(IRState* p)
{
    Logger::println("ContinueStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    if (ident != 0) {
        Logger::println("ident = %s", ident->toChars());

        DtoEnclosingHandlers(enclosinghandler, target->enclosinghandler);

        // get the loop statement the label refers to
        Statement* targetLoopStatement = target->statement;
        ScopeStatement* tmp;
        while(tmp = targetLoopStatement->isScopeStatement())
            targetLoopStatement = tmp->statement;

        // find the right continue block and jump there
        bool found = false;
        IRState::LoopScopeVec::reverse_iterator it;
        for(it = gIR->loopbbs.rbegin(); it != gIR->loopbbs.rend(); ++it) {
            if(it->s == targetLoopStatement) {
                llvm::BranchInst::Create(it->begin, gIR->scopebb());
                found = true;
                break;
            }
        }
        assert(found);
    }
    else {
        // can't 'continue' within switch, so omit them
        IRState::LoopScopeVec::reverse_iterator it;
        for(it = gIR->loopbbs.rbegin(); it != gIR->loopbbs.rend(); ++it) {
            if(!it->isSwitch) {
                break;
            }
        }
        DtoEnclosingHandlers(enclosinghandler, it->enclosinghandler);
        llvm::BranchInst::Create(it->begin, gIR->scopebb());
    }

    // the continue terminated this basicblock, start a new one
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create("aftercontinue", p->topfunc(), oldend);
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

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    // if there's no finalbody or no body, things are simple
    if (!finalbody) {
        if (body)
            body->toIR(p);
        return;
    }
    if (!body) {
        finalbody->toIR(p);
        return;
    }

    // create basic blocks
    llvm::BasicBlock* oldend = p->scopeend();

    llvm::BasicBlock* trybb = llvm::BasicBlock::Create("try", p->topfunc(), oldend);
    llvm::BasicBlock* finallybb = llvm::BasicBlock::Create("finally", p->topfunc(), oldend);
    // the landing pad for statements in the try block
    llvm::BasicBlock* landingpadbb = llvm::BasicBlock::Create("landingpad", p->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("endtryfinally", p->topfunc(), oldend);

    // pass the previous BB into this
    assert(!gIR->scopereturned());
    llvm::BranchInst::Create(trybb, p->scopebb());

    //
    // set up the landing pad
    //
    p->scope() = IRScope(landingpadbb, endbb);

    assert(finalbody);
    gIR->func()->landingPad.addFinally(finalbody);
    gIR->func()->landingPad.push(landingpadbb);

    //
    // do the try block
    //
    p->scope() = IRScope(trybb,finallybb);

    assert(body);
    body->toIR(p);

    // terminate try BB
    if (!p->scopereturned())
        llvm::BranchInst::Create(finallybb, p->scopebb());

    gIR->func()->landingPad.pop();

    //
    // do finally block
    //
    p->scope() = IRScope(finallybb,landingpadbb);
    finalbody->toIR(p);

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

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    // create basic blocks
    llvm::BasicBlock* oldend = p->scopeend();

    llvm::BasicBlock* trybb = llvm::BasicBlock::Create("try", p->topfunc(), oldend);
    // the landing pad will be responsible for branching to the correct catch block
    llvm::BasicBlock* landingpadbb = llvm::BasicBlock::Create("landingpad", p->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("endtrycatch", p->topfunc(), oldend);

    // pass the previous BB into this
    assert(!gIR->scopereturned());
    llvm::BranchInst::Create(trybb, p->scopebb());

    //
    // do catches and the landing pad
    //
    assert(catches);
    gIR->scope() = IRScope(landingpadbb, endbb);

    for (int i = 0; i < catches->dim; i++)
    {
        Catch *c = (Catch *)catches->data[i];
        gIR->func()->landingPad.addCatch(c, endbb);
    }

    gIR->func()->landingPad.push(landingpadbb);

    //
    // do the try block
    //
    p->scope() = IRScope(trybb,landingpadbb);

    assert(body);
    body->toIR(p);

    if (!gIR->scopereturned())
        llvm::BranchInst::Create(endbb, p->scopebb());

    gIR->func()->landingPad.pop();

    // rewrite the scope
    p->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void ThrowStatement::toIR(IRState* p)
{
    Logger::println("ThrowStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    assert(exp);
    DValue* e = exp->toElem(p);
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_throw_exception");
    //Logger::cout() << "calling: " << *fn << '\n';
    LLValue* arg = DtoBitCast(e->getRVal(), fn->getFunctionType()->getParamType(0));
    //Logger::cout() << "arg: " << *arg << '\n';
    gIR->CreateCallOrInvoke(fn, arg);
    gIR->ir->CreateUnreachable();

    // need a block after the throw for now
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create("afterthrow", p->topfunc(), oldend);
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
        Case* c2 = (Case*)obj;
        return str->compare(c2->str);
    }
};

static LLValue* call_string_switch_runtime(llvm::GlobalVariable* table, Expression* e)
{
    Type* dt = e->type->toBasetype();
    Type* dtnext = dt->next->toBasetype();
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
        assert(0 && "not char/wchar/dchar");
    }

    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, fname);

    if (Logger::enabled())
    {
        Logger::cout() << *table->getType() << '\n';
        Logger::cout() << *fn->getFunctionType()->getParamType(0) << '\n';
    }
    assert(table->getType() == fn->getFunctionType()->getParamType(0));

    DValue* val = e->toElem(gIR);
    LLValue* llval;
    if (DSliceValue* sval = val->isSlice())
    {
        // give storage
        llval = DtoAlloca(DtoType(e->type), "tmp");
        DVarValue* vv = new DVarValue(e->type, llval);
        DtoAssign(e->loc, vv, val);
    }
    else
    {
        llval = val->getRVal();
    }
    assert(llval->getType() == fn->getFunctionType()->getParamType(1));

    CallOrInvoke* call = gIR->CreateCallOrInvoke2(fn, table, llval, "tmp");

    llvm::AttrListPtr palist;
    palist = palist.addAttr(1, llvm::Attribute::ByVal);
    palist = palist.addAttr(2, llvm::Attribute::ByVal);
    call->setAttributes(palist);

    return call->get();
}

void SwitchStatement::toIR(IRState* p)
{
    Logger::println("SwitchStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    llvm::BasicBlock* oldend = gIR->scopeend();

    // clear data from previous passes... :/
    for (int i=0; i<cases->dim; ++i)
    {
        CaseStatement* cs = (CaseStatement*)cases->data[i];
        cs->bodyBB = NULL;
        cs->llvmIdx = NULL;
    }

    // string switch?
    llvm::GlobalVariable* switchTable = 0;
    Array caseArray;
    if (!condition->type->isintegral())
    {
        Logger::println("is string switch");
        // build array of the stringexpS
        caseArray.reserve(cases->dim);
        for (int i=0; i<cases->dim; ++i)
        {
            CaseStatement* cs = (CaseStatement*)cases->data[i];

            assert(cs->exp->op == TOKstring);
            caseArray.push(new Case((StringExp*)cs->exp, i));
        }
        // first sort it
        caseArray.sort();
        // iterate and add indices to cases
        std::vector<LLConstant*> inits(caseArray.dim);
        for (size_t i=0; i<caseArray.dim; ++i)
        {
            Case* c = (Case*)caseArray.data[i];
            CaseStatement* cs = (CaseStatement*)cases->data[c->index];
            cs->llvmIdx = DtoConstUint(i);
            inits[i] = c->str->toConstElem(p);
        }
        // build static array for ptr or final array
        const LLType* elemTy = DtoType(condition->type);
        const llvm::ArrayType* arrTy = llvm::ArrayType::get(elemTy, inits.size());
        LLConstant* arrInit = llvm::ConstantArray::get(arrTy, inits);
        llvm::GlobalVariable* arr = new llvm::GlobalVariable(arrTy, true, llvm::GlobalValue::InternalLinkage, arrInit, ".string_switch_table_data", gIR->module);

        const LLType* elemPtrTy = getPtrToType(elemTy);
        LLConstant* arrPtr = llvm::ConstantExpr::getBitCast(arr, elemPtrTy);

        // build the static table
        std::vector<const LLType*> types;
        types.push_back(DtoSize_t());
        types.push_back(elemPtrTy);
        const llvm::StructType* sTy = llvm::StructType::get(types);
        std::vector<LLConstant*> sinits;
        sinits.push_back(DtoConstSize_t(inits.size()));
        sinits.push_back(arrPtr);
        LLConstant* sInit = llvm::ConstantStruct::get(sTy, sinits);

        switchTable = new llvm::GlobalVariable(sTy, true, llvm::GlobalValue::InternalLinkage, sInit, ".string_switch_table", gIR->module);
    }

    // body block
    llvm::BasicBlock* bodybb = llvm::BasicBlock::Create("switchbody", p->topfunc(), oldend);

    // default
    llvm::BasicBlock* defbb = 0;
    if (sdefault) {
        Logger::println("has default");
        defbb = llvm::BasicBlock::Create("default", p->topfunc(), oldend);
        sdefault->bodyBB = defbb;
    }

    // end (break point)
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("switchend", p->topfunc(), oldend);

    // condition var
    LLValue* condVal;
    // integral switch
    if (condition->type->isintegral()) {
        DValue* cond = condition->toElem(p);
        condVal = cond->getRVal();
    }
    // string switch
    else {
        condVal = call_string_switch_runtime(switchTable, condition);
    }
    llvm::SwitchInst* si = llvm::SwitchInst::Create(condVal, defbb ? defbb : endbb, cases->dim, p->scopebb());

    // do switch body
    assert(body);

    p->scope() = IRScope(bodybb, endbb);
    p->loopbbs.push_back(IRLoopScope(this,enclosinghandler,p->scopebb(),endbb,true));
    body->toIR(p);
    p->loopbbs.pop_back();

    if (!p->scopereturned())
        llvm::BranchInst::Create(endbb, p->scopebb());

    // add the cases
    for (int i=0; i<cases->dim; ++i)
    {
        CaseStatement* cs = (CaseStatement*)cases->data[i];
        si->addCase(cs->llvmIdx, cs->bodyBB);
    }

    gIR->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////
void CaseStatement::toIR(IRState* p)
{
    Logger::println("CaseStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    llvm::BasicBlock* nbb = llvm::BasicBlock::Create("case", p->topfunc(), p->scopeend());

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
    statement->toIR(p);
}

//////////////////////////////////////////////////////////////////////////////
void DefaultStatement::toIR(IRState* p)
{
    Logger::println("DefaultStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    assert(bodyBB);

    llvm::BasicBlock* nbb = llvm::BasicBlock::Create("default", p->topfunc(), p->scopeend());

    if (!bodyBB->getTerminator())
    {
        llvm::BranchInst::Create(nbb, bodyBB);
    }
    bodyBB = nbb;

    if (!p->scopereturned())
        llvm::BranchInst::Create(bodyBB, p->scopebb());

    p->scope() = IRScope(bodyBB, p->scopeend());

    assert(statement);
    statement->toIR(p);
}

//////////////////////////////////////////////////////////////////////////////

void UnrolledLoopStatement::toIR(IRState* p)
{
    Logger::println("UnrolledLoopStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("unrolledend", p->topfunc(), oldend);

    p->scope() = IRScope(p->scopebb(),endbb);
    p->loopbbs.push_back(IRLoopScope(this,enclosinghandler,p->scopebb(),endbb));

    for (int i=0; i<statements->dim; ++i)
    {
        Statement* s = (Statement*)statements->data[i];
        s->toIR(p);
    }

    p->loopbbs.pop_back();

    llvm::BranchInst::Create(endbb, p->scopebb());
    p->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void ForeachStatement::toIR(IRState* p)
{
    Logger::println("ForeachStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    //assert(arguments->dim == 1);
    assert(value != 0);
    assert(aggr != 0);
    assert(func != 0);

    //Argument* arg = (Argument*)arguments->data[0];
    //Logger::println("Argument is %s", arg->toChars());

    Logger::println("aggr = %s", aggr->toChars());

    // key
    const LLType* keytype = key ? DtoType(key->type) : DtoSize_t();
    LLValue* keyvar = DtoAlloca(keytype, "foreachkey");
    if (key)
    {
        //key->llvmValue = keyvar;
        assert(!key->ir.irLocal);
        key->ir.irLocal = new IrLocal(key);
        key->ir.irLocal->value = keyvar;
    }
    LLValue* zerokey = llvm::ConstantInt::get(keytype,0,false);

    // value
    Logger::println("value = %s", value->toPrettyChars());
    const LLType* valtype = DtoType(value->type);
    LLValue* valvar = NULL;
    if (!value->isRef() && !value->isOut())
        valvar = DtoAlloca(valtype, "foreachval");
    if (!value->ir.irLocal)
        value->ir.irLocal = new IrLocal(value);

    // what to iterate
    DValue* aggrval = aggr->toElem(p);
    Type* aggrtype = aggr->type->toBasetype();

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

    LLConstant* delta = 0;
    if (op == TOKforeach) {
        new llvm::StoreInst(zerokey, keyvar, p->scopebb());
    }
    else {
        new llvm::StoreInst(niters, keyvar, p->scopebb());
    }

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* condbb = llvm::BasicBlock::Create("foreachcond", p->topfunc(), oldend);
    llvm::BasicBlock* bodybb = llvm::BasicBlock::Create("foreachbody", p->topfunc(), oldend);
    llvm::BasicBlock* nextbb = llvm::BasicBlock::Create("foreachnext", p->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("foreachend", p->topfunc(), oldend);

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
        load = p->ir->CreateSub(load, llvm::ConstantInt::get(keytype, 1, false), "tmp");
        DtoStore(load, keyvar);
    }
    llvm::BranchInst::Create(bodybb, endbb, done, p->scopebb());

    // init body
    p->scope() = IRScope(bodybb,nextbb);

    // get value for this iteration
    LLConstant* zero = llvm::ConstantInt::get(keytype,0,false);
    LLValue* loadedKey = p->ir->CreateLoad(keyvar,"tmp");
    value->ir.irLocal->value = DtoGEP1(val,loadedKey);

    if (!value->isRef() && !value->isOut()) {
        DVarValue dst(value->type, valvar);
        DVarValue src(value->type, value->ir.irLocal->value);
        DtoAssign(loc, &dst, &src);
        value->ir.irLocal->value = valvar;
    }

    // emit body
    p->loopbbs.push_back(IRLoopScope(this,enclosinghandler,nextbb,endbb));
    if(body)
        body->toIR(p);
    p->loopbbs.pop_back();

    if (!p->scopereturned())
        llvm::BranchInst::Create(nextbb, p->scopebb());

    // next
    p->scope() = IRScope(nextbb,endbb);
    if (op == TOKforeach) {
        LLValue* load = DtoLoad(keyvar);
        load = p->ir->CreateAdd(load, llvm::ConstantInt::get(keytype, 1, false), "tmp");
        DtoStore(load, keyvar);
    }
    llvm::BranchInst::Create(condbb, p->scopebb());

    // end
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
        a->code += p->func()->decl->mangle();
        a->code += "_";
        a->code += ident->toChars();
        a->code += ":";
        p->asmBlock->s.push_back(a);
        p->asmBlock->internalLabels.push_back(ident);
    }
    else
    {
        std::string labelname = p->func()->getScopedLabelName(ident->toChars());
        llvm::BasicBlock*& labelBB = p->func()->labelToBB[labelname];

        llvm::BasicBlock* oldend = gIR->scopeend();
        if (labelBB != NULL) {
                labelBB->moveBefore(oldend);
        } else {
                labelBB = llvm::BasicBlock::Create("label", p->topfunc(), oldend);
        }

        if (!p->scopereturned())
                llvm::BranchInst::Create(labelBB, p->scopebb());

        p->scope() = IRScope(labelBB,oldend);
    }

    if (statement)
        statement->toIR(p);
}

//////////////////////////////////////////////////////////////////////////////

void GotoStatement::toIR(IRState* p)
{
    Logger::println("GotoStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create("aftergoto", p->topfunc(), oldend);

    DtoGoto(&loc, label->ident, enclosinghandler, tf);

    p->scope() = IRScope(bb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void GotoDefaultStatement::toIR(IRState* p)
{
    Logger::println("GotoDefaultStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create("aftergotodefault", p->topfunc(), oldend);

    assert(!p->scopereturned());
    assert(sw->sdefault->bodyBB);

    DtoEnclosingHandlers(enclosinghandler, sw->enclosinghandler);

    llvm::BranchInst::Create(sw->sdefault->bodyBB, p->scopebb());
    p->scope() = IRScope(bb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void GotoCaseStatement::toIR(IRState* p)
{
    Logger::println("GotoCaseStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create("aftergotocase", p->topfunc(), oldend);

    assert(!p->scopereturned());
    if (!cs->bodyBB)
    {
        cs->bodyBB = llvm::BasicBlock::Create("goto_case", p->topfunc(), p->scopeend());
    }

    DtoEnclosingHandlers(enclosinghandler, sw->enclosinghandler);

    llvm::BranchInst::Create(cs->bodyBB, p->scopebb());
    p->scope() = IRScope(bb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void WithStatement::toIR(IRState* p)
{
    Logger::println("WithStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    assert(exp);
    assert(body);

    DValue* e = exp->toElem(p);
    assert(!wthis->ir.isSet());
    wthis->ir.irLocal = new IrLocal(wthis);
    wthis->ir.irLocal->value = DtoAlloca(DtoType(wthis->type), wthis->toChars());
    DtoStore(e->getRVal(), wthis->ir.irLocal->value);

    body->toIR(p);
}

//////////////////////////////////////////////////////////////////////////////

static LLConstant* generate_unique_critical_section()
{
    const LLType* Mty = DtoMutexType();
    return new llvm::GlobalVariable(Mty, false, llvm::GlobalValue::InternalLinkage, LLConstant::getNullValue(Mty), ".uniqueCS", gIR->module);
}

void SynchronizedStatement::toIR(IRState* p)
{
    Logger::println("SynchronizedStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

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
    body->toIR(p);

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

void VolatileStatement::toIR(IRState* p)
{
    Logger::println("VolatileStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (global.params.symdebug)
        DtoDwarfStopPoint(loc.linnum);

    // mark in-volatile
    // FIXME

    // has statement
    if (statement != NULL)
    {
        // load-store
        DtoMemoryBarrier(false, true, false, false);

        // do statement
        statement->toIR(p);

        // no point in a unreachable barrier, terminating statements must insert this themselves.
        if (statement->fallOffEnd())
        {
            // store-load
            DtoMemoryBarrier(false, false, true, false);
        }
    }
    // barrier only
    else
    {
        // load-store & store-load
        DtoMemoryBarrier(false, true, true, false);
    }

    // restore volatile state
    // FIXME
}

//////////////////////////////////////////////////////////////////////////////

void SwitchErrorStatement::toIR(IRState* p)
{
    Logger::println("SwitchErrorStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_switch_error");

    // param attrs
    llvm::AttrListPtr palist;
    int idx = 1;

    std::vector<LLValue*> args;

    // file param
    args.push_back(gIR->dmodule->ir.irModule->fileName);
    palist = palist.addAttr(idx++, llvm::Attribute::ByVal);

    // line param
    LLConstant* c = DtoConstUint(loc.linnum);
    args.push_back(c);

    // call
    CallOrInvoke* call = gIR->CreateCallOrInvoke(fn, args.begin(), args.end());
    call->setAttributes(palist);

    gIR->ir->CreateUnreachable();
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
//STUBST(VolatileStatement);
//STUBST(LabelStatement);
//STUBST(ThrowStatement);
//STUBST(GotoCaseStatement);
//STUBST(GotoDefaultStatement);
//STUBST(GotoStatement);
//STUBST(UnrolledLoopStatement);
//STUBST(OnScopeStatement);
