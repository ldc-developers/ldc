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

#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/runtime.h"
#include "gen/arrays.h"
#include "gen/todebug.h"
#include "gen/dvalue.h"

#include "ir/irfunction.h"

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

// generates IR for finally blocks between the 'start' and 'end' statements
// will begin with the finally block belonging to 'start' and does not include
// the finally block of 'end'
void emit_finallyblocks(IRState* p, TryFinallyStatement* start, TryFinallyStatement* end)
{
    // verify that end encloses start
    TryFinallyStatement* endfinally = start;
    while(endfinally != NULL && endfinally != end) {
        endfinally = endfinally->enclosingtryfinally;
    }
    assert(endfinally == end);

    // emit code for finallys between start and end
    TryFinallyStatement* tf = start;
    while(tf != end) {
        tf->finalbody->toIR(p);
        tf = tf->enclosingtryfinally;
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
            assert(f->type->llvmRetInPtr);
            assert(f->decl->ir.irFunc->retArg);

            if (global.params.symdebug) DtoDwarfStopPoint(loc.linnum);

            DValue* rvar = new DVarValue(f->type->next, f->decl->ir.irFunc->retArg, true);

            p->exps.push_back(IRExp(NULL,exp,rvar));
            DValue* e = exp->toElem(p);
            p->exps.pop_back();

            if (!e->inPlace())
                DtoAssign(rvar, e);

            emit_finallyblocks(p, enclosingtryfinally, NULL);

            if (f->inVolatile) {
                // store-load barrier
                DtoMemoryBarrier(false, false, true, false);
            }

            if (global.params.symdebug) DtoDwarfFuncEnd(f->decl);
            llvm::ReturnInst::Create(p->scopebb());

        }
        else {
            if (global.params.symdebug) DtoDwarfStopPoint(loc.linnum);
            DValue* e = exp->toElem(p);
            LLValue* v = e->getRVal();
            delete e;
            Logger::cout() << "return value is '" <<*v << "'\n";

            // can happen for classes
            if (v->getType() != p->topfunc()->getReturnType())
            {
                v = gIR->ir->CreateBitCast(v, p->topfunc()->getReturnType(), "tmp");
                Logger::cout() << "return value after cast: " << *v << '\n';
            }

            emit_finallyblocks(p, enclosingtryfinally, NULL);

            if (gIR->func()->inVolatile) {
                // store-load barrier
                DtoMemoryBarrier(false, false, true, false);
            }

            if (global.params.symdebug) DtoDwarfFuncEnd(p->func()->decl);
            llvm::ReturnInst::Create(v, p->scopebb());
        }
    }
    else
    {
        assert(p->topfunc()->getReturnType() == LLType::VoidTy);
        emit_finallyblocks(p, enclosingtryfinally, NULL);

        if (gIR->func()->inVolatile) {
            // store-load barrier
            DtoMemoryBarrier(false, false, true, false);
        }

        if (global.params.symdebug) DtoDwarfFuncEnd(p->func()->decl);
        llvm::ReturnInst::Create(p->scopebb());
    }
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
        elem* e = exp->toElem(p);
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

    if (match)
    {
        LLValue* allocainst = new llvm::AllocaInst(DtoType(match->type), "._tmp_if_var", p->topallocapoint());
        match->ir.irLocal = new IrLocal(match);
        match->ir.irLocal->value = allocainst;
    }

    DValue* cond_e = condition->toElem(p);
    LLValue* cond_val = cond_e->getRVal();
    delete cond_e;

    llvm::BasicBlock* oldend = gIR->scopeend();

    llvm::BasicBlock* ifbb = llvm::BasicBlock::Create("if", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("endif", gIR->topfunc(), oldend);
    llvm::BasicBlock* elsebb = elsebody ? llvm::BasicBlock::Create("else", gIR->topfunc(), endbb) : endbb;

    if (cond_val->getType() != LLType::Int1Ty) {
        Logger::cout() << "if conditional: " << *cond_val << '\n';
        cond_val = DtoBoolean(cond_val);
    }
    LLValue* ifgoback = llvm::BranchInst::Create(ifbb, elsebb, cond_val, gIR->scopebb());

    // replace current scope
    gIR->scope() = IRScope(ifbb,elsebb);

    // do scoped statements
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

    llvm::BasicBlock* oldend = p->scopeend();

    llvm::BasicBlock* beginbb = 0;

    // remove useless branches by clearing and reusing the current basicblock
    llvm::BasicBlock* bb = p->scopebb();
    if (bb->empty()) {
        beginbb = bb;
    }
    else {
        assert(!p->scopereturned());
        beginbb = llvm::BasicBlock::Create("scope", p->topfunc(), oldend);
        llvm::BranchInst::Create(beginbb, p->scopebb());
    }
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("endscope", p->topfunc(), oldend);

    gIR->scope() = IRScope(beginbb, endbb);

    if (statement)
        statement->toIR(p);

    p->scope() = IRScope(p->scopebb(),oldend);
    endbb->eraseFromParent();
}

//////////////////////////////////////////////////////////////////////////////

void WhileStatement::toIR(IRState* p)
{
    Logger::println("WhileStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

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
    LLValue* cond_val = DtoBoolean(cond_e->getRVal());
    delete cond_e;

    // conditional branch
    LLValue* ifbreak = llvm::BranchInst::Create(whilebodybb, endbb, cond_val, p->scopebb());

    // rewrite scope
    gIR->scope() = IRScope(whilebodybb,endbb);

    // while body code
    p->loopbbs.push_back(IRLoopScope(this,enclosingtryfinally,whilebb,endbb));
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

    // create while blocks
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* dowhilebb = llvm::BasicBlock::Create("dowhile", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("enddowhile", gIR->topfunc(), oldend);

    // move into the while block
    assert(!gIR->scopereturned());
    llvm::BranchInst::Create(dowhilebb, gIR->scopebb());

    // replace current scope
    gIR->scope() = IRScope(dowhilebb,endbb);

    // do-while body code
    p->loopbbs.push_back(IRLoopScope(this,enclosingtryfinally,dowhilebb,endbb));
    body->toIR(p);
    p->loopbbs.pop_back();

    // create the condition
    DValue* cond_e = condition->toElem(p);
    LLValue* cond_val = DtoBoolean(cond_e->getRVal());
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

    p->loopbbs.push_back(IRLoopScope(this,enclosingtryfinally,forincbb,endbb));

    // replace current scope
    gIR->scope() = IRScope(forbb,forbodybb);

    // create the condition
    DValue* cond_e = condition->toElem(p);
    LLValue* cond_val = DtoBoolean(cond_e->getRVal());
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

    if (ident != 0) {
        Logger::println("ident = %s", ident->toChars());

        emit_finallyblocks(p, enclosingtryfinally, target->enclosingtryfinally);

        // get the loop statement the label refers to
        Statement* targetLoopStatement = target->statement;
        ScopeStatement* tmp;
        while(tmp = targetLoopStatement->isScopeStatement())
            targetLoopStatement = tmp->statement;

        // find the right break block and jump there
        IRState::LoopScopeVec::reverse_iterator it;
        for(it = gIR->loopbbs.rbegin(); it != gIR->loopbbs.rend(); ++it) {
            if(it->s == targetLoopStatement) {
                llvm::BranchInst::Create(it->end, gIR->scopebb());
                return;
            }
        }
        assert(0);
    }
    else {
        emit_finallyblocks(p, enclosingtryfinally, gIR->loopbbs.back().enclosingtryfinally);
        llvm::BranchInst::Create(gIR->loopbbs.back().end, gIR->scopebb());
    }
}

//////////////////////////////////////////////////////////////////////////////

void ContinueStatement::toIR(IRState* p)
{
    Logger::println("ContinueStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (ident != 0) {
        Logger::println("ident = %s", ident->toChars());

        emit_finallyblocks(p, enclosingtryfinally, target->enclosingtryfinally);

        // get the loop statement the label refers to
        Statement* targetLoopStatement = target->statement;
        ScopeStatement* tmp;
        while(tmp = targetLoopStatement->isScopeStatement())
            targetLoopStatement = tmp->statement;

        // find the right continue block and jump there
        IRState::LoopScopeVec::reverse_iterator it;
        for(it = gIR->loopbbs.rbegin(); it != gIR->loopbbs.rend(); ++it) {
            if(it->s == targetLoopStatement) {
                llvm::BranchInst::Create(it->begin, gIR->scopebb());
                return;
            }
        }
        assert(0);
    }
    else {
        emit_finallyblocks(p, enclosingtryfinally, gIR->loopbbs.back().enclosingtryfinally);
        llvm::BranchInst::Create(gIR->loopbbs.back().begin, gIR->scopebb());
    }
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

    // create basic blocks
    llvm::BasicBlock* oldend = p->scopeend();

    llvm::BasicBlock* trybb = llvm::BasicBlock::Create("try", p->topfunc(), oldend);
    llvm::BasicBlock* finallybb = llvm::BasicBlock::Create("finally", p->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("endtryfinally", p->topfunc(), oldend);

    // pass the previous BB into this
    assert(!gIR->scopereturned());
    llvm::BranchInst::Create(trybb, p->scopebb());

    // do the try block
    p->scope() = IRScope(trybb,finallybb);

    assert(body);
    body->toIR(p);

    // terminate try BB
    if (!p->scopereturned())
        llvm::BranchInst::Create(finallybb, p->scopebb());

    // do finally block
    p->scope() = IRScope(finallybb,endbb);
    assert(finalbody);
    finalbody->toIR(p);

    // terminate finally
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

    Logger::attention(loc, "try-catch is not yet fully implemented");

    // create basic blocks
    llvm::BasicBlock* oldend = p->scopeend();

    llvm::BasicBlock* trybb = llvm::BasicBlock::Create("try", p->topfunc(), oldend);
    llvm::BasicBlock* catchbb = llvm::BasicBlock::Create("catch", p->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("endtrycatch", p->topfunc(), oldend);

    // pass the previous BB into this
    assert(!gIR->scopereturned());
    llvm::BranchInst::Create(trybb, p->scopebb());

    // do the try block
    p->scope() = IRScope(trybb,catchbb);
    assert(body);
    body->toIR(p);

    if (!gIR->scopereturned())
        llvm::BranchInst::Create(endbb, p->scopebb());

    // do catch
    p->scope() = IRScope(catchbb,oldend);
    llvm::BranchInst::Create(endbb, p->scopebb());
    /*assert(catches);
    for(size_t i=0; i<catches->dim; ++i)
    {
        Catch* c = (Catch*)catches->data[i];
        c->handler->toIR(p);
    }*/

    // rewrite the scope
    p->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void ThrowStatement::toIR(IRState* p)
{
    Logger::println("ThrowStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    Logger::attention(loc, "throw is not yet fully implemented");

    assert(exp);
    DValue* e = exp->toElem(p);
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_throw_exception");
    //Logger::cout() << "calling: " << *fn << '\n';
    LLValue* arg = DtoBitCast(e->getRVal(), fn->getFunctionType()->getParamType(0));
    //Logger::cout() << "arg: " << *arg << '\n';
    gIR->ir->CreateCall(fn, arg, "");
    gIR->ir->CreateUnreachable();
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
    Type* dt = DtoDType(e->type);
    Type* dtnext = DtoDType(dt->next);
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

    Logger::cout() << *table->getType() << '\n';
    Logger::cout() << *fn->getFunctionType()->getParamType(0) << '\n';
    assert(table->getType() == fn->getFunctionType()->getParamType(0));

    DValue* val = e->toElem(gIR);
    LLValue* llval;
    if (DSliceValue* sval = val->isSlice())
    {
        // give storage
        llval = new llvm::AllocaInst(DtoType(e->type), "tmp", gIR->topallocapoint());
        DVarValue* vv = new DVarValue(e->type, llval, true);
        DtoAssign(vv, val);
    }
    else
    {
        llval = val->getRVal();
    }
    assert(llval->getType() == fn->getFunctionType()->getParamType(1));

    return gIR->ir->CreateCall2(fn, table, llval, "tmp");
}

void SwitchStatement::toIR(IRState* p)
{
    Logger::println("SwitchStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    llvm::BasicBlock* oldend = gIR->scopeend();

    // collect the needed cases
    typedef std::pair<llvm::BasicBlock*, std::vector<llvm::ConstantInt*> > CasePair;
    std::vector<CasePair> vcases;
    std::vector<Statement*> vbodies;
    Array caseArray;
    for (int i=0; i<cases->dim; ++i)
    {
        CaseStatement* cs = (CaseStatement*)cases->data[i];

        std::string lblname("case");
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(lblname, p->topfunc(), oldend);
        cs->bodyBB = bb;

        std::vector<llvm::ConstantInt*> tmp;
        CaseStatement* last;
        bool first = true;
        do {
            // integral case
            if (cs->exp->type->isintegral()) {
                LLConstant* c = cs->exp->toConstElem(p);
                tmp.push_back(isaConstantInt(c));
            }
            // string case
            else {
                assert(cs->exp->op == TOKstring);
                // for string switches this is unfortunately necessary or there will be duplicates in the list
                if (first) {
                    caseArray.push(new Case((StringExp*)cs->exp, i));
                    first = false;
                }
            }
            last = cs;
        }
        while (cs = cs->statement->isCaseStatement());

        vcases.push_back(CasePair(bb, tmp));
        vbodies.push_back(last->statement);
    }

    // string switch?
    llvm::GlobalVariable* switchTable = 0;
    if (!condition->type->isintegral())
    {
        // first sort it
        caseArray.sort();
        // iterate and add indices to cases
        std::vector<LLConstant*> inits;
        for (size_t i=0; i<caseArray.dim; ++i)
        {
            Case* c = (Case*)caseArray.data[i];
            vcases[c->index].second.push_back(DtoConstUint(i));
            inits.push_back(c->str->toConstElem(p));
        }
        // build static array for ptr or final array
        const LLType* elemTy = DtoType(condition->type);
        const llvm::ArrayType* arrTy = llvm::ArrayType::get(elemTy, inits.size());
        LLConstant* arrInit = llvm::ConstantArray::get(arrTy, inits);
        llvm::GlobalVariable* arr = new llvm::GlobalVariable(arrTy, true, llvm::GlobalValue::InternalLinkage, arrInit, "string_switch_table_data", gIR->module);

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

        switchTable = new llvm::GlobalVariable(sTy, true, llvm::GlobalValue::InternalLinkage, sInit, "string_switch_table", gIR->module);
    }

    // default
    llvm::BasicBlock* defbb = 0;
    if (!hasNoDefault) {
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

    // add the cases
    size_t n = vcases.size();
    for (size_t i=0; i<n; ++i)
    {
        size_t nc = vcases[i].second.size();
        for (size_t j=0; j<nc; ++j)
        {
            si->addCase(vcases[i].second[j], vcases[i].first);
        }
    }

    // insert case statements
    for (size_t i=0; i<n; ++i)
    {
        llvm::BasicBlock* nextbb = (i == n-1) ? (defbb ? defbb : endbb) : vcases[i+1].first;
        p->scope() = IRScope(vcases[i].first,nextbb);
        p->loopbbs.push_back(IRLoopScope(this,enclosingtryfinally,p->scopebb(),endbb));
        vbodies[i]->toIR(p);
        p->loopbbs.pop_back();

        llvm::BasicBlock* curbb = p->scopebb();
        if (curbb->empty() || !curbb->back().isTerminator())
        {
            llvm::BranchInst::Create(nextbb, curbb);
        }
    }

    // default statement
    if (defbb)
    {
        p->scope() = IRScope(defbb,endbb);
        p->loopbbs.push_back(IRLoopScope(this,enclosingtryfinally,p->scopebb(),endbb));
        Logger::println("doing default statement");
        sdefault->statement->toIR(p);
        p->loopbbs.pop_back();

        llvm::BasicBlock* curbb = p->scopebb();
        if (curbb->empty() || !curbb->back().isTerminator())
        {
            llvm::BranchInst::Create(endbb, curbb);
        }
    }

    gIR->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////
void CaseStatement::toIR(IRState* p)
{
    Logger::println("CaseStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    assert(0);
}

//////////////////////////////////////////////////////////////////////////////

void UnrolledLoopStatement::toIR(IRState* p)
{
    Logger::println("UnrolledLoopStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("unrolledend", p->topfunc(), oldend);

    p->scope() = IRScope(p->scopebb(),endbb);
    p->loopbbs.push_back(IRLoopScope(this,enclosingtryfinally,p->scopebb(),endbb));

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

    //assert(arguments->dim == 1);
    assert(value != 0);
    assert(body != 0);
    assert(aggr != 0);
    assert(func != 0);

    //Argument* arg = (Argument*)arguments->data[0];
    //Logger::println("Argument is %s", arg->toChars());

    Logger::println("aggr = %s", aggr->toChars());

    // key
    const LLType* keytype = key ? DtoType(key->type) : DtoSize_t();
    LLValue* keyvar = new llvm::AllocaInst(keytype, "foreachkey", p->topallocapoint());
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
        valvar = new llvm::AllocaInst(valtype, "foreachval", p->topallocapoint());
    if (!value->ir.irLocal)
        value->ir.irLocal = new IrLocal(value);

    // what to iterate
    DValue* aggrval = aggr->toElem(p);
    Type* aggrtype = DtoDType(aggr->type);

    // get length and pointer
    LLValue* val = 0;
    LLValue* niters = 0;

    // static array
    if (aggrtype->ty == Tsarray)
    {
        Logger::println("foreach over static array");
        val = aggrval->getRVal();
        assert(isaPointer(val->getType()));
        const llvm::ArrayType* arrty = isaArray(val->getType()->getContainedType(0));
        assert(arrty);
        size_t nelems = arrty->getNumElements();
        if(nelems == 0)
            return;
        niters = llvm::ConstantInt::get(keytype,nelems,false);
    }
    // dynamic array
    else if (aggrtype->ty == Tarray)
    {
        if (DSliceValue* slice = aggrval->isSlice()) {
            Logger::println("foreach over slice");
            niters = slice->len;
            assert(niters);
            val = slice->ptr;
            assert(val);
        }
        else {
            Logger::println("foreach over dynamic array");
            val = aggrval->getRVal();
            niters = DtoGEPi(val,0,0);
            niters = DtoLoad(niters, "numiterations");
            val = DtoGEPi(val,0,1);
            val = DtoLoad(val, "collection");
        }
    }
    else
    {
        assert(0 && "aggregate type is not Tarray or Tsarray");
    }

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
        done = new llvm::ICmpInst(llvm::ICmpInst::ICMP_ULT, load, niters, "tmp", p->scopebb());
    }
    else if (op == TOKforeach_reverse) {
        done = new llvm::ICmpInst(llvm::ICmpInst::ICMP_UGT, load, zerokey, "tmp", p->scopebb());
        load = llvm::BinaryOperator::createSub(load,llvm::ConstantInt::get(keytype, 1, false),"tmp",p->scopebb());
        new llvm::StoreInst(load, keyvar, p->scopebb());
    }
    llvm::BranchInst::Create(bodybb, endbb, done, p->scopebb());

    // init body
    p->scope() = IRScope(bodybb,nextbb);

    // get value for this iteration
    LLConstant* zero = llvm::ConstantInt::get(keytype,0,false);
    LLValue* loadedKey = p->ir->CreateLoad(keyvar,"tmp");
    if (aggrtype->ty == Tsarray)
        value->ir.irLocal->value = DtoGEP(val,zero,loadedKey,"tmp");
    else if (aggrtype->ty == Tarray)
        value->ir.irLocal->value = llvm::GetElementPtrInst::Create(val,loadedKey,"tmp",p->scopebb());

    if (!value->isRef() && !value->isOut()) {
        DValue* dst = new DVarValue(value->type, valvar, true);
        DValue* src = new DVarValue(value->type, value->ir.irLocal->value, true);
        DtoAssign(dst, src);
        value->ir.irLocal->value = valvar;
    }

    // emit body
    p->loopbbs.push_back(IRLoopScope(this,enclosingtryfinally,nextbb,endbb));
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
        a->code = ".LDASM_";
        a->code += ident->toChars();
        a->code += ":";
        p->asmBlock->s.push_back(a);
        return;
    }

    assert(tf == NULL);

    llvm::BasicBlock* oldend = gIR->scopeend();
    if (llvmBB)
        llvmBB->moveBefore(oldend);
    else
        llvmBB = llvm::BasicBlock::Create("label", p->topfunc(), oldend);

    if (!p->scopereturned())
        llvm::BranchInst::Create(llvmBB, p->scopebb());

    p->scope() = IRScope(llvmBB,oldend);
    if (statement)
        statement->toIR(p);
}

//////////////////////////////////////////////////////////////////////////////

void GotoStatement::toIR(IRState* p)
{
    Logger::println("GotoStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    assert(tf == NULL);

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create("aftergoto", p->topfunc(), oldend);

    if (label->statement->llvmBB == NULL)
        label->statement->llvmBB = llvm::BasicBlock::Create("label", p->topfunc());
    assert(!p->scopereturned());

    // find finallys between goto and label
    TryFinallyStatement* endfinally = enclosingtryfinally;
    while(endfinally != NULL && endfinally != label->statement->enclosingtryfinally) {
        endfinally = endfinally->enclosingtryfinally;
    }

    // error if didn't find tf statement of label
    if(endfinally != label->statement->enclosingtryfinally)
        error("cannot goto into try block", loc.toChars());

    // emit code for finallys between goto and label
    emit_finallyblocks(p, enclosingtryfinally, endfinally);

    llvm::BranchInst::Create(label->statement->llvmBB, p->scopebb());
    p->scope() = IRScope(bb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void GotoDefaultStatement::toIR(IRState* p)
{
    Logger::println("GotoDefaultStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create("aftergotodefault", p->topfunc(), oldend);

    assert(!p->scopereturned());
    assert(sw->sdefault->bodyBB);

    emit_finallyblocks(p, enclosingtryfinally, sw->enclosingtryfinally);

    llvm::BranchInst::Create(sw->sdefault->bodyBB, p->scopebb());
    p->scope() = IRScope(bb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void GotoCaseStatement::toIR(IRState* p)
{
    Logger::println("GotoCaseStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* bb = llvm::BasicBlock::Create("aftergotocase", p->topfunc(), oldend);

    assert(!p->scopereturned());
    assert(cs->bodyBB);

    emit_finallyblocks(p, enclosingtryfinally, sw->enclosingtryfinally);

    llvm::BranchInst::Create(cs->bodyBB, p->scopebb());
    p->scope() = IRScope(bb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void WithStatement::toIR(IRState* p)
{
    Logger::println("WithStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    assert(exp);
    assert(body);

    DValue* e = exp->toElem(p);
    assert(!wthis->ir.isSet());
    wthis->ir.irLocal = new IrLocal(wthis);
    wthis->ir.irLocal->value = e->getRVal();

    body->toIR(p);
}

//////////////////////////////////////////////////////////////////////////////

void SynchronizedStatement::toIR(IRState* p)
{
    Logger::println("SynchronizedStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    Logger::attention(loc, "synchronized is currently ignored. only the body will be emitted");

    body->toIR(p);
}

//////////////////////////////////////////////////////////////////////////////

/* this has moved to asmstmt.cpp
void AsmStatement::toIR(IRState* p)
{
    Logger::println("AsmStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;
//    error("%s: inline asm is not yet implemented", loc.toChars());
//    fatal();

    assert(!asmcode && !asmalign && !refparam && !naked && !regs);

    Token* t = tokens;
    assert(t);

    std::string asmstr;

    do {
        Logger::println("token: %s", t->toChars());
        asmstr.append(t->toChars());
        asmstr.append(" ");
    } while (t = t->next);

    Logger::println("asm expr = '%s'", asmstr.c_str());

    // create function type
    std::vector<const LLType*> args;
    const llvm::FunctionType* fty = llvm::FunctionType::get(DtoSize_t(), args, false);

    // create inline asm callee
    llvm::InlineAsm* inasm = llvm::InlineAsm::get(fty, asmstr, "r,r", false);

    assert(0);
}
*/
//////////////////////////////////////////////////////////////////////////////

void VolatileStatement::toIR(IRState* p)
{
    Logger::println("VolatileStatement::toIR(): %s", loc.toChars());
    LOG_SCOPE;

    // mark in-volatile
    bool old = gIR->func()->inVolatile;
    gIR->func()->inVolatile = true;

    // has statement
    if (statement != NULL)
    {
        // load-store
        DtoMemoryBarrier(false, true, false, false);

        // do statement
        statement->toIR(p);

        // no point in a unreachable barrier, terminating statements should insert this themselves.
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
    gIR->func()->inVolatile = old;
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
STUBST(DefaultStatement);
//STUBST(CaseStatement);
//STUBST(SwitchStatement);
STUBST(SwitchErrorStatement);
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
