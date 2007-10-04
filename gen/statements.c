// Statements: D -> LLVM glue

#include <stdio.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <iostream>

#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/CallingConv.h"

#include "total.h"
#include "init.h"
#include "symbol.h"
#include "mtype.h"
#include "hdrgen.h"
#include "port.h"

#include "gen/irstate.h"
#include "gen/elem.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/runtime.h"
#include "gen/arrays.h"

//////////////////////////////////////////////////////////////////////////////

void CompoundStatement::toIR(IRState* p)
{
    static int csi = 0;
    Logger::println("CompoundStatement::toIR(%d):\n<<<\n%s>>>", csi++, toChars());
    LOG_SCOPE;

    /*
    const char* labelname;
    bool insterm = false;

    if (!p->scopes()) {
        labelname = "bb";
        insterm = true;
    }
    else
        labelname = "entry";

    //if (!llvm::isa<llvm::TerminatorInst>(p->topfunc()->back().back()))
    //    insterm = true;

    llvm::BasicBlock* bb = new llvm::BasicBlock(labelname, p->topfunc());

    if (insterm) {
        new llvm::BranchInst(bb,p->topbb());
    }

    p->bbs.push(bb);
    */

    size_t n = statements->dim;
    for (size_t i=0; i<n; i++)
    {
        Statement* s = (Statement*)statements->data[i];
        if (s)
        s->toIR(p);
        else
        Logger::println("NULL statement found in CompoundStatement !! :S");
    }

    //p->bbs.pop();
}

//////////////////////////////////////////////////////////////////////////////

void ReturnStatement::toIR(IRState* p)
{
    static int rsi = 0;
    Logger::println("ReturnStatement::toIR(%d): %s", rsi++, toChars());
    LOG_SCOPE;

    if (exp)
    {
        TY expty = exp->type->ty;
        if (p->topfunc()->getReturnType() == llvm::Type::VoidTy) {
            assert(expty == Tstruct || expty == Tdelegate || expty == Tarray);

            TypeFunction* f = p->topfunctype();
            assert(f->llvmRetInPtr && f->llvmRetArg);

            p->lvals.push_back(f->llvmRetArg);
            elem* e = exp->toElem(p);
            p->lvals.pop_back();

            // structliterals do this themselves
            // also they dont produce any value
            if (expty == Tstruct) {
                if (!e->inplace) {
                    TypeStruct* ts = (TypeStruct*)exp->type;
                    assert(e->mem);
                    LLVM_DtoStructCopy(ts,f->llvmRetArg,e->mem);
                }
            }
            else if (expty == Tdelegate) {
                // do nothing, handled by the DelegateExp
                LLVM_DtoDelegateCopy(f->llvmRetArg,e->mem);
            }
            else if (expty == Tarray) {
                if (e->type == elem::SLICE) {
                    LLVM_DtoSetArray(f->llvmRetArg,e->arg,e->mem);
                }
                // else the return value is a variable and should already have been assigned by now
            }
            else
            assert(0);

            new llvm::ReturnInst(p->scopebb());
            delete e;
        }
        else {
            elem* e = exp->toElem(p);
            llvm::Value* v = e->getValue();
            Logger::cout() << *v << '\n';
            new llvm::ReturnInst(v, p->scopebb());
            delete e;
        }
    }
    else
    {
        if (p->topfunc()->getReturnType() == llvm::Type::VoidTy)
            new llvm::ReturnInst(p->scopebb());
        else
            new llvm::UnreachableInst(p->scopebb());
    }

    p->scope().returned = true;
}

//////////////////////////////////////////////////////////////////////////////

void ExpStatement::toIR(IRState* p)
{
    static int esi = 0;
    Logger::println("ExpStatement::toIR(%d): %s", esi++, toChars());
    LOG_SCOPE;

    if (exp != 0) {
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
    static int wsi = 0;
    Logger::println("IfStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    elem* cond_e = condition->toElem(p);
    llvm::Value* cond_val = cond_e->getValue();
    delete cond_e;

    llvm::BasicBlock* oldend = gIR->scopeend();

    llvm::BasicBlock* ifbb = new llvm::BasicBlock("if", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = new llvm::BasicBlock("endif", gIR->topfunc(), oldend);
    llvm::BasicBlock* elsebb = 0;
    if (elsebody) {
        elsebb = new llvm::BasicBlock("else", gIR->topfunc(), endbb);
    }
    else {
        elsebb = endbb;
    }

    if (cond_val->getType() != llvm::Type::Int1Ty) {
        Logger::cout() << "if conditional: " << *cond_val << '\n';
        cond_val = LLVM_DtoBoolean(cond_val);
    }
    llvm::Value* ifgoback = new llvm::BranchInst(ifbb, elsebb, cond_val, gIR->scopebegin());

    // replace current scope
    gIR->scope() = IRScope(ifbb,elsebb);

    bool endifUsed = false;

    // do scoped statements
    ifbody->toIR(p);
    if (!gIR->scopereturned()) {
        new llvm::BranchInst(endbb,gIR->scopebegin());
        endifUsed = true;
    }

    if (elsebody) {
        //assert(0);
        gIR->scope() = IRScope(elsebb,endbb);
        elsebody->toIR(p);
        if (!gIR->scopereturned()) {
            new llvm::BranchInst(endbb,gIR->scopebegin());
            endifUsed = true;
        }
    }

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void ScopeStatement::toIR(IRState* p)
{
    Logger::println("ScopeStatement::toIR(): %s", toChars());
    LOG_SCOPE;

    llvm::BasicBlock* oldend = p->scopeend();

    IRScope irs;
    // remove useless branches by clearing and reusing the current basicblock
    llvm::BasicBlock* bb = p->scopebegin();
    if (bb->empty()) {
        irs.begin = bb;
    }
    else {
        irs.begin = new llvm::BasicBlock("scope", p->topfunc(), oldend);
        new llvm::BranchInst(irs.begin, p->scopebegin());
    }
    irs.end = new llvm::BasicBlock("endscope", p->topfunc(), oldend);

    gIR->scope() = irs;

    statement->toIR(p);

    p->scope() = IRScope(p->scopebb(),oldend);
    irs.end->eraseFromParent();

    /*
    if (!gIR->scopereturned()) {
        new llvm::BranchInst(irs.end, gIR->scopebegin());
    }

    // rewrite the scope
    gIR->scope() = IRScope(irs.end,oldend);
    */
}

//////////////////////////////////////////////////////////////////////////////

void WhileStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("WhileStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    // create while blocks
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* whilebb = new llvm::BasicBlock("whilecond", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = new llvm::BasicBlock("endwhile", gIR->topfunc(), oldend);

    // move into the while block
    new llvm::BranchInst(whilebb, gIR->scopebegin());

    // replace current scope
    gIR->scope() = IRScope(whilebb,endbb);

    // create the condition
    elem* cond_e = condition->toElem(p);
    llvm::Value* cond_val = LLVM_DtoBoolean(cond_e->getValue());
    delete cond_e;

    // while body block
    llvm::BasicBlock* whilebodybb = new llvm::BasicBlock("whilebody", gIR->topfunc(), endbb);

    // conditional branch
    llvm::Value* ifbreak = new llvm::BranchInst(whilebodybb, endbb, cond_val, whilebb);

    // rewrite scope
    gIR->scope() = IRScope(whilebodybb,endbb);

    // do while body code
    body->toIR(p);

    // loop
    new llvm::BranchInst(whilebb, gIR->scopebegin());

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void DoStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("DoStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    // create while blocks
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* dowhilebb = new llvm::BasicBlock("dowhile", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = new llvm::BasicBlock("enddowhile", gIR->topfunc(), oldend);

    // move into the while block
    new llvm::BranchInst(dowhilebb, gIR->scopebegin());

    // replace current scope
    gIR->scope() = IRScope(dowhilebb,endbb);

    // do do-while body code
    body->toIR(p);

    // create the condition
    elem* cond_e = condition->toElem(p);
    llvm::Value* cond_val = LLVM_DtoBoolean(cond_e->getValue());
    delete cond_e;

    // conditional branch
    llvm::Value* ifbreak = new llvm::BranchInst(dowhilebb, endbb, cond_val, gIR->scopebegin());

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void ForStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("ForStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    // create for blocks
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* forbb = new llvm::BasicBlock("forcond", gIR->topfunc(), oldend);
    llvm::BasicBlock* forbodybb = new llvm::BasicBlock("forbody", gIR->topfunc(), oldend);
    llvm::BasicBlock* forincbb = new llvm::BasicBlock("forinc", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = new llvm::BasicBlock("endfor", gIR->topfunc(), oldend);

    // init
    if (init != 0)
    init->toIR(p);

    // move into the for condition block, ie. start the loop
    new llvm::BranchInst(forbb, gIR->scopebegin());

    IRScope loop;
    loop.begin = forincbb;
    loop.end = endbb;
    p->loopbbs.push_back(loop);

    // replace current scope
    gIR->scope() = IRScope(forbb,forbodybb);

    // create the condition
    elem* cond_e = condition->toElem(p);
    llvm::Value* cond_val = LLVM_DtoBoolean(cond_e->getValue());
    delete cond_e;

    // conditional branch
    llvm::Value* ifbreak = new llvm::BranchInst(forbodybb, endbb, cond_val, forbb);

    // rewrite scope
    gIR->scope() = IRScope(forbodybb,forincbb);

    // do for body code
    body->toIR(p);

    // move into the for increment block
    new llvm::BranchInst(forincbb, gIR->scopebegin());
    gIR->scope() = IRScope(forincbb, endbb);

    // increment
    if (increment) {
        elem* inc = increment->toElem(p);
        delete inc;
    }

    // loop
    new llvm::BranchInst(forbb, gIR->scopebegin());

    p->loopbbs.pop_back();

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void BreakStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("BreakStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    if (ident != 0) {
        Logger::println("ident = %s", ident->toChars());
        assert(0);
    }
    else {
        new llvm::BranchInst(gIR->loopbbs.back().end, gIR->scopebegin());
    }
}

//////////////////////////////////////////////////////////////////////////////

void ContinueStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("ContinueStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    if (ident != 0) {
        Logger::println("ident = %s", ident->toChars());
        assert(0);
    }
    else {
        new llvm::BranchInst(gIR->loopbbs.back().begin, gIR->scopebegin());
    }
}

//////////////////////////////////////////////////////////////////////////////

void OnScopeStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("OnScopeStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    assert(statement);
    //statement->toIR(p); // this seems to be redundant
}

//////////////////////////////////////////////////////////////////////////////

void TryFinallyStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("TryFinallyStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    llvm::BasicBlock* oldend = gIR->scopeend();

    llvm::BasicBlock* trybb = new llvm::BasicBlock("try", gIR->topfunc(), oldend);
    llvm::BasicBlock* finallybb = new llvm::BasicBlock("finally", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = new llvm::BasicBlock("endtryfinally", gIR->topfunc(), oldend);

    // pass the previous BB into this
    new llvm::BranchInst(trybb, gIR->scopebegin());

    gIR->scope() = IRScope(trybb,finallybb);

    assert(body);
    body->toIR(p);
    new llvm::BranchInst(finallybb, gIR->scopebegin());

    // rewrite the scope
    gIR->scope() = IRScope(finallybb,endbb);

    assert(finalbody);
    finalbody->toIR(p);
    new llvm::BranchInst(endbb, gIR->scopebegin());

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void TryCatchStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("TryCatchStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    assert(0 && "try-catch is not properly");

    assert(body);
    body->toIR(p);

    assert(catches);
    for(size_t i=0; i<catches->dim; ++i)
    {
        Catch* c = (Catch*)catches->data[i];
        c->handler->toIR(p);
    }
}

//////////////////////////////////////////////////////////////////////////////

void ThrowStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("ThrowStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    assert(0 && "throw is not implemented");

    assert(exp);
    elem* e = exp->toElem(p);
    delete e;
}

//////////////////////////////////////////////////////////////////////////////

void SwitchStatement::toIR(IRState* p)
{
    Logger::println("SwitchStatement::toIR(): %s", toChars());
    LOG_SCOPE;

    llvm::BasicBlock* oldend = gIR->scopeend();

    // collect the needed cases
    typedef std::pair<llvm::BasicBlock*, llvm::ConstantInt*> CasePair;
    std::vector<CasePair> vcases;
    for (int i=0; i<cases->dim; ++i)
    {
        CaseStatement* cs = (CaseStatement*)cases->data[i];

        // get the case value
        elem* e = cs->exp->toElem(p);
        assert(e->val && llvm::isa<llvm::ConstantInt>(e->val));
        llvm::ConstantInt* ec = llvm::cast<llvm::ConstantInt>(e->val);
        delete e;

        // create the case bb with a nice label
        std::string lblname("case"+std::string(cs->exp->toChars()));
        llvm::BasicBlock* bb = new llvm::BasicBlock(lblname, p->topfunc(), oldend);

        vcases.push_back(CasePair(bb,ec));
    }

    // default
    llvm::BasicBlock* defbb = 0;
    if (!hasNoDefault) {
        defbb = new llvm::BasicBlock("default", p->topfunc(), oldend);
    }

    // end (break point)
    llvm::BasicBlock* endbb = new llvm::BasicBlock("switchend", p->topfunc(), oldend);

    // condition var
    elem* cond = condition->toElem(p);
    llvm::SwitchInst* si = new llvm::SwitchInst(cond->getValue(), defbb ? defbb : endbb, cases->dim, p->scopebb());
    delete cond;

    // add the cases
    size_t n = vcases.size();
    for (size_t i=0; i<n; ++i)
    {
        si->addCase(vcases[i].second, vcases[i].first);
    }

    // insert case statements
    for (size_t i=0; i<n; ++i)
    {
        llvm::BasicBlock* nextbb = (i == n-1) ? (defbb ? defbb : endbb) : vcases[i+1].first;
        p->scope() = IRScope(vcases[i].first,nextbb);

        p->loopbbs.push_back(IRScope(p->scopebb(),endbb));
        static_cast<CaseStatement*>(cases->data[i])->statement->toIR(p);
        p->loopbbs.pop_back();

        llvm::BasicBlock* curbb = p->scopebb();
        if (!curbb->empty() && !curbb->back().isTerminator())
        {
            new llvm::BranchInst(nextbb, curbb);
        }
    }

    // default statement
    if (defbb)
    {
        p->scope() = IRScope(defbb,endbb);
        p->loopbbs.push_back(IRScope(defbb,endbb));
        sdefault->statement->toIR(p);
        p->loopbbs.pop_back();

        llvm::BasicBlock* curbb = p->scopebb();
        if (!curbb->empty() && !curbb->back().isTerminator())
        {
            new llvm::BranchInst(endbb, curbb);
        }
    }

    gIR->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void UnrolledLoopStatement::toIR(IRState* p)
{
    Logger::println("UnrolledLoopStatement::toIR(): %s", toChars());
    LOG_SCOPE;

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* endbb = new llvm::BasicBlock("unrolledend", p->topfunc(), oldend);

    p->scope() = IRScope(p->scopebb(),endbb);
    p->loopbbs.push_back(IRScope(p->scopebb(),endbb));

    for (int i=0; i<statements->dim; ++i)
    {
        Statement* s = (Statement*)statements->data[i];
        s->toIR(p);
    }

    p->loopbbs.pop_back();

    new llvm::BranchInst(endbb, p->scopebb());
    p->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

void ForeachStatement::toIR(IRState* p)
{
    Logger::println("ForeachStatement::toIR(): %s", toChars());
    LOG_SCOPE;

    //assert(arguments->dim == 1);
    assert(key == 0);
    assert(value != 0);
    assert(body != 0);
    assert(aggr != 0);
    assert(func != 0);

    //Argument* arg = (Argument*)arguments->data[0];
    //Logger::println("Argument is %s", arg->toChars());

    Logger::println("aggr = %s", aggr->toChars());
    Logger::println("func = %s", func->toChars());

    elem* arr = aggr->toElem(p);
    llvm::Value* val = arr->getValue();
    Logger::cout() << "aggr2llvm = " << *val << '\n';

    llvm::Value* numiters = 0;

    const llvm::Type* keytype = key ? LLVM_DtoType(key->type) : LLVM_DtoSize_t();
    llvm::Value* keyvar = new llvm::AllocaInst(keytype, "foreachkey", p->topallocapoint());

    const llvm::Type* valtype = LLVM_DtoType(value->type);
    llvm::Value* valvar = new llvm::AllocaInst(keytype, "foreachval", p->topallocapoint());

    if (aggr->type->ty == Tsarray)
    {
        assert(llvm::isa<llvm::PointerType>(val->getType()));
        assert(llvm::isa<llvm::ArrayType>(val->getType()->getContainedType(0)));
        size_t n = llvm::cast<llvm::ArrayType>(val->getType()->getContainedType(0))->getNumElements();
        assert(n > 0);
        numiters = llvm::ConstantInt::get(keytype,n,false); 
    }
    else
    {
        assert(0);
    }

    if (op == TOKforeach) {
        new llvm::StoreInst(llvm::ConstantInt::get(keytype,0,false), keyvar, p->scopebb());
    }
    else if (op == TOKforeach_reverse) {
        llvm::Value* v = llvm::BinaryOperator::createSub(numiters, llvm::ConstantInt::get(keytype,1,false),"tmp",p->scopebb());
        new llvm::StoreInst(v, keyvar, p->scopebb());
    }

    delete arr;

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* nexbb = new llvm::BasicBlock("foreachnext", p->topfunc(), oldend);
    llvm::BasicBlock* begbb = new llvm::BasicBlock("foreachbegin", p->topfunc(), oldend);
    llvm::BasicBlock* endbb = new llvm::BasicBlock("foreachend", p->topfunc(), oldend);

    new llvm::BranchInst(begbb, p->scopebb());

    // next
    p->scope() = IRScope(nexbb,begbb);
    llvm::Value* done = 0;
    llvm::Value* load = new llvm::LoadInst(keyvar, "tmp", p->scopebb());
    if (op == TOKforeach) {
        load = llvm::BinaryOperator::createAdd(load,llvm::ConstantInt::get(keytype, 1, false),"tmp",p->scopebb());
        new llvm::StoreInst(load, keyvar, p->scopebb());
        done = new llvm::ICmpInst(llvm::ICmpInst::ICMP_ULT, load, numiters, "tmp", p->scopebb());
    }
    else if (op == TOKforeach_reverse) {
        done = new llvm::ICmpInst(llvm::ICmpInst::ICMP_UGT, load, llvm::ConstantInt::get(keytype, 0, false), "tmp", p->scopebb());
        load = llvm::BinaryOperator::createSub(load,llvm::ConstantInt::get(keytype, 1, false),"tmp",p->scopebb());
        new llvm::StoreInst(load, keyvar, p->scopebb());
    }
    new llvm::BranchInst(begbb, endbb, done, p->scopebb());

    // begin
    p->scope() = IRScope(begbb,nexbb);

    // get value for this iteration
    value->llvmValue = LLVM_DtoGEP(val,llvm::ConstantInt::get(keytype,0,false),new llvm::LoadInst(keyvar,"tmp",p->scopebb()),"tmp",p->scopebb());

    // body
    p->scope() = IRScope(p->scopebb(),endbb);
    p->loopbbs.push_back(IRScope(nexbb,endbb));
    body->toIR(p);
    p->loopbbs.pop_back();

    if (!p->scope().returned)
        new llvm::BranchInst(nexbb, p->scopebb());

    // end
    p->scope() = IRScope(endbb,oldend);
}

//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////

#define STUBST(x) void x::toIR(IRState * p) {error("Statement type "#x" not implemented: %s", toChars());fatal();}
//STUBST(BreakStatement);
//STUBST(ForStatement);
STUBST(WithStatement);
STUBST(SynchronizedStatement);
//STUBST(ReturnStatement);
//STUBST(ContinueStatement);
STUBST(DefaultStatement);
STUBST(CaseStatement);
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
STUBST(AsmStatement);
//STUBST(TryCatchStatement);
//STUBST(TryFinallyStatement);
STUBST(VolatileStatement);
STUBST(LabelStatement);
//STUBST(ThrowStatement);
STUBST(GotoCaseStatement);
STUBST(GotoDefaultStatement);
STUBST(GotoStatement);
//STUBST(UnrolledLoopStatement);
//STUBST(OnScopeStatement);
